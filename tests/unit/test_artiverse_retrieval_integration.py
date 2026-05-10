"""Focused tests for source-constrained Artiverse articulated retrieval."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from omegaconf import OmegaConf

from scenesmith.agent_utils import asset_manager as asset_manager_module
from scenesmith.agent_utils.articulated_retrieval_server.config import (
    ArticulatedConfig,
)
from scenesmith.agent_utils.articulated_retrieval_server.data_loader import (
    ArticulatedMeshMetadata,
    ArticulatedPreprocessedData,
)
from scenesmith.agent_utils.articulated_retrieval_server.dataclasses import (
    ArticulatedRetrievalServerRequest,
)
from scenesmith.agent_utils.articulated_retrieval_server.retrieval import (
    ArticulatedRetriever,
)
from scenesmith.agent_utils.asset_router.dataclasses import AssetItem
from scenesmith.agent_utils.asset_router.router import AssetRouter
from scenesmith.agent_utils.asset_manager import (
    AssetManager,
    _articulated_provenance,
)
from scenesmith.agent_utils.asset_registry import AssetRegistry
from scenesmith.agent_utils.artiverse_visual_normalization import (
    sha256_safe_directory_tree,
)
from scenesmith.agent_utils.room import ObjectType


def _metadata(object_id: str, source: str) -> ArticulatedMeshMetadata:
    return ArticulatedMeshMetadata(
        object_id=object_id,
        source=source,
        category="cabinet",
        description=f"{source} openable cabinet",
        is_manipuland=False,
        placement_type="floor",
        sdf_path=Path(f"/{source}/{object_id}.sdf"),
        bounding_box_min=[0.0, 0.0, 0.0],
        bounding_box_max=[1.0, 0.5, 1.8],
    )


def _mixed_data() -> ArticulatedPreprocessedData:
    artiverse_id = "artiverse/cabinet/fpModel/100"
    artvip_id = "artvip/large_furniture/200"
    return ArticulatedPreprocessedData(
        metadata_by_id={
            artiverse_id: _metadata(artiverse_id, "artiverse"),
            artvip_id: _metadata(artvip_id, "artvip"),
        },
        clip_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        embedding_index=[artiverse_id, artvip_id],
    )


def test_request_serializes_required_source() -> None:
    request = ArticulatedRetrievalServerRequest(
        object_description="openable classroom cabinet",
        object_type="FURNITURE",
        output_dir="/tmp/artiverse",
        required_source="artiverse",
    )

    assert request.to_dict()["required_source"] == "artiverse"


def test_preprocessed_filter_requires_artiverse_source() -> None:
    data = _mixed_data()

    indices = data.filter_by_object_type(
        object_type="FURNITURE", required_source="ARTIVERSE"
    )

    assert indices == [0]
    assert data.embedding_index[indices[0]].startswith("artiverse/")


def test_retriever_never_returns_other_source(monkeypatch) -> None:
    from scenesmith.agent_utils.articulated_retrieval_server import retrieval

    data = _mixed_data()
    retriever = ArticulatedRetriever(
        config=ArticulatedConfig(sources={}, use_top_k=2), clip_device="cpu"
    )
    retriever.preprocessed_data = data
    retriever._initialized = True

    monkeypatch.setattr(
        retrieval,
        "get_text_embedding",
        lambda description, device=None: np.asarray([1.0, 0.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        retrieval,
        "compute_clip_similarities",
        lambda query_embedding, embeddings, indices: {index: 0.9 for index in indices},
    )

    candidates = retriever.retrieve(
        description="openable classroom cabinet",
        object_type="FURNITURE",
        required_source="artiverse",
        top_k=2,
    )

    assert [candidate.source for candidate in candidates] == ["artiverse"]


def test_router_dispatches_artiverse_with_required_source() -> None:
    router = AssetRouter.__new__(AssetRouter)
    router.cfg = OmegaConf.create(
        {
            "asset_manager": {
                "router": {
                    "strategies": {
                        "artiverse_articulated": {
                            "enabled": True,
                            "max_retries": 2,
                        }
                    }
                }
            }
        }
    )
    captured = {}
    expected = object()

    def _fake_articulated(**kwargs):
        captured.update(kwargs)
        return expected

    router._try_articulated_strategy = _fake_articulated
    item = AssetItem(
        description="openable classroom storage cabinet with two doors",
        short_name="classroom_storage_cabinet",
        dimensions=[1.2, 0.5, 1.8],
        object_type=ObjectType.FURNITURE,
        strategies=["artiverse_articulated"],
    )

    result = router.generate_with_validation(
        item=item,
        geometry_client=None,
        image_generator=None,
        images_dir=None,
        geometry_dir=Path("."),
        debug_dir=Path("."),
        articulated_client=object(),
    )

    assert result is expected
    assert captured["required_source"] == "artiverse"


def test_copied_articulated_asset_records_content_bound_provenance(tmp_path) -> None:
    source_dir = tmp_path / "official" / "asset"
    copied_dir = tmp_path / "room" / "asset"
    source_dir.mkdir(parents=True)
    copied_dir.mkdir(parents=True)
    source_sdf = source_dir / "model.sdf"
    copied_sdf = copied_dir / "model.sdf"
    source_sdf.write_text("<sdf><model name='official'/></sdf>", encoding="utf-8")
    (source_dir / "mesh.glb").write_bytes(b"official mesh")
    copied_sdf.write_text(
        "<sdf><model name='official'><collision_filter_group/></model></sdf>",
        encoding="utf-8",
    )
    (copied_dir / "mesh.glb").write_bytes(b"official mesh")
    articulated = SimpleNamespace(
        source="artiverse",
        object_id="artiverse/cabinet/source/model",
        sdf_path=source_sdf,
    )

    evidence = _articulated_provenance(articulated, copied_sdf)

    assert evidence["articulated_source"] == "artiverse"
    assert evidence["articulated_id"] == articulated.object_id
    assert evidence["articulated_source_sdf_path"] == str(source_sdf.resolve())
    assert evidence["articulated_source_sdf_sha256"] != evidence[
        "articulated_copied_sdf_sha256"
    ]
    assert evidence["articulated_source_tree_sha256"] != evidence[
        "articulated_copied_tree_sha256"
    ]


def test_asset_manager_records_visual_normalization_in_artiverse_metadata(
    tmp_path, monkeypatch
) -> None:
    source_dir = tmp_path / "official" / "asset"
    source_dir.mkdir(parents=True)
    source_sdf = source_dir / "model.sdf"
    source_sdf.write_text("<sdf><model name='official'/></sdf>", encoding="utf-8")
    (source_dir / "mesh.bin").write_bytes(b"official-publisher-bytes")
    source_hash = sha256_safe_directory_tree(source_dir)

    manager = AssetManager.__new__(AssetManager)
    manager.geometry_dir = tmp_path / "geometry"
    manager.sdf_dir = tmp_path / "sdf"
    manager.geometry_dir.mkdir()
    manager.sdf_dir.mkdir()
    manager.registry = AssetRegistry()
    assert not hasattr(manager, "agent_type")
    manager.cfg = OmegaConf.create(
        {"asset_manager": {"articulated": {"enable_self_collision_filtering": False}}}
    )
    item = AssetItem(
        description="openable classroom storage cabinet",
        short_name="storage_cabinet",
        dimensions=[1.0, 0.5, 1.5],
        object_type=ObjectType.FURNITURE,
        strategies=["artiverse_articulated"],
    )
    articulated = SimpleNamespace(
        source="artiverse",
        object_id="artiverse/cabinet/source/model",
        sdf_path=source_sdf,
        item=item,
        bounding_box_min=[0.0, 0.0, 0.0],
        bounding_box_max=[1.0, 0.5, 1.5],
    )
    calls = []

    def normalize(copied_sdf, copied_root):
        evidence = {
            "schema_version": 2,
            "status": "pass",
            "policy": "publisher_glb_derived_external_gltf",
            "copied_tree_sha256_after": sha256_safe_directory_tree(copied_root),
        }
        calls.append((copied_sdf, copied_root, evidence))
        return evidence

    class CombinedMesh:
        def export(self, path):
            asset_manager_module.trimesh.creation.box(
                extents=[1.0, 0.5, 1.5]
            ).export(path)

    monkeypatch.setattr(
        asset_manager_module, "normalize_copied_artiverse_visuals", normalize
    )
    monkeypatch.setattr(
        asset_manager_module,
        "combine_sdf_meshes_at_joint_angles",
        lambda *_args, **_kwargs: CombinedMesh(),
    )
    monkeypatch.setattr(asset_manager_module.time, "time", lambda: 1234567890)

    scene_object = manager._convert_articulated_to_scene_object(
        articulated, SimpleNamespace(object_type=ObjectType.FURNITURE)
    )

    assert len(calls) == 1
    assert calls[0][0] == scene_object.sdf_path
    assert scene_object.metadata["artiverse_visual_normalization"] == calls[0][2]
    assert scene_object.metadata["articulated_copied_tree_sha256"] == calls[0][2][
        "copied_tree_sha256_after"
    ]
    assert scene_object.geometry_path.parent.parent == manager.geometry_dir
    assert sha256_safe_directory_tree(source_dir) == source_hash
