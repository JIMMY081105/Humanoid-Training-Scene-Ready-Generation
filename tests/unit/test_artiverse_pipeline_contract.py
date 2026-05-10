import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import artiverse_contract as artiverse_contract_module  # noqa: E402
from assemble_final_house_and_render import (  # noqa: E402
    _collect_artiverse_from_house_state,
    _collect_artiverse_from_room_states,
    _promote_combined_candidate,
    _require_artiverse_records,
    _verify_combined_survival,
    _write_artiverse_usage_manifest,
)
from artiverse_contract import (  # noqa: E402
    ArtiverseContractError,
    OFFICIAL_PACK_SCRIPT_SHA256,
    OFFICIAL_REPOSITORY,
    OFFICIAL_REVISION,
    OFFICIAL_SOURCE_MANIFEST_SHA256,
    load_artiverse_authority,
    require_regular_file_within,
    sha256_directory_tree,
    sha256_file,
    validate_regular_directory_tree,
    validate_sdf_resource_uris,
    validate_usage_manifest,
)
from run_single_room_worker import (  # noqa: E402
    REQUIRED_EMBEDDING_FILES,
    _asset_pipeline_overrides,
    _recover_final_blend,
    _start_worker_services,
    _stop_worker_services,
    _validate_asset_policy,
)
from validate_articulated_router import (  # noqa: E402
    _candidate_passes_source_contract,
    _source_preflight_errors,
    _source_status,
)


class Node(dict):
    """Small attribute-access mapping that behaves like the tested DictConfig nodes."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


TEST_SOURCE_MANIFEST_CONTENT = '{"official": true}\n'
TEST_PACK_SCRIPT_CONTENT = "# official unpacker\n"


@pytest.fixture(autouse=True)
def _bind_synthetic_metadata_to_test_pins(monkeypatch) -> None:
    """Keep synthetic fixtures valid while production constants stay immutable."""

    monkeypatch.setattr(
        artiverse_contract_module,
        "OFFICIAL_SOURCE_MANIFEST_SHA256",
        hashlib.sha256(TEST_SOURCE_MANIFEST_CONTENT.encode("utf-8")).hexdigest(),
    )
    monkeypatch.setattr(
        artiverse_contract_module,
        "OFFICIAL_PACK_SCRIPT_SHA256",
        hashlib.sha256(TEST_PACK_SCRIPT_CONTENT.encode("utf-8")).hexdigest(),
    )


def _touch_embedding_index(embeddings_dir: Path) -> None:
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_EMBEDDING_FILES:
        (embeddings_dir / name).write_bytes(b"test")


def _write_extraction_receipt(dataset_root: Path) -> Path:
    archives = []
    member_total = 0
    directory_total = 0
    for (
        archive,
        archive_bytes,
        archive_sha256,
        model_count,
        file_count,
        input_bytes,
    ) in artiverse_contract_module.OFFICIAL_EXTRACTION_ARCHIVES:
        member_count = model_count + file_count
        member_total += member_count
        directory_total += model_count
        archives.append(
            {
                "archive": archive,
                "archive_bytes": archive_bytes,
                "sha256": archive_sha256,
                "validated_member_count": member_count,
                "validated_directory_count": model_count,
                "validated_regular_file_count": file_count,
                "validated_uncompressed_bytes": input_bytes,
            }
        )
    extractor_path = SCRIPTS_DIR / artiverse_contract_module.SAFE_EXTRACTOR_FILENAME
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "manifest": {
            "path": "dataset_chunks/manifest.json",
            "sha256": artiverse_contract_module.OFFICIAL_SOURCE_MANIFEST_SHA256,
            "format": "artiverse-data-tar-gz-chunks-v1",
        },
        "archives": archives,
        "declared_roots": {
            "count": artiverse_contract_module.OFFICIAL_EXTRACTION_ROOT_COUNT,
            "sha256": artiverse_contract_module.OFFICIAL_EXTRACTION_ROOTS_SHA256,
            "hash_algorithm": artiverse_contract_module.RECEIPT_ROOT_HASH_ALGORITHM,
        },
        "validated_aggregate": {
            "member_count": member_total,
            "directory_count": directory_total,
            "regular_file_count": (
                artiverse_contract_module.OFFICIAL_EXTRACTION_FILE_COUNT
            ),
            "uncompressed_bytes": (
                artiverse_contract_module.OFFICIAL_EXTRACTION_INPUT_BYTES
            ),
        },
        "data_tree_inventory": {
            "directory_count": (
                artiverse_contract_module.OFFICIAL_EXTRACTION_ROOT_COUNT
            ),
            "regular_file_count": (
                artiverse_contract_module.OFFICIAL_EXTRACTION_FILE_COUNT
            ),
            "sha256": "a" * 64,
            "hash_algorithm": artiverse_contract_module.RECEIPT_TREE_HASH_ALGORITHM,
        },
        "safe_extractor": {
            "version": "1.0.0",
            "sha256": sha256_file(extractor_path),
            "filename": extractor_path.name,
        },
    }
    receipt_path = dataset_root / artiverse_contract_module.EXTRACTION_RECEIPT_FILENAME
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt_path


def _write_artiverse_authority(repo_dir: Path):
    dataset_root = (repo_dir / "data" / "artiverse").resolve()
    embeddings = dataset_root / "embeddings"
    model_dir = (
        dataset_root
        / "data"
        / "cabinet"
        / "fpModel"
        / "model_001"
        / "urdf_w_collider"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    embeddings.mkdir(parents=True, exist_ok=True)
    publisher_urdf = model_dir / "model_001.urdf"
    publisher_urdf.write_text(
        """<robot name="official">
  <link name="base">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
      <inertia ixx="0.2" iyy="0.2" izz="0.2" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
    <visual><geometry><mesh filename="mesh.glb"/></geometry></visual>
  </link>
  <link name="door">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.5"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision><geometry><box size="0.1 1 1"/></geometry></collision>
    <visual><geometry><box size="0.1 1 1"/></geometry></visual>
  </link>
  <joint name="door_hinge" type="revolute">
    <parent link="base"/><child link="door"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1.5" effort="100" velocity="5"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    source_sdf = model_dir / "scenesmith_artiverse.sdf"
    source_sdf.write_text(
        """<sdf version="1.7">
  <model name="official">
    <link name="base">
      <inertial>
        <pose>0 0 0 0 0 0</pose>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.2</ixx><iyy>0.2</iyy><izz>0.2</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="base_collision">
        <geometry><box><size>1 1 1</size></box></geometry>
        <surface><friction><ode><mu>0.5</mu><mu2>0.5</mu2></ode></friction></surface>
      </collision>
      <visual name="base_visual">
        <geometry><mesh><uri>mesh.glb</uri></mesh></geometry>
      </visual>
    </link>
    <link name="door">
      <inertial>
        <pose>0 0 0 0 0 0</pose>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="door_collision">
        <geometry><box><size>0.1 1 1</size></box></geometry>
        <surface><friction><ode><mu>0.5</mu><mu2>0.5</mu2></ode></friction></surface>
      </collision>
      <visual name="door_visual">
        <geometry><box><size>0.1 1 1</size></box></geometry>
      </visual>
    </link>
    <joint name="door_hinge" type="revolute">
      <parent>base</parent><child>door</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit><lower>0</lower><upper>1.5</upper></limit>
        <dynamics><damping>0.05</damping><friction>0.05</friction></dynamics>
      </axis>
    </joint>
  </model>
</sdf>
""",
        encoding="utf-8",
    )
    (model_dir / "mesh.glb").write_bytes(b"official-artiverse-mesh")
    source_manifest = dataset_root / "dataset_chunks" / "manifest.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_bytes(TEST_SOURCE_MANIFEST_CONTENT.encode("utf-8"))
    pack_script = dataset_root / "pack_dataset_chunks.py"
    pack_script.write_bytes(TEST_PACK_SCRIPT_CONTENT.encode("utf-8"))
    extraction_receipt = _write_extraction_receipt(dataset_root)

    object_id = "artiverse/cabinet/fpModel/model_001"
    publisher_physics = artiverse_contract_module.publisher_urdf_physics_evidence(
        publisher_urdf
    )
    emitted_physics = artiverse_contract_module.emitted_sdf_physics_evidence(source_sdf)
    metadata = {
        object_id: {
            "category": "cabinet",
            "description": "official cabinet",
            "is_manipuland": False,
            "placement_type": "floor",
            "sdf_path": str(source_sdf.relative_to(dataset_root)),
            "bounding_box_min": [0.0, 0.0, 0.0],
            "bounding_box_max": [1.0, 1.0, 1.0],
            "artiverse_source": "fpModel",
            "artiverse_model_id": "model_001",
            "collision_element_count": 2,
            "source_collision_element_count": 2,
            "physics_source": artiverse_contract_module.PHYSICS_POLICY_ID,
            "physics_link_count": 2,
            "physics_geometry_link_count": 2,
            "physics_total_mass_kg": 1.5,
            "publisher_urdf_path": str(publisher_urdf.relative_to(dataset_root)).replace(
                os.sep, "/"
            ),
            "publisher_urdf_sha256": sha256_file(publisher_urdf),
            "publisher_link_physics_sha256": publisher_physics.sha256,
            "emitted_sdf_physics_sha256": emitted_physics.sha256,
            "movable_joint_count": 1,
            "movable_joint_types": ["revolute"],
            "sdf_sha256": sha256_file(source_sdf),
            "sdf_directory_tree_sha256": sha256_directory_tree(model_dir),
        }
    }
    np.save(embeddings / "clip_embeddings.npy", np.zeros((1, 1024), dtype=np.float32))
    (embeddings / "embedding_index.yaml").write_text(
        yaml.safe_dump([object_id], sort_keys=False), encoding="utf-8"
    )
    (embeddings / "metadata_index.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "status": "pass",
        "source_repository": OFFICIAL_REPOSITORY,
        "source_revision": OFFICIAL_REVISION,
        "source_manifest": {
            "path": str(source_manifest.relative_to(dataset_root)),
            "sha256": sha256_file(source_manifest),
        },
        "source_pack_script_sha256": sha256_file(pack_script),
        "source_extraction_receipt": {
            "path": str(extraction_receipt.relative_to(dataset_root)),
            "sha256": sha256_file(extraction_receipt),
        },
        "safe_extractor_sha256": sha256_file(
            SCRIPTS_DIR / artiverse_contract_module.SAFE_EXTRACTOR_FILENAME
        ),
        "dataset_root": str(dataset_root),
        "output_path": str(embeddings),
        "indexed_count": 1,
        "minimum_indexed": 1,
        "maximum_collision_elements": 32,
        "physics_policy": {
            "id": artiverse_contract_module.PHYSICS_POLICY_ID,
            "required_for_every_link": True,
            "inertial_frame_transform": (
                artiverse_contract_module.INERTIAL_FRAME_TRANSFORM_POLICY
            ),
            "require_zero_emitted_inertial_rpy": True,
            "preserve_source_collision_count": True,
            "preconversion_collision_cap": True,
            "publisher_mass_semantics": (
                artiverse_contract_module.PUBLISHER_MASS_SEMANTICS
            ),
            "joint_dynamics_policy": (
                artiverse_contract_module.JOINT_DYNAMICS_POLICY
            ),
            "collision_friction_policy": (
                artiverse_contract_module.COLLISION_FRICTION_POLICY
            ),
        },
        "physics_bound_indexed_count": 1,
        "physics_binding_sha256": artiverse_contract_module.physics_binding_sha256(
            metadata, [object_id]
        ),
        "index_sha256": {
            filename: sha256_file(embeddings / filename)
            for filename in REQUIRED_EMBEDDING_FILES
        },
    }
    (embeddings / "artiverse_preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return load_artiverse_authority(dataset_root), object_id


def _rewrite_authority_evidence(
    authority,
    object_id: str,
    *,
    record_updates: dict | None = None,
) -> None:
    """Rebind synthetic evidence so a negative reaches the intended invariant."""

    metadata_path = authority.embeddings_path / "metadata_index.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    record = metadata[object_id]
    source_sdf = authority.dataset_root / str(record["sdf_path"])
    record["sdf_sha256"] = sha256_file(source_sdf)
    record["sdf_directory_tree_sha256"] = sha256_directory_tree(source_sdf.parent)
    publisher_urdf = authority.dataset_root / str(record["publisher_urdf_path"])
    record["publisher_urdf_sha256"] = sha256_file(publisher_urdf)
    try:
        record["publisher_link_physics_sha256"] = (
            artiverse_contract_module.publisher_urdf_physics_evidence(
                publisher_urdf
            ).sha256
        )
    except ArtiverseContractError:
        pass
    try:
        record["emitted_sdf_physics_sha256"] = (
            artiverse_contract_module.emitted_sdf_physics_evidence(source_sdf).sha256
        )
    except ArtiverseContractError:
        pass
    if record_updates:
        record.update(record_updates)
    metadata_path.write_text(
        yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8"
    )

    preparation_path = authority.preparation_manifest_path
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["physics_binding_sha256"] = (
        artiverse_contract_module.physics_binding_sha256(metadata, [object_id])
    )
    preparation["index_sha256"] = {
        filename: sha256_file(authority.embeddings_path / filename)
        for filename in REQUIRED_EMBEDDING_FILES
    }
    preparation_path.write_text(
        json.dumps(preparation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _agent(
    *,
    source: str,
    artvip_enabled: bool,
    artiverse_enabled: bool = False,
    artiverse_route_enabled: bool = False,
) -> Node:
    sources = Node(
        artvip=Node(
            enabled=artvip_enabled,
            data_path="data/artvip_sdf",
            embeddings_path="data/artvip_sdf/embeddings",
        )
    )
    if artiverse_enabled:
        sources["artiverse"] = Node(
            enabled=True,
            data_path="data/artiverse",
            embeddings_path="data/artiverse/embeddings",
        )
    return Node(
        asset_manager=Node(
            general_asset_source=source,
            backend="sam3d",
            objaverse=Node(use_top_k=10),
            router=Node(
                strategies=Node(
                    articulated=Node(enabled=True),
                    artiverse_articulated=Node(enabled=artiverse_route_enabled),
                )
            ),
            articulated=Node(sources=sources),
        ),
        collision_geometry=Node(
            coacd=Node(max_convex_hull=32),
            vhacd=Node(max_convex_hulls=32),
        ),
    )


def _full_quality_cfg(*, artiverse_route=True, ceiling_artvip=True) -> Node:
    return Node(
        furniture_agent=_agent(
            source="generated",
            artvip_enabled=True,
            artiverse_enabled=True,
            artiverse_route_enabled=artiverse_route,
        ),
        wall_agent=_agent(source="generated", artvip_enabled=True),
        ceiling_agent=_agent(
            source="generated", artvip_enabled=ceiling_artvip
        ),
        manipuland_agent=_agent(source="objaverse", artvip_enabled=True),
    )


def _prepare_datasets(repo_dir: Path, *, include_artiverse: bool = True) -> None:
    (repo_dir / "data" / "artvip_sdf").mkdir(parents=True)
    _touch_embedding_index(repo_dir / "data" / "artvip_sdf" / "embeddings")
    if include_artiverse:
        _write_artiverse_authority(repo_dir)


def test_official_metadata_digests_match_audited_release() -> None:
    assert OFFICIAL_SOURCE_MANIFEST_SHA256 == (
        "8fa6468254a1f74c58f0c25699598bf88f622fabdaf74f0cd9268ee5663c5586"
    )
    assert OFFICIAL_PACK_SCRIPT_SHA256 == (
        "f438e6fa147514f5260a205bc09d4b6c6ff3c0ce2d3022af424d220a9c933b99"
    )
    assert artiverse_contract_module.OFFICIAL_EXTRACTION_ROOT_COUNT == 3_544
    assert artiverse_contract_module.OFFICIAL_EXTRACTION_ROOTS_SHA256 == (
        "7bdf1be3acc558df62fcc9077b60c71f860227925d35ea05679b6ebbcc9f9182"
    )


def test_generated_worker_uses_real_articulated_source_config() -> None:
    overrides = _asset_pipeline_overrides("generated_sam3d")
    rendered = "\n".join(overrides)

    assert "asset_manager.artiverse_articulated" not in rendered
    assert "asset_manager.articulated.sources.artiverse.enabled=true" in rendered
    assert "asset_manager.articulated.sources.artiverse.embeddings_path=" in rendered
    assert "router.strategies.artiverse_articulated.max_retries=3" in rendered
    assert (
        "experiment.materials_retrieval_server.data_path=data/materials" in rendered
    )
    assert (
        "experiment.materials_retrieval_server.embeddings_path="
        "data/materials_full_quality_contract/embeddings" in rendered
    )


def test_config_only_policy_rejects_missing_artiverse(tmp_path: Path) -> None:
    _prepare_datasets(tmp_path, include_artiverse=False)

    with pytest.raises(RuntimeError, match="artiverse data path is missing"):
        _validate_asset_policy(
            _full_quality_cfg(), "generated_sam3d", repo_dir=tmp_path
        )


def test_config_only_policy_rejects_disabled_route_and_artvip(tmp_path: Path) -> None:
    _prepare_datasets(tmp_path)

    with pytest.raises(RuntimeError) as exc_info:
        _validate_asset_policy(
            _full_quality_cfg(artiverse_route=False, ceiling_artvip=False),
            "generated_sam3d",
            repo_dir=tmp_path,
        )

    message = str(exc_info.value)
    assert "artiverse_articulated strategy is disabled" in message
    assert "ceiling_agent ArtVIP source is disabled" in message


def test_config_only_policy_accepts_complete_strict_contract(tmp_path: Path) -> None:
    _prepare_datasets(tmp_path)

    _validate_asset_policy(
        _full_quality_cfg(), "generated_sam3d", repo_dir=tmp_path
    )


def test_config_only_policy_rejects_hssd_in_generated_service_plan(
    tmp_path: Path, monkeypatch
) -> None:
    import run_single_room_worker as worker

    _prepare_datasets(tmp_path)
    monkeypatch.setattr(
        worker,
        "_service_names",
        lambda _asset_pipeline: ["geometry", "hssd", "articulated", "materials"],
    )

    with pytest.raises(RuntimeError, match="must not start the HSSD server"):
        _validate_asset_policy(
            _full_quality_cfg(), "generated_sam3d", repo_dir=tmp_path
        )


def test_generated_worker_starts_and_stops_all_services_in_reverse_order() -> None:
    class FakeExperiment:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            if name.startswith("_start_") or name.startswith("_stop_"):
                return lambda: self.calls.append(name)
            raise AttributeError(name)

    experiment = FakeExperiment()
    started = _start_worker_services(experiment, "generated_sam3d")
    _stop_worker_services(experiment, started)

    assert started == ["geometry", "objaverse", "articulated", "materials"]
    assert experiment.calls == [
        "_start_geometry_server",
        "_start_objaverse_server",
        "_start_articulated_server",
        "_start_materials_server",
        "_stop_materials_server",
        "_stop_articulated_server",
        "_stop_objaverse_server",
        "_stop_geometry_server",
    ]
    assert not any("hssd" in call for call in experiment.calls)


def test_missing_final_blend_is_reexported_from_exact_saved_state(tmp_path: Path) -> None:
    room_dir = tmp_path / "scene_000" / "room_classroom_01"
    state_dir = room_dir / "scene_states" / "final_scene"
    state_dir.mkdir(parents=True)
    saved_state = {"objects": {"desk_1": {"object_id": "desk_1"}}}
    (state_dir / "scene_state.json").write_text(
        json.dumps(saved_state), encoding="utf-8"
    )
    restored: dict = {}

    class FakeRoomScene:
        def __init__(self, *, room_geometry, scene_dir, room_id):
            restored["init"] = (room_geometry, scene_dir, room_id)

        def restore_from_state_dict(self, state):
            restored["state"] = state

    def fake_export(*, scene, scene_dir, cfg_dict, name):
        restored["export"] = (scene, scene_dir, cfg_dict, name)
        (state_dir / "scene.blend").write_bytes(b"blend")

    blend_path = _recover_final_blend(
        room_dir=room_dir,
        room_id="classroom_01",
        cfg_dict={"furniture_agent": {"rendering": {}}},
        room_scene_cls=FakeRoomScene,
        export_scene_blend=fake_export,
    )

    assert blend_path.read_bytes() == b"blend"
    assert restored["state"] == saved_state
    assert restored["init"] == (None, room_dir, "classroom_01")
    assert restored["export"][3] == "final_scene"


def test_existing_nonempty_final_blend_is_not_reexported(tmp_path: Path) -> None:
    state_dir = tmp_path / "scene_states" / "final_scene"
    state_dir.mkdir(parents=True)
    blend_path = state_dir / "scene.blend"
    blend_path.write_bytes(b"existing")

    result = _recover_final_blend(
        room_dir=tmp_path,
        room_id="classroom_01",
        cfg_dict={},
        room_scene_cls=lambda **_: pytest.fail("scene should not be restored"),
        export_scene_blend=lambda **_: pytest.fail("blend should not be exported"),
    )

    assert result == blend_path
    assert result.read_bytes() == b"existing"


def test_forced_blend_refresh_is_atomic_and_uses_current_state(tmp_path: Path) -> None:
    state_dir = tmp_path / "scene_states" / "final_scene"
    state_dir.mkdir(parents=True)
    (state_dir / "scene_state.json").write_text(
        '{"objects":{"current":{}}}', encoding="utf-8"
    )
    blend_path = state_dir / "scene.blend"
    blend_path.write_bytes(b"stale")
    restored = {}

    class FakeRoomScene:
        def __init__(self, **_kwargs):
            pass

        def restore_from_state_dict(self, state):
            restored["state"] = state

    def export(**_kwargs):
        blend_path.write_bytes(b"current")

    result = _recover_final_blend(
        room_dir=tmp_path,
        room_id="classroom_01",
        cfg_dict={},
        room_scene_cls=FakeRoomScene,
        export_scene_blend=export,
        force_refresh=True,
    )

    assert result.read_bytes() == b"current"
    assert restored["state"] == {"objects": {"current": {}}}
    assert not list(state_dir.glob(".scene.blend.previous.*"))


def test_failed_forced_blend_refresh_restores_previous_blend(tmp_path: Path) -> None:
    state_dir = tmp_path / "scene_states" / "final_scene"
    state_dir.mkdir(parents=True)
    (state_dir / "scene_state.json").write_text(
        '{"objects":{"current":{}}}', encoding="utf-8"
    )
    blend_path = state_dir / "scene.blend"
    blend_path.write_bytes(b"preserve")

    class FakeRoomScene:
        def __init__(self, **_kwargs):
            pass

        def restore_from_state_dict(self, _state):
            pass

    with pytest.raises(RuntimeError, match="injected export failure"):
        _recover_final_blend(
            room_dir=tmp_path,
            room_id="classroom_01",
            cfg_dict={},
            room_scene_cls=FakeRoomScene,
            export_scene_blend=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected export failure")
            ),
            force_refresh=True,
        )

    assert blend_path.read_bytes() == b"preserve"
    assert not list(state_dir.glob(".scene.blend.previous.*"))


def _write_room_state(
    scene_dir: Path,
    authority,
    object_id: str,
    *,
    include_artiverse: bool,
    sdf_exists: bool = True,
    claimed_id: str | None = None,
) -> dict:
    room_dir = scene_dir / "room_room_a"
    state_path = room_dir / "scene_states" / "final_scene" / "scene_state.json"
    state_path.parent.mkdir(parents=True)
    objects = {}
    if include_artiverse:
        expected = authority.asset(object_id)
        sdf_path = (
            room_dir
            / "generated_assets"
            / "sdf"
            / "cabinet"
            / "scenesmith_artiverse.sdf"
        )
        if sdf_exists:
            sdf_path.parent.mkdir(parents=True)
            sdf_path.write_text(
                "<sdf><model name='copied'><link name='base'><collision name='c'/>"
                "<visual name='v'><geometry><mesh><uri>mesh.glb</uri></mesh></geometry></visual>"
                "</link></model></sdf>",
                encoding="utf-8",
            )
            (sdf_path.parent / "mesh.glb").write_bytes(b"official-artiverse-mesh")
            copied_sdf_hash = sha256_file(sdf_path)
            copied_tree_hash = sha256_directory_tree(sdf_path.parent)
        else:
            copied_sdf_hash = "0" * 64
            copied_tree_hash = "0" * 64
        objects["cabinet_1"] = {
            "object_id": "cabinet_1",
            "sdf_path": str(sdf_path.relative_to(room_dir)),
            "metadata": {
                "asset_source": "articulated",
                "articulated_source": "artiverse",
                "articulated_id": claimed_id or object_id,
                "is_articulated": True,
                "articulated_source_sdf_path": str(expected["source_sdf_path"]),
                "articulated_source_sdf_sha256": expected["source_sdf_sha256"],
                "articulated_source_tree_sha256": expected["source_tree_sha256"],
                "articulated_copied_sdf_sha256": copied_sdf_hash,
                "articulated_copied_tree_sha256": copied_tree_hash,
            },
        }
    state = {"objects": objects, "text_description": "classroom"}
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state


def _write_passing_usage_fixture(tmp_path: Path):
    authority, object_id = _write_artiverse_authority(tmp_path)
    scene_dir = tmp_path / "scene_000"
    room_state = _write_room_state(
        scene_dir, authority, object_id, include_artiverse=True
    )
    placed, failures = _collect_artiverse_from_room_states(
        scene_dir, ["room_a"], authority
    )
    _require_artiverse_records(placed, failures, "passing final room states")

    combined_dir = scene_dir / "combined_house"
    combined_dir.mkdir()
    house_state_path = combined_dir / "house_state.json"
    house_state_path.write_text(
        json.dumps({"rooms": {"room_a": room_state}}), encoding="utf-8"
    )
    final, final_failures = _collect_artiverse_from_house_state(
        house_state_path, scene_dir, authority
    )
    _require_artiverse_records(final, final_failures, "combined house state")
    manifest_path = combined_dir / "artiverse_usage.json"
    _write_artiverse_usage_manifest(
        manifest_path,
        placed,
        final,
        authority,
        scene_dir,
        ["room_a"],
        house_state_path,
        house_state_path,
    )
    return authority, scene_dir, house_state_path, manifest_path


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are unavailable for this test user: {exc}")


def test_artiverse_tree_hash_rejects_file_symlink(tmp_path: Path) -> None:
    tree = tmp_path / "model"
    tree.mkdir()
    target = tmp_path / "publisher_mesh.glb"
    target.write_bytes(b"mesh")
    _symlink_or_skip(tree / "mesh.glb", target)

    with pytest.raises(ArtiverseContractError, match="symlink|junction"):
        sha256_directory_tree(tree)


def test_artiverse_tree_validation_rejects_special_file(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable on this platform")
    tree = tmp_path / "model"
    tree.mkdir()
    os.mkfifo(tree / "publisher.pipe")

    with pytest.raises(ArtiverseContractError, match="special filesystem entry"):
        validate_regular_directory_tree(tree, label="publisher model")


def test_sdf_resource_uri_accepts_relative_and_local_file_uri(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    relative_mesh = model_dir / "relative mesh.glb"
    absolute_mesh = model_dir / "absolute.glb"
    relative_mesh.write_bytes(b"relative")
    absolute_mesh.write_bytes(b"absolute")
    sdf_path = model_dir / "asset.sdf"
    sdf_path.write_text(
        "<sdf><model><link><visual><geometry><mesh>"
        "<uri>relative%20mesh.glb</uri>"
        f"<uri>{absolute_mesh.as_uri()}</uri>"
        "</mesh></geometry></visual></link></model></sdf>",
        encoding="utf-8",
    )

    assert validate_sdf_resource_uris(sdf_path, model_dir) == (
        relative_mesh.resolve(),
        absolute_mesh.resolve(),
    )


def test_sdf_resource_uri_rejects_model_tree_escape(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    outside_mesh = tmp_path / "outside.glb"
    outside_mesh.write_bytes(b"outside")
    sdf_path = model_dir / "asset.sdf"
    sdf_path.write_text(
        "<sdf><model><link><visual><geometry><mesh>"
        "<uri>../outside.glb</uri>"
        "</mesh></geometry></visual></link></model></sdf>",
        encoding="utf-8",
    )

    with pytest.raises(ArtiverseContractError, match="outside"):
        validate_sdf_resource_uris(sdf_path, model_dir)


def test_regular_file_boundary_rejects_symlinked_publisher_resource(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    outside_mesh = tmp_path / "outside.glb"
    outside_mesh.write_bytes(b"outside")
    linked_mesh = model_dir / "mesh.glb"
    _symlink_or_skip(linked_mesh, outside_mesh)

    with pytest.raises(ArtiverseContractError, match="symlink|junction"):
        require_regular_file_within(linked_mesh, model_dir, "publisher mesh")


@pytest.mark.parametrize("location", ["outside_scene", "wrong_scene_subtree"])
def test_usage_manifest_rejects_copied_sdf_outside_generated_asset_root(
    tmp_path: Path, location: str
) -> None:
    authority, scene_dir, house_state_path, manifest_path = (
        _write_passing_usage_fixture(tmp_path)
    )
    if location == "outside_scene":
        forged_tree = tmp_path / "forged_copy"
    else:
        forged_tree = scene_dir / "room_room_a" / "assets" / "forged_copy"
    forged_tree.mkdir(parents=True)
    forged_sdf = forged_tree / "scenesmith_artiverse.sdf"
    forged_sdf.write_text(
        "<sdf><model><link><collision name='c'/></link></model></sdf>",
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for collection in ("placed_assets", "final_surviving_assets"):
        manifest[collection][0]["sdf_path"] = str(forged_sdf.resolve())
        manifest[collection][0]["sdf_sha256"] = sha256_file(forged_sdf)
        manifest[collection][0]["sdf_tree_sha256"] = sha256_directory_tree(
            forged_tree
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtiverseContractError, match="outside.*filesystem root"):
        validate_usage_manifest(
            manifest_path, house_state_path, authority, scene_dir=scene_dir
        )


def test_usage_manifest_rejects_source_path_outside_dataset_root(
    tmp_path: Path,
) -> None:
    authority, scene_dir, house_state_path, manifest_path = (
        _write_passing_usage_fixture(tmp_path)
    )
    forged_source = tmp_path / "forged_source.sdf"
    forged_source.write_text("<sdf/>", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for collection in ("placed_assets", "final_surviving_assets"):
        manifest[collection][0]["source_sdf_path"] = str(forged_source.resolve())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtiverseContractError, match="outside.*filesystem root"):
        validate_usage_manifest(
            manifest_path, house_state_path, authority, scene_dir=scene_dir
        )


def test_usage_manifest_rejects_survivor_absent_from_combined_house(
    tmp_path: Path,
) -> None:
    authority, scene_dir, house_state_path, manifest_path = (
        _write_passing_usage_fixture(tmp_path)
    )
    house_state_path.write_text(
        json.dumps({"rooms": {"room_a": {"objects": {}}}}), encoding="utf-8"
    )
    usage = json.loads(manifest_path.read_text(encoding="utf-8"))
    usage["house_state"]["sha256"] = sha256_file(house_state_path)
    manifest_path.write_text(json.dumps(usage), encoding="utf-8")

    with pytest.raises(
        ArtiverseContractError,
        match="final-survivor usage does not exactly match the combined house state",
    ):
        validate_usage_manifest(
            manifest_path, house_state_path, authority, scene_dir=scene_dir
        )


def test_assembler_blocks_zero_artiverse_before_assembly(tmp_path: Path) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    scene_dir = tmp_path / "scene_000"
    _write_room_state(
        scene_dir, authority, object_id, include_artiverse=False
    )

    records, failures = _collect_artiverse_from_room_states(
        scene_dir, ["room_a"], authority
    )

    with pytest.raises(RuntimeError, match="zero surviving Artiverse assets"):
        _require_artiverse_records(records, failures, "passing final room states")


def test_assembler_rejects_artiverse_marker_with_missing_sdf(tmp_path: Path) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    scene_dir = tmp_path / "scene_000"
    _write_room_state(
        scene_dir,
        authority,
        object_id,
        include_artiverse=True,
        sdf_exists=False,
    )

    records, failures = _collect_artiverse_from_room_states(
        scene_dir, ["room_a"], authority
    )

    assert not records
    with pytest.raises(RuntimeError, match="SDF does not exist"):
        _require_artiverse_records(records, failures, "passing final room states")


def test_assembler_rejects_forged_artiverse_id(tmp_path: Path) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    scene_dir = tmp_path / "scene_000"
    _write_room_state(
        scene_dir,
        authority,
        object_id,
        include_artiverse=True,
        claimed_id="artiverse/cabinet/fake/not_indexed",
    )

    records, failures = _collect_artiverse_from_room_states(
        scene_dir, ["room_a"], authority
    )

    assert records == []
    assert any("absent from the prepared metadata index" in error for error in failures)


def test_assembler_rejects_mutated_copied_artiverse_sdf(tmp_path: Path) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    scene_dir = tmp_path / "scene_000"
    room_state = _write_room_state(
        scene_dir, authority, object_id, include_artiverse=True
    )
    copied_sdf = (
        scene_dir
        / "room_room_a"
        / str(room_state["objects"]["cabinet_1"]["sdf_path"])
    )
    copied_sdf.write_text("<sdf><forged/></sdf>", encoding="utf-8")

    records, failures = _collect_artiverse_from_room_states(
        scene_dir, ["room_a"], authority
    )

    assert records == []
    assert any("placed SDF content hash" in error for error in failures)


def test_assembler_rechecks_combined_state_and_writes_manifest(tmp_path: Path) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    scene_dir = tmp_path / "scene_000"
    room_state = _write_room_state(
        scene_dir, authority, object_id, include_artiverse=True
    )
    placed, failures = _collect_artiverse_from_room_states(
        scene_dir, ["room_a"], authority
    )
    _require_artiverse_records(placed, failures, "passing final room states")

    combined_dir = scene_dir / "combined_house"
    combined_dir.mkdir()
    house_state_path = combined_dir / "house_state.json"
    house_state_path.write_text(
        json.dumps({"rooms": {"room_a": room_state}}), encoding="utf-8"
    )
    final, final_failures = _collect_artiverse_from_house_state(
        house_state_path, scene_dir, authority
    )
    _require_artiverse_records(final, final_failures, "combined house state")
    _verify_combined_survival(placed, final)

    manifest_path = combined_dir / "artiverse_usage.json"
    _write_artiverse_usage_manifest(
        manifest_path,
        placed,
        final,
        authority,
        scene_dir,
        ["room_a"],
        house_state_path,
        house_state_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "pass"
    assert manifest["placed_asset_count"] == 1
    assert manifest["final_surviving_asset_count"] == 1
    assert manifest["final_surviving_asset_identifiers"] == [
        object_id
    ]
    assert validate_usage_manifest(
        manifest_path, house_state_path, authority, scene_dir=scene_dir
    )["status"] == "pass"


def test_router_source_preflight_and_candidate_contract(tmp_path: Path) -> None:
    data_path = tmp_path / "data" / "artiverse"
    embeddings_path = data_path / "embeddings"
    missing = _source_status("artiverse", data_path, embeddings_path)
    assert _source_preflight_errors(missing)

    authority, object_id = _write_artiverse_authority(tmp_path)
    sdf_path = authority.asset(object_id)["source_sdf_path"]
    complete = _source_status("artiverse", data_path, embeddings_path)
    assert _source_preflight_errors(complete) == []

    candidate = {
        "source": "artiverse",
        "object_id": object_id,
        "sdf_path": str(sdf_path),
    }
    assert _candidate_passes_source_contract(
        candidate, "artiverse", data_path, authority
    )
    assert not _candidate_passes_source_contract(candidate, "artvip", data_path)


def test_failed_post_promotion_validation_restores_previous_house(tmp_path: Path) -> None:
    target = tmp_path / "combined_house"
    candidate = tmp_path / ".combined_house.staging"
    target.mkdir()
    candidate.mkdir()
    (target / "marker").write_text("previous", encoding="utf-8")
    (candidate / "marker").write_text("candidate", encoding="utf-8")

    with pytest.raises(RuntimeError, match="post-promotion failure"):
        _promote_combined_candidate(
            candidate,
            target,
            lambda _: (_ for _ in ()).throw(RuntimeError("post-promotion failure")),
        )

    assert (target / "marker").read_text(encoding="utf-8") == "previous"


def test_successful_promotion_publishes_validated_candidate(tmp_path: Path) -> None:
    target = tmp_path / "combined_house"
    candidate = tmp_path / ".combined_house.staging"
    target.mkdir()
    candidate.mkdir()
    (target / "marker").write_text("previous", encoding="utf-8")
    (candidate / "marker").write_text("candidate", encoding="utf-8")

    result = _promote_combined_candidate(
        candidate,
        target,
        lambda published: (
            published / "marker"
        ).read_text(encoding="utf-8") == "candidate",
    )

    assert result == target
    assert (target / "marker").read_text(encoding="utf-8") == "candidate"


def test_authority_rejects_index_mutation_after_preparation(tmp_path: Path) -> None:
    authority, _ = _write_artiverse_authority(tmp_path)
    metadata_path = authority.embeddings_path / "metadata_index.yaml"
    metadata_path.write_text("forged: {}\n", encoding="utf-8")

    with pytest.raises(ArtiverseContractError, match="index hash mismatch"):
        load_artiverse_authority(authority.dataset_root)


def test_authority_rejects_wrong_embedding_shape_even_with_updated_hash(tmp_path: Path) -> None:
    authority, _ = _write_artiverse_authority(tmp_path)
    embeddings_file = authority.embeddings_path / "clip_embeddings.npy"
    np.save(embeddings_file, np.zeros((1, 512), dtype=np.float32))
    manifest_path = authority.preparation_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_sha256"]["clip_embeddings.npy"] = sha256_file(embeddings_file)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtiverseContractError, match="shape"):
        load_artiverse_authority(authority.dataset_root)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("missing", r"exactly one <inertial>"),
        ("zero_mass", "mass must be positive"),
        ("nonfinite_tensor", "inertia ixx is not finite"),
        ("nonpositive_tensor", "symmetric positive-definite"),
        ("triangle_invalid", "triangle inequality"),
        ("nonfinite_pose", "inertial pose is not finite"),
        ("rotated_pose", "pose rotation must be zero"),
    ],
)
def test_authority_reparses_every_link_inertial_fail_closed(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    source_sdf = Path(authority.asset(object_id)["source_sdf_path"])
    tree = ET.parse(source_sdf)
    door = tree.getroot().find(".//link[@name='door']")
    assert door is not None
    inertial = door.find("inertial")
    assert inertial is not None
    if mutation == "missing":
        door.remove(inertial)
    elif mutation == "zero_mass":
        inertial.find("mass").text = "0"
    elif mutation == "nonfinite_tensor":
        inertial.find("inertia/ixx").text = "nan"
    elif mutation == "nonpositive_tensor":
        inertial.find("inertia/ixx").text = "-0.1"
    elif mutation == "triangle_invalid":
        inertial.find("inertia/ixx").text = "10"
        inertial.find("inertia/iyy").text = "1"
        inertial.find("inertia/izz").text = "1"
    elif mutation == "nonfinite_pose":
        inertial.find("pose").text = "nan 0 0 0 0 0"
    else:
        inertial.find("pose").text = "0 0 0 0.1 0 0"
    tree.write(source_sdf, encoding="utf-8", xml_declaration=True)
    _rewrite_authority_evidence(authority, object_id)

    with pytest.raises(ArtiverseContractError, match=expected_error):
        load_artiverse_authority(authority.dataset_root)


def test_authority_requires_collision_on_every_geometry_link(tmp_path: Path) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    source_sdf = Path(authority.asset(object_id)["source_sdf_path"])
    tree = ET.parse(source_sdf)
    door = tree.getroot().find(".//link[@name='door']")
    assert door is not None
    collision = door.find("collision")
    assert collision is not None
    door.remove(collision)
    tree.write(source_sdf, encoding="utf-8", xml_declaration=True)
    _rewrite_authority_evidence(
        authority,
        object_id,
        record_updates={
            "collision_element_count": 1,
            "source_collision_element_count": 1,
        },
    )

    with pytest.raises(ArtiverseContractError, match="geometry but no collision"):
        load_artiverse_authority(authority.dataset_root)


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("physics_source", "self_declared", "physics_source is missing or stale"),
        ("physics_link_count", 3, "physics link count is stale"),
        (
            "physics_geometry_link_count",
            1,
            "physics geometry-link count is stale",
        ),
        ("physics_total_mass_kg", 3.0, "physics total mass is stale"),
        (
            "source_collision_element_count",
            1,
            "publisher collision count is stale",
        ),
    ],
)
def test_authority_rejects_stale_physics_metadata(
    tmp_path: Path, field: str, value, expected_error: str
) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    _rewrite_authority_evidence(
        authority, object_id, record_updates={field: value}
    )

    with pytest.raises(ArtiverseContractError, match=expected_error):
        load_artiverse_authority(authority.dataset_root)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("missing_key", "exactly the required policy keys"),
        ("extra_key", "exactly the required policy keys"),
        ("wrong_id", "physics_policy id"),
        ("false_boolean", "required_for_every_link must be exactly true"),
    ],
)
def test_authority_requires_exact_physics_policy(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    authority, _ = _write_artiverse_authority(tmp_path)
    path = authority.preparation_manifest_path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    policy = manifest["physics_policy"]
    if mutation == "missing_key":
        policy.pop("required_for_every_link")
    elif mutation == "extra_key":
        policy["unreviewed_fallback"] = True
    elif mutation == "wrong_id":
        policy["id"] = "self_declared_physics_v1"
    else:
        policy["required_for_every_link"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtiverseContractError, match=expected_error):
        load_artiverse_authority(authority.dataset_root)


def test_authority_exposes_hash_bound_publisher_and_emitted_physics(
    tmp_path: Path,
) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    asset = authority.asset(object_id)
    assert asset["publisher_urdf_path"].is_file()
    assert asset["publisher_urdf_sha256"] == sha256_file(
        asset["publisher_urdf_path"]
    )
    assert len(asset["publisher_link_physics_sha256"]) == 64
    assert len(asset["emitted_sdf_physics_sha256"]) == 64
    assert asset["physics_binding_sha256"] == authority.physics_binding_sha256
    evidence = authority.evidence()
    assert evidence["schema_version"] == 2
    assert evidence["physics_bound_indexed_count"] == 1
    assert evidence["physics_binding_sha256"] == authority.physics_binding_sha256
    assert evidence["physics_policy"]["publisher_mass_semantics"] == (
        artiverse_contract_module.PUBLISHER_MASS_SEMANTICS
    )


@pytest.mark.parametrize(
    "field",
    [
        "publisher_urdf_path",
        "publisher_urdf_sha256",
        "publisher_link_physics_sha256",
        "emitted_sdf_physics_sha256",
    ],
)
def test_authority_requires_every_row_physics_binding_field(
    tmp_path: Path, field: str
) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    metadata_path = authority.embeddings_path / "metadata_index.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata[object_id].pop(field)
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8")
    manifest_path = authority.preparation_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_sha256"]["metadata_index.yaml"] = sha256_file(metadata_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtiverseContractError, match=field):
        load_artiverse_authority(authority.dataset_root)


def test_authority_rejects_publisher_sdf_mass_divergence_even_when_rehashed(
    tmp_path: Path,
) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    publisher_urdf = Path(authority.asset(object_id)["publisher_urdf_path"])
    tree = ET.parse(publisher_urdf)
    tree.getroot().find("./link[@name='door']/inertial/mass").set("value", "0.75")
    tree.write(publisher_urdf, encoding="utf-8", xml_declaration=True)
    _rewrite_authority_evidence(authority, object_id)

    with pytest.raises(ArtiverseContractError, match="emitted mass differs from publisher"):
        load_artiverse_authority(authority.dataset_root)


def test_authority_applies_nonzero_urdf_rpy_before_comparing_emitted_inertia(
    tmp_path: Path,
) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    initial_asset = authority.asset(object_id)
    publisher_urdf = Path(initial_asset["publisher_urdf_path"])
    source_sdf = Path(initial_asset["source_sdf_path"])
    urdf_tree = ET.parse(publisher_urdf)
    inertial = urdf_tree.getroot().find("./link[@name='door']/inertial")
    inertial.find("origin").set("rpy", f"0 0 {np.pi / 2!r}")
    source_inertia = inertial.find("inertia")
    source_inertia.set("ixx", "0.1")
    source_inertia.set("iyy", "0.2")
    source_inertia.set("izz", "0.25")
    urdf_tree.write(publisher_urdf, encoding="utf-8", xml_declaration=True)
    transformed = artiverse_contract_module.publisher_urdf_physics_evidence(
        publisher_urdf
    ).link_values["door"]["inertia"]

    sdf_tree = ET.parse(source_sdf)
    sdf_inertia = sdf_tree.getroot().find(".//link[@name='door']/inertial/inertia")
    for component, value in transformed.items():
        sdf_inertia.find(component).text = f"{value:.6e}"
    sdf_tree.write(source_sdf, encoding="utf-8", xml_declaration=True)
    _rewrite_authority_evidence(authority, object_id)

    loaded = load_artiverse_authority(authority.dataset_root)
    assert loaded.asset(object_id)["publisher_urdf_sha256"] == sha256_file(
        publisher_urdf
    )


@pytest.mark.parametrize(
    ("element_path", "value", "expected"),
    [
        (".//collision/surface/friction/ode/mu", "0.4", "pinned 0.5 friction"),
        (".//joint/axis/dynamics/damping", "0.04", "pinned 0.05 damping"),
    ],
)
def test_authority_enforces_transparent_default_physics_policies(
    tmp_path: Path, element_path: str, value: str, expected: str
) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    source_sdf = Path(authority.asset(object_id)["source_sdf_path"])
    tree = ET.parse(source_sdf)
    tree.getroot().find(element_path).text = value
    tree.write(source_sdf, encoding="utf-8", xml_declaration=True)
    _rewrite_authority_evidence(authority, object_id)

    with pytest.raises(ArtiverseContractError, match=expected):
        load_artiverse_authority(authority.dataset_root)


def test_authority_rejects_stale_physics_binding_aggregate(tmp_path: Path) -> None:
    authority, _ = _write_artiverse_authority(tmp_path)
    path = authority.preparation_manifest_path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["physics_binding_sha256"] = "f" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtiverseContractError, match="physics_binding_sha256 is stale"):
        load_artiverse_authority(authority.dataset_root)


def test_authority_rejects_no_movable_joint_even_with_matching_metadata(
    tmp_path: Path,
) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    source_sdf = Path(authority.asset(object_id)["source_sdf_path"])
    tree = ET.parse(source_sdf)
    joint = tree.getroot().find(".//joint[@name='door_hinge']")
    assert joint is not None
    joint.set("type", "fixed")
    tree.write(source_sdf, encoding="utf-8", xml_declaration=True)
    _rewrite_authority_evidence(
        authority,
        object_id,
        record_updates={"movable_joint_count": 0, "movable_joint_types": []},
    )

    with pytest.raises(ArtiverseContractError, match="no usable movable joint"):
        load_artiverse_authority(authority.dataset_root)


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        ({"movable_joint_count": 2}, "movable-joint count is stale"),
        ({"movable_joint_types": ["prismatic"]}, "movable-joint types are stale"),
    ],
)
def test_authority_rejects_stale_movable_joint_metadata(
    tmp_path: Path, updates: dict, expected_error: str
) -> None:
    authority, object_id = _write_artiverse_authority(tmp_path)
    _rewrite_authority_evidence(authority, object_id, record_updates=updates)

    with pytest.raises(ArtiverseContractError, match=expected_error):
        load_artiverse_authority(authority.dataset_root)


@pytest.mark.parametrize("metadata_kind", ["source_manifest", "pack_script"])
@pytest.mark.parametrize(
    "rewrite_evidence, expected_error",
    [
        (False, "content does not match the pinned digest"),
        (True, "evidence is not bound to the pinned official digest"),
    ],
    ids=["content-tamper", "self-consistent-substitution"],
)
def test_authority_rejects_unpinned_official_metadata(
    tmp_path: Path,
    metadata_kind: str,
    rewrite_evidence: bool,
    expected_error: str,
) -> None:
    authority, _ = _write_artiverse_authority(tmp_path)
    preparation_path = authority.preparation_manifest_path
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))

    if metadata_kind == "source_manifest":
        metadata_path = authority.source_manifest_path
        metadata_path.write_text('{"substituted": true}\n', encoding="utf-8")
        if rewrite_evidence:
            preparation["source_manifest"]["sha256"] = sha256_file(metadata_path)
    else:
        metadata_path = authority.dataset_root / "pack_dataset_chunks.py"
        metadata_path.write_text("# substituted unpacker\n", encoding="utf-8")
        if rewrite_evidence:
            preparation["source_pack_script_sha256"] = sha256_file(metadata_path)

    if rewrite_evidence:
        preparation_path.write_text(
            json.dumps(preparation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ArtiverseContractError, match=expected_error):
        load_artiverse_authority(authority.dataset_root)


def test_authority_rejects_receipt_changed_after_preparation(tmp_path: Path) -> None:
    authority, _ = _write_artiverse_authority(tmp_path)
    receipt_path = authority.source_extraction_receipt_path
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ArtiverseContractError, match="changed after preparation"):
        load_artiverse_authority(authority.dataset_root)


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        ("archive", "archive pin mismatch"),
        ("extractor", "safe extractor does not match"),
    ],
)
def test_authority_rejects_self_consistent_receipt_substitution(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    authority, _ = _write_artiverse_authority(tmp_path)
    receipt_path = authority.source_extraction_receipt_path
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "archive":
        receipt["archives"][0]["archive_bytes"] += 1
    else:
        receipt["safe_extractor"]["sha256"] = "b" * 64
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    preparation_path = authority.preparation_manifest_path
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    preparation["source_extraction_receipt"]["sha256"] = sha256_file(receipt_path)
    if mutation == "extractor":
        preparation["safe_extractor_sha256"] = "b" * 64
    preparation_path.write_text(json.dumps(preparation), encoding="utf-8")

    with pytest.raises(ArtiverseContractError, match=expected_error):
        load_artiverse_authority(authority.dataset_root)


def test_download_wrapper_retries_zero_exit_local_dir_fallback() -> None:
    """A hub local-dir fallback must never masquerade as two complete archives."""

    script = (
        REPO_ROOT / "remote_jobs" / "download_prepare_artiverse_sqz.sh"
    ).read_text(encoding="utf-8")
    function = script.split("download_pinned_chunks_with_retry() {", 1)[1].split(
        "\n}\n", 1
    )[0]

    assert "final_chunks_complete=1" in function
    assert '[[ ! -f "${final_path}" ]]' in function
    assert 'stat -c %s "${final_path}"' in function
    assert "if (( final_chunks_complete == 1 )); then" in function
    assert "exit_code=26" in function
    assert function.index("if (( final_chunks_complete == 1 )); then") < function.index(
        "return 0"
    )
    assert function.index("exit_code=26") < function.index("retrying_chunks")
