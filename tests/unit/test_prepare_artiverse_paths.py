import importlib
import json
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import artiverse_contract as artiverse_contract_module  # noqa: E402
from artiverse_contract import ArtiverseContractError  # noqa: E402


def _load_prepare_module(monkeypatch):
    clip_module = types.ModuleType("scenesmith.agent_utils.clip_embeddings")
    clip_module.get_multiview_image_embedding = lambda _images: None
    mesh_module = types.ModuleType("scenesmith.agent_utils.sdf_mesh_utils")
    mesh_module.combine_sdf_meshes_at_joint_angles = lambda *_args, **_kwargs: None
    converter_module = types.ModuleType("scenesmith.agent_utils.urdf_to_sdf")

    class LinkPhysics:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    converter_module.LinkPhysics = LinkPhysics
    converter_module.convert_urdf_to_sdf = lambda **_kwargs: None
    monkeypatch.setitem(
        sys.modules, "scenesmith.agent_utils.clip_embeddings", clip_module
    )
    monkeypatch.setitem(
        sys.modules, "scenesmith.agent_utils.sdf_mesh_utils", mesh_module
    )
    monkeypatch.setitem(
        sys.modules, "scenesmith.agent_utils.urdf_to_sdf", converter_module
    )
    # Record the previous module state so monkeypatch removes/restores this
    # fake-dependency import at test teardown.
    monkeypatch.setitem(sys.modules, "prepare_artiverse", None)
    sys.modules.pop("prepare_artiverse", None)
    return importlib.import_module("prepare_artiverse")


def test_preparation_physics_policy_is_accepted_by_authority_contract(
    monkeypatch,
) -> None:
    module = _load_prepare_module(monkeypatch)

    assert set(module.PHYSICS_POLICY) == set(
        artiverse_contract_module.PHYSICS_POLICY_KEYS
    )
    assert artiverse_contract_module._exact_physics_policy(
        {"physics_policy": dict(module.PHYSICS_POLICY)}
    ) == module.PHYSICS_POLICY


def _asset(module, model_dir: Path, urdf_path: Path):
    return module.ArtiverseAsset(
        category="cabinet",
        source="publisher",
        model_id="model_001",
        model_dir=model_dir,
        urdf_path=urdf_path,
    )


def _urdf_inertial(
    *,
    mass: str = "1.0",
    xyz: str = "0 0 0",
    rpy: str = "0 0 0",
    ixx: str = "0.1",
    iyy: str = "0.1",
    izz: str = "0.1",
    ixy: str = "0",
    ixz: str = "0",
    iyz: str = "0",
) -> str:
    return (
        f"<inertial><origin xyz='{xyz}' rpy='{rpy}'/>"
        f"<mass value='{mass}'/>"
        f"<inertia ixx='{ixx}' iyy='{iyy}' izz='{izz}' "
        f"ixy='{ixy}' ixz='{ixz}' iyz='{iyz}'/></inertial>"
    )


def _sdf_inertial(
    *,
    mass: str = "1.000000",
    xyz: str = "0 0 0",
    ixx: str = "1.000000e-01",
    iyy: str = "1.000000e-01",
    izz: str = "1.000000e-01",
    ixy: str = "0.000000e+00",
    ixz: str = "0.000000e+00",
    iyz: str = "0.000000e+00",
) -> str:
    return (
        f"<inertial><mass>{mass}</mass><pose>{xyz} 0 0 0</pose><inertia>"
        f"<ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz>"
        f"<ixy>{ixy}</ixy><ixz>{ixz}</ixz><iyz>{iyz}</iyz>"
        "</inertia></inertial>"
    )


def _publisher_urdf(*, joint_type: str = "revolute", collision_count: int = 1) -> str:
    collisions = "".join(
        "<collision><geometry><box size='1 1 1'/></geometry></collision>"
        for _ in range(collision_count)
    )
    return (
        "<robot name='fixture'><link name='base'>"
        + _urdf_inertial()
        + "<visual><geometry><box size='1 1 1'/></geometry></visual>"
        + collisions
        + "</link><link name='door'>"
        + _urdf_inertial()
        + "</link>"
        + f"<joint name='door_joint' type='{joint_type}'>"
        "<parent link='base'/><child link='door'/></joint></robot>"
    )


def _converted_sdf(
    *,
    model_name: str = "fixture",
    joint_type: str = "revolute",
    collision_count: int = 1,
    visual_uri: str | None = None,
    collision_uri: str | None = None,
) -> str:
    if visual_uri is None:
        visual_geometry = "<geometry><box><size>1 1 1</size></box></geometry>"
    else:
        visual_geometry = (
            f"<geometry><mesh><uri>{visual_uri}</uri></mesh></geometry>"
        )
    if collision_uri is None:
        collision_geometry = "<geometry><box><size>1 1 1</size></box></geometry>"
    else:
        collision_geometry = (
            f"<geometry><mesh><uri>{collision_uri}</uri></mesh></geometry>"
        )
    collisions = "".join(
        f"<collision name='body_{index}'>{collision_geometry}</collision>"
        for index in range(collision_count)
    )
    return (
        f"<sdf><model name='{model_name}'><link name='base'>"
        + _sdf_inertial()
        + f"<visual name='body'>{visual_geometry}</visual>"
        + collisions
        + "</link><link name='door'>"
        + _sdf_inertial()
        + "</link>"
        + f"<joint name='door_joint' type='{joint_type}'>"
        "<parent>base</parent><child>door</child></joint></model></sdf>"
    )


def _write_pinned_metadata(module, monkeypatch, dataset_root: Path, roots: list[str]):
    source_manifest = dataset_root / "dataset_chunks" / "manifest.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text(
        json.dumps({"chunks": [{"roots": roots}]}) + "\n",
        encoding="utf-8",
    )
    pack_script = dataset_root / "pack_dataset_chunks.py"
    pack_script.write_text("# audited publisher packer\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "OFFICIAL_SOURCE_MANIFEST_SHA256",
        module.sha256_file(source_manifest),
    )
    monkeypatch.setattr(
        module,
        "OFFICIAL_PACK_SCRIPT_SHA256",
        module.sha256_file(pack_script),
    )
    receipt_path = dataset_root / "artiverse_safe_extraction_receipt.json"
    receipt_path.write_text('{"schema_version": 1, "status": "pass"}\n')
    extraction = types.SimpleNamespace(
        receipt_path=receipt_path.resolve(),
        receipt_sha256=module.sha256_file(receipt_path),
        safe_extractor_sha256="c" * 64,
    )
    monkeypatch.setattr(
        module,
        "validate_extraction_receipt",
        lambda _dataset_root: extraction,
    )
    return source_manifest


def _write_resumable_index(
    module,
    output_path: Path,
    object_id: str,
    sdf_path: Path,
    dataset_root: Path,
) -> None:
    output_path.mkdir(parents=True)
    module.np.save(
        output_path / "clip_embeddings.npy",
        module.np.asarray([[1.0, 0.0]], dtype=module.np.float32),
    )
    (output_path / "embedding_index.yaml").write_text(
        module.yaml.safe_dump([object_id], sort_keys=False), encoding="utf-8"
    )
    (output_path / "metadata_index.yaml").write_text(
        module.yaml.safe_dump(
            {
                object_id: {
                    "category": object_id.split("/")[1],
                    "collision_element_count": 1,
                    "movable_joint_count": 999,
                    "movable_joint_types": ["stale"],
                    "sdf_path": str(sdf_path.relative_to(dataset_root)),
                    "sdf_sha256": module.sha256_file(sdf_path),
                    "sdf_directory_tree_sha256": module.sha256_directory_tree(
                        sdf_path.parent
                    ),
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_resumable_asset(
    module,
    monkeypatch,
    tmp_path: Path,
    *,
    joint_type: str,
) -> tuple[Path, Path, str]:
    dataset_root = tmp_path / "artiverse"
    object_id = "artiverse/cabinet/publisher/model_001"
    _write_pinned_metadata(
        module,
        monkeypatch,
        dataset_root,
        ["data/cabinet/publisher/model_001"],
    )
    urdf_dir = (
        dataset_root
        / "data"
        / "cabinet"
        / "publisher"
        / "model_001"
        / "urdf_w_collider"
    )
    urdf_dir.mkdir(parents=True)
    (urdf_dir / "mobility.urdf").write_text(
        _publisher_urdf(joint_type=joint_type), encoding="utf-8"
    )
    sdf_path = urdf_dir / "scenesmith_artiverse.sdf"
    sdf_path.write_text(
        _converted_sdf(model_name="cabinet", joint_type=joint_type),
        encoding="utf-8",
    )
    output_path = dataset_root / "embeddings"
    _write_resumable_index(module, output_path, object_id, sdf_path, dataset_root)
    return dataset_root, output_path, object_id


def test_prepare_rejects_urdf_outside_publisher_model(monkeypatch, tmp_path: Path) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    model_dir = dataset_root / "data" / "cabinet" / "publisher" / "model_001"
    model_dir.mkdir(parents=True)
    (model_dir / "mesh.glb").write_bytes(b"mesh")
    outside_urdf = dataset_root / "outside.urdf"
    outside_urdf.write_text("<robot/>", encoding="utf-8")
    asset = _asset(module, model_dir, outside_urdf)

    with pytest.raises(ArtiverseContractError, match="outside"):
        module._validate_publisher_asset(
            asset, dataset_root, model_dir / "scenesmith_artiverse.sdf"
        )


def test_prepare_rejects_model_outside_dataset_root(monkeypatch, tmp_path: Path) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    dataset_root.mkdir()
    model_dir = tmp_path / "publisher_model"
    model_dir.mkdir()
    urdf_path = model_dir / "mobility.urdf"
    urdf_path.write_text("<robot/>", encoding="utf-8")
    asset = _asset(module, model_dir, urdf_path)

    with pytest.raises(ArtiverseContractError, match="outside"):
        module._validate_publisher_asset(
            asset, dataset_root, model_dir / "scenesmith_artiverse.sdf"
        )


def test_prepare_rejects_converted_sdf_path_outside_model(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    model_dir = dataset_root / "data" / "cabinet" / "publisher" / "model_001"
    urdf_dir = model_dir / "urdf_w_collider"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "mobility.urdf"
    urdf_path.write_text(_publisher_urdf(), encoding="utf-8")
    asset = _asset(module, model_dir, urdf_path)

    with pytest.raises(RuntimeError, match="escapes its publisher model"):
        module._validate_publisher_asset(
            asset, dataset_root, model_dir / ".." / "escaped.sdf"
        )


def test_prepare_rejects_existing_sdf_symlink(monkeypatch, tmp_path: Path) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    model_dir = dataset_root / "data" / "cabinet" / "publisher" / "model_001"
    urdf_dir = model_dir / "urdf_w_collider"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "mobility.urdf"
    urdf_path.write_text("<robot/>", encoding="utf-8")
    outside_sdf = tmp_path / "outside.sdf"
    outside_sdf.write_text("<sdf/>", encoding="utf-8")
    sdf_path = urdf_dir / "scenesmith_artiverse.sdf"
    try:
        sdf_path.symlink_to(outside_sdf)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are unavailable for this test user: {exc}")
    asset = _asset(module, model_dir, urdf_path)

    with pytest.raises(ArtiverseContractError, match="symlink|junction"):
        module._validate_publisher_asset(asset, dataset_root, sdf_path)


def test_visual_missing_gltf_rewrites_to_existing_obj_without_touching_collision(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    model_dir = tmp_path / "model"
    mesh_dir = model_dir / "urdf_w_collider" / "meshes"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "door.obj").write_text("visual obj", encoding="utf-8")
    (mesh_dir / "collision.gltf").write_text("collision gltf", encoding="utf-8")
    sdf_path = model_dir / "urdf_w_collider" / "scenesmith_artiverse.sdf"
    sdf_path.write_text(
        "<sdf><model><link>"
        "<visual><geometry><mesh><uri>meshes/door.gltf</uri></mesh></geometry></visual>"
        "<collision><geometry><mesh><uri>meshes/collision.gltf</uri></mesh>"
        "</geometry></collision>"
        "</link></model></sdf>",
        encoding="utf-8",
    )

    assert module._rewrite_missing_visual_gltf_uris(sdf_path, model_dir) == (
        ("meshes/door.gltf", "meshes/door.obj"),
    )
    document = module.ET.parse(sdf_path).getroot()
    assert document.find(".//visual/geometry/mesh/uri").text == "meshes/door.obj"
    assert (
        document.find(".//collision/geometry/mesh/uri").text
        == "meshes/collision.gltf"
    )
    assert set(module.validate_sdf_resource_uris(sdf_path, model_dir)) == {
        (mesh_dir / "door.obj").resolve(),
        (mesh_dir / "collision.gltf").resolve(),
    }


def test_existing_visual_gltf_is_preserved_even_when_obj_exists(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    model_dir = tmp_path / "model"
    mesh_dir = model_dir / "urdf_w_collider" / "meshes"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "door.gltf").write_text("visual gltf", encoding="utf-8")
    (mesh_dir / "door.obj").write_text("visual obj", encoding="utf-8")
    sdf_path = model_dir / "urdf_w_collider" / "scenesmith_artiverse.sdf"
    sdf_path.write_text(
        "<sdf><model><link><visual><geometry><mesh>"
        "<uri>meshes/door.gltf</uri>"
        "</mesh></geometry></visual></link></model></sdf>",
        encoding="utf-8",
    )
    original = sdf_path.read_bytes()

    assert module._rewrite_missing_visual_gltf_uris(sdf_path, model_dir) == ()
    assert sdf_path.read_bytes() == original


def test_unindexed_preexisting_sdf_is_rejected(monkeypatch, tmp_path: Path) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    model_dir = dataset_root / "data" / "cabinet" / "publisher" / "model_001"
    urdf_dir = model_dir / "urdf_w_collider"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "mobility.urdf"
    urdf_path.write_text("<robot/>", encoding="utf-8")
    sdf_path = urdf_dir / "scenesmith_artiverse.sdf"
    sdf_path.write_text("<sdf/>", encoding="utf-8")
    asset = _asset(module, model_dir, urdf_path)

    with pytest.raises(RuntimeError, match="unindexed preexisting Artiverse SDF"):
        module._validate_publisher_asset(asset, dataset_root, sdf_path)


@pytest.mark.parametrize("unexpected_write", ["extra_file", "source_mutation"])
def test_conversion_contract_rejects_every_write_except_the_derived_sdf(
    monkeypatch, tmp_path: Path, unexpected_write: str
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    model_dir = dataset_root / "data" / "cabinet" / "publisher" / "model_001"
    urdf_dir = model_dir / "urdf_w_collider"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "mobility.urdf"
    urdf_path.write_text("<robot/>", encoding="utf-8")
    asset = _asset(module, model_dir, urdf_path)
    sdf_path = urdf_dir / "scenesmith_artiverse.sdf"
    publisher_state = module._validate_publisher_asset(
        asset, dataset_root, sdf_path
    )
    sdf_path.write_text("<sdf><model/></sdf>", encoding="utf-8")
    if unexpected_write == "extra_file":
        (urdf_dir / "merged_visual.gltf").write_text("derived", encoding="utf-8")
    else:
        urdf_path.write_text("<robot name='mutated'/>", encoding="utf-8")

    with pytest.raises(RuntimeError, match="single derived SDF contract"):
        module._validate_converted_asset(
            asset,
            dataset_root,
            sdf_path,
            publisher_files_before=publisher_state,
        )


def test_conversion_writes_only_sdf_and_uses_visual_obj_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    _write_pinned_metadata(
        module,
        monkeypatch,
        dataset_root,
        ["data/cabinet/publisher/model_001"],
    )
    model_dir = dataset_root / "data" / "cabinet" / "publisher" / "model_001"
    urdf_dir = model_dir / "urdf_w_collider"
    mesh_dir = urdf_dir / "meshes"
    image_dir = model_dir / "imgs"
    mesh_dir.mkdir(parents=True)
    image_dir.mkdir()
    urdf_path = urdf_dir / "mobility.urdf"
    urdf_path.write_text(_publisher_urdf(), encoding="utf-8")
    (mesh_dir / "door.obj").write_text("visual obj", encoding="utf-8")
    (mesh_dir / "collision.gltf").write_text("collision gltf", encoding="utf-8")
    (image_dir / "reference.png").write_bytes(b"reference image")
    before_paths = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file()
    }
    converter_calls = []

    def fake_convert(**kwargs):
        converter_calls.append(kwargs)
        kwargs["output_path"].write_text(
            _converted_sdf(
                visual_uri="meshes/door.gltf",
                collision_uri="meshes/collision.gltf",
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(module, "convert_urdf_to_sdf", fake_convert)
    monkeypatch.setattr(
        module,
        "combine_sdf_meshes_at_joint_angles",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            bounds=module.np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        ),
    )
    monkeypatch.setattr(
        module,
        "get_multiview_image_embedding",
        lambda _images: module.np.asarray([1.0, 0.0], dtype=module.np.float32),
    )

    manifest = module.prepare(
        dataset_root=dataset_root,
        output_path=dataset_root / "embeddings",
        categories={"cabinet"},
        max_images=1,
        checkpoint_every=1,
        limit=None,
    )

    assert manifest["status"] == "pass"
    assert len(converter_calls) == 1
    assert (
        converter_calls[0]["output_path"].name
        == module.DERIVED_SDF_TEMP_FILENAME
    )
    assert converter_calls[0]["repair_missing_meshes"] is False
    assert converter_calls[0]["merge_visuals"] is False
    assert converter_calls[0]["generate_collision"] is False
    assert set(converter_calls[0]["link_physics"]) == {"base", "door"}
    assert converter_calls[0]["link_physics"]["base"].mass == 1.0
    after_paths = {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file()
    }
    sdf_relative = "urdf_w_collider/scenesmith_artiverse.sdf"
    assert after_paths == before_paths | {sdf_relative}
    document = module.ET.parse(model_dir / sdf_relative).getroot()
    assert document.find(".//visual/geometry/mesh/uri").text == "meshes/door.obj"
    assert (
        document.find(".//collision/geometry/mesh/uri").text
        == "meshes/collision.gltf"
    )
    metadata = module.yaml.safe_load(
        (dataset_root / "embeddings" / "metadata_index.yaml").read_text(
            encoding="utf-8"
        )
    )["artiverse/cabinet/publisher/model_001"]
    assert metadata["physics_source"] == "publisher_urdf_inertial_v1"
    assert metadata["physics_link_count"] == 2
    assert metadata["physics_geometry_link_count"] == 1
    assert metadata["physics_total_mass_kg"] == 2.0
    assert metadata["source_collision_element_count"] == 1
    assert metadata["publisher_urdf_path"].endswith(
        "data/cabinet/publisher/model_001/urdf_w_collider/mobility.urdf"
    )
    assert metadata["publisher_urdf_sha256"] == module.sha256_file(urdf_path)
    assert len(metadata["publisher_link_physics_sha256"]) == 64
    assert len(metadata["emitted_sdf_physics_sha256"]) == 64
    assert manifest["schema_version"] == 2
    assert manifest["physics_bound_indexed_count"] == 1
    assert manifest["physics_binding_sha256"] == module.physics_binding_sha256(
        {"artiverse/cabinet/publisher/model_001": metadata},
        ["artiverse/cabinet/publisher/model_001"],
    )
    assert manifest["physics_policy"] == {
        "id": "publisher_urdf_inertial_v1",
        "required_for_every_link": True,
        "inertial_frame_transform": "urdf_rpy_RzRyRx_to_link_frame_v1",
        "require_zero_emitted_inertial_rpy": True,
        "preserve_source_collision_count": True,
        "preconversion_collision_cap": True,
        "publisher_mass_semantics": "publisher_unit_mass_proxy_not_material_density_v1",
        "joint_dynamics_policy": "scenesmith_defaults_damping_friction_0.05_v1",
        "collision_friction_policy": "scenesmith_default_0.5_v1",
    }


def test_crash_after_sdf_publish_resumes_by_deterministic_regeneration(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    _write_pinned_metadata(
        module,
        monkeypatch,
        dataset_root,
        ["data/cabinet/publisher/model_001"],
    )
    model_dir = dataset_root / "data/cabinet/publisher/model_001"
    urdf_dir = model_dir / "urdf_w_collider"
    image_dir = model_dir / "imgs"
    urdf_dir.mkdir(parents=True)
    image_dir.mkdir()
    urdf_path = urdf_dir / "mobility.urdf"
    urdf_path.write_text(_publisher_urdf(), encoding="utf-8")
    (image_dir / "reference.png").write_bytes(b"reference")
    deterministic_sdf = _converted_sdf(model_name="cabinet")
    converter_outputs: list[Path] = []

    def fake_convert(**kwargs):
        converter_outputs.append(kwargs["output_path"])
        kwargs["output_path"].write_text(deterministic_sdf, encoding="utf-8")

    monkeypatch.setattr(module, "convert_urdf_to_sdf", fake_convert)
    monkeypatch.setattr(
        module,
        "combine_sdf_meshes_at_joint_angles",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            bounds=module.np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        ),
    )
    embedding_attempts = 0

    def fail_once_then_embed(_images):
        nonlocal embedding_attempts
        embedding_attempts += 1
        if embedding_attempts == 1:
            raise RuntimeError("simulated crash before index checkpoint")
        return module.np.asarray([1.0, 0.0], dtype=module.np.float32)

    monkeypatch.setattr(module, "get_multiview_image_embedding", fail_once_then_embed)
    output_path = dataset_root / "embeddings"

    with pytest.raises(ValueError, match="empty Artiverse retrieval index"):
        module.prepare(
            dataset_root=dataset_root,
            output_path=output_path,
            categories={"cabinet"},
            max_images=1,
            checkpoint_every=10,
            limit=None,
        )

    sdf_path = urdf_dir / module.DERIVED_SDF_FILENAME
    assert sdf_path.read_text(encoding="utf-8") == deterministic_sdf
    assert not (urdf_dir / module.DERIVED_SDF_TEMP_FILENAME).exists()

    manifest = module.prepare(
        dataset_root=dataset_root,
        output_path=output_path,
        categories={"cabinet"},
        max_images=1,
        checkpoint_every=10,
        limit=None,
    )

    assert manifest["status"] == "pass"
    assert manifest["added_count"] == 1
    assert len(converter_outputs) == 2
    assert all(
        path.name == module.DERIVED_SDF_TEMP_FILENAME
        for path in converter_outputs
    )
    metadata = module.yaml.safe_load(
        (output_path / "metadata_index.yaml").read_text(encoding="utf-8")
    )
    record = metadata["artiverse/cabinet/publisher/model_001"]
    assert record["sdf_sha256"] == module.sha256_file(sdf_path)
    assert record["sdf_directory_tree_sha256"] == module.sha256_directory_tree(
        sdf_path.parent
    )
    assert not (urdf_dir / module.DERIVED_SDF_TEMP_FILENAME).exists()


def test_unindexed_sdf_must_match_deterministic_regeneration_without_deletion(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    model_dir = dataset_root / "data/cabinet/publisher/model_001"
    urdf_dir = model_dir / "urdf_w_collider"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "mobility.urdf"
    urdf_path.write_text(_publisher_urdf(), encoding="utf-8")
    asset = _asset(module, model_dir, urdf_path)
    sdf_path = urdf_dir / module.DERIVED_SDF_FILENAME
    unexplained = _converted_sdf(model_name="unexplained")
    regenerated = _converted_sdf(model_name="deterministic")
    sdf_path.write_text(unexplained, encoding="utf-8")
    publisher_state = module._validate_publisher_asset(
        asset,
        dataset_root,
        sdf_path,
        recover_unindexed_sdf=True,
    )
    monkeypatch.setattr(
        module,
        "convert_urdf_to_sdf",
        lambda **kwargs: kwargs["output_path"].write_text(
            regenerated, encoding="utf-8"
        ),
    )

    with pytest.raises(RuntimeError, match="does not match deterministic regeneration"):
        module._convert_or_recover_sdf(
            asset,
            dataset_root,
            sdf_path,
            publisher_state,
            module._publisher_physics_contract(
                urdf_path, max_collision_elements=32
            ),
        )

    assert sdf_path.read_text(encoding="utf-8") == unexplained
    assert not (urdf_dir / module.DERIVED_SDF_TEMP_FILENAME).exists()


def test_unindexed_hardlinked_sdf_is_rejected_before_recovery(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    model_dir = dataset_root / "data/cabinet/publisher/model_001"
    urdf_dir = model_dir / "urdf_w_collider"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "mobility.urdf"
    urdf_path.write_text("<robot/>", encoding="utf-8")
    sdf_path = urdf_dir / module.DERIVED_SDF_FILENAME
    sdf_path.write_text("<sdf/>", encoding="utf-8")
    try:
        (urdf_dir / "sdf-hardlink").hardlink_to(sdf_path)
    except OSError as exc:
        pytest.skip(f"hardlinks are unavailable for this test user: {exc}")
    asset = _asset(module, model_dir, urdf_path)

    with pytest.raises(RuntimeError, match="hard-linked"):
        module._validate_publisher_asset(
            asset,
            dataset_root,
            sdf_path,
            recover_unindexed_sdf=True,
        )


def test_movable_joint_summary_accepts_only_supported_types(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    sdf_path = tmp_path / "asset.sdf"
    sdf_path.write_text(
        "<sdf><model>"
        "<joint type='fixed'/><joint type='REVOLUTE'/>"
        "<joint type='continuous'/><joint type='prismatic'/>"
        "<joint type='ball'/></model></sdf>",
        encoding="utf-8",
    )

    assert module._movable_joint_summary(sdf_path) == (
        3,
        ["continuous", "prismatic", "revolute"],
    )


def test_publisher_physics_preserves_every_link_including_geometry_free_proxy(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    urdf_path = tmp_path / "proxy.urdf"
    urdf_path.write_text(
        "<robot><link name='base'>"
        + _urdf_inertial(
            mass="2.5", xyz="0.1 -0.2 0.3", ixx="0.1", iyy="0.2", izz="0.25"
        )
        + "<visual><geometry><box size='1 1 1'/></geometry></visual>"
        "<collision><geometry><box size='1 1 1'/></geometry></collision>"
        "</link><link name='multi_dof_proxy'>"
        + _urdf_inertial(
            mass="1e-6", ixx="1e-9", iyy="1e-9", izz="1e-9"
        )
        + "</link></robot>",
        encoding="utf-8",
    )

    contract = module._publisher_physics_contract(
        urdf_path, max_collision_elements=32
    )

    assert contract.link_names == ("base", "multi_dof_proxy")
    assert contract.geometry_link_names == ("base",)
    assert contract.collision_count_by_link == {"base": 1, "multi_dof_proxy": 0}
    assert contract.source_collision_element_count == 1
    assert contract.link_physics["base"].mass == 2.5
    assert contract.link_physics["base"].center_of_mass == (0.1, -0.2, 0.3)
    assert contract.link_physics["base"].inertia_izz == 0.25
    assert contract.link_physics["multi_dof_proxy"].mass == 1e-6
    assert contract.link_physics["multi_dof_proxy"].inertia_ixx == 1e-9
    assert contract.total_mass_kg == pytest.approx(2.500001)


@pytest.mark.parametrize(
    ("inertial", "error"),
    [
        ("", "exactly one <inertial>"),
        (_urdf_inertial() + _urdf_inertial(), "exactly one <inertial>"),
        (_urdf_inertial(mass="0"), "mass must be positive"),
        (_urdf_inertial(mass="-1"), "mass must be positive"),
        (_urdf_inertial(mass="nan"), "mass is non-finite"),
        (_urdf_inertial(mass="inf"), "mass is non-finite"),
        (_urdf_inertial(xyz="0 0"), "xyz must contain exactly 3"),
        (_urdf_inertial(xyz="0 nan 0"), "xyz contains a non-finite"),
        (
            _urdf_inertial().replace(" iyz='0'", ""),
            "inertia iyz is missing",
        ),
        (_urdf_inertial(ixx="nan"), "inertia ixx is non-finite"),
        (_urdf_inertial(ixx="-0.1"), "not positive definite"),
        (
            _urdf_inertial(ixx="0.1", iyy="0.1", izz="0.3"),
            "triangle inequality",
        ),
    ],
)
def test_publisher_physics_rejects_malformed_inertials(
    monkeypatch, tmp_path: Path, inertial: str, error: str
) -> None:
    module = _load_prepare_module(monkeypatch)
    urdf_path = tmp_path / "bad.urdf"
    urdf_path.write_text(
        "<robot><link name='base'>"
        + inertial
        + "<visual/><collision/></link></robot>",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error):
        module._publisher_physics_contract(urdf_path, max_collision_elements=32)


def test_publisher_inertial_rpy_is_rotated_into_link_frame(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    urdf_path = tmp_path / "rotated.urdf"
    urdf_path.write_text(
        "<robot><link name='base'>"
        + _urdf_inertial(
            rpy="0 0 1.5707963267948966",
            ixx="0.1",
            iyy="0.2",
            izz="0.25",
        )
        + "<visual/><collision/></link></robot>",
        encoding="utf-8",
    )

    contract = module._publisher_physics_contract(
        urdf_path,
        max_collision_elements=32,
    )
    physics = contract.link_physics["base"]

    assert physics.inertia_ixx == pytest.approx(0.2)
    assert physics.inertia_iyy == pytest.approx(0.1)
    assert physics.inertia_izz == pytest.approx(0.25)
    assert physics.inertia_ixy == pytest.approx(0.0, abs=1e-15)
    assert len(contract.publisher_link_physics_sha256) == 64


def test_publisher_physics_rejects_duplicate_link_names(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    urdf_path = tmp_path / "duplicate.urdf"
    link = "<link name='same'>" + _urdf_inertial() + "</link>"
    urdf_path.write_text("<robot>" + link + link + "</robot>", encoding="utf-8")

    with pytest.raises(ValueError, match="repeats link name"):
        module._publisher_physics_contract(urdf_path, max_collision_elements=32)


def test_geometry_link_requires_source_collision(monkeypatch, tmp_path: Path) -> None:
    module = _load_prepare_module(monkeypatch)
    urdf_path = tmp_path / "no_collision.urdf"
    urdf_path.write_text(
        "<robot><link name='base'>"
        + _urdf_inertial()
        + "<visual/></link></robot>",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no source collision"):
        module._publisher_physics_contract(urdf_path, max_collision_elements=32)


def test_source_collision_cap_accepts_32(monkeypatch, tmp_path: Path) -> None:
    module = _load_prepare_module(monkeypatch)
    urdf_path = tmp_path / "thirty_two.urdf"
    urdf_path.write_text(_publisher_urdf(collision_count=32), encoding="utf-8")

    contract = module._publisher_physics_contract(
        urdf_path, max_collision_elements=32
    )

    assert contract.source_collision_element_count == 32
    assert contract.collision_count_by_link["base"] == 32


def test_source_collision_cap_rejects_33_before_converter_or_sdf_creation(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    _write_pinned_metadata(
        module,
        monkeypatch,
        dataset_root,
        ["data/cabinet/publisher/model_001"],
    )
    urdf_dir = (
        dataset_root
        / "data"
        / "cabinet"
        / "publisher"
        / "model_001"
        / "urdf_w_collider"
    )
    urdf_dir.mkdir(parents=True)
    (urdf_dir / "mobility.urdf").write_text(
        _publisher_urdf(collision_count=33), encoding="utf-8"
    )
    converter_called = False

    def forbidden_converter(**_kwargs):
        nonlocal converter_called
        converter_called = True
        pytest.fail("over-cap source must be rejected before conversion")

    monkeypatch.setattr(module, "convert_urdf_to_sdf", forbidden_converter)
    with pytest.raises(ValueError, match="empty Artiverse retrieval index"):
        module.prepare(
            dataset_root=dataset_root,
            output_path=dataset_root / "embeddings",
            categories={"cabinet"},
            max_images=1,
            checkpoint_every=1,
            limit=None,
        )

    assert converter_called is False
    assert not (urdf_dir / module.DERIVED_SDF_FILENAME).exists()
    assert not (urdf_dir / module.DERIVED_SDF_TEMP_FILENAME).exists()


def test_rejected_converted_candidate_leaves_no_final_or_transaction_sdf(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    model_dir = dataset_root / "data/cabinet/publisher/model_001"
    urdf_dir = model_dir / "urdf_w_collider"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "mobility.urdf"
    urdf_path.write_text(_publisher_urdf(), encoding="utf-8")
    asset = _asset(module, model_dir, urdf_path)
    sdf_path = urdf_dir / module.DERIVED_SDF_FILENAME
    publisher_state = module._validate_publisher_asset(
        asset,
        dataset_root,
        sdf_path,
    )
    physics = module._publisher_physics_contract(
        urdf_path,
        max_collision_elements=32,
        require_movable_joint=True,
    )

    def write_wrong_mass(**kwargs):
        kwargs["output_path"].write_text(
            _converted_sdf().replace(
                "<mass>1.000000</mass>",
                "<mass>2.000000</mass>",
                1,
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(module, "convert_urdf_to_sdf", write_wrong_mass)

    with pytest.raises(ValueError, match="mass does not match publisher"):
        module._convert_or_recover_sdf(
            asset,
            dataset_root,
            sdf_path,
            publisher_state,
            physics,
        )

    assert not sdf_path.exists()
    assert not (urdf_dir / module.DERIVED_SDF_TEMP_FILENAME).exists()


@pytest.mark.parametrize(
    ("broken_sdf", "error"),
    [
        (
            _converted_sdf().replace("<mass>1.000000</mass>", "<mass>2</mass>", 1),
            "mass does not match publisher",
        ),
        (
            _converted_sdf().replace("<iyz>0.000000e+00</iyz>", "", 1),
            "exactly one <iyz>",
        ),
        (
            _converted_sdf().replace("<izz>1.000000e-01</izz>", "<izz>3</izz>", 1),
            "triangle inequality",
        ),
        (
            _converted_sdf().replace(
                "<collision name='body_0'><geometry><box><size>1 1 1</size></box></geometry></collision>",
                "",
                1,
            ),
            "collision count does not match publisher",
        ),
        (
            _converted_sdf().replace(
                "</model>",
                "<link name='extra'>" + _sdf_inertial() + "</link></model>",
            ),
            "link set differs from publisher",
        ),
    ],
)
def test_post_conversion_rejects_physics_or_collision_substitution(
    monkeypatch, tmp_path: Path, broken_sdf: str, error: str
) -> None:
    module = _load_prepare_module(monkeypatch)
    urdf_path = tmp_path / "source.urdf"
    sdf_path = tmp_path / "converted.sdf"
    urdf_path.write_text(_publisher_urdf(), encoding="utf-8")
    sdf_path.write_text(broken_sdf, encoding="utf-8")
    contract = module._publisher_physics_contract(
        urdf_path, max_collision_elements=32
    )

    with pytest.raises(ValueError, match=error):
        module._validate_sdf_physics_and_collisions(sdf_path, contract)


def test_post_conversion_happy_path_is_repeatable_and_non_mutating(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    urdf_path = tmp_path / "source.urdf"
    sdf_path = tmp_path / "converted.sdf"
    urdf_path.write_text(_publisher_urdf(), encoding="utf-8")
    sdf_path.write_text(_converted_sdf(), encoding="utf-8")
    contract = module._publisher_physics_contract(
        urdf_path, max_collision_elements=32
    )
    before = sdf_path.read_bytes()

    module._validate_sdf_physics_and_collisions(sdf_path, contract)
    module._validate_sdf_physics_and_collisions(sdf_path, contract)

    assert sdf_path.read_bytes() == before
    evidence = module._physics_metadata(
        contract,
        dataset_root=tmp_path,
        publisher_urdf=urdf_path,
        emitted_sdf=sdf_path,
    )
    assert evidence == {
        "physics_source": "publisher_urdf_inertial_v1",
        "publisher_urdf_path": "source.urdf",
        "publisher_urdf_sha256": module.sha256_file(urdf_path),
        "publisher_link_physics_sha256": contract.publisher_link_physics_sha256,
        "emitted_sdf_physics_sha256": module.emitted_sdf_physics_evidence(
            sdf_path
        ).sha256,
        "physics_link_count": 2,
        "physics_geometry_link_count": 1,
        "physics_total_mass_kg": 2.0,
        "source_collision_element_count": 1,
    }


def test_requested_categories_must_exist_in_pinned_manifest_roots(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    _write_pinned_metadata(
        module,
        monkeypatch,
        dataset_root,
        ["data/cabinet/publisher/model_001"],
    )
    monkeypatch.setattr(
        module,
        "discover_assets",
        lambda *_args, **_kwargs: pytest.fail(
            "asset discovery must not authorize a category absent from the manifest"
        ),
    )

    with pytest.raises(RuntimeError, match="absent from the pinned source manifest"):
        module.prepare(
            dataset_root=dataset_root,
            output_path=dataset_root / "embeddings",
            categories={"cabinet", "invented_category"},
            max_images=1,
            checkpoint_every=1,
            limit=None,
        )


def test_resume_backfills_joint_fields_and_enforces_per_category_minimum(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root, output_path, object_id = _write_resumable_asset(
        module, monkeypatch, tmp_path, joint_type="revolute"
    )
    indexed_sdf = next(dataset_root.rglob(module.DERIVED_SDF_FILENAME))
    indexed_sdf_before = indexed_sdf.read_bytes()
    monkeypatch.setattr(
        module,
        "convert_urdf_to_sdf",
        lambda **_kwargs: pytest.fail(
            "an indexed and hash-bound SDF must never be regenerated or replaced"
        ),
    )

    manifest = module.prepare(
        dataset_root=dataset_root,
        output_path=output_path,
        categories={"cabinet"},
        max_images=1,
        checkpoint_every=1,
        limit=None,
        minimum_indexed_per_category=2,
    )

    assert manifest["status"] == "fail"
    assert indexed_sdf.read_bytes() == indexed_sdf_before
    assert manifest["indexed_target_count"] == 1
    assert manifest["minimum_indexed_per_category"] == 2
    assert manifest["categories_below_minimum_indexed"] == ["cabinet"]
    assert manifest["requested_category_counts"] == {
        "cabinet": {
            "manifest_root_count": 1,
            "discovered_count": 1,
            "indexed_count": 1,
        }
    }
    assert manifest["source_extraction_receipt"] == {
        "path": "artiverse_safe_extraction_receipt.json",
        "sha256": module.sha256_file(
            dataset_root / "artiverse_safe_extraction_receipt.json"
        ),
    }
    assert manifest["safe_extractor_sha256"] == "c" * 64
    metadata = module.yaml.safe_load(
        (output_path / "metadata_index.yaml").read_text(encoding="utf-8")
    )
    assert metadata[object_id]["movable_joint_count"] == 1
    assert metadata[object_id]["movable_joint_types"] == ["revolute"]


def test_resume_entry_without_movable_joint_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root, output_path, _object_id = _write_resumable_asset(
        module, monkeypatch, tmp_path, joint_type="fixed"
    )

    with pytest.raises(ValueError, match="no usable movable joint"):
        module.prepare(
            dataset_root=dataset_root,
            output_path=output_path,
            categories={"cabinet"},
            max_images=1,
            checkpoint_every=1,
            limit=None,
        )


def test_resume_rejects_sdf_that_changed_after_it_was_indexed(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root, output_path, _object_id = _write_resumable_asset(
        module, monkeypatch, tmp_path, joint_type="revolute"
    )
    sdf_path = next(dataset_root.rglob("scenesmith_artiverse.sdf"))
    sdf_path.write_text(
        sdf_path.read_text(encoding="utf-8") + "\n<!-- changed -->\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="SDF hash mismatch"):
        module.prepare(
            dataset_root=dataset_root,
            output_path=output_path,
            categories={"cabinet"},
            max_images=1,
            checkpoint_every=1,
            limit=None,
        )


def test_resume_rejects_sibling_tree_change_after_sdf_was_indexed(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root, output_path, _object_id = _write_resumable_asset(
        module, monkeypatch, tmp_path, joint_type="revolute"
    )
    sdf_path = next(dataset_root.rglob("scenesmith_artiverse.sdf"))
    (sdf_path.parent / "unbound-derived-file.gltf").write_text(
        "unexpected sibling", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="directory-tree hash mismatch"):
        module.prepare(
            dataset_root=dataset_root,
            output_path=output_path,
            categories={"cabinet"},
            max_images=1,
            checkpoint_every=1,
            limit=None,
        )


def test_resume_rejects_metadata_bound_to_a_different_sdf_path(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root, output_path, object_id = _write_resumable_asset(
        module, monkeypatch, tmp_path, joint_type="revolute"
    )
    expected_sdf = next(dataset_root.rglob("scenesmith_artiverse.sdf"))
    alternate_sdf = expected_sdf.with_name("alternate.sdf")
    alternate_sdf.write_bytes(expected_sdf.read_bytes())
    metadata_path = output_path / "metadata_index.yaml"
    metadata = module.yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata[object_id]["sdf_path"] = str(alternate_sdf.relative_to(dataset_root))
    metadata[object_id]["sdf_sha256"] = module.sha256_file(alternate_sdf)
    metadata[object_id]["sdf_directory_tree_sha256"] = module.sha256_directory_tree(
        alternate_sdf.parent
    )
    metadata_path.write_text(
        module.yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="unindexed preexisting Artiverse SDF"):
        module.prepare(
            dataset_root=dataset_root,
            output_path=output_path,
            categories={"cabinet"},
            max_images=1,
            checkpoint_every=1,
            limit=None,
        )


def test_cli_minimum_indexed_per_category_defaults_to_zero(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(module, "prepare", fake_prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_artiverse.py",
            "--dataset-root",
            str(tmp_path / "artiverse"),
        ],
    )

    assert module.main() == 0
    assert captured["minimum_indexed_per_category"] == 0


def test_prepare_requires_safe_extraction_receipt_before_asset_discovery(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    source_manifest = dataset_root / "dataset_chunks" / "manifest.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text('{"chunks": []}\n', encoding="utf-8")
    pack_script = dataset_root / "pack_dataset_chunks.py"
    pack_script.write_text("# official unpacker\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "OFFICIAL_SOURCE_MANIFEST_SHA256",
        module.sha256_file(source_manifest),
    )
    monkeypatch.setattr(
        module,
        "OFFICIAL_PACK_SCRIPT_SHA256",
        module.sha256_file(pack_script),
    )
    monkeypatch.setattr(
        module,
        "validate_extraction_receipt",
        lambda _dataset_root: (_ for _ in ()).throw(
            ArtiverseContractError("safe-extraction receipt is missing")
        ),
    )
    monkeypatch.setattr(
        module,
        "discover_assets",
        lambda *_args, **_kwargs: pytest.fail(
            "asset discovery must not run without extraction provenance"
        ),
    )

    with pytest.raises(ArtiverseContractError, match="receipt is missing"):
        module.prepare(
            dataset_root=dataset_root,
            output_path=dataset_root / "embeddings",
            categories=None,
            max_images=1,
            checkpoint_every=1,
            limit=None,
        )


@pytest.mark.parametrize("metadata_kind", ["source_manifest", "pack_script"])
def test_prepare_rejects_unpinned_metadata_before_asset_discovery(
    monkeypatch, tmp_path: Path, metadata_kind: str
) -> None:
    module = _load_prepare_module(monkeypatch)
    dataset_root = tmp_path / "artiverse"
    source_manifest = dataset_root / "dataset_chunks" / "manifest.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text('{"official": true}\n', encoding="utf-8")
    pack_script = dataset_root / "pack_dataset_chunks.py"
    pack_script.write_text("# official unpacker\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "OFFICIAL_SOURCE_MANIFEST_SHA256",
        module.sha256_file(source_manifest),
    )
    monkeypatch.setattr(
        module,
        "OFFICIAL_PACK_SCRIPT_SHA256",
        module.sha256_file(pack_script),
    )

    if metadata_kind == "source_manifest":
        source_manifest.write_text('{"substituted": true}\n', encoding="utf-8")
    else:
        pack_script.write_text("# substituted unpacker\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "discover_assets",
        lambda *_args, **_kwargs: pytest.fail(
            "asset discovery must not run for unpinned publisher metadata"
        ),
    )

    with pytest.raises(RuntimeError, match="pinned audited digest"):
        module.prepare(
            dataset_root=dataset_root,
            output_path=dataset_root / "embeddings",
            categories=None,
            max_images=1,
            checkpoint_every=1,
            limit=None,
        )
