from __future__ import annotations

import copy
import hashlib
import json
import shutil
import struct
import sys
import xml.etree.ElementTree as ET

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import preflight_artiverse_visual_resources as visual_contract  # noqa: E402
from scenesmith.agent_utils.artiverse_visual_normalization import (  # noqa: E402
    normalize_copied_artiverse_visuals,
)


def _write_glb(
    path: Path,
    *,
    include_normal: bool = True,
    external_image_uri: str | None = None,
    corrupt_buffer_length: bool = False,
) -> None:
    attributes = {"POSITION": 0}
    if include_normal:
        attributes["NORMAL"] = 1
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    normals = struct.pack("<9f", 0, 0, 1, 0, 0, 1, 0, 0, 1)
    indices = struct.pack("<3H", 0, 1, 2)
    padding = b"\x00\x00"
    image = b"\x89PNG\r\n\x1a\nTEST"
    binary = positions + normals + indices + padding + image
    image_record: dict[str, object]
    if external_image_uri is None:
        image_record = {"bufferView": 3, "mimeType": "image/png"}
    else:
        image_record = {"uri": external_image_uri}
    document = {
        "asset": {"version": "2.0"},
        "buffers": [
            {"byteLength": len(binary) + (1 if corrupt_buffer_length else 0)}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(normals), "target": 34962},
            {"buffer": 0, "byteOffset": len(positions) + len(normals), "byteLength": len(indices), "target": 34963},
            {"buffer": 0, "byteOffset": 80, "byteLength": len(image)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "images": [image_record],
        "textures": [{"source": 0}],
        "materials": [
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": 2,
                        "material": 0,
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((-len(payload)) % 4)
    binary += b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(payload) + 8 + len(binary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(payload), visual_contract.GLB_JSON_CHUNK_TYPE)
        + payload
        + struct.pack("<II", len(binary), visual_contract.GLB_BIN_CHUNK_TYPE)
        + binary
    )


def _visual(name: str, uri: str, *, pose: str = "0 0 0 0 0 0") -> str:
    return f"""<visual name="{name}">
      <pose>{pose}</pose>
      <geometry><mesh><uri>{uri}</uri><scale>1 1 1</scale></mesh></geometry>
      <material><diffuse>0.7 0.6 0.5 1</diffuse></material>
    </visual>"""


def _collision(name: str, uri: str, *, pose: str = "0 0 0 0 0 0") -> str:
    return f"""<collision name="{name}">
      <pose>{pose}</pose>
      <geometry><mesh><uri>{uri}</uri><scale>1 1 1</scale></mesh></geometry>
    </collision>"""


def _write_sdf(
    path: Path,
    *,
    visual_uris: tuple[str, ...] = (
        "./objs/1_base__wood.obj",
        "./objs/1_base__metal.obj",
    ),
    collision_uris: tuple[str, ...] | None = None,
    visual_pose: str = "0 0 0 0 0 0",
    collision_pose: str = "0 0 0 0 0 0",
) -> None:
    collisions = collision_uris or visual_uris
    visual_xml = "\n".join(
        _visual(f"visual_{index}", uri, pose=visual_pose)
        for index, uri in enumerate(visual_uris)
    )
    collision_xml = "\n".join(
        _collision(f"collision_{index}", uri, pose=collision_pose)
        for index, uri in enumerate(collisions)
    )
    path.write_text(
        f"""<sdf version="1.7"><model name="fixture">
  <link name="link_1_base">
    <inertial><mass>1</mass><inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz>
      <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
    {visual_xml}
    {collision_xml}
  </link>
  <joint name="fixture_joint" type="revolute"><parent>world</parent>
    <child>link_1_base</child><axis><xyz>0 0 1</xyz>
      <limit><lower>0</lower><upper>1</upper></limit></axis></joint>
</model></sdf>\n""",
        encoding="utf-8",
    )


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(visual_contract.sha256_file(path)))
    return digest.hexdigest()


class _FakeAuthority:
    def __init__(self, dataset_root: Path, object_id: str, sdf_path: Path) -> None:
        self.dataset_root = dataset_root.resolve()
        self.embedding_index = [object_id]
        self.metadata = {
            object_id: {
                "sdf_path": sdf_path.relative_to(dataset_root).as_posix(),
            }
        }

    def asset(self, object_id: str) -> dict[str, object]:
        record = self.metadata[object_id]
        sdf_path = self.dataset_root / record["sdf_path"]
        return {
            "source_sdf_path": sdf_path.resolve(),
            "source_sdf_sha256": visual_contract.sha256_file(sdf_path),
            "source_tree_sha256": _tree_hash(sdf_path.parent),
        }

    def evidence(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "indexed_count": 1,
            "index_sha256": {"fixture": "a" * 64},
            "preparation_manifest": {"sha256": "b" * 64},
        }


def _fixture(tmp_path: Path) -> tuple[_FakeAuthority, str, Path]:
    dataset = tmp_path / "artiverse"
    model = dataset / "data" / "armoire" / "fixture" / "model_001" / "urdf_w_collider"
    (model / "objs").mkdir(parents=True)
    for name in ("1_base__wood.obj", "1_base__metal.obj"):
        (model / "objs" / name).write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
        )
    _write_glb(model / "glbs" / "1_base.glb")
    sdf = model / "scenesmith_artiverse.sdf"
    _write_sdf(sdf)
    object_id = "artiverse/armoire/fixture/model_001"
    return _FakeAuthority(dataset, object_id, sdf), object_id, sdf


def test_audit_maps_every_visual_and_binds_collisions(tmp_path: Path) -> None:
    authority, object_id, sdf = _fixture(tmp_path)
    source_hash = _tree_hash(sdf.parent)

    result = visual_contract.audit_prepared_authority(authority)

    assert result["indexed_asset_count"] == 1
    assert result["audited_asset_count"] == 1
    assert result["mapped_link_count"] == 1
    assert result["visual_mapping_count"] == 2
    assert result["unique_glb_count"] == 1
    assert result["derived_gltf_count"] == 1
    assert result["derived_bin_count"] == 1
    assert result["gltf_material_count"] == 1
    assert result["gltf_image_count"] == 1
    assert result["collision_binding_count"] == 2
    assert result["mapping_target"] == "publisher_glb_derived_external_gltf"
    assert result["publisher_glb_json_bin_resources_validated"] is True
    assert result["derived_external_gltf_hashes_precomputed"] is True
    assert result["source_tree_identity_revalidated_after_audit"] is True
    assert result["source_files_written"] == 0
    assert result["assets"][0]["object_id"] == object_id
    assert result["assets"][0]["source_tree_sha256"] == source_hash
    assert _tree_hash(sdf.parent) == source_hash


def test_receipt_binds_runtime_code_and_attestation(tmp_path: Path) -> None:
    authority, _object_id, _sdf = _fixture(tmp_path)
    runtime = tmp_path / "normalizer.py"
    runtime.write_text("def normalize():\n    return True\n", encoding="utf-8")

    receipt = visual_contract.build_receipt_from_authority(
        authority,
        runtime_code=[runtime],
    )

    assert receipt["status"] == "pass"
    assert receipt["mapping_policy"]["target"] == (
        "publisher_glb_derived_external_gltf"
    )
    assert receipt["mapping_policy"]["target_visual_format"] == (
        "external_gltf_with_external_bin"
    )
    assert receipt["mapping_policy"]["collision_mesh_uris_unchanged"] is True
    assert receipt["runtime_code_revalidated_after_audit"] is True
    assert any(
        item["path"] == str(runtime.resolve()) for item in receipt["runtime_code"]
    )
    visual_contract._validate_saved_receipt(receipt)
    tampered = copy.deepcopy(receipt)
    tampered["audit"]["visual_mapping_count"] = 3
    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="attestation",
    ):
        visual_contract._validate_saved_receipt(tampered)


def test_glb_without_normal_is_rejected(tmp_path: Path) -> None:
    authority, _object_id, sdf = _fixture(tmp_path)
    _write_glb(sdf.parent / "glbs" / "1_base.glb", include_normal=False)

    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="has no POSITION/NORMAL",
    ):
        visual_contract.audit_prepared_authority(authority)


def test_glb_with_external_image_resource_is_rejected(tmp_path: Path) -> None:
    authority, _object_id, sdf = _fixture(tmp_path)
    _write_glb(
        sdf.parent / "glbs" / "1_base.glb",
        external_image_uri="texture.png",
    )

    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="external resource",
    ):
        visual_contract.audit_prepared_authority(authority)


def test_glb_with_inconsistent_bin_length_is_rejected(tmp_path: Path) -> None:
    authority, _object_id, sdf = _fixture(tmp_path)
    _write_glb(
        sdf.parent / "glbs" / "1_base.glb",
        corrupt_buffer_length=True,
    )

    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="BIN chunk length",
    ):
        visual_contract.audit_prepared_authority(authority)


def test_external_gltf_derivation_hashes_are_deterministic(tmp_path: Path) -> None:
    _authority, _object_id, sdf = _fixture(tmp_path)
    glb = sdf.parent / "glbs" / "1_base.glb"

    first, first_gltf, first_bin = visual_contract._read_glb_derivation(
        glb,
        part="1_base",
    )
    second, second_gltf, second_bin = visual_contract._read_glb_derivation(
        glb,
        part="1_base",
    )

    assert first == second
    assert first_gltf == second_gltf
    assert first_bin == second_bin
    derived = json.loads(first_gltf)
    assert derived["buffers"] == [
        {"byteLength": len(first_bin), "uri": "1_base.bin"}
    ]
    preserved = copy.deepcopy(derived)
    del preserved["buffers"][0]["uri"]
    assert first["preserved_semantics_sha256"] == hashlib.sha256(
        visual_contract._canonical_json(preserved) + b"\n"
    ).hexdigest()


def test_one_link_cannot_map_to_multiple_glbs(tmp_path: Path) -> None:
    authority, _object_id, sdf = _fixture(tmp_path)
    second_obj = sdf.parent / "objs" / "2_door__wood.obj"
    second_obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    _write_glb(sdf.parent / "glbs" / "2_door.glb")
    _write_sdf(
        sdf,
        visual_uris=("./objs/1_base__wood.obj", "./objs/2_door__wood.obj"),
        collision_uris=("./objs/1_base__wood.obj", "./objs/2_door__wood.obj"),
    )

    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="multiple GLBs",
    ):
        visual_contract.audit_prepared_authority(authority)


def test_visual_and_collision_pose_must_match(tmp_path: Path) -> None:
    authority, _object_id, sdf = _fixture(tmp_path)
    _write_sdf(sdf, collision_pose="0.1 0 0 0 0 0")

    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="pose/scale-identical collision",
    ):
        visual_contract.audit_prepared_authority(authority)


def test_material_splits_must_have_identical_nonmaterial_structure(
    tmp_path: Path,
) -> None:
    authority, _object_id, sdf = _fixture(tmp_path)
    document = ET.parse(sdf)
    visuals = document.getroot().findall("./model/link/visual")
    assert len(visuals) == 2
    visuals[1].set("cast_shadows", "false")
    document.write(sdf, encoding="utf-8", xml_declaration=True)

    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="render structure",
    ):
        visual_contract.audit_prepared_authority(authority)


def test_unsafe_visual_uri_is_rejected(tmp_path: Path) -> None:
    authority, _object_id, sdf = _fixture(tmp_path)
    _write_sdf(
        sdf,
        visual_uris=("../outside.obj",),
        collision_uris=("../outside.obj",),
    )

    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="escapes",
    ):
        visual_contract.audit_prepared_authority(authority)


def test_normalized_copy_changes_only_allowed_visual_elements(tmp_path: Path) -> None:
    _authority, _object_id, source_sdf = _fixture(tmp_path)
    copied_root = tmp_path / "copied"
    shutil.copytree(source_sdf.parent, copied_root)
    copied_sdf = copied_root / source_sdf.name
    normalization_evidence = normalize_copied_artiverse_visuals(
        copied_sdf,
        copied_root,
    )

    result = visual_contract.validate_normalized_copy(
        source_sdf,
        copied_sdf,
        source_tree_root=source_sdf.parent,
        copied_tree_root=copied_root,
        normalization_evidence=normalization_evidence,
    )

    assert result["status"] == "pass"
    assert result["normalized_link_count"] == 1
    assert result["source_visual_count"] == 2
    assert result["deduplicated_visual_count"] == 1
    assert result["removed_material_override_count"] == 2
    assert result["mapping_target"] == "publisher_glb_derived_external_gltf"
    assert result["publisher_glb_semantics_preserved_in_external_gltf"] is True
    assert result["collision_bindings_unchanged"] is True
    assert result["poses_scales_joints_inertials_unchanged"] is True
    assert result[
        "publisher_source_and_nonderived_copied_resources_unchanged"
    ] is True
    assert result["derived_resource_file_count"] == 3
    assert len(result["runtime_normalization_evidence_sha256"]) == 64

    document = ET.parse(copied_sdf)
    collision_uri = document.getroot().find("./model/link/collision/geometry/mesh/uri")
    assert collision_uri is not None
    collision_uri.text = "./objs/1_base__metal.obj"
    document.write(copied_sdf, encoding="utf-8", xml_declaration=True)
    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="outside the allowed visual",
    ):
        visual_contract.validate_normalized_copy(
            source_sdf,
            copied_sdf,
            source_tree_root=source_sdf.parent,
            copied_tree_root=copied_root,
        )


def test_normalized_copy_rejects_unexpected_or_mutated_derived_resource(
    tmp_path: Path,
) -> None:
    _authority, _object_id, source_sdf = _fixture(tmp_path)
    copied_root = tmp_path / "copied"
    shutil.copytree(source_sdf.parent, copied_root)
    copied_sdf = copied_root / source_sdf.name
    normalize_copied_artiverse_visuals(copied_sdf, copied_root)
    derived = copied_root / visual_contract.DERIVED_RESOURCE_DIRECTORY
    (derived / "unexpected.txt").write_text("not allowed", encoding="utf-8")

    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="inventory is not exact",
    ):
        visual_contract.validate_normalized_copy(
            source_sdf,
            copied_sdf,
            source_tree_root=source_sdf.parent,
            copied_tree_root=copied_root,
        )

    (derived / "unexpected.txt").unlink()
    (derived / "1_base.bin").write_bytes(b"mutated")
    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="derived resource differs",
    ):
        visual_contract.validate_normalized_copy(
            source_sdf,
            copied_sdf,
            source_tree_root=source_sdf.parent,
            copied_tree_root=copied_root,
        )


def test_legacy_direct_glb_copy_migrates_and_validates_against_obj_source(
    tmp_path: Path,
) -> None:
    _authority, _object_id, source_sdf = _fixture(tmp_path)
    copied_root = tmp_path / "copied"
    shutil.copytree(source_sdf.parent, copied_root)
    copied_sdf = copied_root / source_sdf.name
    document = ET.parse(copied_sdf)
    link = document.getroot().find("./model/link")
    assert link is not None
    visuals = link.findall("visual")
    assert len(visuals) == 2
    uri = visuals[0].find("./geometry/mesh/uri")
    assert uri is not None
    uri.text = "./glbs/1_base.glb"
    material = visuals[0].find("material")
    assert material is not None
    visuals[0].remove(material)
    link.remove(visuals[1])
    document.write(copied_sdf, encoding="utf-8", xml_declaration=True)

    evidence = normalize_copied_artiverse_visuals(copied_sdf, copied_root)
    assert evidence["rewritten_link_count"] == 0
    assert evidence["migrated_legacy_glb_link_count"] == 1

    result = visual_contract.validate_normalized_copy(
        source_sdf,
        copied_sdf,
        source_tree_root=source_sdf.parent,
        copied_tree_root=copied_root,
        normalization_evidence=evidence,
    )
    assert result["status"] == "pass"
    assert result["mapping_target"] == "publisher_glb_derived_external_gltf"


def test_receipt_requires_runtime_code(tmp_path: Path) -> None:
    authority, _object_id, _sdf = _fixture(tmp_path)
    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="runtime normalization code",
    ):
        visual_contract.build_receipt_from_authority(authority, runtime_code=[])


def test_runtime_code_change_during_audit_is_rejected(tmp_path: Path) -> None:
    authority, _object_id, _sdf = _fixture(tmp_path)
    runtime = tmp_path / "normalizer.py"
    runtime.write_text("VERSION = 1\n", encoding="utf-8")
    original_asset = authority.asset
    call_count = 0

    def mutating_asset(object_id: str) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        result = original_asset(object_id)
        if call_count == 2:
            runtime.write_text("VERSION = 2\n", encoding="utf-8")
        return result

    authority.asset = mutating_asset  # type: ignore[method-assign]
    with pytest.raises(
        visual_contract.ArtiverseVisualResourceError,
        match="runtime code changed",
    ):
        visual_contract.build_receipt_from_authority(
            authority,
            runtime_code=[runtime],
        )
