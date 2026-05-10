from __future__ import annotations

import json
import shutil
import struct

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from scenesmith.agent_utils.artiverse_visual_normalization import (
    ArtiverseVisualNormalizationError,
    normalize_copied_artiverse_visuals,
    sha256_safe_directory_tree,
)


def _write_glb(path: Path, *, include_normals: bool = True) -> None:
    positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    normals = struct.pack("<9f", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    indices_payload = struct.pack("<3H", 0, 1, 2)
    image_payload = b"PNGproof"
    binary_payload = positions + normals + indices_payload + b"\x00\x00" + image_payload
    attributes = {"POSITION": 0}
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "type": "VEC3",
            "count": 3,
        }
    ]
    if include_normals:
        attributes["NORMAL"] = len(accessors)
        accessors.append(
            {
                "bufferView": 1,
                "componentType": 5126,
                "type": "VEC3",
                "count": 3,
            }
        )
    indices = len(accessors)
    accessors.append(
        {
            "bufferView": 2,
            "componentType": 5123,
            "type": "SCALAR",
            "count": 3,
        }
    )
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary_payload)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {
                "buffer": 0,
                "byteOffset": len(positions),
                "byteLength": len(normals),
            },
            {"buffer": 0, "byteOffset": 72, "byteLength": 6},
            {"buffer": 0, "byteOffset": 80, "byteLength": len(image_payload)},
        ],
        "accessors": accessors,
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "samplers": [{}],
        "textures": [{"sampler": 0, "source": 0}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                }
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": indices,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_chunk = json.dumps(document, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    json_chunk += b" " * (-len(json_chunk) % 4)
    binary_chunk = binary_payload + b"\x00" * (-len(binary_payload) % 4)
    length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, length)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(binary_chunk), 0x004E4942)
        + binary_chunk
    )


def _rewrite_glb_document(path: Path, mutate) -> None:
    payload = path.read_bytes()
    _magic, _version, _length = struct.unpack_from("<4sII", payload, 0)
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    json_start = 20
    json_end = json_start + json_length
    document = json.loads(payload[json_start:json_end].rstrip(b" \x00"))
    mutate(document)
    json_payload = json.dumps(
        document, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    json_payload += b" " * (-len(json_payload) % 4)
    binary_length, binary_type = struct.unpack_from("<II", payload, json_end)
    assert binary_type == 0x004E4942
    binary_payload = payload[json_end + 8 : json_end + 8 + binary_length]
    total = 12 + 8 + len(json_payload) + 8 + len(binary_payload)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_payload), 0x4E4F534A)
        + json_payload
        + struct.pack("<II", len(binary_payload), 0x004E4942)
        + binary_payload
    )


def _write_model_tree(root: Path, *, include_normals: bool = True) -> Path:
    (root / "objs").mkdir(parents=True)
    for name in (
        "1_base__material_red.obj",
        "1_base__material_blue.obj",
        "1_base.obj",
    ):
        (root / "objs" / name).write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
        )
    _write_glb(root / "glbs" / "1_base.glb", include_normals=include_normals)
    sdf = root / "scenesmith_artiverse.sdf"
    sdf.write_text(
        """<sdf version="1.9">
  <model name="fixture">
    <link name="1_base">
      <inertial>
        <pose>0.1 0.2 0.3 0 0 0</pose>
        <mass>2.5</mass>
        <inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz></inertia>
      </inertial>
      <visual name="red_split">
        <pose relative_to="frame">0 0 0 0 0 0</pose>
        <geometry><mesh><uri>./objs/1_base__material_red.obj</uri><scale>0.55 0.55 0.55</scale></mesh></geometry>
        <material><diffuse>1 0 0 1</diffuse></material>
      </visual>
      <collision name="collision">
        <pose relative_to="frame">0 0 0 0 0 0</pose>
        <geometry><mesh><uri>./objs/1_base.obj</uri><scale>0.55 0.55 0.55</scale></mesh></geometry>
      </collision>
      <visual name="blue_split">
        <pose relative_to="frame">0 0 0 0 0 0</pose>
        <geometry><mesh><uri>./objs/1_base__material_blue.obj</uri><scale>0.55 0.55 0.55</scale></mesh></geometry>
        <material><diffuse>0 0 1 1</diffuse></material>
      </visual>
    </link>
    <link name="anchor"/>
    <joint name="fixture_joint" type="fixed">
      <parent>anchor</parent><child>1_base</child>
      <pose>0 0 0 0 0 0</pose>
    </joint>
  </model>
</sdf>
""",
        encoding="utf-8",
    )
    return sdf


def _local_name(element: ET.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local_name(child) == name]


def _collision_bytes(sdf: Path) -> list[bytes]:
    root = ET.parse(sdf).getroot()
    payloads: list[bytes] = []
    for element in root.iter():
        if _local_name(element) == "collision":
            element.tail = None
            payloads.append(ET.tostring(element, encoding="utf-8"))
    return payloads


def _protected_element_bytes(sdf: Path) -> list[bytes]:
    root = ET.parse(sdf).getroot()
    payloads: list[bytes] = []
    for element in root.iter():
        if _local_name(element) in {"collision", "inertial", "joint"}:
            element.tail = None
            payloads.append(ET.tostring(element, encoding="utf-8"))
    return payloads


def test_multimaterial_visuals_derive_external_gltf_without_touching_source(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "publisher" / "asset"
    source_sdf = _write_model_tree(source_root)
    source_sdf_bytes = source_sdf.read_bytes()
    source_tree_hash = sha256_safe_directory_tree(source_root)
    copied_root = tmp_path / "room" / "asset"
    shutil.copytree(source_root, copied_root)
    copied_sdf = copied_root / source_sdf.name
    collisions_before = _collision_bytes(copied_sdf)
    protected_before = _protected_element_bytes(copied_sdf)

    evidence = normalize_copied_artiverse_visuals(copied_sdf, copied_root)

    assert source_sdf.read_bytes() == source_sdf_bytes
    assert sha256_safe_directory_tree(source_root) == source_tree_hash
    assert evidence["status"] == "pass"
    assert evidence["rewritten_link_count"] == 1
    assert evidence["deduplicated_visual_count"] == 1
    assert evidence["removed_material_override_count"] == 2
    assert evidence["migrated_legacy_glb_link_count"] == 0
    assert evidence["derived_publication_state"] == "published"
    assert evidence["derived_part_count"] == 1
    assert evidence["derived_resource_count"] == 2
    assert evidence["collision_element_count"] == 1
    assert evidence["copied_tree_sha256_before"] != evidence[
        "copied_tree_sha256_after"
    ]
    assert evidence["copied_tree_sha256_after"] == sha256_safe_directory_tree(
        copied_root
    )
    record = evidence["links"][0]
    assert record["source_obj_visual_uris"] == [
        "./objs/1_base__material_red.obj",
        "./objs/1_base__material_blue.obj",
    ]
    assert record["publisher_glb_uri"] == "./glbs/1_base.glb"
    assert record["derived_gltf_uri"] == (
        "./scenesmith_artiverse_visuals_v2/1_base.gltf"
    )
    assert record["derived_bin_uri"] == (
        "./scenesmith_artiverse_visuals_v2/1_base.bin"
    )
    assert record["normal_accessor_count"] == 1
    assert record["embedded_image_count"] == 1

    document = ET.parse(copied_sdf).getroot()
    link = next(element for element in document.iter() if _local_name(element) == "link")
    visuals = _children(link, "visual")
    assert len(visuals) == 1
    uri = next(element for element in visuals[0].iter() if _local_name(element) == "uri")
    scale = next(
        element for element in visuals[0].iter() if _local_name(element) == "scale"
    )
    pose = next(
        element for element in visuals[0].iter() if _local_name(element) == "pose"
    )
    assert uri.text == "./scenesmith_artiverse_visuals_v2/1_base.gltf"
    assert scale.text == "0.55 0.55 0.55"
    assert pose.text == "0 0 0 0 0 0"
    assert pose.attrib == {"relative_to": "frame"}
    assert not _children(visuals[0], "material")
    assert _collision_bytes(copied_sdf) == collisions_before
    assert _protected_element_bytes(copied_sdf) == protected_before

    derived_dir = copied_root / "scenesmith_artiverse_visuals_v2"
    derived_document = json.loads((derived_dir / "1_base.gltf").read_text("utf-8"))
    assert derived_document["buffers"] == [
        {"byteLength": 88, "uri": "1_base.bin"}
    ]
    assert derived_document["materials"][0]["pbrMetallicRoughness"][
        "baseColorTexture"
    ] == {"index": 0}
    assert derived_document["images"] == [
        {"bufferView": 3, "mimeType": "image/png"}
    ]
    assert (derived_dir / "1_base.bin").stat().st_size == 88
    assert sorted(path.name for path in derived_dir.iterdir()) == [
        "1_base.bin",
        "1_base.gltf",
        "_derivation_manifest.json",
    ]


def test_normalization_is_byte_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    first = normalize_copied_artiverse_visuals(sdf, root)
    first_bytes = sdf.read_bytes()
    first_tree_hash = sha256_safe_directory_tree(root)

    second = normalize_copied_artiverse_visuals(sdf, root)

    assert sdf.read_bytes() == first_bytes
    assert sha256_safe_directory_tree(root) == first_tree_hash
    assert second["rewritten_link_count"] == 0
    assert second["already_normalized_link_count"] == 1
    assert second["derived_publication_state"] == "verified_existing"
    assert second["sdf_sha256_before"] == first["sdf_sha256_after"]
    assert second["sdf_sha256_after"] == first["sdf_sha256_after"]
    assert second["copied_tree_sha256_before"] == first_tree_hash
    assert second["copied_tree_sha256_after"] == first_tree_hash


def test_exact_crash_orphan_directory_is_recovered_before_sdf_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    original_sdf = sdf.read_bytes()
    normalize_copied_artiverse_visuals(sdf, root)
    sdf.write_bytes(original_sdf)

    evidence = normalize_copied_artiverse_visuals(sdf, root)

    assert evidence["derived_publication_state"] == "recovered_unreferenced"
    assert evidence["rewritten_link_count"] == 1
    uri = next(
        element for element in ET.parse(sdf).getroot().iter() if _local_name(element) == "uri"
    )
    assert uri.text == "./scenesmith_artiverse_visuals_v2/1_base.gltf"


def test_stale_crash_orphan_directory_fails_before_sdf_rewrite(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    original_sdf = sdf.read_bytes()
    normalize_copied_artiverse_visuals(sdf, root)
    sdf.write_bytes(original_sdf)
    derived_bin = root / "scenesmith_artiverse_visuals_v2" / "1_base.bin"
    derived_bin.write_bytes(derived_bin.read_bytes() + b"stale")

    with pytest.raises(
        ArtiverseVisualNormalizationError, match="stale derived-resource mismatch"
    ):
        normalize_copied_artiverse_visuals(sdf, root)

    assert sdf.read_bytes() == original_sdf


def test_legacy_direct_glb_state_is_migrated_to_external_gltf(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    document = ET.parse(sdf)
    link = next(
        element for element in document.getroot().iter() if _local_name(element) == "link"
    )
    visuals = _children(link, "visual")
    first_uri = next(
        element for element in visuals[0].iter() if _local_name(element) == "uri"
    )
    first_uri.text = "./glbs/1_base.glb"
    for material in _children(visuals[0], "material"):
        visuals[0].remove(material)
    link.remove(visuals[1])
    document.write(sdf, encoding="utf-8", xml_declaration=True)

    evidence = normalize_copied_artiverse_visuals(sdf, root)

    assert evidence["rewritten_link_count"] == 0
    assert evidence["migrated_legacy_glb_link_count"] == 1
    assert evidence["already_normalized_link_count"] == 0
    uri = next(
        element for element in ET.parse(sdf).getroot().iter() if _local_name(element) == "uri"
    )
    assert uri.text == "./scenesmith_artiverse_visuals_v2/1_base.gltf"


def test_idempotence_validates_every_derived_resource_hash(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    normalize_copied_artiverse_visuals(sdf, root)
    gltf = root / "scenesmith_artiverse_visuals_v2" / "1_base.gltf"
    gltf.write_bytes(gltf.read_bytes() + b" ")

    with pytest.raises(
        ArtiverseVisualNormalizationError, match="stale derived-resource mismatch"
    ):
        normalize_copied_artiverse_visuals(sdf, root)


def test_derivation_evidence_binds_publisher_glb_and_final_tree(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    evidence = normalize_copied_artiverse_visuals(sdf, root)
    bound_tree_hash = evidence["copied_tree_sha256_after"]
    bound_glb_hash = evidence["links"][0]["publisher_glb_sha256"]

    glb = root / "glbs" / "1_base.glb"
    glb.write_bytes(glb.read_bytes() + b"tampered")

    assert sha256_safe_directory_tree(root) != bound_tree_hash
    assert evidence["links"][0]["publisher_glb_sha256"] == bound_glb_hash
    with pytest.raises(
        ArtiverseVisualNormalizationError, match="exact GLB 2.0 payload"
    ):
        normalize_copied_artiverse_visuals(sdf, root)


def test_ambiguous_part_mapping_fails_closed_without_rewrite(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    original = sdf.read_bytes()
    document = ET.parse(sdf)
    uris = [
        element
        for element in document.getroot().iter()
        if _local_name(element) == "uri"
        and (element.text or "").endswith("material_blue.obj")
    ]
    assert len(uris) == 1
    other = root / "objs" / "2_drawer__material_blue.obj"
    other.write_bytes((root / "objs" / "1_base__material_blue.obj").read_bytes())
    uris[0].text = "./objs/2_drawer__material_blue.obj"
    document.write(sdf, encoding="utf-8", xml_declaration=True)
    ambiguous = sdf.read_bytes()

    with pytest.raises(
        ArtiverseVisualNormalizationError, match="ambiguous part stems"
    ):
        normalize_copied_artiverse_visuals(sdf, root)

    assert original != ambiguous
    assert sdf.read_bytes() == ambiguous


def test_unequal_split_transform_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    document = ET.parse(sdf)
    visual_scales = [
        element
        for visual in document.getroot().iter()
        if _local_name(visual) == "visual"
        for element in visual.iter()
        if _local_name(element) == "scale"
    ]
    visual_scales[1].text = "1 1 1"
    document.write(sdf, encoding="utf-8", xml_declaration=True)

    with pytest.raises(
        ArtiverseVisualNormalizationError, match="unequal pose/scale"
    ):
        normalize_copied_artiverse_visuals(sdf, root)


def test_publisher_glb_without_authored_normals_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root, include_normals=False)

    with pytest.raises(
        ArtiverseVisualNormalizationError, match="POSITION/NORMAL"
    ):
        normalize_copied_artiverse_visuals(sdf, root)


def test_publisher_normal_accessor_must_be_float_or_normalized_signed_int(
    tmp_path: Path,
) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    _rewrite_glb_document(
        root / "glbs" / "1_base.glb",
        lambda document: document["accessors"][1].update(
            {"componentType": 5122, "normalized": False}
        ),
    )

    with pytest.raises(
        ArtiverseVisualNormalizationError,
        match="POSITION/NORMAL component normalization",
    ):
        normalize_copied_artiverse_visuals(sdf, root)


def test_publisher_accessor_range_must_fit_its_buffer_view(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    _rewrite_glb_document(
        root / "glbs" / "1_base.glb",
        lambda document: document["bufferViews"][1].update({"byteLength": 8}),
    )

    with pytest.raises(
        ArtiverseVisualNormalizationError, match="accessor range/stride"
    ):
        normalize_copied_artiverse_visuals(sdf, root)


def test_publisher_glb_rejects_external_image_dependency(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    _rewrite_glb_document(
        root / "glbs" / "1_base.glb",
        lambda document: document["images"].__setitem__(0, {"uri": "texture.png"}),
    )

    with pytest.raises(
        ArtiverseVisualNormalizationError, match="external resource"
    ):
        normalize_copied_artiverse_visuals(sdf, root)


def test_visual_uri_tree_escape_fails_without_rewrite(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    document = ET.parse(sdf)
    first_uri = next(
        element for element in document.getroot().iter() if _local_name(element) == "uri"
    )
    first_uri.text = "../outside.obj"
    document.write(sdf, encoding="utf-8", xml_declaration=True)
    unsafe_bytes = sdf.read_bytes()

    with pytest.raises(ArtiverseVisualNormalizationError, match="exact ./objs/"):
        normalize_copied_artiverse_visuals(sdf, root)

    assert sdf.read_bytes() == unsafe_bytes


def test_stale_atomic_transaction_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    orphan = root / f".{sdf.name}.visual-normalization.crashed.tmp"
    orphan.write_bytes(b"orphan")

    with pytest.raises(
        ArtiverseVisualNormalizationError, match="stale Artiverse visual-normalization"
    ):
        normalize_copied_artiverse_visuals(sdf, root)


def test_deduplication_rejects_differing_nonmaterial_render_properties(
    tmp_path: Path,
) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    document = ET.parse(sdf)
    visuals = [
        element for element in document.getroot().iter() if _local_name(element) == "visual"
    ]
    ET.SubElement(visuals[1], "transparency").text = "0.25"
    document.write(sdf, encoding="utf-8", xml_declaration=True)

    with pytest.raises(
        ArtiverseVisualNormalizationError, match="non-material render structure"
    ):
        normalize_copied_artiverse_visuals(sdf, root)


def test_idempotent_state_rejects_lingering_material_override(tmp_path: Path) -> None:
    root = tmp_path / "copied"
    sdf = _write_model_tree(root)
    normalize_copied_artiverse_visuals(sdf, root)
    document = ET.parse(sdf)
    visual = next(
        element for element in document.getroot().iter() if _local_name(element) == "visual"
    )
    ET.SubElement(visual, "material")
    document.write(sdf, encoding="utf-8", xml_declaration=True)

    with pytest.raises(
        ArtiverseVisualNormalizationError, match="lingering material override"
    ):
        normalize_copied_artiverse_visuals(sdf, root)
