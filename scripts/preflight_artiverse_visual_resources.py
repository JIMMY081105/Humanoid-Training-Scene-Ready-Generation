#!/usr/bin/env python3
"""Prove deterministic Drake-compatible glTF derivation for Artiverse visuals.

The Artiverse publisher trees used by SceneSmith contain OBJ collision/visual
meshes without ``vn`` records and, alongside them, per-link GLB files with
explicit glTF ``NORMAL`` attributes.  Drake accepts the OBJ collision geometry,
but its glTF-client renderer both rejects a visual OBJ without normals and ignores
direct ``.glb`` mesh URIs.  This preflight therefore proves that every publisher
GLB can be losslessly externalized into deterministic room-local ``.gltf`` and
``.bin`` bytes.  A copied SDF may replace each material-split OBJ visual group
only with that derived external glTF.  Redundant visual elements and their SDF
material overrides are removed so that the publisher GLB's complete authored
material assignment survives; source data, collisions, poses, scales, joints,
inertials, and every nonvisual SDF field remain unchanged.

The check is intentionally read-only with respect to the Artiverse dataset.  A
passing receipt binds the already validated Artiverse authority, the complete
indexed asset set, per-asset mapping/collision digests, this verifier, and the
runtime normalization code supplied with ``--runtime-code``.  ``--verify-only``
recomputes all evidence and compares it byte-for-byte with the saved receipt.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import hashlib
import json
import math
import os
import stat
import struct
import sys
import xml.etree.ElementTree as ET

from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, unquote_to_bytes, urlsplit

from artiverse_contract import (
    ArtiverseContractError,
    load_artiverse_authority,
    require_regular_file_within,
    sha256_file,
)


SCHEMA_VERSION = 2
ATTESTATION_SCHEMA_VERSION = 1
CONTRACT_NAME = "artiverse-visual-external-gltf-derivation"
MAPPING_POLICY_ID = "artiverse-copied-obj-via-publisher-glb-to-external-gltf-v2"
RUNTIME_HELPER_POLICY = "publisher_glb_derived_external_gltf"
DERIVED_RESOURCE_DIRECTORY = "scenesmith_artiverse_visuals_v2"
DERIVATION_MANIFEST = "_derivation_manifest.json"
HASH_ALGORITHM = "sha256"
GLB_MAGIC = b"glTF"
GLB_VERSION = 2
GLB_JSON_CHUNK_TYPE = 0x4E4F534A
GLB_BIN_CHUNK_TYPE = 0x004E4942
GLTF_FLOAT = 5126
GLTF_BYTE = 5120
GLTF_SHORT = 5122
ALLOWED_NORMAL_COMPONENT_TYPES = frozenset({GLTF_FLOAT, GLTF_BYTE, GLTF_SHORT})
GLTF_COMPONENT_BYTES = {
    5120: 1,
    5121: 1,
    5122: 2,
    5123: 2,
    5125: 4,
    5126: 4,
}
GLTF_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
DEFAULT_RUNTIME_CODE = (
    Path("scenesmith/agent_utils/artiverse_visual_normalization.py"),
    Path("scenesmith/agent_utils/asset_manager.py"),
)


class ArtiverseVisualResourceError(RuntimeError):
    """Raised when visual-resource substitution cannot be proven exactly."""


def mapping_policy() -> dict[str, Any]:
    """Return the exact policy independently required by every downstream gate."""

    return {
        "id": MAPPING_POLICY_ID,
        "source_scope": "prepared_official_artiverse_tree_read_only",
        "mutation_scope": (
            "room_local_copied_sdf_visual_elements_and_exact_derived_"
            "resource_directory_only"
        ),
        "source_visual_format": "obj",
        "authoritative_visual_format": "publisher_glb",
        "target": RUNTIME_HELPER_POLICY,
        "target_visual_format": "external_gltf_with_external_bin",
        "derived_resource_directory": DERIVED_RESOURCE_DIRECTORY,
        "derived_resource_manifest": DERIVATION_MANIFEST,
        "publisher_glb_requires_normal_on_every_primitive": True,
        "publisher_glb_must_be_self_contained": True,
        "lossless_json_bin_externalization": True,
        "material_split_visuals_collapsed_to_one_gltf_per_link": True,
        "sdf_material_overrides_removed_for_publisher_gltf_materials": True,
        "collision_mesh_uris_unchanged": True,
        "visual_pose_and_scale_unchanged": True,
        "publisher_source_tree_unchanged": True,
        "joints_inertials_and_nonvisual_xml_unchanged": True,
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _attestation(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "algorithm": HASH_ALGORITHM,
        "sha256": _sha256_bytes(_canonical_json(payload)),
    }


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _exact_child(element: ET.Element, name: str, label: str) -> ET.Element:
    children = _direct_children(element, name)
    if len(children) != 1:
        raise ArtiverseVisualResourceError(
            f"{label} must contain exactly one <{name}>; found {len(children)}"
        )
    return children[0]


def _optional_vector(
    element: ET.Element,
    child_name: str,
    *,
    length: int,
    default: tuple[float, ...],
    label: str,
) -> tuple[float, ...]:
    children = _direct_children(element, child_name)
    if not children:
        return default
    if len(children) != 1:
        raise ArtiverseVisualResourceError(
            f"{label} must contain at most one <{child_name}>"
        )
    tokens = (children[0].text or "").split()
    if len(tokens) != length:
        raise ArtiverseVisualResourceError(
            f"{label} <{child_name}> must contain exactly {length} numbers"
        )
    try:
        values = tuple(float(token) for token in tokens)
    except ValueError as exc:
        raise ArtiverseVisualResourceError(
            f"{label} <{child_name}> is not numeric"
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise ArtiverseVisualResourceError(
            f"{label} <{child_name}> contains a non-finite number"
        )
    return values


def _pose(element: ET.Element, label: str) -> dict[str, Any]:
    children = _direct_children(element, "pose")
    if not children:
        return {"relative_to": None, "values": [0.0] * 6}
    if len(children) != 1:
        raise ArtiverseVisualResourceError(f"{label} has multiple <pose> elements")
    pose = children[0]
    unexpected = sorted(set(pose.attrib) - {"relative_to"})
    if unexpected:
        raise ArtiverseVisualResourceError(
            f"{label} pose has unsupported attributes: {', '.join(unexpected)}"
        )
    values = _optional_vector(
        element,
        "pose",
        length=6,
        default=(0.0,) * 6,
        label=label,
    )
    return {"relative_to": pose.get("relative_to"), "values": list(values)}


def _mesh_record(element: ET.Element, *, label: str) -> dict[str, Any]:
    geometry = _exact_child(element, "geometry", label)
    mesh = _exact_child(geometry, "mesh", f"{label} geometry")
    uri = _exact_child(mesh, "uri", f"{label} mesh")
    raw_uri = (uri.text or "").strip()
    if not raw_uri:
        raise ArtiverseVisualResourceError(f"{label} mesh URI is empty")
    scale = _optional_vector(
        mesh,
        "scale",
        length=3,
        default=(1.0, 1.0, 1.0),
        label=f"{label} mesh",
    )
    if not all(value > 0.0 for value in scale):
        raise ArtiverseVisualResourceError(f"{label} mesh scale must be positive")
    return {
        "name": element.get("name"),
        "uri": raw_uri,
        "pose": _pose(element, label),
        "scale": list(scale),
    }


def _visual_structure_signature(visual: ET.Element) -> tuple[Any, ...]:
    """Bind every split-visual property that deduplication may not discard."""

    def signature(element: ET.Element, *, visual_root: bool = False) -> tuple[Any, ...]:
        name = _local_name(element.tag)
        attributes = tuple(
            sorted(
                (str(key), str(value))
                for key, value in element.attrib.items()
                if not (visual_root and key == "name")
            )
        )
        text = " ".join((element.text or "").split())
        if name == "uri":
            text = "<AUDITED_VISUAL_URI>"
        children = tuple(
            signature(child)
            for child in list(element)
            if not (visual_root and _local_name(child.tag) == "material")
        )
        return name, attributes, text, children

    return signature(visual, visual_root=True)


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath((os.fspath(root), os.fspath(candidate))) == os.fspath(
            root
        )
    except ValueError:
        return False


def _resolve_local_resource(uri: str, model_root: Path, *, label: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ArtiverseVisualResourceError(f"{label} is not a plain local URI: {uri}")
    decoded = unquote(parsed.path)
    if not decoded or "\\" in decoded or "\x00" in decoded:
        raise ArtiverseVisualResourceError(f"{label} is unsafe: {uri}")
    relative = PurePosixPath(decoded)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtiverseVisualResourceError(f"{label} escapes its model tree: {uri}")
    candidate = model_root.joinpath(*relative.parts)
    try:
        resolved = require_regular_file_within(candidate, model_root, label)
    except ArtiverseContractError as exc:
        raise ArtiverseVisualResourceError(str(exc)) from exc
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise ArtiverseVisualResourceError(f"Cannot inspect {label}: {candidate}") from exc
    if not stat.S_ISREG(metadata.st_mode) or candidate.is_symlink():
        raise ArtiverseVisualResourceError(f"{label} is not an ordinary file: {candidate}")
    return resolved


def _obj_to_glb_relative(obj_path: Path, model_root: Path, *, label: str) -> str:
    try:
        relative = obj_path.relative_to(model_root)
    except ValueError as exc:
        raise ArtiverseVisualResourceError(f"{label} escapes its model root") from exc
    if len(relative.parts) != 2 or relative.parts[0] != "objs":
        raise ArtiverseVisualResourceError(
            f"{label} must be directly under the model's objs directory: {relative}"
        )
    if relative.suffix.lower() != ".obj":
        raise ArtiverseVisualResourceError(f"{label} is not an OBJ: {relative}")
    part_name = relative.stem.split("__", 1)[0]
    if not part_name or Path(part_name).name != part_name:
        raise ArtiverseVisualResourceError(
            f"{label} has no canonical Artiverse part name: {relative}"
        )
    return f"glbs/{part_name}.glb"


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtiverseVisualResourceError(
                f"GLB JSON contains duplicate key: {key}"
            )
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ArtiverseVisualResourceError(f"GLB JSON contains non-finite number: {value}")


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtiverseVisualResourceError(f"{label} is not an integer >= {minimum}")
    return value


def _index(value: Any, inventory: Sequence[Any], *, label: str) -> int:
    result = _integer(value, label=label)
    if result >= len(inventory):
        raise ArtiverseVisualResourceError(f"{label} is out of range")
    return result


def _image_payload(uri: str, *, label: str) -> tuple[str, bytes]:
    if not uri.startswith("data:") or "," not in uri:
        raise ArtiverseVisualResourceError(f"{label} references an external resource")
    header, encoded = uri[5:].split(",", 1)
    tokens = header.split(";")
    mime_type = tokens[0]
    if not mime_type.startswith("image/") or not mime_type[6:]:
        raise ArtiverseVisualResourceError(f"{label} has an invalid data-image MIME type")
    try:
        if tokens[-1].lower() == "base64":
            payload = base64.b64decode(encoded, validate=True)
        else:
            payload = unquote_to_bytes(encoded)
    except (ValueError, binascii.Error) as exc:
        raise ArtiverseVisualResourceError(f"{label} data URI is malformed") from exc
    if not payload:
        raise ArtiverseVisualResourceError(f"{label} image payload is empty")
    return mime_type.lower(), payload


def _validate_image_signature(mime_type: str, payload: bytes, *, label: str) -> None:
    signatures = {
        "image/png": payload.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": payload.startswith(b"\xff\xd8\xff"),
        "image/webp": (
            len(payload) >= 12
            and payload.startswith(b"RIFF")
            and payload[8:12] == b"WEBP"
        ),
        "image/ktx2": payload.startswith(b"\xabKTX 20\xbb\r\n\x1a\n"),
    }
    if mime_type not in signatures or not signatures[mime_type]:
        raise ArtiverseVisualResourceError(
            f"{label} bytes do not match supported MIME type {mime_type!r}"
        )


def _read_glb_derivation(
    path: Path,
    *,
    part: str,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Independently validate and derive the exact runtime external resources."""

    try:
        before = path.lstat()
        data = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise ArtiverseVisualResourceError(f"Cannot read GLB: {path}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ArtiverseVisualResourceError(f"GLB changed while being read: {path}")
    if len(data) < 12:
        raise ArtiverseVisualResourceError(f"GLB header is truncated: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != GLB_MAGIC or version != GLB_VERSION or declared_length != len(data):
        raise ArtiverseVisualResourceError(f"GLB header is invalid: {path}")

    chunks: list[tuple[int, bytes]] = []
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise ArtiverseVisualResourceError(f"GLB chunk header is truncated: {path}")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_length % 4 or chunk_end > len(data):
            raise ArtiverseVisualResourceError(f"GLB chunk length is invalid: {path}")
        chunks.append((chunk_type, data[offset:chunk_end]))
        offset = chunk_end
    if (
        offset != len(data)
        or len(chunks) != 2
        or chunks[0][0] != GLB_JSON_CHUNK_TYPE
        or chunks[1][0] != GLB_BIN_CHUNK_TYPE
    ):
        raise ArtiverseVisualResourceError(
            f"GLB must contain exactly one JSON and one BIN chunk: {path}"
        )
    raw_json = chunks[0][1]
    binary_chunk = chunks[1][1]
    try:
        document = json.loads(
            raw_json.decode("utf-8").rstrip(" \t\r\n\x00"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except ArtiverseVisualResourceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ArtiverseVisualResourceError(f"GLB JSON is malformed: {path}") from exc
    if not isinstance(document, dict):
        raise ArtiverseVisualResourceError(f"GLB JSON root is not an object: {path}")
    asset = document.get("asset")
    if not isinstance(asset, dict) or str(asset.get("version", "")) != "2.0":
        raise ArtiverseVisualResourceError(f"GLB is not glTF 2.0: {path}")

    buffers = document.get("buffers")
    if not isinstance(buffers, list) or len(buffers) != 1 or not isinstance(buffers[0], dict):
        raise ArtiverseVisualResourceError(f"GLB must have exactly one buffer: {path}")
    if "uri" in buffers[0]:
        raise ArtiverseVisualResourceError(f"GLB buffer references an external resource: {path}")
    buffer_length = _integer(
        buffers[0].get("byteLength"), label="GLB buffer byteLength", minimum=1
    )
    if not buffer_length <= len(binary_chunk) <= buffer_length + 3:
        raise ArtiverseVisualResourceError(f"GLB BIN chunk length is invalid: {path}")
    if any(binary_chunk[buffer_length:]):
        raise ArtiverseVisualResourceError(f"GLB BIN padding is nonzero: {path}")
    binary_payload = binary_chunk[:buffer_length]

    buffer_views = document.get("bufferViews", [])
    if not isinstance(buffer_views, list):
        raise ArtiverseVisualResourceError(f"GLB bufferViews inventory is invalid: {path}")
    view_bounds: list[tuple[int, int, int | None]] = []
    for view_index, view in enumerate(buffer_views):
        if not isinstance(view, dict) or view.get("buffer") != 0:
            raise ArtiverseVisualResourceError(
                f"GLB bufferView {view_index} does not use buffer 0: {path}"
            )
        byte_offset = _integer(
            view.get("byteOffset", 0), label=f"GLB bufferView {view_index} byteOffset"
        )
        byte_length = _integer(
            view.get("byteLength"),
            label=f"GLB bufferView {view_index} byteLength",
            minimum=1,
        )
        if byte_offset + byte_length > buffer_length:
            raise ArtiverseVisualResourceError(
                f"GLB bufferView {view_index} escapes buffer 0: {path}"
            )
        stride = view.get("byteStride")
        if stride is not None:
            stride = _integer(stride, label=f"GLB bufferView {view_index} byteStride")
            if stride < 4 or stride > 252 or stride % 4:
                raise ArtiverseVisualResourceError(
                    f"GLB bufferView {view_index} byteStride is invalid: {path}"
                )
        target = view.get("target")
        if target is not None and target not in {34962, 34963}:
            raise ArtiverseVisualResourceError(
                f"GLB bufferView {view_index} target is invalid: {path}"
            )
        view_bounds.append((byte_offset, byte_length, stride))

    accessors = document.get("accessors")
    if not isinstance(accessors, list) or not accessors:
        raise ArtiverseVisualResourceError(f"GLB has no accessor inventory: {path}")
    for accessor_index, accessor in enumerate(accessors):
        if not isinstance(accessor, dict):
            raise ArtiverseVisualResourceError(
                f"GLB accessor {accessor_index} is malformed: {path}"
            )
        component_type = accessor.get("componentType")
        accessor_type = accessor.get("type")
        if component_type not in GLTF_COMPONENT_BYTES or accessor_type not in GLTF_TYPE_COMPONENTS:
            raise ArtiverseVisualResourceError(
                f"GLB accessor {accessor_index} has invalid type metadata: {path}"
            )
        count = _integer(
            accessor.get("count"), label=f"GLB accessor {accessor_index} count", minimum=1
        )
        accessor_offset = _integer(
            accessor.get("byteOffset", 0),
            label=f"GLB accessor {accessor_index} byteOffset",
        )
        element_size = GLTF_COMPONENT_BYTES[component_type] * GLTF_TYPE_COMPONENTS[accessor_type]
        view_value = accessor.get("bufferView")
        if view_value is None:
            if "sparse" not in accessor:
                raise ArtiverseVisualResourceError(
                    f"GLB accessor {accessor_index} has no bufferView or sparse data: {path}"
                )
        else:
            view_index = _index(
                view_value, buffer_views, label=f"GLB accessor {accessor_index} bufferView"
            )
            _view_offset, view_length, stride = view_bounds[view_index]
            item_stride = stride or element_size
            if item_stride < element_size or accessor_offset + (count - 1) * item_stride + element_size > view_length:
                raise ArtiverseVisualResourceError(
                    f"GLB accessor {accessor_index} escapes its bufferView: {path}"
                )
        sparse = accessor.get("sparse")
        if sparse is not None:
            if not isinstance(sparse, dict):
                raise ArtiverseVisualResourceError(
                    f"GLB accessor {accessor_index} sparse record is malformed: {path}"
                )
            sparse_count = _integer(
                sparse.get("count"),
                label=f"GLB accessor {accessor_index} sparse count",
                minimum=1,
            )
            if sparse_count > count:
                raise ArtiverseVisualResourceError(
                    f"GLB accessor {accessor_index} sparse count exceeds count: {path}"
                )
            indices = sparse.get("indices")
            values = sparse.get("values")
            if not isinstance(indices, dict) or not isinstance(values, dict):
                raise ArtiverseVisualResourceError(
                    f"GLB accessor {accessor_index} sparse payload is malformed: {path}"
                )
            index_component = indices.get("componentType")
            if index_component not in {5121, 5123, 5125}:
                raise ArtiverseVisualResourceError(
                    f"GLB accessor {accessor_index} sparse index type is invalid: {path}"
                )
            indices_view = _index(
                indices.get("bufferView"),
                buffer_views,
                label=f"GLB accessor {accessor_index} sparse indices bufferView",
            )
            values_view = _index(
                values.get("bufferView"),
                buffer_views,
                label=f"GLB accessor {accessor_index} sparse values bufferView",
            )
            indices_offset = _integer(
                indices.get("byteOffset", 0),
                label=f"GLB accessor {accessor_index} sparse indices byteOffset",
            )
            values_offset = _integer(
                values.get("byteOffset", 0),
                label=f"GLB accessor {accessor_index} sparse values byteOffset",
            )
            if indices_offset + sparse_count * GLTF_COMPONENT_BYTES[index_component] > view_bounds[indices_view][1] or values_offset + sparse_count * element_size > view_bounds[values_view][1]:
                raise ArtiverseVisualResourceError(
                    f"GLB accessor {accessor_index} sparse payload escapes bufferView: {path}"
                )

    images = document.get("images", [])
    if not isinstance(images, list):
        raise ArtiverseVisualResourceError(f"GLB image inventory is invalid: {path}")
    image_records: list[dict[str, Any]] = []
    for image_index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ArtiverseVisualResourceError(f"GLB image {image_index} is malformed: {path}")
        has_uri = "uri" in image
        has_view = "bufferView" in image
        if has_uri == has_view:
            raise ArtiverseVisualResourceError(
                f"GLB image {image_index} must have exactly one embedded source: {path}"
            )
        if has_uri:
            uri = image.get("uri")
            if not isinstance(uri, str):
                raise ArtiverseVisualResourceError(f"GLB image {image_index} URI is invalid: {path}")
            mime_type, payload = _image_payload(uri, label=f"GLB image {image_index}")
            source = "data_uri"
        else:
            view_index = _index(
                image.get("bufferView"),
                buffer_views,
                label=f"GLB image {image_index} bufferView",
            )
            mime_type = image.get("mimeType")
            if not isinstance(mime_type, str) or not mime_type:
                raise ArtiverseVisualResourceError(
                    f"GLB image {image_index} MIME type is invalid: {path}"
                )
            view_offset, view_length, _stride = view_bounds[view_index]
            payload = binary_payload[view_offset : view_offset + view_length]
            source = "buffer_view"
        mime_type = mime_type.lower()
        _validate_image_signature(mime_type, payload, label=f"GLB image {image_index}")
        image_records.append(
            {
                "index": image_index,
                "source": source,
                "mime_type": mime_type,
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )

    samplers = document.get("samplers", [])
    textures = document.get("textures", [])
    materials = document.get("materials", [])
    if not isinstance(samplers, list) or not isinstance(textures, list) or not isinstance(materials, list):
        raise ArtiverseVisualResourceError(f"GLB material/texture inventories are malformed: {path}")
    for sampler_index, sampler in enumerate(samplers):
        if not isinstance(sampler, dict):
            raise ArtiverseVisualResourceError(f"GLB sampler {sampler_index} is malformed: {path}")
        if sampler.get("magFilter") not in {None, 9728, 9729} or sampler.get("minFilter") not in {None, 9728, 9729, 9984, 9985, 9986, 9987} or sampler.get("wrapS", 10497) not in {33071, 33648, 10497} or sampler.get("wrapT", 10497) not in {33071, 33648, 10497}:
            raise ArtiverseVisualResourceError(f"GLB sampler {sampler_index} has invalid enums: {path}")
    for texture_index, texture in enumerate(textures):
        if not isinstance(texture, dict):
            raise ArtiverseVisualResourceError(f"GLB texture {texture_index} is malformed: {path}")
        if "sampler" in texture:
            _index(texture["sampler"], samplers, label=f"GLB texture {texture_index} sampler")
        source = texture.get("source")
        basisu = texture.get("extensions", {}).get("KHR_texture_basisu") if isinstance(texture.get("extensions"), dict) else None
        basisu_source = basisu.get("source") if isinstance(basisu, dict) else None
        if source is None and basisu_source is None:
            raise ArtiverseVisualResourceError(f"GLB texture {texture_index} has no image source: {path}")
        if source is not None:
            _index(source, images, label=f"GLB texture {texture_index} source")
        if basisu_source is not None:
            _index(basisu_source, images, label=f"GLB texture {texture_index} KTX2 source")

    def validate_texture_infos(value: Any, trail: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_trail = f"{trail}.{key}"
                if key.endswith("Texture") and isinstance(child, dict):
                    _index(child.get("index"), textures, label=f"GLB {child_trail} index")
                    if "texCoord" in child:
                        _integer(child["texCoord"], label=f"GLB {child_trail} texCoord")
                validate_texture_infos(child, child_trail)
        elif isinstance(value, list):
            for index_value, child in enumerate(value):
                validate_texture_infos(child, f"{trail}[{index_value}]")

    for material_index, material in enumerate(materials):
        if not isinstance(material, dict):
            raise ArtiverseVisualResourceError(f"GLB material {material_index} is malformed: {path}")
        validate_texture_infos(material, f"material[{material_index}]")

    meshes = document.get("meshes")
    if not isinstance(meshes, list) or not meshes:
        raise ArtiverseVisualResourceError(f"GLB has no mesh inventory: {path}")
    primitive_count = 0
    normal_counts: list[int] = []
    material_binding_count = 0
    for mesh_index, mesh in enumerate(meshes):
        if not isinstance(mesh, dict) or not isinstance(mesh.get("primitives"), list) or not mesh["primitives"]:
            raise ArtiverseVisualResourceError(f"GLB mesh {mesh_index} has no primitives: {path}")
        for primitive_index, primitive in enumerate(mesh["primitives"]):
            if not isinstance(primitive, dict) or primitive.get("mode", 4) != 4:
                raise ArtiverseVisualResourceError(f"GLB primitive is not TRIANGLES: {path}")
            attributes = primitive.get("attributes")
            if not isinstance(attributes, dict) or "POSITION" not in attributes or "NORMAL" not in attributes:
                raise ArtiverseVisualResourceError(
                    f"GLB primitive {mesh_index}/{primitive_index} has no POSITION/NORMAL: {path}"
                )
            for semantic, accessor_value in attributes.items():
                if not isinstance(semantic, str) or not semantic:
                    raise ArtiverseVisualResourceError(f"GLB primitive attribute is malformed: {path}")
                _index(accessor_value, accessors, label=f"GLB primitive {mesh_index}/{primitive_index} {semantic}")
            for target_index, target in enumerate(primitive.get("targets", [])):
                if not isinstance(target, dict):
                    raise ArtiverseVisualResourceError(f"GLB morph target is malformed: {path}")
                for semantic, accessor_value in target.items():
                    _index(accessor_value, accessors, label=f"GLB morph target {target_index} {semantic}")
            position_accessor = accessors[attributes["POSITION"]]
            normal_accessor = accessors[attributes["NORMAL"]]
            if position_accessor.get("type") != "VEC3" or normal_accessor.get("type") != "VEC3":
                raise ArtiverseVisualResourceError(f"GLB POSITION/NORMAL accessor is not VEC3: {path}")
            if normal_accessor.get("componentType") not in ALLOWED_NORMAL_COMPONENT_TYPES:
                raise ArtiverseVisualResourceError(f"GLB NORMAL accessor has unsupported component type: {path}")
            if normal_accessor.get("componentType") != GLTF_FLOAT and normal_accessor.get("normalized") is not True:
                raise ArtiverseVisualResourceError(f"Integer GLB NORMAL accessor is not normalized: {path}")
            if position_accessor.get("count") != normal_accessor.get("count"):
                raise ArtiverseVisualResourceError(f"GLB POSITION/NORMAL accessor counts differ: {path}")
            count = normal_accessor["count"]
            indices_value = primitive.get("indices")
            if indices_value is None:
                if count < 3 or count % 3:
                    raise ArtiverseVisualResourceError(f"GLB unindexed triangle count is invalid: {path}")
            else:
                indices_index = _index(indices_value, accessors, label="GLB triangle indices accessor")
                indices_accessor = accessors[indices_index]
                if indices_accessor.get("type") != "SCALAR" or indices_accessor.get("componentType") not in {5121, 5123, 5125} or indices_accessor.get("count", 0) < 3 or indices_accessor["count"] % 3:
                    raise ArtiverseVisualResourceError(f"GLB triangle index accessor is invalid: {path}")
            if "material" in primitive:
                _index(primitive["material"], materials, label="GLB primitive material")
                material_binding_count += 1
            primitive_count += 1
            normal_counts.append(count)

    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    if not isinstance(nodes, list) or not isinstance(scenes, list):
        raise ArtiverseVisualResourceError(f"GLB node/scene inventory is malformed: {path}")
    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ArtiverseVisualResourceError(f"GLB node {node_index} is malformed: {path}")
        if "mesh" in node:
            _index(node["mesh"], meshes, label=f"GLB node {node_index} mesh")
        for child in node.get("children", []):
            _index(child, nodes, label=f"GLB node {node_index} child")
    for scene_index, scene in enumerate(scenes):
        if not isinstance(scene, dict) or not isinstance(scene.get("nodes", []), list):
            raise ArtiverseVisualResourceError(f"GLB scene {scene_index} is malformed: {path}")
        for node in scene.get("nodes", []):
            _index(node, nodes, label=f"GLB scene {scene_index} node")
    if "scene" in document:
        _index(document["scene"], scenes, label="GLB default scene")

    derived_document = copy.deepcopy(document)
    derived_document["buffers"][0]["uri"] = f"{part}.bin"
    preserved_document = copy.deepcopy(derived_document)
    del preserved_document["buffers"][0]["uri"]
    if preserved_document != document:
        raise ArtiverseVisualResourceError(
            f"External glTF derivation changed publisher document semantics: {path}"
        )
    derived_gltf = _canonical_json(derived_document) + b"\n"
    semantic_sha256 = _sha256_bytes(_canonical_json(preserved_document) + b"\n")
    evidence = {
        "publisher_glb_size_bytes": before.st_size,
        "publisher_glb_sha256": _sha256_bytes(data),
        "publisher_json_chunk_sha256": _sha256_bytes(raw_json),
        "publisher_document_sha256": semantic_sha256,
        "publisher_bin_chunk_size_bytes": len(binary_chunk),
        "publisher_bin_chunk_sha256": _sha256_bytes(binary_chunk),
        "declared_buffer_size_bytes": buffer_length,
        "mesh_count": len(meshes),
        "primitive_count": primitive_count,
        "normal_accessor_element_count": sum(normal_counts),
        "buffer_view_count": len(buffer_views),
        "accessor_count": len(accessors),
        "material_count": len(materials),
        "material_binding_count": material_binding_count,
        "texture_count": len(textures),
        "image_count": len(images),
        "image_inventory_sha256": _sha256_bytes(_canonical_json(image_records)),
        "preserved_semantics_sha256": semantic_sha256,
        "derived_gltf_name": f"{part}.gltf",
        "derived_gltf_size_bytes": len(derived_gltf),
        "derived_gltf_sha256": _sha256_bytes(derived_gltf),
        "derived_bin_name": f"{part}.bin",
        "derived_bin_size_bytes": len(binary_payload),
        "derived_bin_sha256": _sha256_bytes(binary_payload),
        "derived_resource_pair_sha256": _sha256_bytes(
            _canonical_json(
                {
                    f"{part}.bin": _sha256_bytes(binary_payload),
                    f"{part}.gltf": _sha256_bytes(derived_gltf),
                }
            )
        ),
    }
    return evidence, derived_gltf, binary_payload


def _read_glb_normal_evidence(path: Path, *, part: str | None = None) -> dict[str, Any]:
    """Compatibility wrapper returning the stronger derivation evidence."""

    selected_part = part or path.stem
    evidence, _gltf, _binary = _read_glb_derivation(path, part=selected_part)
    return evidence


def _snapshot_regular_tree(root: Path) -> dict[str, tuple[int, ...]]:
    try:
        canonical = root.resolve(strict=True)
    except OSError as exc:
        raise ArtiverseVisualResourceError(f"Model tree is missing: {root}") from exc
    if not canonical.is_dir():
        raise ArtiverseVisualResourceError(f"Model tree is not a directory: {canonical}")
    result: dict[str, tuple[int, ...]] = {}
    pending = [canonical]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ArtiverseVisualResourceError(
                f"Cannot scan Artiverse model tree: {directory}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArtiverseVisualResourceError(
                    f"Cannot inspect Artiverse model entry: {path}"
                ) from exc
            relative = path.relative_to(canonical).as_posix()
            if entry.is_symlink():
                raise ArtiverseVisualResourceError(
                    f"Artiverse model tree contains a link: {path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                result[f"F:{relative}"] = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                    metadata.st_nlink,
                )
            else:
                raise ArtiverseVisualResourceError(
                    f"Artiverse model tree contains a special entry: {path}"
                )
    if not any(key.startswith("F:") for key in result):
        raise ArtiverseVisualResourceError(f"Artiverse model tree is empty: {canonical}")
    return result


def _audit_sdf_visual_mapping(
    sdf_path: Path,
    model_root: Path,
    *,
    object_id: str,
) -> dict[str, Any]:
    try:
        document = ET.parse(sdf_path)
    except (OSError, ET.ParseError) as exc:
        raise ArtiverseVisualResourceError(
            f"Cannot parse prepared SDF for {object_id}: {exc}"
        ) from exc
    root = document.getroot()
    links = [element for element in root.iter() if _local_name(element.tag) == "link"]
    if not links:
        raise ArtiverseVisualResourceError(f"Prepared SDF has no links: {object_id}")

    mappings: list[dict[str, Any]] = []
    collision_bindings: list[dict[str, Any]] = []
    glb_cache: dict[Path, dict[str, Any]] = {}
    mapped_links = 0
    for link_index, link in enumerate(links):
        link_name = link.get("name")
        if not link_name:
            raise ArtiverseVisualResourceError(
                f"Prepared SDF link {link_index} has no name: {object_id}"
            )
        visuals = _direct_children(link, "visual")
        collisions = _direct_children(link, "collision")
        if not visuals:
            continue
        if not collisions:
            raise ArtiverseVisualResourceError(
                f"Visual link has no collision elements: {object_id}/{link_name}"
            )

        visual_records: list[dict[str, Any]] = []
        link_targets: set[str] = set()
        for visual_index, visual in enumerate(visuals):
            label = f"{object_id}/{link_name}/visual[{visual_index}]"
            record = _mesh_record(visual, label=label)
            obj_path = _resolve_local_resource(record["uri"], model_root, label=label)
            glb_relative = _obj_to_glb_relative(obj_path, model_root, label=label)
            glb_path = _resolve_local_resource(glb_relative, model_root, label=label)
            evidence = glb_cache.get(glb_path)
            if evidence is None:
                evidence = _read_glb_normal_evidence(
                    glb_path,
                    part=PurePosixPath(glb_relative).stem,
                )
                glb_cache[glb_path] = evidence
            link_targets.add(glb_relative)
            visual_records.append(record)
            record["structure"] = _visual_structure_signature(visual)
            mappings.append(
                {
                    "object_id": object_id,
                    "link_name": link_name,
                    "visual_index": visual_index,
                    "visual_name": record["name"],
                    "source_obj_uri": record["uri"],
                    "publisher_glb_uri": glb_relative,
                    "derived_gltf_uri": (
                        f"{DERIVED_RESOURCE_DIRECTORY}/"
                        f"{PurePosixPath(glb_relative).stem}.gltf"
                    ),
                    "pose": record["pose"],
                    "scale": record["scale"],
                    "derivation": evidence,
                }
            )
        if len(link_targets) != 1:
            raise ArtiverseVisualResourceError(
                f"Visuals in one link map to multiple GLBs: {object_id}/{link_name}"
            )
        visual_pose_scales = {
            _canonical_json({"pose": item["pose"], "scale": item["scale"]})
            for item in visual_records
        }
        if len(visual_pose_scales) != 1:
            raise ArtiverseVisualResourceError(
                f"Visuals in one link disagree on pose/scale: {object_id}/{link_name}"
            )
        visual_structures = {item["structure"] for item in visual_records}
        if len(visual_structures) != 1:
            raise ArtiverseVisualResourceError(
                "Material-split visuals in one link differ in render structure: "
                f"{object_id}/{link_name}"
            )

        collision_records: list[dict[str, Any]] = []
        for collision_index, collision in enumerate(collisions):
            label = f"{object_id}/{link_name}/collision[{collision_index}]"
            record = _mesh_record(collision, label=label)
            _resolve_local_resource(record["uri"], model_root, label=label)
            collision_record = {
                "object_id": object_id,
                "link_name": link_name,
                "collision_index": collision_index,
                "collision_name": record["name"],
                "uri": record["uri"],
                "pose": record["pose"],
                "scale": record["scale"],
            }
            collision_records.append(collision_record)
            collision_bindings.append(collision_record)

        comparable_collisions = {
            _canonical_json(
                {"uri": item["uri"], "pose": item["pose"], "scale": item["scale"]}
            )
            for item in collision_records
        }
        for visual in visual_records:
            comparable = _canonical_json(
                {
                    "uri": visual["uri"],
                    "pose": visual["pose"],
                    "scale": visual["scale"],
                }
            )
            if comparable not in comparable_collisions:
                raise ArtiverseVisualResourceError(
                    "Visual OBJ has no pose/scale-identical collision binding: "
                    f"{object_id}/{link_name}/{visual['uri']}"
                )
        mapped_links += 1

    if not mappings:
        raise ArtiverseVisualResourceError(
            f"Prepared Artiverse asset has no mesh visuals: {object_id}"
        )
    mappings.sort(
        key=lambda item: (
            item["link_name"],
            item["visual_index"],
            item["source_obj_uri"],
        )
    )
    collision_bindings.sort(
        key=lambda item: (
            item["link_name"],
            item["collision_index"],
            item["uri"],
        )
    )
    return {
        "mapped_link_count": mapped_links,
        "visual_mapping_count": len(mappings),
        "unique_glb_count": len(glb_cache),
        "glb_primitive_count": sum(
            item["primitive_count"] for item in glb_cache.values()
        ),
        "derived_gltf_count": len(glb_cache),
        "derived_bin_count": len(glb_cache),
        "gltf_buffer_view_count": sum(
            item["buffer_view_count"] for item in glb_cache.values()
        ),
        "gltf_accessor_count": sum(
            item["accessor_count"] for item in glb_cache.values()
        ),
        "gltf_material_count": sum(
            item["material_count"] for item in glb_cache.values()
        ),
        "gltf_texture_count": sum(
            item["texture_count"] for item in glb_cache.values()
        ),
        "gltf_image_count": sum(
            item["image_count"] for item in glb_cache.values()
        ),
        "derived_resource_inventory_sha256": _sha256_bytes(
            _canonical_json(
                sorted(
                    (
                        {
                            "publisher_glb_uri": glb_path.relative_to(
                                model_root
                            ).as_posix(),
                            "publisher_glb_sha256": item[
                                "publisher_glb_sha256"
                            ],
                            "preserved_semantics_sha256": item[
                                "preserved_semantics_sha256"
                            ],
                            "derived_gltf_name": item["derived_gltf_name"],
                            "derived_gltf_sha256": item["derived_gltf_sha256"],
                            "derived_bin_name": item["derived_bin_name"],
                            "derived_bin_sha256": item["derived_bin_sha256"],
                        }
                        for glb_path, item in glb_cache.items()
                    ),
                    key=lambda record: record["publisher_glb_uri"],
                )
            )
        ),
        "collision_binding_count": len(collision_bindings),
        "mapping_sha256": _sha256_bytes(_canonical_json(mappings)),
        "collision_binding_sha256": _sha256_bytes(
            _canonical_json(collision_bindings)
        ),
        "mappings": mappings,
        "collision_bindings": collision_bindings,
    }


def audit_prepared_authority(authority: Any) -> dict[str, Any]:
    """Audit every indexed Artiverse asset without modifying its source tree."""

    object_ids = list(authority.embedding_index)
    if not object_ids or len(object_ids) != len(set(object_ids)):
        raise ArtiverseVisualResourceError(
            "Artiverse authority has an empty or duplicate embedding index"
        )
    if set(object_ids) != set(authority.metadata):
        raise ArtiverseVisualResourceError(
            "Artiverse authority embedding and metadata indexes differ"
        )

    assets: list[dict[str, Any]] = []
    aggregate_mappings: list[dict[str, Any]] = []
    aggregate_collisions: list[dict[str, Any]] = []
    totals = {
        "mapped_link_count": 0,
        "visual_mapping_count": 0,
        "unique_glb_count": 0,
        "glb_primitive_count": 0,
        "derived_gltf_count": 0,
        "derived_bin_count": 0,
        "gltf_buffer_view_count": 0,
        "gltf_accessor_count": 0,
        "gltf_material_count": 0,
        "gltf_texture_count": 0,
        "gltf_image_count": 0,
        "collision_binding_count": 0,
    }
    for object_id in object_ids:
        metadata = authority.metadata.get(object_id)
        if not isinstance(metadata, dict):
            raise ArtiverseVisualResourceError(
                f"Artiverse metadata row is malformed: {object_id}"
            )
        raw_sdf = metadata.get("sdf_path")
        if not isinstance(raw_sdf, str) or not raw_sdf:
            raise ArtiverseVisualResourceError(
                f"Artiverse metadata has no SDF path: {object_id}"
            )
        expected_sdf = require_regular_file_within(
            authority.dataset_root / raw_sdf,
            authority.dataset_root,
            f"Artiverse SDF for {object_id}",
        )
        model_root = expected_sdf.parent
        before = _snapshot_regular_tree(model_root)
        expected = authority.asset(object_id)
        source_sdf = Path(expected["source_sdf_path"])
        if source_sdf != expected_sdf:
            raise ArtiverseVisualResourceError(
                f"Authority returned a different source SDF: {object_id}"
            )
        audited = _audit_sdf_visual_mapping(
            source_sdf,
            model_root,
            object_id=object_id,
        )
        revalidated = authority.asset(object_id)
        for field in (
            "source_sdf_path",
            "source_sdf_sha256",
            "source_tree_sha256",
        ):
            if revalidated[field] != expected[field]:
                raise ArtiverseVisualResourceError(
                    f"Artiverse source authority changed during visual audit: "
                    f"{object_id}/{field}"
                )
        after = _snapshot_regular_tree(model_root)
        if before != after:
            raise ArtiverseVisualResourceError(
                f"Artiverse source tree changed during visual audit: {object_id}"
            )

        asset_record = {
            "object_id": object_id,
            "source_sdf_path": source_sdf.relative_to(
                authority.dataset_root
            ).as_posix(),
            "source_sdf_sha256": expected["source_sdf_sha256"],
            "source_tree_sha256": expected["source_tree_sha256"],
            **{
                key: audited[key]
                for key in (
                    "mapped_link_count",
                    "visual_mapping_count",
                    "unique_glb_count",
                    "glb_primitive_count",
                    "derived_gltf_count",
                    "derived_bin_count",
                    "gltf_buffer_view_count",
                    "gltf_accessor_count",
                    "gltf_material_count",
                    "gltf_texture_count",
                    "gltf_image_count",
                    "collision_binding_count",
                    "derived_resource_inventory_sha256",
                    "mapping_sha256",
                    "collision_binding_sha256",
                )
            },
        }
        assets.append(asset_record)
        aggregate_mappings.extend(audited["mappings"])
        aggregate_collisions.extend(audited["collision_bindings"])
        for key in totals:
            totals[key] += int(audited[key])

    assets.sort(key=lambda item: item["object_id"])
    aggregate_mappings.sort(
        key=lambda item: (
            item["object_id"],
            item["link_name"],
            item["visual_index"],
            item["source_obj_uri"],
        )
    )
    aggregate_collisions.sort(
        key=lambda item: (
            item["object_id"],
            item["link_name"],
            item["collision_index"],
            item["uri"],
        )
    )
    return {
        "indexed_asset_count": len(object_ids),
        "audited_asset_count": len(assets),
        **totals,
        "mapping_inventory_sha256": _sha256_bytes(
            _canonical_json(aggregate_mappings)
        ),
        "collision_inventory_sha256": _sha256_bytes(
            _canonical_json(aggregate_collisions)
        ),
        "asset_inventory_sha256": _sha256_bytes(_canonical_json(assets)),
        "assets": assets,
        "mapping_target": RUNTIME_HELPER_POLICY,
        "publisher_glb_json_bin_resources_validated": True,
        "publisher_position_normal_accessors_validated": True,
        "publisher_material_image_references_validated": True,
        "derived_external_gltf_hashes_precomputed": True,
        "source_tree_identity_revalidated_after_audit": True,
        "source_hash_revalidated_after_each_asset_audit": True,
        "source_files_written": 0,
    }


def _canonical_xml(element: ET.Element) -> Any:
    return {
        "tag": _local_name(element.tag),
        "attributes": dict(sorted(element.attrib.items())),
        "text": (element.text or "").strip(),
        "children": [_canonical_xml(child) for child in element],
    }


def _planned_visual_normalization(
    source_root: ET.Element,
    source_model_root: Path,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for link_index, link in enumerate(
        element for element in source_root.iter() if _local_name(element.tag) == "link"
    ):
        link_name = link.get("name") or f"link[{link_index}]"
        visuals = _direct_children(link, "visual")
        if not visuals:
            continue
        parsed: list[dict[str, Any]] = []
        for visual_index, visual in enumerate(visuals):
            label = f"{link_name}/visual[{visual_index}]"
            record = _mesh_record(visual, label=label)
            geometry = _exact_child(visual, "geometry", label)
            mesh = _exact_child(geometry, "mesh", f"{label} geometry")
            uri_element = _exact_child(mesh, "uri", f"{label} mesh")
            raw_uri = record["uri"]
            obj_path = _resolve_local_resource(
                raw_uri,
                source_model_root,
                label=f"{label} source OBJ",
            )
            publisher_glb = _obj_to_glb_relative(
                obj_path,
                source_model_root,
                label=f"{label} source OBJ",
            )
            part = PurePosixPath(publisher_glb).stem
            parsed.append(
                {
                    "visual": visual,
                    "uri_element": uri_element,
                    "source_obj_uri": raw_uri,
                    "publisher_glb_uri": publisher_glb,
                    "part": part,
                    "derived_gltf_uri": (
                        f"./{DERIVED_RESOURCE_DIRECTORY}/{part}.gltf"
                    ),
                    "pose": record["pose"],
                    "scale": record["scale"],
                    "structure": _visual_structure_signature(visual),
                }
            )
        publishers = {item["publisher_glb_uri"] for item in parsed}
        targets = {item["derived_gltf_uri"] for item in parsed}
        transforms = {
            _canonical_json({"pose": item["pose"], "scale": item["scale"]})
            for item in parsed
        }
        structures = {item["structure"] for item in parsed}
        if (
            len(publishers) != 1
            or len(targets) != 1
            or len(transforms) != 1
            or len(structures) != 1
        ):
            raise ArtiverseVisualResourceError(
                f"Source visual group is ambiguous: {link_name}"
            )
        target = next(iter(targets))
        publisher_glb = next(iter(publishers))
        first = parsed[0]
        first["uri_element"].text = target
        removed_material_override_count = 0
        for material in _direct_children(first["visual"], "material"):
            first["visual"].remove(material)
            removed_material_override_count += 1
        for duplicate in parsed[1:]:
            removed_material_override_count += len(
                _direct_children(duplicate["visual"], "material")
            )
            link.remove(duplicate["visual"])
        groups.append(
            {
                "link_name": link_name,
                "source_obj_visual_uris": [
                    item["source_obj_uri"] for item in parsed
                ],
                "publisher_glb_uri": f"./{publisher_glb}",
                "part": first["part"],
                "derived_gltf_uri": target,
                "pose": first["pose"],
                "scale": first["scale"],
                "deduplicated_visual_count": len(parsed) - 1,
                "removed_material_override_count": removed_material_override_count,
            }
        )
    if not groups:
        raise ArtiverseVisualResourceError("SDF has no visual OBJ URIs to rewrite")
    return groups


def _expected_derived_resources(
    model_root: Path,
    groups: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, bytes], dict[str, Any], dict[str, dict[str, Any]]]:
    derivations: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    resources: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda item: str(item["part"])):
        part = str(group["part"])
        if part in derivations:
            continue
        publisher_uri = str(group["publisher_glb_uri"])
        publisher = _resolve_local_resource(
            publisher_uri,
            model_root,
            label=f"publisher Artiverse GLB for {part}",
        )
        evidence, gltf_payload, bin_payload = _read_glb_derivation(
            publisher,
            part=part,
        )
        gltf_name = f"{part}.gltf"
        bin_name = f"{part}.bin"
        for name, payload in ((gltf_name, gltf_payload), (bin_name, bin_payload)):
            previous = next(
                (existing for existing in payloads if existing.casefold() == name.casefold()),
                None,
            )
            if previous is not None:
                raise ArtiverseVisualResourceError(
                    "Derived Artiverse resource has a case-fold collision: "
                    f"{previous!r} versus {name!r}"
                )
            payloads[name] = payload
        record = {
            "part": part,
            "publisher_glb_sha256": evidence["publisher_glb_sha256"],
            "publisher_glb_size_bytes": evidence["publisher_glb_size_bytes"],
            "publisher_document_sha256": evidence["publisher_document_sha256"],
            "preserved_semantics_sha256": evidence["preserved_semantics_sha256"],
            "derived_gltf": gltf_name,
            "derived_gltf_sha256": evidence["derived_gltf_sha256"],
            "derived_gltf_size_bytes": evidence["derived_gltf_size_bytes"],
            "derived_bin": bin_name,
            "derived_bin_sha256": evidence["derived_bin_sha256"],
            "derived_bin_size_bytes": evidence["derived_bin_size_bytes"],
        }
        resources.append(record)
        derivations[part] = {**evidence, **record}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "policy": RUNTIME_HELPER_POLICY,
        "resource_directory": DERIVED_RESOURCE_DIRECTORY,
        "resources": resources,
    }
    payloads[DERIVATION_MANIFEST] = _canonical_json(manifest) + b"\n"
    return payloads, manifest, derivations


def _regular_content_inventory(
    root: Path,
    *,
    excluded_prefix: str | None = None,
) -> dict[str, dict[str, Any]]:
    canonical = root.resolve(strict=True)
    result: dict[str, dict[str, Any]] = {}
    for identity in _snapshot_regular_tree(canonical):
        if not identity.startswith("F:"):
            continue
        relative = identity[2:]
        if excluded_prefix is not None and (
            relative == excluded_prefix or relative.startswith(excluded_prefix + "/")
        ):
            continue
        path = canonical.joinpath(*PurePosixPath(relative).parts)
        result[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _content_inventory_tree_sha256(inventory: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, record in sorted(inventory.items()):
        relative = name.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(str(record["sha256"])))
    return digest.hexdigest()


def _validate_exact_derived_directory(
    copied_root: Path,
    expected_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    directory = copied_root / DERIVED_RESOURCE_DIRECTORY
    try:
        canonical = directory.resolve(strict=True)
    except OSError as exc:
        raise ArtiverseVisualResourceError(
            f"Copied Artiverse derived-resource directory is missing: {directory}"
        ) from exc
    if not canonical.is_dir() or directory.is_symlink():
        raise ArtiverseVisualResourceError(
            f"Copied Artiverse derived-resource path is not a real directory: {directory}"
        )
    try:
        direct_entries = list(os.scandir(canonical))
    except OSError as exc:
        raise ArtiverseVisualResourceError(
            f"Cannot scan copied Artiverse derived resources: {canonical}"
        ) from exc
    if any(not entry.is_file(follow_symlinks=False) for entry in direct_entries):
        raise ArtiverseVisualResourceError(
            "Copied Artiverse derived-resource directory must contain only flat "
            "regular files"
        )
    inventory = _regular_content_inventory(canonical)
    if sorted(inventory) != sorted(expected_payloads):
        raise ArtiverseVisualResourceError(
            "Copied Artiverse derived-resource inventory is not exact"
        )
    records: list[dict[str, Any]] = []
    for name in sorted(expected_payloads):
        payload = expected_payloads[name]
        actual = inventory[name]
        expected_hash = _sha256_bytes(payload)
        if actual != {"size_bytes": len(payload), "sha256": expected_hash}:
            raise ArtiverseVisualResourceError(
                f"Copied Artiverse derived resource differs: {name}"
            )
        records.append({"path": name, **actual})
    tree_sha256 = _content_inventory_tree_sha256(
        {
            str(record["path"]): {
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
            for record in records
        }
    )
    return {
        "directory": DERIVED_RESOURCE_DIRECTORY,
        "file_count": len(records),
        "files": records,
        "inventory_sha256": _sha256_bytes(_canonical_json(records)),
        "tree_sha256": tree_sha256,
    }


def validate_normalized_copy(
    source_sdf: Path,
    copied_sdf: Path,
    *,
    source_tree_root: Path | None = None,
    copied_tree_root: Path | None = None,
    normalization_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove the copied SDF changed only its allowed visual-element surface."""

    source_root_dir = (source_tree_root or source_sdf.parent).resolve(strict=True)
    copied_root_dir = (copied_tree_root or copied_sdf.parent).resolve(strict=True)
    source_file = require_regular_file_within(
        source_sdf, source_root_dir, "Source Artiverse SDF"
    )
    copied_file = require_regular_file_within(
        copied_sdf, copied_root_dir, "Copied Artiverse SDF"
    )
    try:
        source_document = ET.parse(source_file)
        copied_document = ET.parse(copied_file)
    except (OSError, ET.ParseError) as exc:
        raise ArtiverseVisualResourceError(f"Cannot parse source/copied SDF: {exc}") from exc

    expected_root = copy.deepcopy(source_document.getroot())
    groups = _planned_visual_normalization(expected_root, source_root_dir)
    expected_payloads, expected_manifest, derivations = _expected_derived_resources(
        source_root_dir,
        groups,
    )
    derived_directory = _validate_exact_derived_directory(
        copied_root_dir,
        expected_payloads,
    )

    expected = _canonical_xml(expected_root)
    actual = _canonical_xml(copied_document.getroot())
    if expected != actual:
        raise ArtiverseVisualResourceError(
            "Copied Artiverse SDF differs from source outside the allowed visual "
            "external-glTF grouping/material normalization"
        )
    source_relative_sdf = source_file.relative_to(source_root_dir).as_posix()
    copied_relative_sdf = copied_file.relative_to(copied_root_dir).as_posix()
    if source_relative_sdf != copied_relative_sdf:
        raise ArtiverseVisualResourceError(
            "Source/copied Artiverse SDF relative paths differ"
        )
    source_inventory = _regular_content_inventory(source_root_dir)
    copied_full_inventory = _regular_content_inventory(copied_root_dir)
    copied_full_tree_sha256 = _content_inventory_tree_sha256(copied_full_inventory)
    copied_inventory = _regular_content_inventory(
        copied_root_dir,
        excluded_prefix=DERIVED_RESOURCE_DIRECTORY,
    )
    source_inventory.pop(source_relative_sdf, None)
    copied_inventory.pop(copied_relative_sdf, None)
    if copied_inventory != source_inventory:
        raise ArtiverseVisualResourceError(
            "Copied Artiverse tree differs from source outside its SDF and exact "
            "derived-resource directory"
        )
    groups.sort(key=lambda item: item["link_name"])
    source_hash = sha256_file(source_file)
    copied_hash = sha256_file(copied_file)
    result: dict[str, Any] = {
        "status": "pass",
        "normalized_link_count": len(groups),
        "source_visual_count": sum(
            1 + int(group["deduplicated_visual_count"]) for group in groups
        ),
        "deduplicated_visual_count": sum(
            int(group["deduplicated_visual_count"]) for group in groups
        ),
        "removed_material_override_count": sum(
            int(group["removed_material_override_count"]) for group in groups
        ),
        "normalization_sha256": _sha256_bytes(_canonical_json(groups)),
        "mapping_target": RUNTIME_HELPER_POLICY,
        "source_sdf_sha256": source_hash,
        "copied_sdf_sha256": copied_hash,
        "derived_resource_directory": DERIVED_RESOURCE_DIRECTORY,
        "derived_resource_directory_sha256": derived_directory[
            "tree_sha256"
        ],
        "derived_resource_file_count": derived_directory["file_count"],
        "derived_resource_manifest_sha256": _sha256_bytes(
            expected_payloads[DERIVATION_MANIFEST]
        ),
        "expected_derivation_manifest": expected_manifest,
        "derived_semantics_sha256": _sha256_bytes(
            _canonical_json(
                {
                    part: evidence["preserved_semantics_sha256"]
                    for part, evidence in sorted(derivations.items())
                }
            )
        ),
        "only_allowed_visual_elements_changed": True,
        "publisher_glb_semantics_preserved_in_external_gltf": True,
        "collision_bindings_unchanged": True,
        "poses_scales_joints_inertials_unchanged": True,
        "publisher_source_and_nonderived_copied_resources_unchanged": True,
    }
    if normalization_evidence is not None:
        rewritten = int(normalization_evidence.get("rewritten_link_count", -1))
        migrated = int(
            normalization_evidence.get("migrated_legacy_glb_link_count", -1)
        )
        already = int(
            normalization_evidence.get("already_normalized_link_count", -1)
        )
        runtime_links = normalization_evidence.get("links")
        expected_links = {
            str(group["link_name"]): {
                "part": str(group["part"]),
                "publisher_glb_uri": str(group["publisher_glb_uri"]),
                "derived_gltf_uri": str(group["derived_gltf_uri"]),
            }
            for group in groups
        }
        runtime_link_map = {
            str(item.get("link_name")): item
            for item in runtime_links
            if isinstance(item, dict) and isinstance(item.get("link_name"), str)
        } if isinstance(runtime_links, list) else {}
        links_match = set(runtime_link_map) == set(expected_links)
        if links_match:
            for link_name, expected_link in expected_links.items():
                item = runtime_link_map[link_name]
                part = expected_link["part"]
                derivation = derivations[part]
                if (
                    item.get("state")
                    not in {"rewritten", "migrated_legacy_glb", "already_normalized"}
                    or item.get("part") != part
                    or item.get("publisher_glb_uri")
                    != expected_link["publisher_glb_uri"]
                    or item.get("publisher_glb_size_bytes")
                    != derivation["publisher_glb_size_bytes"]
                    or item.get("publisher_glb_sha256")
                    != derivation["publisher_glb_sha256"]
                    or item.get("publisher_document_sha256")
                    != derivation["publisher_document_sha256"]
                    or item.get("preserved_semantics_sha256")
                    != derivation["preserved_semantics_sha256"]
                    or item.get("derived_gltf_uri")
                    != expected_link["derived_gltf_uri"]
                    or item.get("derived_gltf_size_bytes")
                    != derivation["derived_gltf_size_bytes"]
                    or item.get("derived_gltf_sha256")
                    != derivation["derived_gltf_sha256"]
                    or item.get("derived_bin_uri")
                    != f"./{DERIVED_RESOURCE_DIRECTORY}/{part}.bin"
                    or item.get("derived_bin_size_bytes")
                    != derivation["derived_bin_size_bytes"]
                    or item.get("derived_bin_sha256")
                    != derivation["derived_bin_sha256"]
                ):
                    links_match = False
                    break
        if (
            normalization_evidence.get("schema_version") != SCHEMA_VERSION
            or normalization_evidence.get("status") != "pass"
            or normalization_evidence.get("policy") != RUNTIME_HELPER_POLICY
            or Path(str(normalization_evidence.get("copied_sdf_path", ""))).resolve()
            != copied_file
            or normalization_evidence.get("sdf_sha256_after") != copied_hash
            or normalization_evidence.get("copied_tree_sha256_after")
            != copied_full_tree_sha256
            or rewritten + migrated + already != len(groups)
            or (
                rewritten == len(groups)
                and (
                    int(normalization_evidence.get("deduplicated_visual_count", -1))
                    != result["deduplicated_visual_count"]
                    or int(
                        normalization_evidence.get(
                            "removed_material_override_count", -1
                        )
                    )
                    != result["removed_material_override_count"]
                )
            )
            or (
                rewritten == 0
                and (
                    normalization_evidence.get("deduplicated_visual_count") != 0
                    or normalization_evidence.get(
                        "removed_material_override_count"
                    )
                    != 0
                )
            )
            or Path(
                str(normalization_evidence.get("derived_resource_directory", ""))
            ).resolve()
            != (copied_root_dir / DERIVED_RESOURCE_DIRECTORY).resolve()
            or normalization_evidence.get("derived_resource_directory_sha256")
            != result["derived_resource_directory_sha256"]
            or normalization_evidence.get("derivation_manifest_sha256")
            != result["derived_resource_manifest_sha256"]
            or normalization_evidence.get("derived_resource_count")
            != result["derived_resource_file_count"] - 1
            or normalization_evidence.get("derived_part_count") != len(derivations)
            or normalization_evidence.get("derived_publication_state")
            not in {"published", "verified_existing", "recovered_unreferenced"}
            or not isinstance(
                normalization_evidence.get("protected_nonvisual_sdf_sha256"), str
            )
            or len(normalization_evidence.get("protected_nonvisual_sdf_sha256", ""))
            != 64
            or not links_match
        ):
            raise ArtiverseVisualResourceError(
                "Runtime Artiverse visual-normalization evidence is missing or stale"
            )
        result["runtime_normalization_evidence_sha256"] = _sha256_bytes(
            _canonical_json(normalization_evidence)
        )
    return result


def _code_artifact(path: Path) -> dict[str, Any]:
    try:
        supplied = path.expanduser().absolute()
        resolved = supplied.resolve(strict=True)
        metadata = supplied.lstat()
    except OSError as exc:
        raise ArtiverseVisualResourceError(f"Runtime code file is missing: {path}") from exc
    if supplied.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ArtiverseVisualResourceError(
            f"Runtime code must be an ordinary file: {supplied}"
        )
    return {
        "path": str(resolved),
        "size_bytes": metadata.st_size,
        "sha256": sha256_file(resolved),
    }


def build_receipt_from_authority(
    authority: Any,
    *,
    runtime_code: Iterable[Path],
) -> dict[str, Any]:
    runtime_paths = list(runtime_code)
    if not runtime_paths:
        raise ArtiverseVisualResourceError(
            "At least one runtime normalization code path is required"
        )
    script_path = Path(__file__).resolve(strict=True)
    contract_path = Path(sys.modules[load_artiverse_authority.__module__].__file__).resolve(
        strict=True
    )
    code_paths = {script_path, contract_path}
    code_paths.update(path.expanduser().absolute().resolve(strict=True) for path in runtime_paths)
    code_artifacts = sorted(
        (_code_artifact(path) for path in code_paths),
        key=lambda item: item["path"],
    )
    audit = audit_prepared_authority(authority)
    code_artifacts_after = sorted(
        (_code_artifact(path) for path in code_paths),
        key=lambda item: item["path"],
    )
    if code_artifacts_after != code_artifacts:
        raise ArtiverseVisualResourceError(
            "Visual-normalization runtime code changed during source audit"
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "status": "pass",
        "hash_algorithm": HASH_ALGORITHM,
        "mapping_policy": mapping_policy(),
        "authority": authority.evidence(),
        "runtime_code": code_artifacts,
        "runtime_code_revalidated_after_audit": True,
        "audit": audit,
    }
    return {**payload, "attestation": _attestation(payload)}


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except ArtiverseVisualResourceError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtiverseVisualResourceError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtiverseVisualResourceError(f"{label} must be a JSON object")
    return value


def _validate_saved_receipt(receipt: Mapping[str, Any]) -> None:
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("contract") != CONTRACT_NAME
        or receipt.get("status") != "pass"
    ):
        raise ArtiverseVisualResourceError(
            "Saved visual-resource receipt is not a passing supported contract"
        )
    attestation = receipt.get("attestation")
    payload = {key: value for key, value in receipt.items() if key != "attestation"}
    if attestation != _attestation(payload):
        raise ArtiverseVisualResourceError(
            "Saved visual-resource receipt attestation is missing or stale"
        )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().absolute()
    if path.exists() and path.is_symlink():
        raise ArtiverseVisualResourceError(f"Output receipt is a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() and temporary.is_symlink():
        raise ArtiverseVisualResourceError(f"Temporary receipt is a symlink: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(
    *,
    dataset_root: Path,
    embeddings_path: Path | None,
    runtime_code: Sequence[Path],
    output: Path,
    verify_only: bool = False,
) -> dict[str, Any]:
    canonical_dataset = dataset_root.expanduser().resolve(strict=True)
    output_path = output.expanduser().absolute()
    if _inside(canonical_dataset, output_path):
        raise ArtiverseVisualResourceError(
            "Visual-resource receipt must be written outside the read-only dataset"
        )
    saved: dict[str, Any] | None = None
    if verify_only:
        saved = _load_json_object(output_path, label="saved visual-resource receipt")
        _validate_saved_receipt(saved)
    try:
        authority = load_artiverse_authority(dataset_root, embeddings_path)
    except ArtiverseContractError as exc:
        raise ArtiverseVisualResourceError(str(exc)) from exc
    current = build_receipt_from_authority(authority, runtime_code=runtime_code)
    if saved is not None:
        if saved != current:
            raise ArtiverseVisualResourceError(
                "Saved visual-resource receipt differs from the current authority, "
                "mapping inventory, collision bindings, or runtime code"
            )
        return saved
    _atomic_write_json(output_path, current)
    return current


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/artiverse"))
    parser.add_argument("--embeddings-path", type=Path)
    parser.add_argument(
        "--runtime-code",
        type=Path,
        action="append",
        dest="runtime_code",
        help=(
            "Runtime normalizer/helper file to hash-bind. Repeat for every executing "
            "normalization component. Defaults to the canonical helper and asset manager."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime_code = args.runtime_code or list(DEFAULT_RUNTIME_CODE)
    try:
        receipt = run(
            dataset_root=args.dataset_root,
            embeddings_path=args.embeddings_path,
            runtime_code=runtime_code,
            output=args.output,
            verify_only=args.verify_only,
        )
    except (ArtiverseVisualResourceError, OSError) as exc:
        if not args.verify_only:
            try:
                canonical_dataset = args.dataset_root.expanduser().resolve(strict=True)
                output_path = args.output.expanduser().absolute()
                if not _inside(canonical_dataset, output_path):
                    _atomic_write_json(
                        output_path,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "contract": CONTRACT_NAME,
                            "status": "fail",
                            "error": str(exc),
                        },
                    )
            except Exception:
                pass
        print(f"FAIL: {exc}")
        return 2
    action = "verified" if args.verify_only else "created"
    audit = receipt["audit"]
    print(
        f"PASS: {action} Artiverse visual mapping receipt for "
        f"{audit['audited_asset_count']} assets / "
        f"{audit['visual_mapping_count']} visuals"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
