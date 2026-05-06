"""Derive Drake-compatible external glTF visuals from publisher Artiverse GLBs.

Artiverse's converted SDF can contain one OBJ visual per material split while the
same copied model tree also contains one publisher ``glbs/<part>.glb`` with authored
normals and materials for that complete part.  Drake's glTF render client accepts
external ``.gltf`` but not direct ``.glb`` SDF mesh URIs, so this module losslessly
externalizes each authoritative GLB's JSON and BIN chunks into one room-local,
atomically published resource directory.  It then replaces each link's split OBJ
visual group with one external glTF visual.  Collision elements and publisher files
are never edited.

The mapping is deliberately narrow and fail-closed.  A link must name one unique
part, all split visuals must have identical pose and scale, the matching GLB must be
a self-contained, regular in-tree GLB 2.0 file, and every triangle primitive must
contain a NORMAL attribute.  The exact derived directory is published before the
rewritten SDF.  A retry validates and reuses an exact crash orphan, but rejects any
stale mismatch.  SDF publication uses an atomic sibling replacement.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import secrets
import stat
import struct
import tempfile

from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET


POLICY = "publisher_glb_derived_external_gltf"
SCHEMA_VERSION = 2
DERIVED_RESOURCE_DIRECTORY = "scenesmith_artiverse_visuals_v2"
DERIVATION_MANIFEST = "_derivation_manifest.json"
_HASH_CHUNK_SIZE = 1024 * 1024
_GLB_HEADER = struct.Struct("<4sII")
_GLB_CHUNK_HEADER = struct.Struct("<II")
_GLB_MAGIC = b"glTF"
_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_BIN_CHUNK = 0x004E4942
_SAFE_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ArtiverseVisualNormalizationError(RuntimeError):
    """A copied Artiverse SDF cannot be normalized without ambiguity."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _real_directory(path: Path, *, label: str) -> Path:
    supplied = path.expanduser().absolute()
    if _is_link_like(supplied):
        raise ArtiverseVisualNormalizationError(
            f"{label} must not be link-like: {supplied}"
        )
    try:
        metadata = supplied.stat(follow_symlinks=False)
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ArtiverseVisualNormalizationError(
            f"cannot inspect {label} {supplied}: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArtiverseVisualNormalizationError(
            f"{label} is not a real directory: {supplied}"
        )
    if os.path.normcase(os.fspath(supplied)) != os.path.normcase(os.fspath(resolved)):
        raise ArtiverseVisualNormalizationError(
            f"{label} contains a link-like parent: {supplied}"
        )
    return resolved


def _inside(root: Path, candidate: Path) -> bool:
    try:
        return os.path.commonpath((os.fspath(root), os.fspath(candidate))) == os.fspath(
            root
        )
    except ValueError:
        return False


def _regular_file_within(path: Path, root: Path, *, label: str) -> Path:
    supplied = path.absolute()
    current = root
    try:
        relative = supplied.relative_to(root)
    except ValueError as exc:
        raise ArtiverseVisualNormalizationError(
            f"{label} escapes copied Artiverse tree: {supplied}"
        ) from exc
    for part in relative.parts:
        current = current / part
        if _is_link_like(current):
            raise ArtiverseVisualNormalizationError(
                f"{label} contains a link-like component: {current}"
            )
    try:
        metadata = supplied.stat(follow_symlinks=False)
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ArtiverseVisualNormalizationError(
            f"cannot inspect {label} {supplied}: {exc}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < 1:
        raise ArtiverseVisualNormalizationError(
            f"{label} must be a nonempty regular file: {supplied}"
        )
    if metadata.st_nlink != 1:
        raise ArtiverseVisualNormalizationError(
            f"{label} must not be hard-linked: {supplied}"
        )
    if not _inside(root, resolved):
        raise ArtiverseVisualNormalizationError(
            f"{label} resolves outside copied Artiverse tree: {resolved}"
        )
    return resolved


def _safe_tree_files(root: Path) -> list[Path]:
    selected: list[Path] = []
    pending = [root]
    casefolded: dict[str, str] = {}
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ArtiverseVisualNormalizationError(
                f"cannot scan copied Artiverse tree {directory}: {exc}"
            ) from exc
        for entry in entries:
            candidate = Path(entry.path)
            if entry.is_symlink() or _is_link_like(candidate):
                raise ArtiverseVisualNormalizationError(
                    f"copied Artiverse tree contains a link: {candidate}"
                )
            # ``DirEntry.stat().st_nlink`` is reported as zero by some Windows
            # filesystems; Path.stat returns the authoritative link count.
            metadata = candidate.stat(follow_symlinks=False)
            relative = candidate.relative_to(root).as_posix()
            collision_key = relative.casefold()
            previous = casefolded.setdefault(collision_key, relative)
            if previous != relative:
                raise ArtiverseVisualNormalizationError(
                    "copied Artiverse tree has a case-fold path collision: "
                    f"{previous!r} versus {relative!r}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(candidate)
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise ArtiverseVisualNormalizationError(
                        f"copied Artiverse tree contains a hard link: {candidate}"
                    )
                selected.append(candidate.resolve(strict=True))
            else:
                raise ArtiverseVisualNormalizationError(
                    f"copied Artiverse tree contains a special file: {candidate}"
                )
    if not selected:
        raise ArtiverseVisualNormalizationError(
            f"copied Artiverse tree is empty: {root}"
        )
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def sha256_safe_directory_tree(root: Path) -> str:
    """Hash an exact regular, unlinked directory inventory deterministically."""

    resolved_root = _real_directory(root, label="Artiverse tree")
    digest = hashlib.sha256()
    for path in _safe_tree_files(resolved_root):
        relative = path.relative_to(resolved_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _local_name(element: ET.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local_name(child) == name]


def _one_direct_child(element: ET.Element, name: str, *, label: str) -> ET.Element:
    children = _direct_children(element, name)
    if len(children) != 1:
        raise ArtiverseVisualNormalizationError(
            f"{label} must contain exactly one direct {name} element"
        )
    return children[0]


def _finite_vector(
    element: ET.Element | None,
    *,
    count: int,
    default: tuple[float, ...],
    label: str,
    strictly_positive: bool,
) -> dict[str, Any]:
    if element is None:
        values = default
        attributes: tuple[tuple[str, str], ...] = ()
    else:
        tokens = (element.text or "").split()
        if len(tokens) != count:
            raise ArtiverseVisualNormalizationError(
                f"{label} must contain exactly {count} values"
            )
        try:
            values = tuple(float(token) for token in tokens)
        except ValueError as exc:
            raise ArtiverseVisualNormalizationError(
                f"{label} contains a nonnumeric value"
            ) from exc
        attributes = tuple(sorted((str(key), str(value)) for key, value in element.attrib.items()))
    if not all(math.isfinite(value) for value in values):
        raise ArtiverseVisualNormalizationError(f"{label} contains a nonfinite value")
    if strictly_positive and not all(value > 0.0 for value in values):
        raise ArtiverseVisualNormalizationError(f"{label} must be strictly positive")
    return {"values": list(values), "attributes": [list(item) for item in attributes]}


def _visual_transform(visual: ET.Element, mesh: ET.Element, *, label: str) -> dict[str, Any]:
    poses = _direct_children(visual, "pose")
    scales = _direct_children(mesh, "scale")
    if len(poses) > 1 or len(scales) > 1:
        raise ArtiverseVisualNormalizationError(
            f"{label} has duplicate pose or mesh scale elements"
        )
    return {
        "pose": _finite_vector(
            poses[0] if poses else None,
            count=6,
            default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            label=f"{label} pose",
            strictly_positive=False,
        ),
        "scale": _finite_vector(
            scales[0] if scales else None,
            count=3,
            default=(1.0, 1.0, 1.0),
            label=f"{label} mesh scale",
            strictly_positive=True,
        ),
    }


def _visual_structure_signature(visual: ET.Element) -> tuple[Any, ...]:
    """Bind every visual property that deduplication is not allowed to discard."""

    def signature(element: ET.Element, *, visual_root: bool = False) -> tuple[Any, ...]:
        name = _local_name(element)
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
            if not (visual_root and _local_name(child) == "material")
        )
        return name, attributes, text, children

    return signature(visual, visual_root=True)


def _mesh_uri(visual: ET.Element, *, label: str) -> tuple[ET.Element, ET.Element, str]:
    geometry = _one_direct_child(visual, "geometry", label=label)
    mesh = _one_direct_child(geometry, "mesh", label=f"{label} geometry")
    uri = _one_direct_child(mesh, "uri", label=f"{label} mesh")
    raw_uri = (uri.text or "").strip()
    if not raw_uri:
        raise ArtiverseVisualNormalizationError(f"{label} mesh URI is empty")
    return mesh, uri, raw_uri


def _safe_relative_uri(raw_uri: str, *, directory: str, suffix: str, label: str) -> PurePosixPath:
    parsed = urlsplit(raw_uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ArtiverseVisualNormalizationError(f"{label} URI is not a plain relative path")
    decoded = unquote(parsed.path)
    if "\\" in decoded or "\x00" in decoded:
        raise ArtiverseVisualNormalizationError(f"{label} URI contains an unsafe character")
    while decoded.startswith("./"):
        decoded = decoded[2:]
    relative = PurePosixPath(decoded)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != directory
        or relative.suffix.lower() != suffix
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ArtiverseVisualNormalizationError(
            f"{label} URI is outside the exact ./{directory}/<file>{suffix} contract: {raw_uri}"
        )
    return relative


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _glb_external_derivation(path: Path, *, part: str) -> dict[str, Any]:
    """Validate one self-contained GLB and externalize its semantic payload."""

    data = path.read_bytes()
    if len(data) < _GLB_HEADER.size + _GLB_CHUNK_HEADER.size:
        raise ArtiverseVisualNormalizationError(f"publisher GLB is truncated: {path}")
    magic, version, declared_length = _GLB_HEADER.unpack_from(data, 0)
    if magic != _GLB_MAGIC or version != 2 or declared_length != len(data):
        raise ArtiverseVisualNormalizationError(
            f"publisher file is not an exact GLB 2.0 payload: {path}"
        )
    chunks: list[tuple[int, bytes]] = []
    offset = _GLB_HEADER.size
    while offset < len(data):
        if offset + _GLB_CHUNK_HEADER.size > len(data):
            raise ArtiverseVisualNormalizationError(
                f"publisher GLB has a truncated chunk header: {path}"
            )
        chunk_length, chunk_type = _GLB_CHUNK_HEADER.unpack_from(data, offset)
        offset += _GLB_CHUNK_HEADER.size
        chunk_end = offset + chunk_length
        if chunk_length % 4 or chunk_end > len(data):
            raise ArtiverseVisualNormalizationError(
                f"publisher GLB has an invalid chunk length: {path}"
            )
        chunks.append((chunk_type, data[offset:chunk_end]))
        offset = chunk_end
    if offset != len(data) or len(chunks) != 2:
        raise ArtiverseVisualNormalizationError(
            f"publisher GLB must contain exactly JSON and BIN chunks: {path}"
        )
    if chunks[0][0] != _GLB_JSON_CHUNK or chunks[1][0] != _GLB_BIN_CHUNK:
        raise ArtiverseVisualNormalizationError(
            f"publisher GLB lacks the exact leading JSON/following BIN chunks: {path}"
        )
    json_payload = chunks[0][1]
    binary_chunk = chunks[1][1]
    try:
        document = json.loads(
            json_payload.rstrip(b" \t\r\n\x00").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtiverseVisualNormalizationError(
            f"publisher GLB JSON is malformed: {path}: {exc}"
        ) from exc
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("asset"), dict)
        or str(document["asset"].get("version", "")) != "2.0"
        or not isinstance(document.get("accessors"), list)
    ):
        raise ArtiverseVisualNormalizationError(
            f"publisher GLB lacks a glTF 2.0 asset/accessor inventory: {path}"
        )
    buffers = document.get("buffers")
    if (
        not isinstance(buffers, list)
        or len(buffers) != 1
        or not isinstance(buffers[0], dict)
        or "uri" in buffers[0]
        or isinstance(buffers[0].get("byteLength"), bool)
        or not isinstance(buffers[0].get("byteLength"), int)
        or buffers[0]["byteLength"] < 1
    ):
        raise ArtiverseVisualNormalizationError(
            f"publisher GLB must contain one embedded nonempty buffer: {path}"
        )
    buffer_length = buffers[0]["byteLength"]
    if not buffer_length <= len(binary_chunk) <= buffer_length + 3:
        raise ArtiverseVisualNormalizationError(
            f"publisher GLB BIN chunk does not match buffer byteLength: {path}"
        )
    if any(binary_chunk[buffer_length:]):
        raise ArtiverseVisualNormalizationError(
            f"publisher GLB BIN padding is nonzero: {path}"
        )
    binary_payload = binary_chunk[:buffer_length]

    buffer_views = document.get("bufferViews", [])
    if not isinstance(buffer_views, list):
        raise ArtiverseVisualNormalizationError(
            f"publisher GLB bufferViews inventory is invalid: {path}"
        )
    for index, view in enumerate(buffer_views):
        if not isinstance(view, dict) or view.get("buffer") != 0:
            raise ArtiverseVisualNormalizationError(
                f"publisher GLB bufferView {index} does not use embedded buffer 0: {path}"
            )
        byte_offset = view.get("byteOffset", 0)
        byte_length = view.get("byteLength")
        if (
            isinstance(byte_offset, bool)
            or not isinstance(byte_offset, int)
            or byte_offset < 0
            or isinstance(byte_length, bool)
            or not isinstance(byte_length, int)
            or byte_length < 1
            or byte_offset + byte_length > buffer_length
        ):
            raise ArtiverseVisualNormalizationError(
                f"publisher GLB bufferView {index} escapes its buffer: {path}"
            )

    images = document.get("images", [])
    if not isinstance(images, list):
        raise ArtiverseVisualNormalizationError(
            f"publisher GLB image inventory is invalid: {path}"
        )
    embedded_image_count = 0
    data_image_count = 0
    for image_index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ArtiverseVisualNormalizationError(
                f"publisher GLB image {image_index} is invalid: {path}"
            )
        has_uri = "uri" in image
        has_view = "bufferView" in image
        if has_uri == has_view:
            raise ArtiverseVisualNormalizationError(
                f"publisher GLB image {image_index} must use exactly one embedded source: {path}"
            )
        if has_uri:
            uri = image["uri"]
            if not isinstance(uri, str) or not uri.startswith("data:"):
                raise ArtiverseVisualNormalizationError(
                    f"publisher GLB image {image_index} references an external resource: {path}"
                )
            data_image_count += 1
        else:
            view_index = image["bufferView"]
            if (
                isinstance(view_index, bool)
                or not isinstance(view_index, int)
                or not 0 <= view_index < len(buffer_views)
                or not isinstance(image.get("mimeType"), str)
                or not image["mimeType"]
            ):
                raise ArtiverseVisualNormalizationError(
                    f"publisher GLB image {image_index} has an invalid embedded bufferView: {path}"
                )
            embedded_image_count += 1

    accessors = document["accessors"]

    component_sizes = {
        5120: 1,
        5121: 1,
        5122: 2,
        5123: 2,
        5125: 4,
        5126: 4,
    }

    def validate_stored_accessor(
        accessor_index: int, *, semantic: str, allowed_component_types: set[int]
    ) -> dict[str, Any]:
        accessor = accessors[accessor_index]
        if not isinstance(accessor, dict) or "sparse" in accessor:
            raise ArtiverseVisualNormalizationError(
                f"publisher GLB {semantic} accessor is invalid or sparse: {path}"
            )
        view_index = accessor.get("bufferView")
        component_type = accessor.get("componentType")
        count = accessor.get("count")
        accessor_offset = accessor.get("byteOffset", 0)
        if (
            isinstance(view_index, bool)
            or not isinstance(view_index, int)
            or not 0 <= view_index < len(buffer_views)
            or component_type not in allowed_component_types
            or isinstance(component_type, bool)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or isinstance(accessor_offset, bool)
            or not isinstance(accessor_offset, int)
            or accessor_offset < 0
        ):
            raise ArtiverseVisualNormalizationError(
                f"publisher GLB {semantic} accessor storage is invalid: {path}"
            )
        if accessor.get("type") == "VEC3":
            component_count = 3
        elif accessor.get("type") == "SCALAR":
            component_count = 1
        else:
            raise ArtiverseVisualNormalizationError(
                f"publisher GLB {semantic} accessor type is invalid: {path}"
            )
        component_size = component_sizes[component_type]
        element_size = component_size * component_count
        view = buffer_views[view_index]
        stride = view.get("byteStride", element_size)
        absolute_offset = view.get("byteOffset", 0) + accessor_offset
        if (
            isinstance(stride, bool)
            or not isinstance(stride, int)
            or stride < element_size
            or stride > 252
            or stride % component_size
            or absolute_offset % component_size
            or accessor_offset + stride * (count - 1) + element_size
            > view["byteLength"]
            or (
                "byteStride" in view
                and (semantic == "indices" or stride % 4)
            )
        ):
            raise ArtiverseVisualNormalizationError(
                f"publisher GLB {semantic} accessor range/stride is invalid: {path}"
            )
        normalized = accessor.get("normalized", False)
        if not isinstance(normalized, bool):
            raise ArtiverseVisualNormalizationError(
                f"publisher GLB {semantic} normalized flag is invalid: {path}"
            )
        return {
            "component_type": component_type,
            "count": count,
            "normalized": normalized,
        }

    meshes = document.get("meshes")
    if not isinstance(meshes, list) or not meshes:
        raise ArtiverseVisualNormalizationError(f"publisher GLB contains no meshes: {path}")
    primitive_count = 0
    normal_accessor_count = 0
    for mesh_index, mesh in enumerate(meshes):
        primitives = mesh.get("primitives") if isinstance(mesh, dict) else None
        if not isinstance(primitives, list) or not primitives:
            raise ArtiverseVisualNormalizationError(
                f"publisher GLB mesh {mesh_index} has no primitives: {path}"
            )
        for primitive_index, primitive in enumerate(primitives):
            if not isinstance(primitive, dict) or primitive.get("mode", 4) != 4:
                raise ArtiverseVisualNormalizationError(
                    f"publisher GLB primitive is not TRIANGLES: {path}"
                )
            attributes = primitive.get("attributes")
            if not isinstance(attributes, dict):
                raise ArtiverseVisualNormalizationError(
                    f"publisher GLB primitive lacks attributes: {path}"
                )
            position_index = attributes.get("POSITION")
            normal_index = attributes.get("NORMAL")
            if (
                isinstance(position_index, bool)
                or not isinstance(position_index, int)
                or isinstance(normal_index, bool)
                or not isinstance(normal_index, int)
                or not 0 <= position_index < len(accessors)
                or not 0 <= normal_index < len(accessors)
            ):
                raise ArtiverseVisualNormalizationError(
                    "publisher GLB triangle primitive lacks valid POSITION/NORMAL "
                    f"accessors: {path} mesh={mesh_index} primitive={primitive_index}"
                )
            position = accessors[position_index]
            normal = accessors[normal_index]
            if (
                not isinstance(position, dict)
                or not isinstance(normal, dict)
                or position.get("type") != "VEC3"
                or normal.get("type") != "VEC3"
                or isinstance(position.get("count"), bool)
                or not isinstance(position.get("count"), int)
                or position["count"] < 3
                or normal.get("count") != position["count"]
            ):
                raise ArtiverseVisualNormalizationError(
                    f"publisher GLB POSITION/NORMAL accessor contract is invalid: {path}"
                )
            position_storage = validate_stored_accessor(
                position_index,
                semantic="POSITION",
                allowed_component_types={5126},
            )
            normal_storage = validate_stored_accessor(
                normal_index,
                semantic="NORMAL",
                allowed_component_types={5120, 5122, 5126},
            )
            if position_storage["normalized"] or (
                normal_storage["component_type"] in {5120, 5122}
                and not normal_storage["normalized"]
            ) or (
                normal_storage["component_type"] == 5126
                and normal_storage["normalized"]
            ):
                raise ArtiverseVisualNormalizationError(
                    f"publisher GLB POSITION/NORMAL component normalization is invalid: {path}"
                )
            indices_index = primitive.get("indices")
            if indices_index is None:
                if position["count"] % 3:
                    raise ArtiverseVisualNormalizationError(
                        f"publisher GLB unindexed triangle count is invalid: {path}"
                    )
            elif (
                isinstance(indices_index, bool)
                or not isinstance(indices_index, int)
                or not 0 <= indices_index < len(accessors)
                or not isinstance(accessors[indices_index], dict)
                or accessors[indices_index].get("type") != "SCALAR"
                or isinstance(accessors[indices_index].get("count"), bool)
                or not isinstance(accessors[indices_index].get("count"), int)
                or accessors[indices_index]["count"] < 3
                or accessors[indices_index]["count"] % 3
            ):
                raise ArtiverseVisualNormalizationError(
                    f"publisher GLB triangle index accessor is invalid: {path}"
                )
            if indices_index is not None:
                indices_storage = validate_stored_accessor(
                    indices_index,
                    semantic="indices",
                    allowed_component_types={5121, 5123, 5125},
                )
                if indices_storage["normalized"]:
                    raise ArtiverseVisualNormalizationError(
                        f"publisher GLB index accessor cannot be normalized: {path}"
                    )
            primitive_count += 1
            normal_accessor_count += 1
    derived_document = copy.deepcopy(document)
    derived_document["buffers"][0]["uri"] = f"{part}.bin"
    preserved_document = copy.deepcopy(derived_document)
    del preserved_document["buffers"][0]["uri"]
    if preserved_document != document:
        raise ArtiverseVisualNormalizationError(
            f"external glTF derivation changed publisher document semantics: {path}"
        )
    gltf_payload = _canonical_json_bytes(derived_document)
    return {
        "triangle_primitive_count": primitive_count,
        "normal_accessor_count": normal_accessor_count,
        "embedded_image_count": embedded_image_count,
        "data_image_count": data_image_count,
        "publisher_document_sha256": hashlib.sha256(
            _canonical_json_bytes(document)
        ).hexdigest(),
        "preserved_semantics_sha256": hashlib.sha256(
            _canonical_json_bytes(preserved_document)
        ).hexdigest(),
        "derived_gltf_bytes": gltf_payload,
        "derived_gltf_sha256": hashlib.sha256(gltf_payload).hexdigest(),
        "derived_bin_bytes": binary_payload,
        "derived_bin_sha256": hashlib.sha256(binary_payload).hexdigest(),
    }


def _collision_inventory(root: ET.Element) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for link in (element for element in root.iter() if _local_name(element) == "link"):
        link_name = (link.get("name") or "").strip()
        for index, collision in enumerate(_direct_children(link, "collision")):
            detached = copy.deepcopy(collision)
            detached.tail = None
            payload = ET.tostring(detached, encoding="utf-8")
            records.append(
                {
                    "link_name": link_name,
                    "index": index,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return records


def _protected_sdf_sha256(root: ET.Element) -> str:
    """Bind every SDF field except the intentionally replaced visual groups."""

    def signature(element: ET.Element) -> tuple[Any, ...]:
        children = tuple(
            signature(child)
            for child in list(element)
            if _local_name(child) != "visual"
        )
        return (
            _local_name(element),
            tuple(sorted((str(key), str(value)) for key, value in element.attrib.items())),
            " ".join((element.text or "").split()),
            children,
        )

    return hashlib.sha256(_canonical_json_bytes(signature(root))).hexdigest()


def _parse_sdf(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    try:
        return ET.parse(path, parser=parser)
    except (ET.ParseError, OSError) as exc:
        raise ArtiverseVisualNormalizationError(
            f"cannot parse copied Artiverse SDF {path}: {exc}"
        ) from exc


def _atomic_replace_xml(path: Path, root: ET.Element, mode: int) -> None:
    payload = ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    ) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.visual-normalization.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _derived_resource_payloads(
    derivations: dict[str, dict[str, Any]],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    payloads: dict[str, bytes] = {}
    resources: list[dict[str, Any]] = []
    casefolded: set[str] = set()
    for part in sorted(derivations):
        derivation = derivations[part]
        gltf_name = f"{part}.gltf"
        bin_name = f"{part}.bin"
        for name in (gltf_name, bin_name):
            if name.casefold() in casefolded:
                raise ArtiverseVisualNormalizationError(
                    f"derived Artiverse resource has a case-fold collision: {name}"
                )
            casefolded.add(name.casefold())
        payloads[gltf_name] = derivation["derived_gltf_bytes"]
        payloads[bin_name] = derivation["derived_bin_bytes"]
        resources.append(
            {
                "part": part,
                "publisher_glb_sha256": derivation["publisher_glb_sha256"],
                "publisher_glb_size_bytes": derivation[
                    "publisher_glb_size_bytes"
                ],
                "publisher_document_sha256": derivation[
                    "publisher_document_sha256"
                ],
                "preserved_semantics_sha256": derivation[
                    "preserved_semantics_sha256"
                ],
                "derived_gltf": gltf_name,
                "derived_gltf_sha256": derivation["derived_gltf_sha256"],
                "derived_gltf_size_bytes": len(derivation["derived_gltf_bytes"]),
                "derived_bin": bin_name,
                "derived_bin_sha256": derivation["derived_bin_sha256"],
                "derived_bin_size_bytes": len(derivation["derived_bin_bytes"]),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "policy": POLICY,
        "resource_directory": DERIVED_RESOURCE_DIRECTORY,
        "resources": resources,
    }
    payloads[DERIVATION_MANIFEST] = _canonical_json_bytes(manifest)
    return payloads, manifest


def _validate_derived_directory(
    directory: Path, expected_payloads: dict[str, bytes]
) -> Path:
    try:
        resolved = _real_directory(directory, label="derived Artiverse resource directory")
        actual_files = _safe_tree_files(resolved)
    except ArtiverseVisualNormalizationError as exc:
        raise ArtiverseVisualNormalizationError(
            f"stale derived-resource mismatch at {directory}: {exc}"
        ) from exc
    actual_names = [path.relative_to(resolved).as_posix() for path in actual_files]
    expected_names = sorted(expected_payloads)
    if actual_names != expected_names:
        raise ArtiverseVisualNormalizationError(
            "stale derived-resource mismatch: exact inventory differs at "
            f"{directory}: expected={expected_names!r} actual={actual_names!r}"
        )
    for name in expected_names:
        path = _regular_file_within(
            resolved.joinpath(*PurePosixPath(name).parts),
            resolved,
            label=f"derived Artiverse resource {name}",
        )
        expected = expected_payloads[name]
        if path.stat().st_size != len(expected) or _sha256_file(path) != hashlib.sha256(
            expected
        ).hexdigest():
            raise ArtiverseVisualNormalizationError(
                f"stale derived-resource mismatch: content differs for {path}"
            )
    return resolved


def _publish_or_validate_derived_directory(
    sdf_parent: Path,
    expected_payloads: dict[str, bytes],
    *,
    sdf_was_normalized: bool,
) -> tuple[Path, str]:
    final = sdf_parent / DERIVED_RESOURCE_DIRECTORY
    transaction_prefix = f".{DERIVED_RESOURCE_DIRECTORY}."
    for sibling in sdf_parent.iterdir():
        if sibling.name.startswith(transaction_prefix) and sibling.name.endswith(".tmp"):
            raise ArtiverseVisualNormalizationError(
                f"stale Artiverse derived-resource transaction exists: {sibling}"
            )

    if final.exists() or _is_link_like(final):
        resolved = _validate_derived_directory(final, expected_payloads)
        return resolved, (
            "verified_existing" if sdf_was_normalized else "recovered_unreferenced"
        )

    temporary: Path | None = None
    for _ in range(100):
        candidate = sdf_parent / (
            f"{transaction_prefix}{secrets.token_hex(12)}.tmp"
        )
        try:
            candidate.mkdir(mode=0o755)
        except FileExistsError:
            continue
        temporary = candidate
        break
    if temporary is None:
        raise ArtiverseVisualNormalizationError(
            "cannot allocate a unique Artiverse derived-resource transaction"
        )
    try:
        for name in sorted(expected_payloads):
            relative = PurePosixPath(name)
            if len(relative.parts) != 1 or relative.name != name:
                raise ArtiverseVisualNormalizationError(
                    f"derived Artiverse resource name is unsafe: {name!r}"
                )
            path = temporary / name
            with path.open("xb") as stream:
                stream.write(expected_payloads[name])
                stream.flush()
                os.fsync(stream.fileno())
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        os.replace(temporary, final)
        if hasattr(os, "O_DIRECTORY"):
            parent_fd = os.open(sdf_parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    except Exception:
        if temporary.exists():
            try:
                for child in temporary.iterdir():
                    child.unlink(missing_ok=True)
                temporary.rmdir()
            except OSError:
                pass
        raise
    return _validate_derived_directory(final, expected_payloads), "published"


def normalize_copied_artiverse_visuals(
    sdf_path: Path, copied_tree_root: Path
) -> dict[str, Any]:
    """Atomically derive and use external glTF visuals in a copied Artiverse tree."""

    tree_root = _real_directory(copied_tree_root, label="copied Artiverse tree")
    sdf = _regular_file_within(sdf_path, tree_root, label="copied Artiverse SDF")
    transaction_prefix = f".{sdf.name}.visual-normalization."
    for sibling in sdf.parent.iterdir():
        if sibling.name.startswith(transaction_prefix) and sibling.name.endswith(".tmp"):
            raise ArtiverseVisualNormalizationError(
                f"stale Artiverse visual-normalization transaction exists: {sibling}"
            )
    tree_sha_before = sha256_safe_directory_tree(tree_root)
    sdf_sha_before = _sha256_file(sdf)
    sdf_mode = stat.S_IMODE(sdf.stat(follow_symlinks=False).st_mode)
    document = _parse_sdf(sdf)
    xml_root = document.getroot()
    collision_before = _collision_inventory(xml_root)
    protected_sdf_before = _protected_sdf_sha256(xml_root)

    link_names: set[str] = set()
    records: list[dict[str, Any]] = []
    derivations: dict[str, dict[str, Any]] = {}
    sdf_input_kinds: set[str] = set()
    rewritten_links = 0
    migrated_legacy_glb_links = 0
    already_normalized_links = 0
    deduplicated_visuals = 0
    removed_material_overrides = 0

    links = [element for element in xml_root.iter() if _local_name(element) == "link"]
    for link in links:
        link_name = (link.get("name") or "").strip()
        if not link_name or link_name in link_names:
            raise ArtiverseVisualNormalizationError(
                f"copied Artiverse SDF has an empty or duplicate link name: {link_name!r}"
            )
        link_names.add(link_name)
        visuals = _direct_children(link, "visual")
        if not visuals:
            continue

        parsed_visuals: list[dict[str, Any]] = []
        kinds: set[str] = set()
        for index, visual in enumerate(visuals):
            label = f"link {link_name!r} visual {index}"
            mesh, uri_element, raw_uri = _mesh_uri(visual, label=label)
            lower_path = urlsplit(raw_uri).path.lower()
            if lower_path.endswith(".obj"):
                relative = _safe_relative_uri(
                    raw_uri, directory="objs", suffix=".obj", label=label
                )
                stem = relative.stem.split("__", 1)[0]
                kind = "obj"
            elif lower_path.endswith(".glb"):
                relative = _safe_relative_uri(
                    raw_uri, directory="glbs", suffix=".glb", label=label
                )
                stem = relative.stem
                kind = "glb"
            elif lower_path.endswith(".gltf"):
                relative = _safe_relative_uri(
                    raw_uri,
                    directory=DERIVED_RESOURCE_DIRECTORY,
                    suffix=".gltf",
                    label=label,
                )
                stem = relative.stem
                kind = "gltf"
            else:
                raise ArtiverseVisualNormalizationError(
                    f"{label} is neither an audited OBJ, legacy GLB, nor derived glTF visual: {raw_uri}"
                )
            if not _SAFE_PART.fullmatch(stem):
                raise ArtiverseVisualNormalizationError(
                    f"{label} has an unsafe/ambiguous Artiverse part stem: {stem!r}"
                )
            referenced = _regular_file_within(
                tree_root.joinpath(*relative.parts),
                tree_root,
                label=f"{label} resource",
            )
            parsed_visuals.append(
                {
                    "visual": visual,
                    "uri_element": uri_element,
                    "raw_uri": raw_uri,
                    "relative": relative,
                    "resource": referenced,
                    "resource_sha256": _sha256_file(referenced),
                    "stem": stem,
                    "kind": kind,
                    "transform": _visual_transform(visual, mesh, label=label),
                    "structure": _visual_structure_signature(visual),
                }
            )
            kinds.add(kind)

        if len(kinds) != 1:
            raise ArtiverseVisualNormalizationError(
                f"link {link_name!r} mixes OBJ, legacy GLB, or derived glTF visual states"
            )
        stems = {item["stem"] for item in parsed_visuals}
        transforms = {
            json.dumps(item["transform"], sort_keys=True, separators=(",", ":"))
            for item in parsed_visuals
        }
        structures = {item["structure"] for item in parsed_visuals}
        if len(stems) != 1 or len(transforms) != 1 or len(structures) != 1:
            raise ArtiverseVisualNormalizationError(
                f"link {link_name!r} has ambiguous part stems, unequal pose/scale, "
                "or unequal non-material render structure"
            )
        part = next(iter(stems))
        publisher_relative = PurePosixPath("glbs") / f"{part}.glb"
        publisher_glb = _regular_file_within(
            tree_root.joinpath(*publisher_relative.parts),
            tree_root,
            label=f"link {link_name!r} publisher GLB",
        )
        publisher_glb_sha256 = _sha256_file(publisher_glb)
        publisher_glb_size_bytes = publisher_glb.stat().st_size
        derivation = _glb_external_derivation(publisher_glb, part=part)
        derivation.update(
            {
                "publisher_glb_sha256": publisher_glb_sha256,
                "publisher_glb_size_bytes": publisher_glb_size_bytes,
            }
        )
        previous_derivation = derivations.setdefault(part, derivation)
        if (
            previous_derivation["publisher_glb_sha256"] != publisher_glb_sha256
            or previous_derivation["derived_gltf_sha256"]
            != derivation["derived_gltf_sha256"]
            or previous_derivation["derived_bin_sha256"]
            != derivation["derived_bin_sha256"]
        ):
            raise ArtiverseVisualNormalizationError(
                f"part {part!r} maps to inconsistent publisher/derived resources"
            )
        publisher_uri = f"./{publisher_relative.as_posix()}"
        derived_relative = (
            PurePosixPath(DERIVED_RESOURCE_DIRECTORY) / f"{part}.gltf"
        )
        derived_uri = f"./{derived_relative.as_posix()}"
        kind = next(iter(kinds))
        sdf_input_kinds.add(kind)

        if kind == "obj":
            first = parsed_visuals[0]
            first["uri_element"].text = derived_uri
            # Publisher GLB primitives carry the complete authored multi-material
            # assignment.  Retaining one split OBJ's SDF override would flatten it.
            removed_material_overrides += sum(
                len(_direct_children(item["visual"], "material"))
                for item in parsed_visuals
            )
            for material in _direct_children(first["visual"], "material"):
                first["visual"].remove(material)
            for duplicate in parsed_visuals[1:]:
                link.remove(duplicate["visual"])
            rewritten_links += 1
            deduplicated_visuals += len(parsed_visuals) - 1
            state = "rewritten"
        elif kind == "glb":
            if (
                len(parsed_visuals) != 1
                or parsed_visuals[0]["raw_uri"] != publisher_uri
                or _direct_children(parsed_visuals[0]["visual"], "material")
            ):
                raise ArtiverseVisualNormalizationError(
                    f"link {link_name!r} has a noncanonical legacy GLB visual "
                    "or lingering material override"
                )
            parsed_visuals[0]["uri_element"].text = derived_uri
            migrated_legacy_glb_links += 1
            state = "migrated_legacy_glb"
        else:
            if (
                len(parsed_visuals) != 1
                or parsed_visuals[0]["raw_uri"] != derived_uri
                or _direct_children(parsed_visuals[0]["visual"], "material")
            ):
                raise ArtiverseVisualNormalizationError(
                    f"link {link_name!r} has a noncanonical derived glTF visual "
                    "or lingering material override"
                )
            already_normalized_links += 1
            state = "already_normalized"

        records.append(
            {
                "link_name": link_name,
                "state": state,
                "part": part,
                "source_obj_visual_uris": (
                    [item["raw_uri"] for item in parsed_visuals] if kind == "obj" else []
                ),
                "source_obj_sha256": (
                    [item["resource_sha256"] for item in parsed_visuals]
                    if kind == "obj"
                    else []
                ),
                "publisher_glb_uri": publisher_uri,
                "publisher_glb_size_bytes": publisher_glb_size_bytes,
                "publisher_glb_sha256": publisher_glb_sha256,
                "publisher_document_sha256": derivation[
                    "publisher_document_sha256"
                ],
                "preserved_semantics_sha256": derivation[
                    "preserved_semantics_sha256"
                ],
                "derived_gltf_uri": derived_uri,
                "derived_gltf_size_bytes": len(derivation["derived_gltf_bytes"]),
                "derived_gltf_sha256": derivation["derived_gltf_sha256"],
                "derived_bin_uri": (
                    f"./{DERIVED_RESOURCE_DIRECTORY}/{part}.bin"
                ),
                "derived_bin_size_bytes": len(derivation["derived_bin_bytes"]),
                "derived_bin_sha256": derivation["derived_bin_sha256"],
                "transform": parsed_visuals[0]["transform"],
                "triangle_primitive_count": derivation[
                    "triangle_primitive_count"
                ],
                "normal_accessor_count": derivation["normal_accessor_count"],
                "embedded_image_count": derivation["embedded_image_count"],
                "data_image_count": derivation["data_image_count"],
            }
        )

    if not records:
        raise ArtiverseVisualNormalizationError(
            "copied Artiverse SDF contains no visual links to normalize or verify"
        )

    if len(sdf_input_kinds) != 1:
        raise ArtiverseVisualNormalizationError(
            "copied Artiverse SDF mixes original, legacy, and normalized lifecycle states"
        )

    expected_payloads, derivation_manifest = _derived_resource_payloads(derivations)
    derived_directory, derived_publication_state = (
        _publish_or_validate_derived_directory(
            sdf.parent,
            expected_payloads,
            sdf_was_normalized=sdf_input_kinds == {"gltf"},
        )
    )
    derived_directory_sha256 = sha256_safe_directory_tree(derived_directory)

    if rewritten_links or migrated_legacy_glb_links:
        _atomic_replace_xml(sdf, xml_root, sdf_mode)
    reparsed = _parse_sdf(sdf).getroot()
    collision_after = _collision_inventory(reparsed)
    if collision_after != collision_before:
        raise ArtiverseVisualNormalizationError(
            "Artiverse visual normalization changed collision elements"
        )
    protected_sdf_after = _protected_sdf_sha256(reparsed)
    if protected_sdf_after != protected_sdf_before:
        raise ArtiverseVisualNormalizationError(
            "Artiverse visual normalization changed protected non-visual SDF content"
        )
    tree_sha_after = sha256_safe_directory_tree(tree_root)
    sdf_sha_after = _sha256_file(sdf)
    if not rewritten_links and not migrated_legacy_glb_links and (
        sdf_sha_after != sdf_sha_before or tree_sha_after != tree_sha_before
    ):
        raise ArtiverseVisualNormalizationError(
            "idempotent Artiverse visual verification unexpectedly changed copied bytes"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "policy": POLICY,
        "copied_sdf_path": str(sdf),
        "sdf_sha256_before": sdf_sha_before,
        "sdf_sha256_after": sdf_sha_after,
        "copied_tree_sha256_before": tree_sha_before,
        "copied_tree_sha256_after": tree_sha_after,
        "derived_resource_directory": str(derived_directory),
        "derived_resource_directory_sha256": derived_directory_sha256,
        "derived_publication_state": derived_publication_state,
        "derivation_manifest_sha256": hashlib.sha256(
            expected_payloads[DERIVATION_MANIFEST]
        ).hexdigest(),
        "derived_resource_count": len(expected_payloads) - 1,
        "derived_part_count": len(derivation_manifest["resources"]),
        "rewritten_link_count": rewritten_links,
        "migrated_legacy_glb_link_count": migrated_legacy_glb_links,
        "already_normalized_link_count": already_normalized_links,
        "deduplicated_visual_count": deduplicated_visuals,
        "removed_material_override_count": removed_material_overrides,
        "collision_element_count": len(collision_after),
        "protected_nonvisual_sdf_sha256": protected_sdf_after,
        "collision_elements_sha256": hashlib.sha256(
            json.dumps(collision_after, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "links": records,
    }
