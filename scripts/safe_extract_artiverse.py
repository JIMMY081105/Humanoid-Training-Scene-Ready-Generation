#!/usr/bin/env python3
"""Safely extract the pinned official Artiverse dataset chunk release.

The publisher's chunk packer is useful release metadata, but its unpack command
uses :mod:`tarfile`'s legacy ``extractall`` behavior.  This extractor instead
authenticates the complete pinned release before creating a staging directory,
accepts only ordinary directories and regular files beneath declared model
roots, writes without following links, and publishes only the completed
``data`` tree.

No third-party packages are required.  The publisher manifest, packer, and
chunk archives are read-only inputs and remain in place.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import stat
import struct
import tarfile
import tempfile
import unicodedata
import uuid

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable


EXTRACTOR_VERSION = "1.2.0"
RECEIPT_FILENAME = "artiverse_safe_extraction_receipt.json"
MANIFEST_FORMAT = "artiverse-data-tar-gz-chunks-v1"
PINNED_MANIFEST_SHA256 = (
    "8fa6468254a1f74c58f0c25699598bf88f622fabdaf74f0cd9268ee5663c5586"
)
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_PATH_BYTES = 4096
MAX_COMPONENT_BYTES = 255
MAX_SINGLE_FILE_BYTES = 16 * 1024**3
COPY_BUFFER_BYTES = 1024 * 1024
ROOT_HASH_ALGORITHM = "sha256-u64be-length-prefixed-utf8-sorted-v1"
TREE_HASH_ALGORITHM = "sha256-sorted-utf8-path-directory-and-regular-file-content-v1"
DERIVED_SDF_FILENAME = "scenesmith_artiverse.sdf"
DERIVED_SDF_TEMP_FILENAME = f".{DERIVED_SDF_FILENAME}.tmp"
DERIVED_SDF_FILENAMES = frozenset(
    {DERIVED_SDF_FILENAME, DERIVED_SDF_TEMP_FILENAME}
)


class ArtiverseExtractionError(RuntimeError):
    """The release failed authentication or safe extraction validation."""


@dataclass(frozen=True)
class ChunkSpec:
    archive: str
    sha256: str
    archive_bytes: int
    model_count: int
    file_count: int
    input_bytes: int
    estimated_tar_bytes: int


@dataclass(frozen=True)
class ReleaseSpec:
    manifest_sha256: str
    chunks: tuple[ChunkSpec, ...]
    model_count: int
    file_count: int
    input_bytes: int
    estimated_tar_bytes: int


@dataclass(frozen=True)
class PinnedCasefoldMember:
    """One authenticated regular file in an approved case-fold collision."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class PinnedCasefoldCollision:
    """A release-, archive-, root-, and content-bound filename exception."""

    manifest_sha256: str
    archive: str
    archive_sha256: str
    model_root: str
    members: tuple[PinnedCasefoldMember, PinnedCasefoldMember]


PINNED_RELEASE = ReleaseSpec(
    manifest_sha256=PINNED_MANIFEST_SHA256,
    chunks=(
        ChunkSpec(
            archive="artiverse_data-00001-of-00002.tar.gz",
            sha256=(
                "695d2d602faafab922ce66359ea104d81505f5b0fdee8f461d8905f0ccb4ef3b"
            ),
            archive_bytes=38_163_580_631,
            model_count=1_561,
            file_count=219_064,
            input_bytes=47_866_437_602,
            estimated_tar_bytes=48_261_014_528,
        ),
        ChunkSpec(
            archive="artiverse_data-00002-of-00002.tar.gz",
            sha256=(
                "56dffa50f1c8c20d3b1eef626046805a6c7cd997141e8ab5fac9ebdae8ffab81"
            ),
            archive_bytes=27_170_560_473,
            model_count=1_983,
            file_count=312_873,
            input_bytes=39_126_315_288,
            estimated_tar_bytes=39_689_876_992,
        ),
    ),
    model_count=3_544,
    file_count=531_937,
    input_bytes=86_992_752_890,
    estimated_tar_bytes=87_950_891_520,
)


# An exhaustive authenticated scan of the pinned release found exactly these
# two collision groups in chunk 2 (four distinct regular files) and no other
# case-fold collisions.  They are not byte-identical, so both official
# spellings must be preserved on the case-sensitive production filesystem.
# Never add a path-only exception: every entry must pin the exact release,
# archive, model root, member sizes, and member content hashes.
_OFFICIAL_TOASTER_ROOT = "data/toaster/objaverse/9662f5cc6cf6456292613d8c5075fcfb"
_OFFICIAL_TOASTER_OBJS = _OFFICIAL_TOASTER_ROOT + "/urdf_w_collider/objs"
PINNED_CASEFOLD_COLLISIONS: tuple[PinnedCasefoldCollision, ...] = (
    PinnedCasefoldCollision(
        manifest_sha256=PINNED_MANIFEST_SHA256,
        archive="artiverse_data-00002-of-00002.tar.gz",
        archive_sha256=(
            "56dffa50f1c8c20d3b1eef626046805a6c7cd997141e8ab5fac9ebdae8ffab81"
        ),
        model_root=_OFFICIAL_TOASTER_ROOT,
        members=(
            PinnedCasefoldMember(
                path=_OFFICIAL_TOASTER_OBJS + "/1_base__Material.mtl",
                size=176,
                sha256=(
                    "3ecce3aba2a39f02f63ba0fedde49a663464411625fd24ad0abb74d6c145ef0a"
                ),
            ),
            PinnedCasefoldMember(
                path=_OFFICIAL_TOASTER_OBJS + "/1_base__material.mtl",
                size=174,
                sha256=(
                    "cf70acff54721bb8fa667fabf88a34d0a506980a8fa1a67113dffece9ccc1fec"
                ),
            ),
        ),
    ),
    PinnedCasefoldCollision(
        manifest_sha256=PINNED_MANIFEST_SHA256,
        archive="artiverse_data-00002-of-00002.tar.gz",
        archive_sha256=(
            "56dffa50f1c8c20d3b1eef626046805a6c7cd997141e8ab5fac9ebdae8ffab81"
        ),
        model_root=_OFFICIAL_TOASTER_ROOT,
        members=(
            PinnedCasefoldMember(
                path=_OFFICIAL_TOASTER_OBJS + "/1_base__Material.obj",
                size=27_919,
                sha256=(
                    "643bb5505d1d281a5996220f3d0c38b782315d9020a1a89a12760e94c3940273"
                ),
            ),
            PinnedCasefoldMember(
                path=_OFFICIAL_TOASTER_OBJS + "/1_base__material.obj",
                size=21_915,
                sha256=(
                    "e6aa38146df6728147fd20c6937a0c4858e9b0e51d7cb92f16f704437aa0b866"
                ),
            ),
        ),
    ),
)


MANIFEST_KEYS = {
    "format",
    "created_utc",
    "data_dir",
    "max_chunk_gb",
    "chunk_count",
    "model_count",
    "file_count",
    "input_bytes",
    "estimated_tar_bytes",
    "chunks",
}
CHUNK_KEYS = {
    "index",
    "archive",
    "sha256",
    "archive_bytes",
    "model_count",
    "file_count",
    "input_bytes",
    "estimated_tar_bytes",
    "roots",
}
RECEIPT_KEYS = {
    "schema_version",
    "status",
    "manifest",
    "archives",
    "declared_roots",
    "validated_aggregate",
    "data_tree_inventory",
    "safe_extractor",
}
RECEIPT_MANIFEST_KEYS = {"path", "sha256", "format"}
RECEIPT_ARCHIVE_KEYS = {
    "archive",
    "archive_bytes",
    "sha256",
    "validated_member_count",
    "validated_directory_count",
    "validated_regular_file_count",
    "validated_uncompressed_bytes",
}
RECEIPT_ROOT_KEYS = {"count", "sha256", "hash_algorithm"}
RECEIPT_AGGREGATE_KEYS = {
    "member_count",
    "directory_count",
    "regular_file_count",
    "uncompressed_bytes",
}
RECEIPT_INVENTORY_KEYS = {
    "directory_count",
    "regular_file_count",
    "sha256",
    "hash_algorithm",
}
RECEIPT_EXTRACTOR_KEYS = {"version", "sha256", "filename"}


@dataclass
class ValidatedChunk:
    spec: ChunkSpec
    roots: tuple[str, ...]


@dataclass
class VerifiedArchive:
    chunk: ValidatedChunk
    path: Path
    file: BinaryIO
    device: int
    inode: int


@dataclass
class ChunkResult:
    archive: str
    member_count: int = 0
    directory_count: int = 0
    regular_file_count: int = 0
    uncompressed_bytes: int = 0
    roots_seen: set[str] = field(default_factory=set)


@dataclass
class TreeState:
    casefold_allowances: tuple[PinnedCasefoldCollision, ...] = ()
    verified_casefold_members: set[str] = field(default_factory=set)
    explicit_members: set[str] = field(default_factory=set)
    spelling_by_casefold: dict[str, str] = field(default_factory=dict)
    directories: set[str] = field(default_factory=set)
    files: dict[str, tuple[int, bytes]] = field(default_factory=dict)


class _AuthenticatedArchiveReader:
    """Non-closing proxy that hashes the compressed bytes consumed by tarfile."""

    def __init__(self, source: BinaryIO):
        self.source = source
        self.byte_count = 0
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        block = self.source.read(size)
        if block:
            self.byte_count += len(block)
            self.digest.update(block)
        return block

    def readinto(self, buffer: bytearray | memoryview) -> int | None:
        count = self.source.readinto(buffer)
        if count:
            self.byte_count += count
            self.digest.update(memoryview(buffer)[:count])
        return count

    def drain(self) -> None:
        for _block in iter(lambda: self.read(COPY_BUFFER_BYTES), b""):
            pass

    def validate(self, spec: ChunkSpec) -> None:
        digest = self.digest.hexdigest()
        if self.byte_count != spec.archive_bytes or digest != spec.sha256:
            raise ArtiverseExtractionError(
                "Compressed archive bytes consumed during extraction failed "
                f"authentication for {spec.archive}: bytes={self.byte_count} "
                f"sha256={digest}"
            )

    def authenticate(self, spec: ChunkSpec) -> None:
        # Tar streams may stop after the end-of-archive blocks while gzip still
        # has a trailer or buffered tail. Hash every remaining compressed byte
        # through this same proxy before accepting the extraction.
        self.drain()
        self.validate(spec)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(COPY_BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtiverseExtractionError(f"Duplicate JSON key in manifest: {key!r}")
        result[key] = value
    return result


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtiverseExtractionError(
            f"{label} schema mismatch (missing={missing}, extra={extra})"
        )


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtiverseExtractionError(
            f"{label} must be an integer of at least {minimum}"
        )
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtiverseExtractionError(f"{label} must be a non-empty string")
    return value


def _validate_sha256(value: Any, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ArtiverseExtractionError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def _validate_component(component: str, label: str) -> None:
    if not component or component in {".", ".."}:
        raise ArtiverseExtractionError(f"{label} contains an empty/dot path component")
    if unicodedata.normalize("NFC", component) != component:
        raise ArtiverseExtractionError(f"{label} is not NFC-normalized")
    if any(ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF for character in component):
        raise ArtiverseExtractionError(f"{label} contains a control/surrogate character")
    if len(component.encode("utf-8")) > MAX_COMPONENT_BYTES:
        raise ArtiverseExtractionError(f"{label} has an overlong path component")


def _normalize_posix_path(name: str, label: str, *, directory: bool) -> tuple[str, ...]:
    if not isinstance(name, str) or not name:
        raise ArtiverseExtractionError(f"{label} has an empty path")
    if "\x00" in name:
        raise ArtiverseExtractionError(f"{label} contains NUL")
    if "\\" in name:
        raise ArtiverseExtractionError(f"{label} contains a backslash")
    if directory and name.endswith("/"):
        name = name[:-1]
    if not name or name.endswith("/"):
        raise ArtiverseExtractionError(f"{label} has a non-canonical trailing slash")
    path = PurePosixPath(name)
    if path.is_absolute() or name.startswith("/"):
        raise ArtiverseExtractionError(f"{label} is absolute")
    raw_parts = name.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ArtiverseExtractionError(f"{label} contains empty/dot/dotdot components")
    if raw_parts[0].endswith(":"):
        raise ArtiverseExtractionError(f"{label} contains a drive-qualified path")
    for component in raw_parts:
        _validate_component(component, label)
    canonical = "/".join(raw_parts)
    if len(canonical.encode("utf-8")) > MAX_PATH_BYTES:
        raise ArtiverseExtractionError(f"{label} exceeds the maximum path length")
    return tuple(raw_parts)


def _validate_root(root: Any, label: str) -> str:
    text = _string(root, label)
    parts = _normalize_posix_path(text, label, directory=False)
    if len(parts) != 4 or parts[0] != "data":
        raise ArtiverseExtractionError(
            f"{label} must be canonical data/<category>/<source>/<model_id>"
        )
    return "/".join(parts)


def _manifest_bytes(path: Path) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ArtiverseExtractionError(f"Cannot stat manifest {path}: {exc}") from exc
    if size > MAX_MANIFEST_BYTES:
        raise ArtiverseExtractionError("Manifest exceeds the safe size limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ArtiverseExtractionError(f"Cannot read manifest {path}: {exc}") from exc


def _validate_manifest(
    manifest_payload: bytes, expected: ReleaseSpec
) -> tuple[dict[str, Any], tuple[ValidatedChunk, ...], tuple[str, ...]]:
    digest = _sha256_bytes(manifest_payload)
    if digest != expected.manifest_sha256:
        raise ArtiverseExtractionError(
            f"Manifest SHA-256 mismatch: expected {expected.manifest_sha256}, got {digest}"
        )
    try:
        manifest = json.loads(
            manifest_payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtiverseExtractionError(f"Invalid UTF-8 JSON manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtiverseExtractionError("Manifest root must be a JSON object")
    _exact_keys(manifest, MANIFEST_KEYS, "Manifest")
    if manifest["format"] != MANIFEST_FORMAT:
        raise ArtiverseExtractionError("Unexpected Artiverse manifest format")
    _string(manifest["created_utc"], "created_utc")
    if manifest["data_dir"] != "data":
        raise ArtiverseExtractionError("Manifest data_dir must be exactly 'data'")
    max_chunk_gb = manifest["max_chunk_gb"]
    if isinstance(max_chunk_gb, bool) or not isinstance(max_chunk_gb, (int, float)) or max_chunk_gb <= 0:
        raise ArtiverseExtractionError("max_chunk_gb must be a positive number")

    aggregate_fields = {
        "chunk_count": len(expected.chunks),
        "model_count": expected.model_count,
        "file_count": expected.file_count,
        "input_bytes": expected.input_bytes,
        "estimated_tar_bytes": expected.estimated_tar_bytes,
    }
    for field_name, expected_value in aggregate_fields.items():
        actual = _integer(manifest[field_name], field_name)
        if actual != expected_value:
            raise ArtiverseExtractionError(
                f"Manifest {field_name} mismatch: expected {expected_value}, got {actual}"
            )
    chunks_value = manifest["chunks"]
    if not isinstance(chunks_value, list) or len(chunks_value) != len(expected.chunks):
        raise ArtiverseExtractionError("Manifest chunks do not match the pinned release")

    validated: list[ValidatedChunk] = []
    all_roots: list[str] = []
    root_spellings: dict[str, str] = {}
    for index, (entry, chunk_spec) in enumerate(zip(chunks_value, expected.chunks), start=1):
        label = f"chunks[{index - 1}]"
        if not isinstance(entry, dict):
            raise ArtiverseExtractionError(f"{label} must be an object")
        _exact_keys(entry, CHUNK_KEYS, label)
        actual_index = _integer(entry["index"], f"{label}.index", minimum=1)
        if actual_index != index:
            raise ArtiverseExtractionError(
                f"{label}.index mismatch: expected {index}, got {actual_index}"
            )
        scalar_expected: dict[str, Any] = {
            "archive": chunk_spec.archive,
            "sha256": chunk_spec.sha256,
            "archive_bytes": chunk_spec.archive_bytes,
            "model_count": chunk_spec.model_count,
            "file_count": chunk_spec.file_count,
            "input_bytes": chunk_spec.input_bytes,
            "estimated_tar_bytes": chunk_spec.estimated_tar_bytes,
        }
        for field_name, expected_value in scalar_expected.items():
            actual = entry[field_name]
            if field_name == "archive":
                actual = _string(actual, f"{label}.{field_name}")
            elif field_name == "sha256":
                actual = _validate_sha256(actual, f"{label}.{field_name}")
            else:
                actual = _integer(actual, f"{label}.{field_name}")
            if actual != expected_value:
                raise ArtiverseExtractionError(
                    f"{label}.{field_name} mismatch: expected {expected_value!r}, got {actual!r}"
                )
        archive_parts = _normalize_posix_path(
            chunk_spec.archive, f"{label}.archive", directory=False
        )
        if len(archive_parts) != 1:
            raise ArtiverseExtractionError(f"{label}.archive must be a basename")
        roots_value = entry["roots"]
        if not isinstance(roots_value, list) or len(roots_value) != chunk_spec.model_count:
            raise ArtiverseExtractionError(
                f"{label}.roots must contain exactly {chunk_spec.model_count} roots"
            )
        roots: list[str] = []
        for root_index, root_value in enumerate(roots_value):
            root = _validate_root(root_value, f"{label}.roots[{root_index}]")
            folded = root.casefold()
            previous = root_spellings.get(folded)
            if previous is not None:
                collision = "duplicate" if previous == root else "case-fold collision"
                raise ArtiverseExtractionError(
                    f"Declared model root {collision}: {previous!r} and {root!r}"
                )
            root_spellings[folded] = root
            roots.append(root)
            all_roots.append(root)
        validated.append(ValidatedChunk(spec=chunk_spec, roots=tuple(roots)))

    if len(all_roots) != expected.model_count:
        raise ArtiverseExtractionError("Declared root aggregate count mismatch")
    sum_checks = {
        "model_count": sum(chunk.spec.model_count for chunk in validated),
        "file_count": sum(chunk.spec.file_count for chunk in validated),
        "input_bytes": sum(chunk.spec.input_bytes for chunk in validated),
        "estimated_tar_bytes": sum(
            chunk.spec.estimated_tar_bytes for chunk in validated
        ),
    }
    for field_name, actual in sum_checks.items():
        if actual != getattr(expected, field_name):
            raise ArtiverseExtractionError(f"Pinned {field_name} aggregate is inconsistent")
    return manifest, tuple(validated), tuple(all_roots)


def _validate_casefold_allowance_shape(
    allowance: PinnedCasefoldCollision,
) -> None:
    """Validate the code-pinned exception itself before it can weaken a guard."""

    if not isinstance(allowance, PinnedCasefoldCollision):
        raise ArtiverseExtractionError("Invalid pinned case-fold collision record")
    _validate_sha256(
        allowance.manifest_sha256,
        "Pinned case-fold collision manifest_sha256",
    )
    archive_parts = _normalize_posix_path(
        allowance.archive,
        "Pinned case-fold collision archive",
        directory=False,
    )
    if len(archive_parts) != 1:
        raise ArtiverseExtractionError(
            "Pinned case-fold collision archive must be a basename"
        )
    _validate_sha256(
        allowance.archive_sha256,
        "Pinned case-fold collision archive_sha256",
    )
    model_root = _validate_root(
        allowance.model_root,
        "Pinned case-fold collision model_root",
    )
    if (
        not isinstance(allowance.members, tuple)
        or len(allowance.members) != 2
        or any(
            not isinstance(member, PinnedCasefoldMember)
            for member in allowance.members
        )
    ):
        raise ArtiverseExtractionError(
            "Pinned case-fold collision must contain exactly two member records"
        )
    normalized_paths: list[str] = []
    member_parts: list[tuple[str, ...]] = []
    for member in allowance.members:
        parts = _normalize_posix_path(
            member.path,
            "Pinned case-fold collision member path",
            directory=False,
        )
        normalized = "/".join(parts)
        if len(parts) < 5 or "/".join(parts[:4]) != model_root:
            raise ArtiverseExtractionError(
                "Pinned case-fold collision member is outside its exact model root"
            )
        if _is_explicit_scenesmith_derived_file(parts):
            raise ArtiverseExtractionError(
                "Pinned case-fold collision cannot exempt a derived-SDF path"
            )
        if (
            isinstance(member.size, bool)
            or not isinstance(member.size, int)
            or member.size < 0
            or member.size > MAX_SINGLE_FILE_BYTES
        ):
            raise ArtiverseExtractionError(
                "Pinned case-fold collision member size is outside safe limits"
            )
        _validate_sha256(
            member.sha256,
            "Pinned case-fold collision member sha256",
        )
        normalized_paths.append(normalized)
        member_parts.append(parts)
    first, second = normalized_paths
    if first == second or first.casefold() != second.casefold():
        raise ArtiverseExtractionError(
            "Pinned case-fold collision paths must be distinct exact spellings "
            "with one identical case-folded path"
        )
    if member_parts[0][:-1] != member_parts[1][:-1]:
        raise ArtiverseExtractionError(
            "Pinned case-fold collision may differ only in its regular-file basename"
        )


def _casefold_allowances_for_release(
    expected: ReleaseSpec,
    chunks: tuple[ValidatedChunk, ...],
) -> tuple[PinnedCasefoldCollision, ...]:
    """Select only exceptions cryptographically bound to this exact release."""

    selected: list[PinnedCasefoldCollision] = []
    pair_keys: set[tuple[str, str]] = set()
    folded_keys: set[str] = set()
    member_paths: set[str] = set()
    chunks_by_archive = {chunk.spec.archive: chunk for chunk in chunks}
    for allowance in PINNED_CASEFOLD_COLLISIONS:
        _validate_casefold_allowance_shape(allowance)
        if allowance.manifest_sha256 != expected.manifest_sha256:
            continue
        chunk = chunks_by_archive.get(allowance.archive)
        if chunk is None:
            raise ArtiverseExtractionError(
                "Pinned case-fold collision names an archive outside its release"
            )
        if chunk.spec.sha256 != allowance.archive_sha256:
            raise ArtiverseExtractionError(
                "Pinned case-fold collision archive SHA-256 does not match its release"
            )
        if allowance.model_root not in chunk.roots:
            raise ArtiverseExtractionError(
                "Pinned case-fold collision model root is not declared by its archive"
            )
        paths = tuple(member.path for member in allowance.members)
        pair_key = tuple(sorted(paths))
        folded_key = paths[0].casefold()
        if (
            pair_key in pair_keys
            or folded_key in folded_keys
            or any(path in member_paths for path in paths)
        ):
            raise ArtiverseExtractionError(
                "Overlapping or duplicate pinned case-fold collision records"
            )
        pair_keys.add(pair_key)
        folded_keys.add(folded_key)
        member_paths.update(paths)
        selected.append(allowance)
    return tuple(selected)


def _is_reparse_point(file_stat: os.stat_result) -> bool:
    attributes = getattr(file_stat, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def _require_regular_unlinked_file(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ArtiverseExtractionError(f"Cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or _is_reparse_point(info):
        raise ArtiverseExtractionError(f"{label} must be a regular, non-link file: {path}")
    return info


def _require_real_directory(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ArtiverseExtractionError(f"Cannot stat {label} {path}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode) or _is_reparse_point(info):
        raise ArtiverseExtractionError(f"{label} must be a real directory: {path}")
    return info


def _within(path: Path, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ArtiverseExtractionError(f"{label} is outside the repository root") from exc
    return resolved


def _verify_archives(
    chunks_dir: Path,
    chunks: tuple[ValidatedChunk, ...],
    stack: contextlib.ExitStack,
) -> tuple[VerifiedArchive, ...]:
    verified: list[VerifiedArchive] = []
    for chunk in chunks:
        archive_path = chunks_dir / chunk.spec.archive
        lexical = archive_path.absolute()
        try:
            lexical.relative_to(chunks_dir.absolute())
        except ValueError as exc:  # defense in depth after basename validation
            raise ArtiverseExtractionError("Archive path escapes chunks directory") from exc
        before = _require_regular_unlinked_file(archive_path, "Chunk archive")
        if before.st_size != chunk.spec.archive_bytes:
            raise ArtiverseExtractionError(
                f"Archive size mismatch for {chunk.spec.archive}: expected "
                f"{chunk.spec.archive_bytes}, got {before.st_size}"
            )
        try:
            handle = stack.enter_context(archive_path.open("rb"))
        except OSError as exc:
            raise ArtiverseExtractionError(
                f"Cannot open chunk archive {archive_path}: {exc}"
            ) from exc
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ArtiverseExtractionError(f"Opened archive is not regular: {archive_path}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ArtiverseExtractionError(f"Chunk archive changed while opening: {archive_path}")
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(COPY_BUFFER_BYTES), b""):
            digest.update(block)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != chunk.spec.sha256:
            raise ArtiverseExtractionError(
                f"Archive SHA-256 mismatch for {chunk.spec.archive}: expected "
                f"{chunk.spec.sha256}, got {actual_sha256}"
            )
        handle.seek(0)
        verified.append(
            VerifiedArchive(
                chunk=chunk,
                path=archive_path,
                file=handle,
                device=opened.st_dev,
                inode=opened.st_ino,
            )
        )
    return tuple(verified)


def _casefold_allowance_for_pair(
    state: TreeState,
    first_path: str,
    second_path: str,
) -> PinnedCasefoldCollision | None:
    pair = {first_path, second_path}
    for allowance in state.casefold_allowances:
        if pair == {member.path for member in allowance.members}:
            return allowance
    return None


def _casefold_member_record(
    state: TreeState,
    normalized: str,
) -> tuple[PinnedCasefoldCollision, PinnedCasefoldMember] | None:
    for allowance in state.casefold_allowances:
        for member in allowance.members:
            if member.path == normalized:
                return allowance, member
    return None


def _verify_casefold_member(
    state: TreeState,
    normalized: str,
    *,
    archive: str | None,
    size: int,
    content_digest: bytes,
) -> None:
    pinned = _casefold_member_record(state, normalized)
    if pinned is None:
        return
    allowance, member = pinned
    if archive is not None and archive != allowance.archive:
        raise ArtiverseExtractionError(
            "Pinned case-fold collision member appeared in the wrong archive: "
            f"{normalized!r}"
        )
    if size != member.size or content_digest.hex() != member.sha256:
        raise ArtiverseExtractionError(
            "Pinned case-fold collision member content mismatch: "
            f"{normalized!r}"
        )
    if normalized in state.verified_casefold_members:
        raise ArtiverseExtractionError(
            f"Pinned case-fold collision member was verified twice: {normalized!r}"
        )
    state.verified_casefold_members.add(normalized)


def _require_complete_casefold_allowances(state: TreeState) -> None:
    expected = {
        member.path
        for allowance in state.casefold_allowances
        for member in allowance.members
    }
    if state.verified_casefold_members != expected:
        missing = sorted(expected - state.verified_casefold_members)
        extra = sorted(state.verified_casefold_members - expected)
        raise ArtiverseExtractionError(
            "Pinned case-fold collision membership mismatch "
            f"(missing={missing}, extra={extra})"
        )


def _register_path_spelling(
    state: TreeState,
    parts: tuple[str, ...],
    *,
    archive: str | None = None,
    is_directory: bool = False,
) -> None:
    for length in range(1, len(parts) + 1):
        prefix = "/".join(parts[:length])
        folded = prefix.casefold()
        previous = state.spelling_by_casefold.get(folded)
        if previous is not None and previous != prefix:
            allowance = None
            if length == len(parts) and not is_directory:
                allowance = _casefold_allowance_for_pair(state, previous, prefix)
            if allowance is not None:
                if archive is not None and archive != allowance.archive:
                    raise ArtiverseExtractionError(
                        "Pinned case-fold collision pair appeared in the wrong "
                        f"archive: {previous!r} and {prefix!r}"
                    )
                if previous not in state.files:
                    raise ArtiverseExtractionError(
                        "Pinned case-fold collision is ambiguous with a directory: "
                        f"{previous!r} and {prefix!r}"
                    )
                # Keep the first spelling in the fold map.  A third spelling is
                # not one of the exact two-member set and will still fail.
                continue
            raise ArtiverseExtractionError(
                f"Case-fold path collision: {previous!r} and {prefix!r}"
            )
        state.spelling_by_casefold[folded] = prefix


def _register_member(
    state: TreeState,
    parts: tuple[str, ...],
    *,
    is_directory: bool,
    archive: str | None = None,
) -> str:
    normalized = "/".join(parts)
    if normalized in state.explicit_members:
        raise ArtiverseExtractionError(f"Duplicate normalized tar member: {normalized!r}")
    state.explicit_members.add(normalized)
    _register_path_spelling(
        state,
        parts,
        archive=archive,
        is_directory=is_directory,
    )
    for length in range(1, len(parts)):
        parent = "/".join(parts[:length])
        if parent in state.files:
            raise ArtiverseExtractionError(
                f"Tar member descends through a regular file: {parent!r}"
            )
        state.directories.add(parent)
    if is_directory:
        if normalized in state.files:
            raise ArtiverseExtractionError(f"File/directory collision: {normalized!r}")
        state.directories.add(normalized)
    elif normalized in state.directories:
        raise ArtiverseExtractionError(f"Directory/file collision: {normalized!r}")
    return normalized


def _secure_dir_fd_supported() -> bool:
    return (
        hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
        and os.open in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
    )


def _open_directory_chain(root_fd: int, parts: Iterable[str]) -> int:
    current = os.dup(root_fd)
    try:
        for component in parts:
            try:
                os.mkdir(component, mode=0o755, dir_fd=current)
            except FileExistsError:
                pass
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            next_fd = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        return current
    except Exception:
        os.close(current)
        raise


def _copy_stream(source: BinaryIO, destination: BinaryIO, size: int) -> bytes:
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        block = source.read(min(COPY_BUFFER_BYTES, remaining))
        if not block:
            raise ArtiverseExtractionError("Unexpected end of a regular tar member")
        destination.write(block)
        digest.update(block)
        remaining -= len(block)
    if source.read(1):
        raise ArtiverseExtractionError("Tar member yielded more data than its declared size")
    return digest.digest()


def _write_member_posix(
    staging_root: Path, parts: tuple[str, ...], source: BinaryIO, size: int
) -> bytes:
    root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(staging_root, root_flags)
    try:
        parent_fd = _open_directory_chain(root_fd, parts[:-1])
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            file_fd = os.open(parts[-1], flags, 0o644, dir_fd=parent_fd)
            with os.fdopen(file_fd, "wb") as destination:
                return _copy_stream(source, destination, size)
        finally:
            os.close(parent_fd)
    finally:
        os.close(root_fd)


def _ensure_directory_posix(staging_root: Path, parts: tuple[str, ...]) -> None:
    root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(staging_root, root_flags)
    try:
        directory_fd = _open_directory_chain(root_fd, parts)
        os.close(directory_fd)
    finally:
        os.close(root_fd)


def _ensure_directory_fallback(staging_root: Path, parts: tuple[str, ...]) -> None:
    current = staging_root
    for component in parts:
        current = current / component
        try:
            current.mkdir(mode=0o755)
        except FileExistsError:
            info = current.lstat()
            if not stat.S_ISDIR(info.st_mode) or _is_reparse_point(info):
                raise ArtiverseExtractionError(
                    f"Extraction path component is not a real directory: {current}"
                )


def _write_member_fallback(
    staging_root: Path, parts: tuple[str, ...], source: BinaryIO, size: int
) -> bytes:
    _ensure_directory_fallback(staging_root, parts[:-1])
    destination_path = staging_root.joinpath(*parts)
    try:
        with destination_path.open("xb") as destination:
            return _copy_stream(source, destination, size)
    except FileExistsError as exc:
        raise ArtiverseExtractionError(
            f"Extraction destination already exists: {destination_path}"
        ) from exc


def _ensure_directory(staging_root: Path, parts: tuple[str, ...]) -> None:
    try:
        if _secure_dir_fd_supported():
            _ensure_directory_posix(staging_root, parts)
        else:
            _ensure_directory_fallback(staging_root, parts)
    except (OSError, ValueError) as exc:
        raise ArtiverseExtractionError(
            f"Could not safely create directory {'/'.join(parts)!r}: {exc}"
        ) from exc


def _write_regular_member(
    staging_root: Path, parts: tuple[str, ...], source: BinaryIO, size: int
) -> bytes:
    try:
        if _secure_dir_fd_supported():
            return _write_member_posix(staging_root, parts, source, size)
        return _write_member_fallback(staging_root, parts, source, size)
    except ArtiverseExtractionError:
        raise
    except (OSError, ValueError) as exc:
        raise ArtiverseExtractionError(
            f"Could not safely write regular file {'/'.join(parts)!r}: {exc}"
        ) from exc


def _reject_special_member(member: tarfile.TarInfo) -> None:
    sparse_headers = any(
        key.startswith("GNU.sparse") or key in {"SCHILY.realsize", "GNU.sparse.realsize"}
        for key in member.pax_headers
    )
    is_sparse = (
        member.type == tarfile.GNUTYPE_SPARSE
        or bool(getattr(member, "issparse", lambda: False)())
        or sparse_headers
    )
    if is_sparse:
        raise ArtiverseExtractionError(f"Sparse tar member is forbidden: {member.name!r}")
    if not (member.isdir() or member.isreg()):
        raise ArtiverseExtractionError(
            f"Non-directory/non-regular tar member is forbidden: {member.name!r}"
        )


def _extract_archive(
    archive: VerifiedArchive,
    staging_root: Path,
    state: TreeState,
) -> ChunkResult:
    spec = archive.chunk.spec
    declared_roots = set(archive.chunk.roots)
    result = ChunkResult(archive=spec.archive)
    max_members = spec.file_count * 4 + spec.model_count * 64
    archive.file.seek(0)
    authenticated_stream = _AuthenticatedArchiveReader(archive.file)
    try:
        with tarfile.open(fileobj=authenticated_stream, mode="r|gz") as tar:
            for member in tar:
                result.member_count += 1
                if result.member_count > max_members:
                    raise ArtiverseExtractionError(
                        f"Archive {spec.archive} exceeds the safe member-count limit"
                    )
                _reject_special_member(member)
                parts = _normalize_posix_path(
                    member.name,
                    f"Tar member in {spec.archive}",
                    directory=member.isdir(),
                )
                if _is_explicit_scenesmith_derived_file(parts):
                    raise ArtiverseExtractionError(
                        "Publisher archive uses a reserved SceneSmith derived-SDF "
                        f"path: {member.name!r}"
                    )
                if len(parts) < 4 or parts[0] != "data":
                    raise ArtiverseExtractionError(
                        f"Tar member is not inside a declared model root: {member.name!r}"
                    )
                model_root = "/".join(parts[:4])
                if model_root not in declared_roots:
                    raise ArtiverseExtractionError(
                        f"Tar member is outside this chunk's declared roots: {member.name!r}"
                    )
                if not member.isdir() and len(parts) < 5:
                    raise ArtiverseExtractionError(
                        "Regular tar member is not below its declared model root: "
                        f"{member.name!r}"
                    )
                result.roots_seen.add(model_root)
                normalized = _register_member(
                    state,
                    parts,
                    is_directory=member.isdir(),
                    archive=spec.archive,
                )
                if member.isdir():
                    if member.size != 0:
                        raise ArtiverseExtractionError(
                            f"Directory tar member has non-zero size: {member.name!r}"
                        )
                    result.directory_count += 1
                    _ensure_directory(staging_root, parts)
                    continue
                if member.size < 0 or member.size > MAX_SINGLE_FILE_BYTES:
                    raise ArtiverseExtractionError(
                        f"Regular tar member size is outside safe limits: {member.name!r}"
                    )
                result.regular_file_count += 1
                result.uncompressed_bytes += member.size
                if result.regular_file_count > spec.file_count:
                    raise ArtiverseExtractionError(
                        f"Archive {spec.archive} contains too many regular files"
                    )
                if result.uncompressed_bytes > spec.input_bytes:
                    raise ArtiverseExtractionError(
                        f"Archive {spec.archive} exceeds its declared uncompressed bytes"
                    )
                source = tar.extractfile(member)
                if source is None:
                    raise ArtiverseExtractionError(
                        f"Could not stream regular tar member: {member.name!r}"
                    )
                with source:
                    content_digest = _write_regular_member(
                        staging_root, parts, source, member.size
                    )
                _verify_casefold_member(
                    state,
                    normalized,
                    archive=spec.archive,
                    size=member.size,
                    content_digest=content_digest,
                )
                state.files[normalized] = (member.size, content_digest)
        authenticated_stream.authenticate(spec)
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise ArtiverseExtractionError(
            f"Cannot safely read archive {spec.archive}: {exc}"
        ) from exc
    if result.regular_file_count != spec.file_count:
        raise ArtiverseExtractionError(
            f"Archive {spec.archive} regular-file count mismatch: expected "
            f"{spec.file_count}, got {result.regular_file_count}"
        )
    if result.uncompressed_bytes != spec.input_bytes:
        raise ArtiverseExtractionError(
            f"Archive {spec.archive} uncompressed-byte mismatch: expected "
            f"{spec.input_bytes}, got {result.uncompressed_bytes}"
        )
    if result.roots_seen != declared_roots:
        missing = sorted(declared_roots - result.roots_seen)
        raise ArtiverseExtractionError(
            f"Archive {spec.archive} does not contain all declared roots: {missing[:5]}"
        )
    return result


def _length_prefixed_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted((item.encode("utf-8") for item in values)):
        digest.update(struct.pack(">Q", len(value)))
        digest.update(value)
    return digest.hexdigest()


def _tree_inventory_hash(state: TreeState) -> str:
    digest = hashlib.sha256()
    all_paths = sorted(
        set(state.directories).union(state.files), key=lambda value: value.encode("utf-8")
    )
    for path in all_paths:
        encoded = path.encode("utf-8")
        if path in state.files:
            size, content_sha256 = state.files[path]
            digest.update(b"F")
            digest.update(struct.pack(">Q", len(encoded)))
            digest.update(encoded)
            digest.update(struct.pack(">Q", size))
            digest.update(content_sha256)
        else:
            digest.update(b"D")
            digest.update(struct.pack(">Q", len(encoded)))
            digest.update(encoded)
    return digest.hexdigest()


def _receipt(
    expected: ReleaseSpec,
    roots: tuple[str, ...],
    results: tuple[ChunkResult, ...],
    state: TreeState,
    extractor_sha256: str,
) -> dict[str, Any]:
    by_archive = {result.archive: result for result in results}
    archives: list[dict[str, Any]] = []
    for chunk in expected.chunks:
        result = by_archive[chunk.archive]
        archives.append(
            {
                "archive": chunk.archive,
                "archive_bytes": chunk.archive_bytes,
                "sha256": chunk.sha256,
                "validated_member_count": result.member_count,
                "validated_directory_count": result.directory_count,
                "validated_regular_file_count": result.regular_file_count,
                "validated_uncompressed_bytes": result.uncompressed_bytes,
            }
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "manifest": {
            "path": "dataset_chunks/manifest.json",
            "sha256": expected.manifest_sha256,
            "format": MANIFEST_FORMAT,
        },
        "archives": archives,
        "declared_roots": {
            "count": len(roots),
            "sha256": _length_prefixed_hash(roots),
            "hash_algorithm": ROOT_HASH_ALGORITHM,
        },
        "validated_aggregate": {
            "member_count": sum(result.member_count for result in results),
            "directory_count": sum(result.directory_count for result in results),
            "regular_file_count": sum(
                result.regular_file_count for result in results
            ),
            "uncompressed_bytes": sum(
                result.uncompressed_bytes for result in results
            ),
        },
        "data_tree_inventory": {
            "directory_count": len(state.directories),
            "regular_file_count": len(state.files),
            "sha256": _tree_inventory_hash(state),
            "hash_algorithm": TREE_HASH_ALGORITHM,
        },
        "safe_extractor": {
            "version": EXTRACTOR_VERSION,
            "sha256": extractor_sha256,
            "filename": Path(__file__).name,
        },
    }


def _write_staged_receipt(path: Path, receipt: dict[str, Any]) -> None:
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def _load_receipt(path: Path) -> dict[str, Any]:
    _require_regular_unlinked_file(path, "Extraction receipt")
    try:
        size = path.stat().st_size
        if size > 1024 * 1024:
            raise ArtiverseExtractionError("Extraction receipt exceeds the safe size limit")
        payload = path.read_bytes()
        receipt = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except ArtiverseExtractionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtiverseExtractionError(f"Cannot read extraction receipt: {exc}") from exc
    if not isinstance(receipt, dict):
        raise ArtiverseExtractionError("Extraction receipt root must be an object")
    return receipt


def _validate_receipt(
    receipt: dict[str, Any],
    expected: ReleaseSpec,
    roots: tuple[str, ...],
    extractor_sha256: str,
) -> None:
    _exact_keys(receipt, RECEIPT_KEYS, "Extraction receipt")
    if _integer(receipt["schema_version"], "receipt.schema_version") != 1:
        raise ArtiverseExtractionError("Unsupported extraction receipt schema version")
    if receipt["status"] != "pass":
        raise ArtiverseExtractionError("Extraction receipt status is not pass")

    receipt_manifest = receipt["manifest"]
    if not isinstance(receipt_manifest, dict):
        raise ArtiverseExtractionError("receipt.manifest must be an object")
    _exact_keys(receipt_manifest, RECEIPT_MANIFEST_KEYS, "receipt.manifest")
    expected_manifest = {
        "path": "dataset_chunks/manifest.json",
        "sha256": expected.manifest_sha256,
        "format": MANIFEST_FORMAT,
    }
    if receipt_manifest != expected_manifest:
        raise ArtiverseExtractionError("Extraction receipt manifest pin mismatch")

    receipt_archives = receipt["archives"]
    if not isinstance(receipt_archives, list) or len(receipt_archives) != len(expected.chunks):
        raise ArtiverseExtractionError("Extraction receipt archive list mismatch")
    archive_member_total = 0
    archive_directory_total = 0
    archive_file_total = 0
    archive_byte_total = 0
    for index, (record, chunk) in enumerate(zip(receipt_archives, expected.chunks)):
        label = f"receipt.archives[{index}]"
        if not isinstance(record, dict):
            raise ArtiverseExtractionError(f"{label} must be an object")
        _exact_keys(record, RECEIPT_ARCHIVE_KEYS, label)
        pinned = {
            "archive": chunk.archive,
            "archive_bytes": chunk.archive_bytes,
            "sha256": chunk.sha256,
            "validated_regular_file_count": chunk.file_count,
            "validated_uncompressed_bytes": chunk.input_bytes,
        }
        for field_name, expected_value in pinned.items():
            if record[field_name] != expected_value:
                raise ArtiverseExtractionError(f"{label}.{field_name} pin mismatch")
        member_count = _integer(
            record["validated_member_count"], f"{label}.validated_member_count"
        )
        directory_count = _integer(
            record["validated_directory_count"],
            f"{label}.validated_directory_count",
        )
        file_count = _integer(
            record["validated_regular_file_count"],
            f"{label}.validated_regular_file_count",
        )
        byte_count = _integer(
            record["validated_uncompressed_bytes"],
            f"{label}.validated_uncompressed_bytes",
        )
        if member_count != directory_count + file_count:
            raise ArtiverseExtractionError(f"{label} member counts are inconsistent")
        archive_member_total += member_count
        archive_directory_total += directory_count
        archive_file_total += file_count
        archive_byte_total += byte_count

    receipt_roots = receipt["declared_roots"]
    if not isinstance(receipt_roots, dict):
        raise ArtiverseExtractionError("receipt.declared_roots must be an object")
    _exact_keys(receipt_roots, RECEIPT_ROOT_KEYS, "receipt.declared_roots")
    expected_roots = {
        "count": expected.model_count,
        "sha256": _length_prefixed_hash(roots),
        "hash_algorithm": ROOT_HASH_ALGORITHM,
    }
    if receipt_roots != expected_roots:
        raise ArtiverseExtractionError("Extraction receipt declared-root pin mismatch")

    aggregate = receipt["validated_aggregate"]
    if not isinstance(aggregate, dict):
        raise ArtiverseExtractionError("receipt.validated_aggregate must be an object")
    _exact_keys(aggregate, RECEIPT_AGGREGATE_KEYS, "receipt.validated_aggregate")
    expected_aggregate = {
        "member_count": archive_member_total,
        "directory_count": archive_directory_total,
        "regular_file_count": archive_file_total,
        "uncompressed_bytes": archive_byte_total,
    }
    for field_name, expected_value in expected_aggregate.items():
        actual = _integer(aggregate[field_name], f"receipt.validated_aggregate.{field_name}")
        if actual != expected_value:
            raise ArtiverseExtractionError(
                f"receipt.validated_aggregate.{field_name} mismatch"
            )
    if archive_file_total != expected.file_count or archive_byte_total != expected.input_bytes:
        raise ArtiverseExtractionError("Extraction receipt aggregate release counts mismatch")

    inventory = receipt["data_tree_inventory"]
    if not isinstance(inventory, dict):
        raise ArtiverseExtractionError("receipt.data_tree_inventory must be an object")
    _exact_keys(inventory, RECEIPT_INVENTORY_KEYS, "receipt.data_tree_inventory")
    _integer(inventory["directory_count"], "receipt.data_tree_inventory.directory_count", minimum=1)
    if _integer(
        inventory["regular_file_count"],
        "receipt.data_tree_inventory.regular_file_count",
    ) != expected.file_count:
        raise ArtiverseExtractionError("Extraction receipt tree file count mismatch")
    _validate_sha256(inventory["sha256"], "receipt.data_tree_inventory.sha256")
    if inventory["hash_algorithm"] != TREE_HASH_ALGORITHM:
        raise ArtiverseExtractionError("Extraction receipt tree hash algorithm mismatch")

    extractor_record = receipt["safe_extractor"]
    if not isinstance(extractor_record, dict):
        raise ArtiverseExtractionError("receipt.safe_extractor must be an object")
    _exact_keys(extractor_record, RECEIPT_EXTRACTOR_KEYS, "receipt.safe_extractor")
    expected_extractor = {
        "version": EXTRACTOR_VERSION,
        "sha256": extractor_sha256,
        "filename": Path(__file__).name,
    }
    if extractor_record != expected_extractor:
        raise ArtiverseExtractionError("Extraction receipt safe-extractor pin mismatch")


def _hash_existing_regular_file(path: Path) -> tuple[int, bytes]:
    before = _require_regular_unlinked_file(path, "Extracted data file")
    if before.st_nlink != 1:
        raise ArtiverseExtractionError(f"Hard-linked extracted data file is forbidden: {path}")
    if before.st_size > MAX_SINGLE_FILE_BYTES:
        raise ArtiverseExtractionError(f"Extracted data file exceeds size limit: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ArtiverseExtractionError(f"Cannot safely open extracted file {path}: {exc}") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ArtiverseExtractionError(f"Extracted data file changed while opening: {path}")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            block = os.read(fd, COPY_BUFFER_BYTES)
            if not block:
                break
            byte_count += len(block)
            digest.update(block)
        after = os.fstat(fd)
        if (
            byte_count != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise ArtiverseExtractionError(f"Extracted data file changed while hashing: {path}")
        return byte_count, digest.digest()
    finally:
        os.close(fd)


def _is_explicit_scenesmith_derived_file(parts: tuple[str, ...]) -> bool:
    """Return whether ``parts`` names one narrowly allowed derived SDF file.

    SceneSmith writes the converted SDF beside the selected publisher URDF.  A
    process crash can also leave the one fixed-name transactional file beside
    that URDF.  These two regular files are not members of the publisher
    release and therefore must be omitted when re-hashing the immutable source
    inventory.  No other generated basename or location is exempted.
    """

    return (
        len(parts) >= 6
        and parts[0] == "data"
        and parts[4] == "urdf_w_collider"
        and parts[-1] in DERIVED_SDF_FILENAMES
    )


def _inventory_existing_tree(
    repository_root: Path,
    roots: tuple[str, ...],
    expected: ReleaseSpec,
    casefold_allowances: tuple[PinnedCasefoldCollision, ...],
) -> TreeState:
    data_root = repository_root / "data"
    _require_real_directory(data_root, "Extracted data root")
    declared_roots = set(roots)
    allowed_prefixes = {"data"}
    for root in roots:
        root_parts = root.split("/")
        for length in range(2, 5):
            allowed_prefixes.add("/".join(root_parts[:length]))
    state = TreeState(casefold_allowances=casefold_allowances)
    state.directories.add("data")
    state.spelling_by_casefold["data"] = "data"
    seen_roots: set[str] = set()
    derived_files: list[tuple[str, Path]] = []
    stack = [data_root]
    aggregate_bytes = 0
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ArtiverseExtractionError(f"Cannot safely scan {directory}: {exc}") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArtiverseExtractionError(f"Cannot stat extracted path {path}: {exc}") from exc
            if entry.is_symlink() or _is_reparse_point(info):
                raise ArtiverseExtractionError(f"Link/reparse point in extracted data: {path}")
            is_directory = stat.S_ISDIR(info.st_mode)
            is_regular = stat.S_ISREG(info.st_mode)
            if not (is_directory or is_regular):
                raise ArtiverseExtractionError(f"Special file in extracted data: {path}")
            relative_parts = path.relative_to(repository_root).parts
            parts = _normalize_posix_path(
                "/".join(relative_parts),
                f"Extracted path {path}",
                directory=is_directory,
            )
            normalized = "/".join(parts)
            _register_path_spelling(
                state,
                parts,
                is_directory=is_directory,
            )
            if len(parts) < 4:
                if not is_directory or normalized not in allowed_prefixes:
                    raise ArtiverseExtractionError(
                        f"Extracted path is outside declared root prefixes: {normalized!r}"
                    )
            else:
                model_root = "/".join(parts[:4])
                if model_root not in declared_roots:
                    raise ArtiverseExtractionError(
                        f"Extracted path is outside declared model roots: {normalized!r}"
                    )
                if normalized == model_root and not is_directory:
                    raise ArtiverseExtractionError(
                        f"Declared model root is not a directory: {normalized!r}"
                    )
                seen_roots.add(model_root)
            if is_directory:
                state.directories.add(normalized)
                stack.append(path)
            else:
                if len(parts) < 5:
                    raise ArtiverseExtractionError(
                        f"Regular file is not below a declared model root: {normalized!r}"
                    )
                size, content_digest = _hash_existing_regular_file(path)
                if _is_explicit_scenesmith_derived_file(parts):
                    derived_files.append((normalized, path))
                    continue
                _verify_casefold_member(
                    state,
                    normalized,
                    archive=None,
                    size=size,
                    content_digest=content_digest,
                )
                aggregate_bytes += size
                if len(state.files) + 1 > expected.file_count or aggregate_bytes > expected.input_bytes:
                    raise ArtiverseExtractionError("Existing data tree exceeds release counts")
                state.files[normalized] = (size, content_digest)
    for normalized, path in derived_files:
        parent = normalized.rsplit("/", 1)[0]
        if not any(
            source_path.rsplit("/", 1)[0] == parent
            and source_path.lower().endswith(".urdf")
            for source_path in state.files
        ):
            raise ArtiverseExtractionError(
                "SceneSmith derived SDF is not beside a publisher URDF: "
                f"{path}"
            )
    _require_complete_casefold_allowances(state)
    if seen_roots != declared_roots:
        missing = sorted(declared_roots - seen_roots)
        raise ArtiverseExtractionError(f"Existing data tree is missing model roots: {missing[:5]}")
    if len(state.files) != expected.file_count or aggregate_bytes != expected.input_bytes:
        raise ArtiverseExtractionError("Existing data tree file/byte counts mismatch")
    return state


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    info = path.lstat()
    if stat.S_ISDIR(info.st_mode) and not _is_reparse_point(info):
        shutil.rmtree(path)
    else:
        path.unlink()


def _validate_publish_targets(repository_root: Path, overwrite: bool) -> None:
    data_target = repository_root / "data"
    receipt_target = repository_root / RECEIPT_FILENAME
    if data_target.exists() or data_target.is_symlink():
        if not overwrite:
            raise ArtiverseExtractionError(
                "Target data directory already exists; use --overwrite for backup/rollback replacement"
            )
        _require_real_directory(data_target, "Existing data target")
    if receipt_target.exists() or receipt_target.is_symlink():
        if not overwrite:
            raise ArtiverseExtractionError(
                f"Extraction receipt already exists: {receipt_target}"
            )
        _require_regular_unlinked_file(receipt_target, "Existing extraction receipt")


def _publish(
    repository_root: Path, staging_root: Path, *, overwrite: bool
) -> None:
    data_target = repository_root / "data"
    receipt_target = repository_root / RECEIPT_FILENAME
    staged_data = staging_root / "data"
    staged_receipt = staging_root / RECEIPT_FILENAME
    if not staged_data.is_dir() or staged_data.is_symlink():
        raise ArtiverseExtractionError("Validated staging tree has no real data directory")
    if not staged_receipt.is_file() or staged_receipt.is_symlink():
        raise ArtiverseExtractionError("Validated staging tree has no regular receipt")
    suffix = uuid.uuid4().hex
    data_backup = repository_root / f".artiverse-data-backup-{suffix}"
    receipt_backup = repository_root / f".artiverse-receipt-backup-{suffix}.json"
    had_data = data_target.exists()
    had_receipt = receipt_target.exists()
    published_data = False
    published_receipt = False
    try:
        if had_data:
            if not overwrite:
                raise ArtiverseExtractionError("Data target appeared during extraction")
            os.replace(data_target, data_backup)
        if had_receipt:
            if not overwrite:
                raise ArtiverseExtractionError("Receipt target appeared during extraction")
            os.replace(receipt_target, receipt_backup)
        os.replace(staged_data, data_target)
        published_data = True
        os.replace(staged_receipt, receipt_target)
        published_receipt = True
    except Exception as exc:
        rollback_errors: list[str] = []
        try:
            if published_receipt:
                _remove_path(receipt_target)
            if receipt_backup.exists():
                os.replace(receipt_backup, receipt_target)
        except Exception as rollback_exc:  # pragma: no cover - catastrophic FS failure
            rollback_errors.append(f"receipt rollback: {rollback_exc}")
        try:
            if published_data:
                _remove_path(data_target)
            if data_backup.exists():
                os.replace(data_backup, data_target)
        except Exception as rollback_exc:  # pragma: no cover - catastrophic FS failure
            rollback_errors.append(f"data rollback: {rollback_exc}")
        detail = f"; {'; '.join(rollback_errors)}" if rollback_errors else ""
        raise ArtiverseExtractionError(f"Atomic publish failed: {exc}{detail}") from exc
    if data_backup.exists():
        _remove_path(data_backup)
    if receipt_backup.exists():
        _remove_path(receipt_backup)


def extract_release(
    repository_root: Path,
    *,
    manifest_path: Path | None = None,
    chunks_dir: Path | None = None,
    overwrite: bool = False,
    expected_release: ReleaseSpec = PINNED_RELEASE,
) -> dict[str, Any]:
    """Authenticate, validate, safely extract, and atomically publish Artiverse."""

    repository_root = repository_root.absolute()
    _require_real_directory(repository_root, "Repository root")
    repository_root = repository_root.resolve(strict=True)
    manifest_path = manifest_path or repository_root / "dataset_chunks" / "manifest.json"
    chunks_dir = chunks_dir or repository_root / "dataset_chunks"
    _require_regular_unlinked_file(manifest_path, "Manifest")
    _require_real_directory(chunks_dir, "Chunks directory")
    manifest_path = _within(manifest_path, repository_root, "Manifest")
    chunks_dir = _within(chunks_dir, repository_root, "Chunks directory")
    _validate_publish_targets(repository_root, overwrite)
    extractor_sha256 = _sha256_file(Path(__file__).resolve())

    payload = _manifest_bytes(manifest_path)
    _manifest, chunks, roots = _validate_manifest(payload, expected_release)
    casefold_allowances = _casefold_allowances_for_release(
        expected_release,
        chunks,
    )
    staging_root: Path | None = None
    with contextlib.ExitStack() as stack:
        # Keep the authenticated file descriptions open through extraction.
        # No staging or target mutation occurs before every archive passes.
        verified = _verify_archives(chunks_dir, chunks, stack)
        try:
            staging_root = Path(
                tempfile.mkdtemp(prefix=".artiverse-data-staging-", dir=repository_root)
            )
            state = TreeState(casefold_allowances=casefold_allowances)
            results = tuple(
                _extract_archive(archive, staging_root, state) for archive in verified
            )
            _require_complete_casefold_allowances(state)
            if len(state.files) != expected_release.file_count:
                raise ArtiverseExtractionError("Aggregate extracted regular-file count mismatch")
            if sum(size for size, _digest in state.files.values()) != expected_release.input_bytes:
                raise ArtiverseExtractionError("Aggregate extracted byte count mismatch")
            receipt = _receipt(
                expected_release, roots, results, state, extractor_sha256
            )
            _write_staged_receipt(staging_root / RECEIPT_FILENAME, receipt)
            _publish(repository_root, staging_root, overwrite=overwrite)
            return receipt
        finally:
            if staging_root is not None and staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)


def verify_existing_release(
    repository_root: Path,
    *,
    manifest_path: Path | None = None,
    expected_release: ReleaseSpec = PINNED_RELEASE,
) -> dict[str, Any]:
    """Re-hash an existing extracted tree and match its strict receipt."""

    repository_root = repository_root.absolute()
    _require_real_directory(repository_root, "Repository root")
    repository_root = repository_root.resolve(strict=True)
    manifest_path = manifest_path or repository_root / "dataset_chunks" / "manifest.json"
    _require_regular_unlinked_file(manifest_path, "Manifest")
    manifest_path = _within(manifest_path, repository_root, "Manifest")
    payload = _manifest_bytes(manifest_path)
    _manifest, chunks, roots = _validate_manifest(payload, expected_release)
    casefold_allowances = _casefold_allowances_for_release(
        expected_release,
        chunks,
    )
    extractor_sha256 = _sha256_file(Path(__file__).resolve())
    receipt = _load_receipt(repository_root / RECEIPT_FILENAME)
    _validate_receipt(receipt, expected_release, roots, extractor_sha256)
    state = _inventory_existing_tree(
        repository_root,
        roots,
        expected_release,
        casefold_allowances,
    )
    inventory = receipt["data_tree_inventory"]
    if len(state.directories) != inventory["directory_count"]:
        raise ArtiverseExtractionError("Existing data tree directory count mismatch")
    if len(state.files) != inventory["regular_file_count"]:
        raise ArtiverseExtractionError("Existing data tree regular-file count mismatch")
    if _tree_inventory_hash(state) != inventory["sha256"]:
        raise ArtiverseExtractionError("Existing data tree inventory SHA-256 mismatch")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--chunks-dir", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing data tree using backup/rollback publication.",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Re-hash and validate an existing data tree and extraction receipt.",
    )
    args = parser.parse_args()
    if args.verify_existing and args.overwrite:
        parser.error("--verify-existing and --overwrite are mutually exclusive")
    try:
        if args.verify_existing:
            receipt = verify_existing_release(
                args.repository_root,
                manifest_path=args.manifest,
            )
        else:
            receipt = extract_release(
                args.repository_root,
                manifest_path=args.manifest,
                chunks_dir=args.chunks_dir,
                overwrite=args.overwrite,
            )
    except ArtiverseExtractionError as exc:
        parser.exit(2, f"Artiverse safe extraction failed: {exc}\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
