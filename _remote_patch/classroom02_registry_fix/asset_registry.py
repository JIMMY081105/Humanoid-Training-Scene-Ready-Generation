"""Asset registry for tracking generated assets during a session."""

import json
import hashlib
import copy
import base64
import logging
import math
import os
import re
import uuid
import xml.etree.ElementTree as ET

from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from pydrake.all import Quaternion, RigidTransform, RotationMatrix

from scenesmith.agent_utils.room import ObjectType, SceneObject, UniqueID

console_logger = logging.getLogger(__name__)

_REGISTRY_SCHEMA_VERSION = 2
_REGISTRY_ENTRY_KEYS = frozenset(
    {
        "object_id",
        "object_type",
        "name",
        "description",
        "transform",
        "geometry_path",
        "sdf_path",
        "image_path",
        "metadata",
        "bbox_min",
        "bbox_max",
        "scale_factor",
    }
)
_GEOMETRY_SUFFIXES = frozenset({".gltf", ".glb", ".obj", ".stl", ".ply"})
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_DEPENDENCY_SUFFIXES = frozenset(
    {
        *_GEOMETRY_SUFFIXES,
        *_IMAGE_SUFFIXES,
        ".bin",
        ".mtl",
        ".tga",
        ".bmp",
        ".exr",
        ".ktx2",
        ".sdf",
    }
)
_LEGACY_INTERRUPTED_REGISTRY_SHA256 = (
    "030e43be8bf0467f8338ee361aa8dacc395374d80e026368adfb32f22f74cc5f"
)
_LEGACY_INTERRUPTED_REGISTRY_IDS = frozenset(
    {
        "office_supply_box_0",
        "binder_clip_0",
        "file_folder_0",
        "manila_file_folder_0",
    }
)
_LEGACY_INTERRUPTED_DIRECTORY_MANIFESTS = {
    "binder_clip_0": (42, 1930681, "0fa0b2b3d980a4c60a60591d36a81f90450173d27adee2e585d31bc4d90457d6"),
    "file_folder_0": (42, 509116, "4e6b1871bdeecc32b8028e23f6e6e546cd61f34334a7d284b34a73d1cca87f97"),
    "manila_file_folder_0": (42, 230799, "1e2859cb4e068722e4b473e3aa30ca9216854a4d6a2c4da0f4fd95df443a378e"),
    "office_supply_box_0": (42, 1330876, "05b7137e3a27cff0df9a70e8ec97f4a067bcdbc9c7ec7ce9423da0f8944ee86a"),
}
_LEGACY_INTERRUPTED_ENTRY_SHA256 = {
    "binder_clip_0": "9855cf2bd799cc51d83b0340c68ac23c086387aaec55a8f64f4bc1c1c8862fa5",
    "file_folder_0": "cc69ead5039f449f139c81ae90a82ed0cf5d184c9a10e44ffecd97fc7f4d70c7",
    "manila_file_folder_0": "8a9b90bebab6ff32343903ffe2391eb6d7a348083d08f1a96e2da8b4575d9a4a",
    "office_supply_box_0": "4dff9432c9bb57576c5896ec071ce948377b854857f9b3eddd5fa4103826e477",
}
_LEGACY_INTERRUPTED_AGGREGATE_SHA256 = (
    "2670eb42cf715166d9105bc21107db156aec4f4b131a9c65345abcb2b010e250"
)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"Asset registry contains duplicate key: {key}")
        result[key] = value
    return result


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_payload_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class AssetRegistry:
    """Registry for tracking generated assets within a session (tracked in memory)."""

    def __init__(
        self,
        auto_save_path: Path | None = None,
        required_root: Path | None = None,
        allowed_object_types: frozenset[ObjectType] | None = None,
    ) -> None:
        """Initialize empty registry.

        Args:
            auto_save_path: If provided, registry will auto-save to this path
                after each registration. This ensures the registry is saved
                incrementally, even if scene generation fails partway through.
            required_root: Optional room output root.  Every persisted registry
                path and asset dependency must remain inside this directory.
        """
        self._assets: dict[UniqueID, SceneObject] = {}
        self.auto_save_path = auto_save_path
        self.required_root = required_root
        self.allowed_object_types = allowed_object_types
        self._immutable_loaded_entries: dict[UniqueID, dict[str, Any]] = {}
        self._immutable_loaded_files: dict[Path, tuple[int, str]] = {}
        self._revision = 0
        self._previous_attestation: str | None = None
        self._attestation_history: list[str] = []
        self._lineage_root: str | None = None
        self._last_save_committed = False
        self._persisted_file_sha256: str | None = None
        self._legacy_source_b64: str | None = None
        console_logger.info(
            f"Initialized AssetRegistry"
            f"{' with auto-save to ' + str(auto_save_path) if auto_save_path else ''}"
        )

    def register(self, asset: SceneObject) -> None:
        """Register a generated asset.

        Args:
            asset: SceneObject to register for reuse.

        Raises:
            ValueError: If asset_id already exists in registry.
        """
        asset_id = asset.object_id
        self._validate_template_shape(asset)
        if asset_id in self._assets:
            raise ValueError(
                f"Asset {asset_id} already registered. Use generate_unique_id() "
                f"to generate collision-free IDs."
            )

        previous_assets = self._assets
        stored_asset = (
            copy.deepcopy(asset)
            if self.allowed_object_types is not None
            else asset
        )
        self._assets = {**self._assets, asset_id: stored_asset}
        console_logger.info(f"Registered asset {asset_id} ({asset.name})")

        # Auto-save if path is configured (fail-fast on errors).
        if self.auto_save_path:
            try:
                self.save_to_file(file_path=self.auto_save_path)
            except Exception:
                if not self._last_save_committed:
                    self._assets = previous_assets
                raise
            console_logger.debug(f"Auto-saved registry to {self.auto_save_path}")

    def _validate_template_shape(self, asset: SceneObject) -> None:
        if self.allowed_object_types is None:
            return
        if (
            asset.object_type not in self.allowed_object_types
            or asset.support_surfaces
            or asset.placement_info is not None
            or asset.metadata.get("composite_type") is not None
            or asset.immutable
        ):
            raise RuntimeError(
                f"Strict registry asset is not a reusable identity template: "
                f"{asset.object_id}"
            )

    def generate_unique_id(self, name: str) -> UniqueID:
        """Generate unique ID that doesn't conflict with registered assets.

        Args:
            name: Human-readable name for the asset.

        Returns:
            UniqueID guaranteed not to conflict with existing assets.
        """
        return UniqueID.generate_unique(name, self._assets)

    def get(self, asset_id: UniqueID) -> SceneObject | None:
        """Get asset by ID.

        Args:
            asset_id: Unique identifier of the asset.

        Returns:
            SceneObject if found, None otherwise.
        """
        asset = self._assets.get(asset_id)
        if asset is not None and self.allowed_object_types is not None:
            asset = copy.deepcopy(asset)
        if asset:
            console_logger.debug(f"Retrieved asset {asset_id}")
        else:
            console_logger.debug(f"Asset {asset_id} not found in registry")
        return asset

    def list_all(self) -> list[SceneObject]:
        """List all registered assets.

        Returns:
            List of all registered SceneObjects.
        """
        if self.allowed_object_types is None:
            # Preserve historical session-cache identity semantics for the
            # furniture/wall/ceiling managers.  Strict detached definitions are
            # confined to the resumable MANIPULAND registry.
            assets = list(self._assets.values())
        else:
            assets = [self.get(asset_id) for asset_id in self._assets]
        console_logger.debug(f"Listed {len(assets)} available assets")
        return assets

    def exists(self, asset_id: UniqueID) -> bool:
        """Check if asset exists in registry.

        Args:
            asset_id: Unique identifier to check.

        Returns:
            True if asset exists, False otherwise.
        """
        return asset_id in self._assets

    def clear(self) -> None:
        """Clear all registered assets."""
        if self._immutable_loaded_entries:
            raise RuntimeError("Refusing to clear immutable loaded registry entries")
        count = len(self._assets)
        self._assets.clear()
        console_logger.info(f"Cleared {count} assets from registry")

    def size(self) -> int:
        """Get number of registered assets."""
        return len(self._assets)

    def apply_scale_by_sdf_path(self, sdf_path: Path, scale_factor: float) -> int:
        """Apply scale to all registry entries with matching sdf_path.

        Updates bbox_min, bbox_max, and scale_factor for all assets that share
        the given SDF file. This keeps the registry in sync after rescaling.

        Args:
            sdf_path: Path to the SDF file that was rescaled.
            scale_factor: Scale multiplier that was applied.

        Returns:
            Number of registry entries updated.
        """
        target = sdf_path.resolve(strict=False)
        immutable_matches = [
            str(asset_id)
            for asset_id in self._immutable_loaded_entries
            if self._assets[asset_id].sdf_path is not None
            and self._assets[asset_id].sdf_path.resolve(strict=False) == target
        ]
        if immutable_matches:
            raise RuntimeError(
                "Refusing to rescale immutable loaded registry assets: "
                f"{sorted(immutable_matches)}"
            )
        originals: dict[UniqueID, SceneObject] = {}
        updated_count = 0
        for asset_id, asset in self._assets.items():
            if asset.sdf_path == sdf_path:
                originals[asset_id] = copy.deepcopy(asset)
                asset.apply_scale(scale_factor)
                updated_count += 1

        if updated_count > 0:
            console_logger.info(
                f"Updated {updated_count} registry entries for rescaled asset "
                f"{sdf_path.name}"
            )

            # Auto-save if path is configured.
            if self.auto_save_path:
                try:
                    self.save_to_file(file_path=self.auto_save_path)
                except Exception:
                    if not self._last_save_committed:
                        self._assets.update(originals)
                    raise
                console_logger.debug(f"Auto-saved registry after rescale")

        return updated_count

    def _registry_root(self, file_path: Path) -> Path:
        root = self.required_root or file_path.parent
        lexical = root if root.is_absolute() else Path.cwd() / root
        current = Path(lexical.anchor)
        for part in lexical.parts[1:]:
            current = current / part
            if current.is_symlink():
                raise RuntimeError(
                    f"Asset registry root contains a symlink component: {current}"
                )
        try:
            resolved = lexical.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"Asset registry root is missing: {root}") from exc
        if not resolved.is_dir() or resolved.is_symlink():
            raise RuntimeError(f"Asset registry root is not a regular directory: {root}")
        return resolved

    @staticmethod
    def _regular_file_no_links(path: Path, *, label: str) -> Path:
        lexical = path if path.is_absolute() else Path.cwd() / path
        current = Path(lexical.anchor)
        for part in lexical.parts[1:]:
            current = current / part
            if current.is_symlink():
                raise RuntimeError(f"{label} contains a symlink component: {current}")
        try:
            resolved = lexical.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"{label} is missing: {lexical}") from exc
        stat = resolved.stat()
        if not resolved.is_file() or stat.st_size <= 0 or stat.st_nlink != 1:
            raise RuntimeError(f"{label} is not a nonempty unlinked regular file")
        return resolved

    def _canonical_asset_path(
        self,
        raw_path: Any,
        *,
        root: Path,
        label: str,
        suffixes: frozenset[str],
        allow_absolute: bool = False,
    ) -> Path | None:
        if raw_path is None:
            return None
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or raw_path != raw_path.strip()
            or "%" in raw_path
            or (
                self.allowed_object_types is not None
                and not allow_absolute
                and "\\" in raw_path
            )
        ):
            raise RuntimeError(f"{label} is not a canonical path")
        path = Path(raw_path)
        pure = PurePosixPath(raw_path) if "\\" not in raw_path else None
        if (
            any(part in {"", ".", ".."} for part in path.parts)
            or (
                self.allowed_object_types is not None
                and not allow_absolute
                and (
                    pure is None
                    or pure.is_absolute()
                    or pure.as_posix() != raw_path
                )
            )
        ):
            raise RuntimeError(f"{label} is not a normalized path")
        if path.is_absolute() and not allow_absolute:
            raise RuntimeError(f"{label} must be namespace-relative")
        if not path.is_absolute():
            path = root / path
        resolved = self._regular_file_no_links(path, label=label)
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"{label} escapes the room output: {resolved}")
        if resolved.suffix.lower() not in suffixes:
            raise RuntimeError(f"{label} has unsupported suffix: {resolved.suffix}")
        return resolved

    def _dependency_path(
        self,
        raw_uri: Any,
        *,
        owner: Path,
        root: Path,
        label: str,
    ) -> Path:
        if (
            not isinstance(raw_uri, str)
            or not raw_uri
            or raw_uri != raw_uri.strip()
            or "\\" in raw_uri
            or "%" in raw_uri
        ):
            raise RuntimeError(f"{label} URI is not canonical")
        parsed = urlsplit(raw_uri)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise RuntimeError(f"{label} URI is not a local file")
        path_text = parsed.path[2:] if parsed.path.startswith("./") else parsed.path
        relative = PurePosixPath(path_text)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.as_posix() != path_text
            or relative.suffix.lower() not in _DEPENDENCY_SUFFIXES
        ):
            raise RuntimeError(f"{label} URI is unsafe or unsupported")
        path = owner.parent.joinpath(*relative.parts)
        resolved = self._regular_file_no_links(path, label=label)
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"{label} dependency escapes the room output")
        return resolved

    def _validate_dependency_tree(
        self, initial: list[Path], *, root: Path
    ) -> set[Path]:
        pending = list(initial)
        visited: set[Path] = set()
        dependency_graph: dict[Path, set[Path]] = {}
        while pending:
            path = pending.pop()
            if path in visited:
                continue
            visited.add(path)
            suffix = path.suffix.lower()
            dependencies: list[Path] = []
            if suffix == ".sdf":
                try:
                    document = ET.parse(path).getroot()
                except (OSError, ET.ParseError) as exc:
                    raise RuntimeError(f"Asset registry SDF is invalid: {path}") from exc
                for uri in document.iter():
                    if uri.tag.rsplit("}", 1)[-1] != "uri":
                        continue
                    dependencies.append(
                        self._dependency_path(
                            uri.text or "",
                            owner=path,
                            root=root,
                            label=f"registry SDF {path.name}",
                        )
                    )
            elif suffix == ".gltf":
                try:
                    document = json.loads(
                        path.read_text(encoding="utf-8"),
                        object_pairs_hook=_reject_duplicate_json_keys,
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"Asset registry glTF is invalid: {path}") from exc
                uris: list[Any] = []

                def collect(value: Any) -> None:
                    if isinstance(value, dict):
                        for key, child in value.items():
                            if key == "uri":
                                uris.append(child)
                            else:
                                collect(child)
                    elif isinstance(value, list):
                        for child in value:
                            collect(child)

                collect(document)
                for uri in uris:
                    if isinstance(uri, str) and uri.startswith("data:"):
                        continue
                    dependencies.append(
                        self._dependency_path(
                            uri,
                            owner=path,
                            root=root,
                            label=f"registry glTF {path.name}",
                        )
                    )
            elif suffix == ".obj":
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeDecodeError) as exc:
                    raise RuntimeError(f"Asset registry OBJ is invalid: {path}") from exc
                for line in lines:
                    tokens = line.strip().split()
                    if tokens and tokens[0].lower() == "mtllib":
                        if len(tokens) != 2:
                            raise RuntimeError(f"Registry OBJ mtllib is ambiguous: {path}")
                        dependencies.append(
                            self._dependency_path(
                                tokens[1],
                                owner=path,
                                root=root,
                                label=f"registry OBJ {path.name}",
                            )
                        )
            elif suffix == ".mtl":
                texture_commands = {
                    "map_ka", "map_kd", "map_ks", "map_ke", "map_ns", "map_d",
                    "bump", "map_bump", "disp", "decal", "norm", "refl",
                }
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeDecodeError) as exc:
                    raise RuntimeError(f"Asset registry MTL is invalid: {path}") from exc
                for line in lines:
                    tokens = line.strip().split()
                    if tokens and tokens[0].lower() in texture_commands:
                        if len(tokens) < 2:
                            raise RuntimeError(f"Registry MTL texture is empty: {path}")
                        dependencies.append(
                            self._dependency_path(
                                tokens[-1],
                                owner=path,
                                root=root,
                                label=f"registry MTL {path.name}",
                            )
                        )
            for dependency in dependencies:
                search = [dependency]
                seen: set[Path] = set()
                while search:
                    current = search.pop()
                    if current == path:
                        raise RuntimeError(
                            f"Asset registry dependency cycle includes {path}"
                        )
                    if current in seen:
                        continue
                    seen.add(current)
                    search.extend(dependency_graph.get(current, set()))
                dependency_graph.setdefault(path, set()).add(dependency)
            pending.extend(dependencies)
        return visited

    @staticmethod
    def _validate_vector(value: Any, *, length: int, label: str) -> list[float]:
        if not isinstance(value, list) or len(value) != length:
            raise RuntimeError(f"{label} has invalid dimensions")
        if any(isinstance(item, bool) for item in value):
            raise RuntimeError(f"{label} contains boolean values")
        try:
            result = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{label} is not numeric") from exc
        if not all(math.isfinite(item) for item in result):
            raise RuntimeError(f"{label} is not finite")
        return result

    def _deserialize_entry(
        self,
        asset_id_str: str,
        asset_data: Any,
        *,
        root: Path,
        allow_absolute_paths: bool = False,
    ) -> SceneObject:
        strict_namespace = self.allowed_object_types is not None
        if not strict_namespace:
            # Preserve the pre-attestation registry contract for furniture, wall,
            # and ceiling agents.  Those registries are session-local caches and
            # historically allowed absent/external paths plus older entries that
            # omit fields added later.  Strict restart validation is deliberately
            # confined to MANIPULAND registries constructed with an allowed type.
            if not isinstance(asset_data, dict):
                raise RuntimeError(
                    f"Asset registry entry {asset_id_str!r} is not a mapping"
                )
            transform_data = asset_data["transform"]
            translation = np.array(transform_data["translation"])
            quaternion = Quaternion(wxyz=transform_data["rotation_wxyz"])
            return SceneObject(
                object_id=UniqueID(asset_data["object_id"]),
                object_type=ObjectType(asset_data["object_type"]),
                name=asset_data["name"],
                description=asset_data["description"],
                transform=RigidTransform(
                    RotationMatrix(quaternion), translation
                ),
                geometry_path=(
                    Path(asset_data["geometry_path"])
                    if asset_data.get("geometry_path")
                    else None
                ),
                sdf_path=(
                    Path(asset_data["sdf_path"])
                    if asset_data.get("sdf_path")
                    else None
                ),
                image_path=(
                    Path(asset_data["image_path"])
                    if asset_data.get("image_path")
                    else None
                ),
                support_surfaces=[],
                metadata=copy.deepcopy(asset_data.get("metadata", {})),
                bbox_min=(
                    np.array(asset_data["bbox_min"])
                    if asset_data.get("bbox_min")
                    else None
                ),
                bbox_max=(
                    np.array(asset_data["bbox_max"])
                    if asset_data.get("bbox_max")
                    else None
                ),
                scale_factor=asset_data.get("scale_factor", 1.0),
            )
        if (
            not isinstance(asset_id_str, str)
            or not asset_id_str
            or not isinstance(asset_data, dict)
            or set(asset_data) != _REGISTRY_ENTRY_KEYS
            or asset_data.get("object_id") != asset_id_str
        ):
            raise RuntimeError(f"Asset registry entry {asset_id_str!r} has invalid schema")
        if strict_namespace and re.fullmatch(r"[A-Za-z0-9_.-]+", asset_id_str) is None:
            raise RuntimeError(f"Asset registry entry {asset_id_str!r} has unsafe ID")
        if not isinstance(asset_data["name"], str) or not asset_data["name"]:
            raise RuntimeError(f"Asset registry entry {asset_id_str} has invalid name")
        if not isinstance(asset_data["description"], str) or not asset_data["description"]:
            raise RuntimeError(
                f"Asset registry entry {asset_id_str} has invalid description"
            )
        try:
            object_type = ObjectType(asset_data["object_type"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Asset registry entry {asset_id_str} has invalid object type"
            ) from exc
        if strict_namespace and object_type not in self.allowed_object_types:
            raise RuntimeError(
                f"Asset registry entry {asset_id_str} has out-of-namespace object type"
            )
        transform_data = asset_data["transform"]
        if not isinstance(transform_data, dict) or set(transform_data) != {
            "translation", "rotation_wxyz"
        }:
            raise RuntimeError(f"Asset registry entry {asset_id_str} has invalid transform")
        translation = self._validate_vector(
            transform_data["translation"], length=3, label=f"{asset_id_str} translation"
        )
        rotation_wxyz = self._validate_vector(
            transform_data["rotation_wxyz"],
            length=4,
            label=f"{asset_id_str} quaternion",
        )
        if abs(sum(item * item for item in rotation_wxyz) - 1.0) > 1e-6:
            raise RuntimeError(f"Asset registry entry {asset_id_str} quaternion is not unit")
        if strict_namespace and (max(abs(item) for item in translation) > 1e-9 or min(
            max(abs(left - right) for left, right in zip(rotation_wxyz, expected))
            for expected in ([1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0])
        ) > 1e-9):
            raise RuntimeError(
                f"Asset registry entry {asset_id_str} is not an identity template"
            )
        bbox_min_raw = asset_data["bbox_min"]
        bbox_max_raw = asset_data["bbox_max"]
        if strict_namespace and (bbox_min_raw is None or bbox_max_raw is None):
            raise RuntimeError(f"Asset registry entry {asset_id_str} lacks bounds")
        bbox_min = (
            None
            if bbox_min_raw is None
            else np.array(
                self._validate_vector(
                    bbox_min_raw, length=3, label=f"{asset_id_str} bbox_min"
                )
            )
        )
        bbox_max = (
            None
            if bbox_max_raw is None
            else np.array(
                self._validate_vector(
                    bbox_max_raw, length=3, label=f"{asset_id_str} bbox_max"
                )
            )
        )
        if bbox_min is not None and np.any(bbox_min > bbox_max):
            raise RuntimeError(f"Asset registry entry {asset_id_str} bounds are reversed")
        scale_factor = asset_data["scale_factor"]
        if (
            isinstance(scale_factor, bool)
            or not isinstance(scale_factor, (int, float))
            or not math.isfinite(float(scale_factor))
            or float(scale_factor) <= 0
        ):
            raise RuntimeError(f"Asset registry entry {asset_id_str} has invalid scale")
        if not isinstance(asset_data["metadata"], dict):
            raise RuntimeError(f"Asset registry entry {asset_id_str} metadata is invalid")

        def verify_json_finite(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise RuntimeError("Asset registry metadata key is not a string")
                    verify_json_finite(child)
            elif isinstance(value, list):
                for child in value:
                    verify_json_finite(child)
            elif isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError("Asset registry metadata is non-finite")
            elif not isinstance(value, (str, int, float, bool, type(None))):
                raise RuntimeError("Asset registry metadata is not JSON-native")

        verify_json_finite(asset_data["metadata"])
        geometry_path = self._canonical_asset_path(
            asset_data["geometry_path"],
            root=root,
            label=f"registry geometry {asset_id_str}",
            suffixes=frozenset(_GEOMETRY_SUFFIXES - {".glb"}),
            allow_absolute=allow_absolute_paths,
        )
        sdf_path = self._canonical_asset_path(
            asset_data["sdf_path"],
            root=root,
            label=f"registry SDF {asset_id_str}",
            suffixes=frozenset({".sdf"}),
            allow_absolute=allow_absolute_paths,
        )
        image_path = self._canonical_asset_path(
            asset_data["image_path"],
            root=root,
            label=f"registry image {asset_id_str}",
            suffixes=_IMAGE_SUFFIXES,
            allow_absolute=allow_absolute_paths,
        )
        if strict_namespace and (geometry_path is None or sdf_path is None):
            raise RuntimeError(
                f"Asset registry entry {asset_id_str} lacks reusable geometry/SDF"
            )
        if strict_namespace and geometry_path is not None and sdf_path is not None:
            expected_sdf_root = (root / "sdf").resolve(strict=True)
            namespace = sdf_path.parent
            if (
                namespace.parent != expected_sdf_root
                or geometry_path.parent != namespace
            ):
                raise RuntimeError(
                    f"Asset registry entry {asset_id_str} has a cross-namespace path"
                )
        if strict_namespace:
            self._validate_dependency_tree(
                [path for path in (geometry_path, sdf_path, image_path) if path is not None],
                root=root,
            )
        quaternion = Quaternion(wxyz=rotation_wxyz)
        return SceneObject(
            object_id=UniqueID(asset_id_str),
            object_type=object_type,
            name=asset_data["name"],
            description=asset_data["description"],
            transform=RigidTransform(RotationMatrix(quaternion), np.array(translation)),
            geometry_path=geometry_path,
            sdf_path=sdf_path,
            image_path=image_path,
            support_surfaces=[],
            metadata=asset_data["metadata"],
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            scale_factor=float(scale_factor),
        )

    def _serialize_asset(
        self, asset: SceneObject, *, root: Path | None = None
    ) -> dict[str, Any]:
        quaternion = asset.transform.rotation().ToQuaternion()

        def serialize_path(path: Path | None) -> str | None:
            if path is None:
                return None
            if root is None:
                return str(path)
            resolved = self._regular_file_no_links(
                path, label=f"registry asset {asset.object_id}"
            )
            if not resolved.is_relative_to(root):
                raise RuntimeError("Registry asset escaped its namespace")
            return resolved.relative_to(root).as_posix()

        return {
            "object_id": str(asset.object_id),
            "object_type": asset.object_type.value,
            "name": asset.name,
            "description": asset.description,
            "transform": {
                "translation": asset.transform.translation().tolist(),
                "rotation_wxyz": [
                    quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()
                ],
            },
            "geometry_path": serialize_path(asset.geometry_path),
            "sdf_path": serialize_path(asset.sdf_path),
            "image_path": serialize_path(asset.image_path),
            "metadata": copy.deepcopy(asset.metadata),
            "bbox_min": asset.bbox_min.tolist() if asset.bbox_min is not None else None,
            "bbox_max": asset.bbox_max.tolist() if asset.bbox_max is not None else None,
            "scale_factor": asset.scale_factor,
        }

    def _asset_file_inventory(
        self, assets: dict[UniqueID, SceneObject], *, root: Path
    ) -> list[dict[str, Any]]:
        paths: set[Path] = set()
        namespaces: dict[Path, UniqueID] = {}
        for asset in assets.values():
            direct = []
            for path in (asset.geometry_path, asset.sdf_path, asset.image_path):
                if path is None:
                    continue
                resolved = self._regular_file_no_links(
                    path, label=f"registry asset {asset.object_id}"
                )
                if not resolved.is_relative_to(root):
                    raise RuntimeError("Registry asset escaped its namespace")
                direct.append(resolved)
            paths.update(self._validate_dependency_tree(direct, root=root))
            if self.allowed_object_types is not None and asset.sdf_path is not None:
                namespace = asset.sdf_path.resolve(strict=True).parent
                if namespace == root or not namespace.is_relative_to(root):
                    raise RuntimeError(
                        f"Registry asset {asset.object_id} lacks an exclusive namespace"
                    )
                previous_owner = namespaces.setdefault(namespace, asset.object_id)
                if previous_owner != asset.object_id:
                    raise RuntimeError(
                        f"Registry assets {previous_owner} and {asset.object_id} "
                        "share one asset namespace"
                    )
                for candidate in namespace.rglob("*"):
                    if candidate.is_symlink():
                        raise RuntimeError(
                            f"Registry namespace {asset.object_id} contains a symlink"
                        )
                    if candidate.is_dir():
                        continue
                    paths.add(
                        self._regular_file_no_links(
                            candidate,
                            label=f"registry namespace {asset.object_id}",
                        )
                    )
        records = []
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix()
            records.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        return records

    def _legacy_directory_manifest(self, directory: Path, *, root: Path) -> tuple[int, int, str]:
        directory = directory.resolve(strict=True)
        if not directory.is_relative_to(root):
            raise RuntimeError("Legacy registry asset directory escapes namespace")
        lines: list[bytes] = []
        total_bytes = 0
        count = 0
        for path in sorted(directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()):
            if path.is_symlink():
                raise RuntimeError("Legacy registry asset directory contains a symlink")
            if path.is_dir():
                continue
            resolved = self._regular_file_no_links(path, label="legacy registry asset")
            relative = resolved.relative_to(directory).as_posix()
            size = resolved.stat().st_size
            lines.append(
                relative.encode("utf-8")
                + b"\0"
                + str(size).encode("ascii")
                + b"\0"
                + _sha256_file(resolved).encode("ascii")
                + b"\n"
            )
            count += 1
            total_bytes += size
        return count, total_bytes, hashlib.sha256(b"".join(lines)).hexdigest()

    @staticmethod
    def empty_snapshot() -> dict[str, Any]:
        """Return the exact authority record for a pristine absent registry."""
        return {
            "assets": {},
            "asset_files": [],
            "head": {
                "revision": 0,
                "attestation": None,
                "attestation_history": [],
                "lineage_root": None,
                "persisted_file_sha256": None,
            },
        }

    def snapshot(self) -> dict[str, Any]:
        """Return exact registry semantics, bytes, and accepted persisted head."""
        if self.auto_save_path is None:
            raise RuntimeError("Cannot snapshot a registry without a persistence path")
        root = self._registry_root(self.auto_save_path)
        return {
            "assets": {
                str(asset_id): self._serialize_asset(asset, root=root)
                for asset_id, asset in sorted(
                    self._assets.items(), key=lambda item: str(item[0])
                )
            },
            "asset_files": self._asset_file_inventory(self._assets, root=root),
            "head": {
                "revision": self._revision,
                "attestation": self._previous_attestation,
                "attestation_history": list(self._attestation_history),
                "lineage_root": self._lineage_root,
                "persisted_file_sha256": self._persisted_file_sha256,
            },
        }

    def verify_snapshot(self, snapshot: Any) -> None:
        """Require baseline definitions/files unchanged while allowing additions."""
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "assets",
            "asset_files",
            "head",
        }:
            raise RuntimeError("Asset registry scope snapshot is malformed")
        if self.auto_save_path is None:
            raise RuntimeError("Cannot verify a registry without a persistence path")
        root = self._registry_root(self.auto_save_path)
        baseline = snapshot["assets"]
        if not isinstance(baseline, dict) or not set(baseline).issubset(
            {str(asset_id) for asset_id in self._assets}
        ):
            raise RuntimeError("Asset registry removed baseline definitions")
        for asset_id, expected in baseline.items():
            current = self._serialize_asset(
                self._assets[UniqueID(asset_id)], root=root
            )
            if current != expected:
                raise RuntimeError(f"Asset registry mutated baseline definition {asset_id}")
        baseline_assets = {
            UniqueID(asset_id): self._assets[UniqueID(asset_id)] for asset_id in baseline
        }
        if self._asset_file_inventory(baseline_assets, root=root) != snapshot["asset_files"]:
            raise RuntimeError("Asset registry mutated baseline asset files")
        head = snapshot["head"]
        head_keys = {
            "revision",
            "attestation",
            "attestation_history",
            "lineage_root",
            "persisted_file_sha256",
        }
        if (
            not isinstance(head, dict)
            or set(head) != head_keys
            or isinstance(head["revision"], bool)
            or not isinstance(head["revision"], int)
            or head["revision"] < 0
            or not isinstance(head["attestation_history"], list)
            or len(head["attestation_history"]) != max(0, head["revision"] - 1)
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in head["attestation_history"]
            )
            or len(set(head["attestation_history"]))
            != len(head["attestation_history"])
            or (
                head["attestation"] is not None
                and (
                    not isinstance(head["attestation"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", head["attestation"])
                    is None
                )
            )
            or (
                head["lineage_root"] is not None
                and (
                    not isinstance(head["lineage_root"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", head["lineage_root"])
                    is None
                )
            )
            or (
                head["persisted_file_sha256"] is not None
                and (
                    not isinstance(head["persisted_file_sha256"], str)
                    or re.fullmatch(
                        r"[0-9a-f]{64}", head["persisted_file_sha256"]
                    )
                    is None
                )
            )
            or (head["revision"] == 0 and head != self.empty_snapshot()["head"])
            or (head["revision"] > 0 and head["attestation"] is None)
            or (
                head["revision"] > 0
                and head["persisted_file_sha256"] is None
            )
            or self._revision < head["revision"]
            or self._lineage_root != head["lineage_root"]
        ):
            raise RuntimeError("Asset registry snapshot head is malformed")
        if self._revision == head["revision"]:
            if (
                self._previous_attestation != head["attestation"]
                or self._attestation_history != head["attestation_history"]
                or self._persisted_file_sha256
                != head["persisted_file_sha256"]
            ):
                raise RuntimeError("Asset registry snapshot head changed")
        elif head["revision"] > 0:
            # A later additive registry must retain the exact checkpoint head as
            # its predecessor at the checkpoint revision and preserve its prefix.
            if (
                self._attestation_history[: head["revision"] - 1]
                != head["attestation_history"]
                or self._attestation_history[head["revision"] - 1]
                != head["attestation"]
            ):
                raise RuntimeError("Asset registry snapshot head is not an ancestor")
        self.verify_persisted()

    def verify_persisted(self) -> None:
        """Reparse the persisted envelope and match exact in-memory definitions."""
        if self.auto_save_path is None:
            raise RuntimeError("Registry has no persistence path")
        if not self.auto_save_path.exists() and not self.auto_save_path.is_symlink():
            if (
                not self._assets
                and self._revision == 0
                and self._previous_attestation is None
                and self._persisted_file_sha256 is None
            ):
                return
            raise RuntimeError("Nonempty asset registry has no persisted envelope")
        candidate = AssetRegistry(
            auto_save_path=self.auto_save_path,
            required_root=self.required_root,
            allowed_object_types=self.allowed_object_types,
        )
        candidate.load_from_file(self.auto_save_path)
        if (
            candidate.snapshot() != self.snapshot()
            or candidate._revision != self._revision
            or candidate._previous_attestation != self._previous_attestation
            or candidate._attestation_history != self._attestation_history
            or candidate._lineage_root != self._lineage_root
            or candidate._persisted_file_sha256 != self._persisted_file_sha256
            or candidate._legacy_source_b64 != self._legacy_source_b64
        ):
            raise RuntimeError("Persisted asset registry differs from memory")

    def references_immutable_sdf(self, sdf_path: Path) -> bool:
        target = sdf_path.resolve(strict=False)
        return any(
            self._assets[asset_id].sdf_path is not None
            and self._assets[asset_id].sdf_path.resolve(strict=False) == target
            for asset_id in self._immutable_loaded_entries
        )

    def snapshot_sdf_paths(self, snapshot: Any) -> set[Path]:
        """Resolve the SDF files protected by one exact scope snapshot."""
        if (
            not isinstance(snapshot, dict)
            or not isinstance(snapshot.get("assets"), dict)
            or self.auto_save_path is None
        ):
            raise RuntimeError("Asset registry scope snapshot is malformed")
        root = self._registry_root(self.auto_save_path)
        result: set[Path] = set()
        for asset_id, record in snapshot["assets"].items():
            if not isinstance(record, dict):
                raise RuntimeError(f"Registry snapshot entry is malformed: {asset_id}")
            path = self._canonical_asset_path(
                record.get("sdf_path"),
                root=root,
                label=f"registry snapshot SDF {asset_id}",
                suffixes=frozenset({".sdf"}),
            )
            if path is not None:
                result.add(path)
        return result

    def freeze_asset(self, asset_id: UniqueID) -> None:
        """Make a validated reusable definition immutable for this process."""
        asset = self._assets.get(asset_id)
        if asset is None or self.auto_save_path is None:
            raise RuntimeError(f"Cannot freeze missing registry asset: {asset_id}")
        root = self._registry_root(self.auto_save_path)
        self._immutable_loaded_entries[asset_id] = self._serialize_asset(
            asset, root=root
        )
        for record in self._asset_file_inventory({asset_id: asset}, root=root):
            path = root.joinpath(*PurePosixPath(record["path"]).parts)
            self._immutable_loaded_files[path] = (
                record["size_bytes"],
                record["sha256"],
            )

    def register_many_immutable(self, assets: list[SceneObject]) -> None:
        """Validate and atomically persist a batch of checkpoint leaf templates."""
        if not assets:
            return
        ids = [asset.object_id for asset in assets]
        if len(ids) != len(set(ids)) or any(asset_id in self._assets for asset_id in ids):
            raise RuntimeError("Immutable registry batch contains duplicate IDs")
        for asset in assets:
            self._validate_template_shape(asset)
        previous_assets = self._assets
        self._assets = {
            **self._assets,
            **{
                asset.object_id: (
                    copy.deepcopy(asset)
                    if self.allowed_object_types is not None
                    else asset
                )
                for asset in assets
            },
        }
        try:
            if self.auto_save_path is not None:
                self.save_to_file(self.auto_save_path)
        except Exception:
            if not self._last_save_committed:
                self._assets = previous_assets
            else:
                for asset_id in ids:
                    self.freeze_asset(asset_id)
            raise
        for asset_id in ids:
            self.freeze_asset(asset_id)

    def save_to_file(self, file_path: Path) -> None:
        """Atomically save an attested registry without mutating loaded entries."""
        self._last_save_committed = False
        if self.allowed_object_types is None:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                str(asset_id): self._serialize_asset(asset)
                for asset_id, asset in self._assets.items()
            }
            encoded = json.dumps(data, indent=2).encode("utf-8")
            stage = file_path.with_name(
                f".{file_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            )
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                descriptor = os.open(stage, flags, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(stage, file_path)
                self._last_save_committed = True
                _fsync_directory(file_path.parent)
            except Exception:
                if stage.exists():
                    stage.unlink()
                raise
            console_logger.info(f"Saved {len(data)} assets to {file_path}")
            return
        if self.required_root is None:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        elif not file_path.parent.exists():
            raise RuntimeError("Strict asset registry parent is missing")
        if file_path.exists() or file_path.is_symlink():
            self._regular_file_no_links(file_path, label="existing asset registry")
        root = self._registry_root(file_path)
        parent = file_path.parent.resolve(strict=True)
        if (
            not parent.is_relative_to(root)
            or parent.is_symlink()
            or (self.allowed_object_types is not None and file_path.name != "asset_registry.json")
        ):
            raise RuntimeError("Asset registry destination escapes its namespace")
        for asset_id, expected in self._immutable_loaded_entries.items():
            asset = self._assets.get(asset_id)
            if asset is None or self._serialize_asset(asset, root=root) != expected:
                raise RuntimeError(f"Immutable registry entry changed: {asset_id}")
        for path, expected in self._immutable_loaded_files.items():
            resolved = self._regular_file_no_links(
                path, label="immutable loaded registry asset"
            )
            current = (resolved.stat().st_size, _sha256_file(resolved))
            if current != expected:
                raise RuntimeError(f"Immutable registry asset bytes changed: {path}")

        for asset in self._assets.values():
            self._validate_template_shape(asset)

        assets = {
            str(asset_id): self._serialize_asset(asset, root=root)
            for asset_id, asset in sorted(
                self._assets.items(), key=lambda item: str(item[0])
            )
        }
        # Re-deserialize current state before publishing it.  This applies the
        # exact same schema/path/dependency checks used on restart.
        for asset_id, data in assets.items():
            self._deserialize_entry(asset_id, data, root=root)
        next_revision = self._revision + 1
        attestation_history = list(self._attestation_history)
        if next_revision > 1:
            if self._previous_attestation is None:
                raise RuntimeError("Asset registry predecessor is missing")
            attestation_history.append(self._previous_attestation)
        payload = {
            "schema_version": _REGISTRY_SCHEMA_VERSION,
            "revision": next_revision,
            "previous_attestation": self._previous_attestation,
            "attestation_history": attestation_history,
            "lineage_root": self._lineage_root,
            "legacy_source_b64": self._legacy_source_b64,
            "assets": assets,
            "asset_files": self._asset_file_inventory(self._assets, root=root),
        }
        document = {**payload, "attestation": _canonical_payload_sha256(payload)}
        encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
        stage = file_path.with_name(f".{file_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
        if stage.exists() or stage.is_symlink():
            raise RuntimeError(f"Asset registry staging path exists: {stage}")
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(stage, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise
            os.replace(stage, file_path)
            self._last_save_committed = True
            self._revision = next_revision
            self._previous_attestation = document["attestation"]
            self._attestation_history = attestation_history
            self._persisted_file_sha256 = hashlib.sha256(encoded).hexdigest()
            _fsync_directory(file_path.parent)
        except Exception:
            if stage.exists():
                stage.unlink()
            raise
        console_logger.info(f"Saved {len(assets)} assets to {file_path}")

    def load_from_file(self, file_path: Path) -> None:
        """Load a strict attested registry or the one pinned legacy migration."""
        if not file_path.exists():
            raise FileNotFoundError(f"Registry file not found: {file_path}")
        if self.allowed_object_types is None:
            try:
                document = json.loads(
                    file_path.read_text(encoding="utf-8"),
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Asset registry is not valid JSON") from exc
            if not isinstance(document, dict):
                raise RuntimeError("Asset registry mapping is malformed")
            root = file_path.parent.resolve(strict=True)
            loaded: dict[UniqueID, SceneObject] = {}
            for asset_id, data in document.items():
                asset = self._deserialize_entry(asset_id, data, root=root)
                loaded[asset.object_id] = asset
            self._assets = loaded
            console_logger.info(f"Loaded {len(loaded)} assets from {file_path}")
            return
        resolved_registry = self._regular_file_no_links(
            file_path, label="asset registry"
        )
        root = self._registry_root(file_path)
        if not resolved_registry.is_relative_to(root):
            raise RuntimeError("Asset registry file escapes its required namespace")
        raw = resolved_registry.read_bytes()
        try:
            document = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    RuntimeError(f"Asset registry contains {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Asset registry is not strict JSON") from exc
        if not isinstance(document, dict) or not document:
            raise RuntimeError("Asset registry document is malformed")

        legacy = "schema_version" not in document
        if legacy:
            raw_sha = hashlib.sha256(raw).hexdigest()
            if (
                raw_sha != _LEGACY_INTERRUPTED_REGISTRY_SHA256
                or set(document) != _LEGACY_INTERRUPTED_REGISTRY_IDS
            ):
                raise RuntimeError("Unattested legacy asset registry is rejected")
            registry_data = document
            revision = 0
            previous_attestation = raw_sha
            attestation_history: list[str] = []
            lineage_root = raw_sha
            legacy_source_b64 = base64.b64encode(raw).decode("ascii")
        else:
            expected_keys = {
                "schema_version", "revision", "previous_attestation", "assets",
                "attestation_history", "lineage_root", "legacy_source_b64",
                "asset_files", "attestation",
            }
            if (
                set(document) != expected_keys
                or document["schema_version"] != _REGISTRY_SCHEMA_VERSION
                or isinstance(document["revision"], bool)
                or not isinstance(document["revision"], int)
                or document["revision"] < 1
                or (
                    document["lineage_root"] is not None
                    and (
                        not isinstance(document["lineage_root"], str)
                        or re.fullmatch(r"[0-9a-f]{64}", document["lineage_root"])
                        is None
                    )
                )
                or (
                    document["previous_attestation"] is not None
                    and (
                        not isinstance(document["previous_attestation"], str)
                        or re.fullmatch(
                            r"[0-9a-f]{64}", document["previous_attestation"]
                        ) is None
                    )
                )
                or (
                    document["revision"] > 1
                    and document["previous_attestation"] is None
                )
                or not isinstance(document["attestation_history"], list)
                or len(document["attestation_history"])
                != document["revision"] - 1
                or any(
                    not isinstance(item, str)
                    or re.fullmatch(r"[0-9a-f]{64}", item) is None
                    for item in document["attestation_history"]
                )
                or len(set(document["attestation_history"]))
                != len(document["attestation_history"])
                or (
                    document["revision"] > 1
                    and document["previous_attestation"]
                    != document["attestation_history"][-1]
                )
            ):
                raise RuntimeError(
                    "Asset registry envelope lineage/predecessor is malformed"
                )
            payload = {key: value for key, value in document.items() if key != "attestation"}
            if document["attestation"] != _canonical_payload_sha256(payload):
                raise RuntimeError("Asset registry attestation mismatch")
            registry_data = document["assets"]
            revision = document["revision"]
            previous_attestation = document["attestation"]
            attestation_history = list(document["attestation_history"])
            lineage_root = document["lineage_root"]
            legacy_source_b64 = document["legacy_source_b64"]
            if self.allowed_object_types is not None:
                if lineage_root == _LEGACY_INTERRUPTED_REGISTRY_SHA256:
                    if not isinstance(legacy_source_b64, str):
                        raise RuntimeError("Asset registry legacy source is missing")
                    try:
                        legacy_source_raw = base64.b64decode(
                            legacy_source_b64, validate=True
                        )
                    except (ValueError, TypeError) as exc:
                        raise RuntimeError("Asset registry legacy source is invalid") from exc
                    if hashlib.sha256(legacy_source_raw).hexdigest() != lineage_root:
                        raise RuntimeError("Asset registry legacy source hash mismatch")
                    if (
                        revision == 1
                        and document["previous_attestation"]
                        != _LEGACY_INTERRUPTED_REGISTRY_SHA256
                    ):
                        raise RuntimeError("Asset registry legacy lineage is broken")
                elif lineage_root is None:
                    if legacy_source_b64 is not None:
                        raise RuntimeError("Fresh asset registry has a legacy source")
                    # Object IDs are room-local semantic names, not globally
                    # reserved identifiers.  A freshly retrieved asset may
                    # legitimately be called ``binder_clip_0``.  Reject only an
                    # exact legacy baseline entry that lost its lineage envelope,
                    # not an unrelated fresh asset with the same local ID.
                    matching_legacy_ids = {
                        asset_id
                        for asset_id in _LEGACY_INTERRUPTED_REGISTRY_IDS.intersection(
                            registry_data
                        )
                        if _canonical_payload_sha256(registry_data[asset_id])
                        == _LEGACY_INTERRUPTED_ENTRY_SHA256[asset_id]
                    }
                    if matching_legacy_ids or (
                        revision == 1
                        and document["previous_attestation"] is not None
                    ):
                        raise RuntimeError("Fresh asset registry lineage is malformed")
                else:
                    raise RuntimeError("Asset registry lineage root is unauthorized")

        if not isinstance(registry_data, dict) or not registry_data:
            raise RuntimeError("Asset registry has no exact asset mapping")
        loaded: dict[UniqueID, SceneObject] = {}
        for asset_id, asset_data in registry_data.items():
            scene_object = self._deserialize_entry(
                asset_id,
                asset_data,
                root=root,
                allow_absolute_paths=legacy,
            )
            if scene_object.object_id in loaded:
                raise RuntimeError(f"Duplicate registry asset ID: {scene_object.object_id}")
            loaded[scene_object.object_id] = scene_object
        inventory = self._asset_file_inventory(loaded, root=root)
        if not legacy:
            if document["asset_files"] != inventory:
                raise RuntimeError("Asset registry file inventory mismatch")
        if legacy or (
            self.allowed_object_types is not None
            and lineage_root == _LEGACY_INTERRUPTED_REGISTRY_SHA256
        ):
            if not _LEGACY_INTERRUPTED_REGISTRY_IDS.issubset(
                {str(asset_id) for asset_id in loaded}
            ):
                raise RuntimeError("Asset registry lost legacy baseline definitions")
            normalized_baseline = {
                asset_id: self._serialize_asset(
                    loaded[UniqueID(asset_id)], root=root
                )
                for asset_id in sorted(_LEGACY_INTERRUPTED_REGISTRY_IDS)
            }
            for asset_id, normalized in normalized_baseline.items():
                if (
                    _canonical_payload_sha256(normalized)
                    != _LEGACY_INTERRUPTED_ENTRY_SHA256[asset_id]
                ):
                    raise RuntimeError(
                        f"Legacy registry baseline semantics changed: {asset_id}"
                    )
            if (
                _canonical_payload_sha256(normalized_baseline)
                != _LEGACY_INTERRUPTED_AGGREGATE_SHA256
            ):
                raise RuntimeError("Legacy registry aggregate semantics changed")
            for asset_id, asset in loaded.items():
                if str(asset_id) not in _LEGACY_INTERRUPTED_REGISTRY_IDS:
                    continue
                if asset.sdf_path is None:
                    raise RuntimeError("Legacy registry asset lacks SDF")
                if self._legacy_directory_manifest(asset.sdf_path.parent, root=root) != (
                    _LEGACY_INTERRUPTED_DIRECTORY_MANIFESTS[str(asset_id)]
                ):
                    raise RuntimeError(
                        f"Legacy registry asset directory mismatch: {asset_id}"
                    )

        loaded_entries = {
            asset_id: self._serialize_asset(asset, root=root)
            for asset_id, asset in loaded.items()
        }
        loaded_files = {
            root.joinpath(*PurePosixPath(record["path"]).parts): (
                record["size_bytes"], record["sha256"]
            )
            for record in inventory
        }
        previous_state = (
            self._assets,
            self._revision,
            self._previous_attestation,
            self._attestation_history,
            self._lineage_root,
            self._legacy_source_b64,
            self._immutable_loaded_entries,
            self._immutable_loaded_files,
            self._persisted_file_sha256,
        )
        self._assets = loaded
        self._revision = revision
        self._previous_attestation = previous_attestation
        self._attestation_history = attestation_history
        self._lineage_root = lineage_root
        self._legacy_source_b64 = legacy_source_b64
        self._immutable_loaded_entries = loaded_entries
        self._immutable_loaded_files = loaded_files
        if legacy:
            # Convert the one exact interrupted-run mapping immediately into the
            # attested atomic format; asset definitions remain immutable.
            try:
                self.save_to_file(file_path)
            except Exception:
                if not self._last_save_committed:
                    (
                        self._assets,
                        self._revision,
                        self._previous_attestation,
                        self._attestation_history,
                        self._lineage_root,
                        self._legacy_source_b64,
                        self._immutable_loaded_entries,
                        self._immutable_loaded_files,
                        self._persisted_file_sha256,
                    ) = previous_state
                raise
        else:
            self._persisted_file_sha256 = hashlib.sha256(raw).hexdigest()
        console_logger.info(f"Loaded {len(self._assets)} assets from {file_path}")
