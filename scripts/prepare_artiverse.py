#!/usr/bin/env python3
"""Prepare the official Artiverse release for SceneSmith articulated retrieval.

The official release is arranged as ``data/<category>/<source>/<model_id>`` and
ships simulator URDFs plus reference renders. SceneSmith retrieves articulated
assets from Drake-compatible SDF files and a CLIP embedding index. This script:

1. discovers Artiverse model folders;
2. converts each simulator URDF to SDF next to the URDF, preserving relative
   mesh references and the supplied collision geometry;
3. computes a CLIP embedding from the supplied reference renders;
4. computes the closed-pose bounding box from the converted SDF; and
5. writes the three index files consumed by SceneSmith's generic multi-source
   articulated loader.

It is resumable and installs nothing. Download and unpack the access-gated
Artiverse dataset before running it.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import stat
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
import yaml

try:
    from .artiverse_contract import (
        OFFICIAL_PACK_SCRIPT_SHA256,
        OFFICIAL_REPOSITORY,
        OFFICIAL_REVISION,
        OFFICIAL_SOURCE_MANIFEST_SHA256,
        emitted_sdf_physics_evidence,
        physics_binding_sha256,
        publisher_urdf_physics_evidence,
        require_regular_file_within,
        sha256_directory_tree,
        sha256_file,
        validate_extraction_receipt,
        validate_regular_directory_tree,
        validate_sdf_resource_uris,
    )
except ImportError:  # Direct execution: python scripts/prepare_artiverse.py
    from artiverse_contract import (  # type: ignore[no-redef]
        OFFICIAL_PACK_SCRIPT_SHA256,
        OFFICIAL_REPOSITORY,
        OFFICIAL_REVISION,
        OFFICIAL_SOURCE_MANIFEST_SHA256,
        emitted_sdf_physics_evidence,
        physics_binding_sha256,
        publisher_urdf_physics_evidence,
        require_regular_file_within,
        sha256_directory_tree,
        sha256_file,
        validate_extraction_receipt,
        validate_regular_directory_tree,
        validate_sdf_resource_uris,
    )
from scenesmith.agent_utils.clip_embeddings import get_multiview_image_embedding
from scenesmith.agent_utils.sdf_mesh_utils import combine_sdf_meshes_at_joint_angles
from scenesmith.agent_utils.urdf_to_sdf import LinkPhysics, convert_urdf_to_sdf


LOGGER = logging.getLogger("prepare_artiverse")
INDEX_FILENAMES = (
    "clip_embeddings.npy",
    "embedding_index.yaml",
    "metadata_index.yaml",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MOVABLE_JOINT_TYPES = frozenset({"continuous", "prismatic", "revolute"})
DERIVED_SDF_FILENAME = "scenesmith_artiverse.sdf"
DERIVED_SDF_TEMP_FILENAME = f".{DERIVED_SDF_FILENAME}.tmp"
PHYSICS_POLICY = {
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


@dataclass(frozen=True)
class PublisherPhysicsContract:
    """Validated, deterministic physics facts bound to one publisher URDF."""

    link_physics: dict[str, LinkPhysics]
    link_names: tuple[str, ...]
    geometry_link_names: tuple[str, ...]
    collision_count_by_link: dict[str, int]
    source_collision_element_count: int
    source_movable_joint_count: int
    total_mass_kg: float
    publisher_link_physics_sha256: str


@dataclass(frozen=True)
class ArtiverseAsset:
    category: str
    source: str
    model_id: str
    model_dir: Path
    urdf_path: Path

    @property
    def object_id(self) -> str:
        return f"artiverse/{self.category}/{self.source}/{self.model_id}"


def _data_root(dataset_root: Path) -> Path:
    nested = dataset_root / "data"
    return nested if nested.is_dir() else dataset_root


def _select_urdf(model_dir: Path) -> Path | None:
    urdf_dir = model_dir / "urdf_w_collider"
    if not urdf_dir.is_dir():
        return None
    candidates = [
        path
        for path in sorted(urdf_dir.rglob("*.urdf"))
        if "repaired" not in path.name and "scenesmith" not in path.name
    ]
    if not candidates:
        return None
    preferred = [
        path
        for path in candidates
        if path.stem.lower() in {"mobility", "model", "object"}
    ]
    return preferred[0] if preferred else candidates[0]


def discover_assets(
    dataset_root: Path, categories: set[str] | None = None
) -> list[ArtiverseAsset]:
    root = _data_root(dataset_root)
    assets: list[ArtiverseAsset] = []
    if not root.is_dir():
        return assets
    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        if categories and category_dir.name.lower() not in categories:
            continue
        for source_dir in sorted(category_dir.iterdir()):
            if not source_dir.is_dir() or source_dir.name.startswith("."):
                continue
            for model_dir in sorted(source_dir.iterdir()):
                if not model_dir.is_dir() or model_dir.name.startswith("."):
                    continue
                urdf_path = _select_urdf(model_dir)
                if urdf_path is None:
                    continue
                assets.append(
                    ArtiverseAsset(
                        category=category_dir.name,
                        source=source_dir.name,
                        model_id=model_dir.name,
                        model_dir=model_dir,
                        urdf_path=urdf_path,
                    )
                )
    return assets


def _reference_images(asset: ArtiverseAsset, max_images: int) -> list[Path]:
    image_dir = asset.model_dir / "imgs"
    if not image_dir.is_dir():
        return []
    images = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return images[:max_images]


def _collision_count(sdf_path: Path) -> int:
    return len(ET.parse(sdf_path).getroot().findall(".//collision"))


def _local_name(element: ET.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def _direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local_name(child) == name]


def _exact_child(element: ET.Element, name: str, label: str) -> ET.Element:
    children = _direct_children(element, name)
    if len(children) != 1:
        raise ValueError(f"{label} must contain exactly one <{name}> element")
    return children[0]


def _finite_vector(raw: str | None, length: int, label: str) -> tuple[float, ...]:
    if raw is None:
        raise ValueError(f"{label} is missing")
    fields = raw.split()
    if len(fields) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    try:
        values = tuple(float(field) for field in fields)
    except ValueError as exc:
        raise ValueError(f"{label} contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} contains a non-finite value")
    return values


def _finite_float(raw: str | None, label: str) -> float:
    if raw is None:
        raise ValueError(f"{label} is missing")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} is non-finite")
    return value


def _validate_inertia_tensor(
    terms: dict[str, float], label: str
) -> np.ndarray:
    tensor = np.asarray(
        [
            [terms["ixx"], terms["ixy"], terms["ixz"]],
            [terms["ixy"], terms["iyy"], terms["iyz"]],
            [terms["ixz"], terms["iyz"], terms["izz"]],
        ],
        dtype=np.float64,
    )
    if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError(f"{label} is not a finite symmetric 3x3 tensor")
    if not np.array_equal(tensor, tensor.T):
        raise ValueError(f"{label} is not symmetric")
    eigenvalues = np.linalg.eigvalsh(tensor)
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[0]) <= 0.0:
        raise ValueError(f"{label} is not positive definite")
    tolerance = 1e-9 * max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    for index in range(3):
        other = [value for ordinal, value in enumerate(eigenvalues) if ordinal != index]
        if float(other[0] + other[1]) + tolerance < float(eigenvalues[index]):
            raise ValueError(f"{label} violates the principal-moment triangle inequality")
    return tensor


def _urdf_rpy_rotation(rpy: tuple[float, ...]) -> np.ndarray:
    """Return URDF's fixed-axis RPY rotation as ``Rz(yaw) Ry(pitch) Rx(roll)``."""

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _publisher_physics_contract(
    urdf_path: Path,
    *,
    max_collision_elements: int,
    require_movable_joint: bool = False,
) -> PublisherPhysicsContract:
    """Strictly bind every publisher link's inertia and collision inventory."""

    document = ET.parse(urdf_path).getroot()
    if _local_name(document) != "robot":
        raise ValueError(f"Publisher URDF root is not <robot>: {urdf_path}")
    links = _direct_children(document, "link")
    if not links:
        raise ValueError(f"Publisher URDF has no links: {urdf_path}")

    link_physics: dict[str, LinkPhysics] = {}
    geometry_link_names: list[str] = []
    collision_count_by_link: dict[str, int] = {}
    link_names: list[str] = []
    total_mass_kg = 0.0

    for link in links:
        link_name = str(link.get("name", "")).strip()
        if not link_name:
            raise ValueError("Publisher URDF link has no non-empty name")
        if link_name in link_physics:
            raise ValueError(f"Publisher URDF repeats link name: {link_name}")

        inertial = _exact_child(link, "inertial", f"Publisher link {link_name}")
        origins = _direct_children(inertial, "origin")
        if len(origins) > 1:
            raise ValueError(
                f"Publisher link {link_name} inertial contains multiple <origin> elements"
            )
        origin = origins[0] if origins else None
        center_of_mass = _finite_vector(
            origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
            3,
            f"Publisher link {link_name} inertial xyz",
        )
        inertial_rpy = _finite_vector(
            origin.get("rpy", "0 0 0") if origin is not None else "0 0 0",
            3,
            f"Publisher link {link_name} inertial rpy",
        )
        mass_element = _exact_child(
            inertial, "mass", f"Publisher link {link_name} inertial"
        )
        mass = _finite_float(
            mass_element.get("value"), f"Publisher link {link_name} mass"
        )
        if mass <= 0.0:
            raise ValueError(f"Publisher link {link_name} mass must be positive")

        inertia_element = _exact_child(
            inertial, "inertia", f"Publisher link {link_name} inertial"
        )
        terms = {
            term: _finite_float(
                inertia_element.get(term),
                f"Publisher link {link_name} inertia {term}",
            )
            for term in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        }
        source_tensor = _validate_inertia_tensor(
            terms, f"Publisher link {link_name} inertia"
        )
        rotation = _urdf_rpy_rotation(inertial_rpy)
        rotated_tensor = rotation @ source_tensor @ rotation.T
        rotated_tensor = (rotated_tensor + rotated_tensor.T) / 2.0
        rotated_terms = {
            "ixx": float(rotated_tensor[0, 0]),
            "iyy": float(rotated_tensor[1, 1]),
            "izz": float(rotated_tensor[2, 2]),
            "ixy": float(rotated_tensor[0, 1]),
            "ixz": float(rotated_tensor[0, 2]),
            "iyz": float(rotated_tensor[1, 2]),
        }
        _validate_inertia_tensor(
            rotated_terms,
            f"Publisher link {link_name} link-frame inertia",
        )
        link_physics[link_name] = LinkPhysics(
            mass=mass,
            inertia_ixx=rotated_terms["ixx"],
            inertia_iyy=rotated_terms["iyy"],
            inertia_izz=rotated_terms["izz"],
            inertia_ixy=rotated_terms["ixy"],
            inertia_ixz=rotated_terms["ixz"],
            inertia_iyz=rotated_terms["iyz"],
            center_of_mass=center_of_mass,
        )
        link_names.append(link_name)
        total_mass_kg += mass

        has_geometry = bool(
            _direct_children(link, "visual") or _direct_children(link, "collision")
        )
        collision_count = len(_direct_children(link, "collision"))
        collision_count_by_link[link_name] = collision_count
        if has_geometry:
            geometry_link_names.append(link_name)
            if collision_count < 1:
                raise ValueError(
                    f"Publisher geometry link {link_name} has no source collision"
                )

    source_collision_element_count = sum(collision_count_by_link.values())
    if not 1 <= source_collision_element_count <= max_collision_elements:
        raise ValueError(
            "Publisher URDF source collision count is outside "
            f"1..{max_collision_elements}: {source_collision_element_count}"
        )
    source_movable_joint_count = sum(
        str(joint.get("type", "")).strip().lower() in MOVABLE_JOINT_TYPES
        for joint in _direct_children(document, "joint")
    )
    if require_movable_joint and source_movable_joint_count < 1:
        raise ValueError("Publisher URDF contains no usable movable joint")
    publisher_evidence = publisher_urdf_physics_evidence(urdf_path)
    local_components = {
        name: {
            "mass": float(physics.mass),
            "pose_xyz": tuple(float(value) for value in physics.center_of_mass),
            "inertia": {
                "ixx": float(physics.inertia_ixx),
                "iyy": float(physics.inertia_iyy),
                "izz": float(physics.inertia_izz),
                "ixy": float(physics.inertia_ixy),
                "ixz": float(physics.inertia_ixz),
                "iyz": float(physics.inertia_iyz),
            },
        }
        for name, physics in link_physics.items()
    }
    if (
        set(publisher_evidence.link_values) != set(link_physics)
        or publisher_evidence.collision_count_by_link != collision_count_by_link
        or publisher_evidence.geometry_links != frozenset(geometry_link_names)
        or not math.isclose(
            publisher_evidence.total_mass_kg,
            math.fsum(float(value.mass) for value in link_physics.values()),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("Publisher physics digest disagrees with converter inputs")
    for name, actual in local_components.items():
        expected = publisher_evidence.link_values[name]
        scalar_pairs = [(actual["mass"], expected["mass"])]
        scalar_pairs.extend(zip(actual["pose_xyz"], expected["pose_xyz"], strict=True))
        scalar_pairs.extend(
            (actual["inertia"][term], expected["inertia"][term])
            for term in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        )
        if any(
            not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
            for left, right in scalar_pairs
        ):
            raise RuntimeError(
                f"Publisher physics transform disagrees for link {name}"
            )
    return PublisherPhysicsContract(
        link_physics=link_physics,
        link_names=tuple(link_names),
        geometry_link_names=tuple(geometry_link_names),
        collision_count_by_link=collision_count_by_link,
        source_collision_element_count=source_collision_element_count,
        source_movable_joint_count=source_movable_joint_count,
        total_mass_kg=math.fsum(float(value.mass) for value in link_physics.values()),
        publisher_link_physics_sha256=publisher_evidence.sha256,
    )


def _close_converter_value(
    actual: float, expected: float, *, absolute_tolerance: float
) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=1e-6,
        abs_tol=absolute_tolerance,
    )


def _validate_sdf_physics_and_collisions(
    sdf_path: Path, contract: PublisherPhysicsContract
) -> None:
    """Require a converted SDF to preserve all publisher physics and collisions."""

    root = ET.parse(sdf_path).getroot()
    links = [element for element in root.iter() if _local_name(element) == "link"]
    sdf_by_name: dict[str, ET.Element] = {}
    for link in links:
        link_name = str(link.get("name", "")).strip()
        if not link_name:
            raise ValueError("Converted SDF link has no non-empty name")
        if link_name in sdf_by_name:
            raise ValueError(f"Converted SDF repeats link name: {link_name}")
        sdf_by_name[link_name] = link
    if set(sdf_by_name) != set(contract.link_names):
        missing = sorted(set(contract.link_names) - set(sdf_by_name))
        extra = sorted(set(sdf_by_name) - set(contract.link_names))
        raise ValueError(
            f"Converted SDF link set differs from publisher (missing={missing}, extra={extra})"
        )

    expected_geometry_links = set(contract.geometry_link_names)
    actual_geometry_links: set[str] = set()
    total_collisions = 0
    for link_name in contract.link_names:
        link = sdf_by_name[link_name]
        expected = contract.link_physics[link_name]
        inertial = _exact_child(link, "inertial", f"Converted SDF link {link_name}")
        mass_element = _exact_child(
            inertial, "mass", f"Converted SDF link {link_name} inertial"
        )
        mass = _finite_float(
            (mass_element.text or "").strip(), f"Converted SDF link {link_name} mass"
        )
        if mass <= 0.0 or not _close_converter_value(
            mass, float(expected.mass), absolute_tolerance=5.1e-7
        ):
            raise ValueError(
                f"Converted SDF link {link_name} mass does not match publisher"
            )

        pose = _exact_child(
            inertial, "pose", f"Converted SDF link {link_name} inertial"
        )
        pose_values = _finite_vector(
            (pose.text or "").strip(), 6, f"Converted SDF link {link_name} inertial pose"
        )
        if any(value != 0.0 for value in pose_values[3:]):
            raise ValueError(f"Converted SDF link {link_name} inertial rpy is nonzero")
        if any(
            not _close_converter_value(actual, expected_value, absolute_tolerance=5.1e-7)
            for actual, expected_value in zip(
                pose_values[:3], expected.center_of_mass, strict=True
            )
        ):
            raise ValueError(
                f"Converted SDF link {link_name} center of mass does not match publisher"
            )

        inertia = _exact_child(
            inertial, "inertia", f"Converted SDF link {link_name} inertial"
        )
        actual_terms = {}
        for term in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
            element = _exact_child(
                inertia, term, f"Converted SDF link {link_name} inertia"
            )
            actual_terms[term] = _finite_float(
                (element.text or "").strip(),
                f"Converted SDF link {link_name} inertia {term}",
            )
        _validate_inertia_tensor(actual_terms, f"Converted SDF link {link_name} inertia")
        expected_terms = {
            "ixx": expected.inertia_ixx,
            "iyy": expected.inertia_iyy,
            "izz": expected.inertia_izz,
            "ixy": expected.inertia_ixy,
            "ixz": expected.inertia_ixz,
            "iyz": expected.inertia_iyz,
        }
        if any(
            not _close_converter_value(
                actual_terms[term], float(expected_terms[term]), absolute_tolerance=1e-12
            )
            for term in actual_terms
        ):
            raise ValueError(
                f"Converted SDF link {link_name} inertia does not match publisher"
            )

        visuals = _direct_children(link, "visual")
        collisions = _direct_children(link, "collision")
        if visuals or collisions:
            actual_geometry_links.add(link_name)
        expected_collision_count = contract.collision_count_by_link[link_name]
        if len(collisions) != expected_collision_count:
            raise ValueError(
                f"Converted SDF link {link_name} collision count does not match publisher: "
                f"expected {expected_collision_count}, got {len(collisions)}"
            )
        if link_name in expected_geometry_links and not collisions:
            raise ValueError(
                f"Converted SDF geometry link {link_name} has no collision"
            )
        total_collisions += len(collisions)

    if actual_geometry_links != expected_geometry_links:
        raise ValueError("Converted SDF geometry-link set differs from publisher")
    if total_collisions != contract.source_collision_element_count:
        raise ValueError(
            "Converted SDF total collision count does not match publisher: "
            f"expected {contract.source_collision_element_count}, got {total_collisions}"
        )


def _physics_metadata(
    contract: PublisherPhysicsContract,
    *,
    dataset_root: Path,
    publisher_urdf: Path,
    emitted_sdf: Path,
) -> dict[str, Any]:
    publisher_relative = publisher_urdf.resolve(strict=True).relative_to(dataset_root)
    publisher_evidence = publisher_urdf_physics_evidence(publisher_urdf)
    if publisher_evidence.sha256 != contract.publisher_link_physics_sha256:
        raise RuntimeError("Publisher URDF physics changed during preparation")
    emitted_evidence = emitted_sdf_physics_evidence(emitted_sdf)
    return {
        "physics_source": PHYSICS_POLICY["id"],
        "publisher_urdf_path": publisher_relative.as_posix(),
        "publisher_urdf_sha256": sha256_file(publisher_urdf),
        "publisher_link_physics_sha256": contract.publisher_link_physics_sha256,
        "emitted_sdf_physics_sha256": emitted_evidence.sha256,
        "physics_link_count": len(contract.link_names),
        "physics_geometry_link_count": len(contract.geometry_link_names),
        "physics_total_mass_kg": float(contract.total_mass_kg),
        "source_collision_element_count": contract.source_collision_element_count,
    }


def _movable_joint_summary(sdf_path: Path) -> tuple[int, list[str]]:
    """Return the count and distinct types of usable movable SDF joints."""

    root = ET.parse(sdf_path).getroot()
    movable_types = [
        str(element.get("type", "")).strip().lower()
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "joint"
        and str(element.get("type", "")).strip().lower() in MOVABLE_JOINT_TYPES
    ]
    return len(movable_types), sorted(set(movable_types))


def _manifest_root_category_counts(source_manifest_path: Path) -> dict[str, int]:
    """Count canonical model roots per category in the pinned source manifest."""

    try:
        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Official Artiverse source manifest is not valid JSON") from exc
    chunks = manifest.get("chunks") if isinstance(manifest, dict) else None
    if not isinstance(chunks, list) or not chunks:
        raise RuntimeError("Official Artiverse source manifest has no chunk roots")

    counts: dict[str, int] = {}
    seen_roots: set[str] = set()
    for chunk in chunks:
        roots = chunk.get("roots") if isinstance(chunk, dict) else None
        if not isinstance(roots, list):
            raise RuntimeError("Official Artiverse source manifest has invalid chunk roots")
        for root in roots:
            components = root.split("/") if isinstance(root, str) else []
            if (
                len(components) != 4
                or components[0] != "data"
                or any(
                    not value
                    or value in {".", ".."}
                    or Path(value).name != value
                    for value in components[1:]
                )
            ):
                raise RuntimeError(
                    f"Official Artiverse source manifest has a non-canonical root: {root!r}"
                )
            if root in seen_roots:
                raise RuntimeError(
                    f"Official Artiverse source manifest repeats model root: {root}"
                )
            seen_roots.add(root)
            category = components[1].lower()
            counts[category] = counts.get(category, 0) + 1
    if not counts:
        raise RuntimeError("Official Artiverse source manifest has no model roots")
    return counts


def _description(asset: ArtiverseAsset) -> str:
    category = asset.category.replace("_", " ").replace("-", " ").strip()
    return f"Articulated {category} with functional doors or drawers"


def _sdf_transaction_path(sdf_path: Path) -> Path:
    return sdf_path.with_name(DERIVED_SDF_TEMP_FILENAME)


def _require_unlinked_regular_file(path: Path, root: Path, label: str) -> Path:
    """Require an ordinary, single-link file beneath ``root``."""

    canonical = require_regular_file_within(path, root, label)
    file_stat = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(file_stat.st_mode):  # defensive after the shared helper
        raise RuntimeError(f"{label} is not a regular file: {path}")
    if file_stat.st_nlink != 1:
        raise RuntimeError(f"{label} is hard-linked: {path}")
    return canonical


def _validate_publisher_asset(
    asset: ArtiverseAsset,
    dataset_root: Path,
    sdf_path: Path,
    *,
    existing_sdf_is_bound: bool = False,
    recover_unindexed_sdf: bool = False,
) -> dict[str, tuple[int, int, int, int, int, int]]:
    """Reject publisher path indirection before invoking the converter."""

    publisher_files = validate_regular_directory_tree(
        asset.model_dir,
        containment_root=dataset_root,
        label=f"Artiverse publisher model tree for {asset.object_id}",
    )
    require_regular_file_within(
        asset.urdf_path,
        asset.model_dir,
        f"Artiverse publisher URDF for {asset.object_id}",
    )
    if sdf_path.is_symlink():
        raise RuntimeError(
            f"Artiverse converted SDF path is a symlink: {asset.object_id}"
        )
    lexical_sdf = Path(os.path.abspath(os.fspath(sdf_path)))
    lexical_model = Path(os.path.abspath(os.fspath(asset.model_dir)))
    try:
        lexical_sdf.relative_to(lexical_model)
        sdf_path.resolve(strict=False).relative_to(
            asset.model_dir.resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Artiverse converted SDF path escapes its publisher model: {asset.object_id}"
        ) from exc
    transaction_path = _sdf_transaction_path(sdf_path)
    if sdf_path.exists() or sdf_path.is_symlink():
        _require_unlinked_regular_file(
            sdf_path,
            asset.model_dir,
            f"Existing Artiverse SDF for {asset.object_id}",
        )
        if not existing_sdf_is_bound and not recover_unindexed_sdf:
            raise RuntimeError(
                "Refusing unindexed preexisting Artiverse SDF for "
                f"{asset.object_id}: {sdf_path}"
            )
    if transaction_path.exists() or transaction_path.is_symlink():
        _require_unlinked_regular_file(
            transaction_path,
            asset.model_dir,
            f"Artiverse SDF transaction for {asset.object_id}",
        )
    resolved_model_dir = asset.model_dir.resolve(strict=True)
    derived_paths = {
        Path(os.path.abspath(os.fspath(sdf_path))),
        Path(os.path.abspath(os.fspath(transaction_path))),
    }
    publisher_state: dict[str, tuple[int, int, int, int, int, int]] = {}
    for path in publisher_files:
        if Path(os.path.abspath(os.fspath(path))) in derived_paths:
            continue
        file_stat = path.stat(follow_symlinks=False)
        if file_stat.st_nlink != 1:
            raise RuntimeError(
                f"Artiverse publisher file is hard-linked for {asset.object_id}: {path}"
            )
        publisher_state[path.relative_to(resolved_model_dir).as_posix()] = (
            file_stat.st_mode,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
            file_stat.st_dev,
            file_stat.st_ino,
        )
    return publisher_state


def _rewrite_missing_visual_gltf_uris(
    sdf_path: Path, model_dir: Path, *, atomic: bool = True
) -> tuple[tuple[str, str], ...]:
    """Redirect missing visual GLTFs to existing same-path OBJ files only."""

    document = ET.parse(sdf_path)
    rewrites: list[tuple[str, str]] = []
    for visual in document.getroot().iter():
        if str(visual.tag).rsplit("}", 1)[-1] != "visual":
            continue
        for element in visual.iter():
            if str(element.tag).rsplit("}", 1)[-1] != "uri":
                continue
            raw_uri = (element.text or "").strip()
            parsed = urlsplit(raw_uri)
            if (
                not raw_uri
                or parsed.scheme
                or parsed.netloc
                or parsed.query
                or parsed.fragment
                or not parsed.path.lower().endswith(".gltf")
            ):
                continue
            decoded_path = unquote(parsed.path)
            if "\\" in decoded_path:
                continue
            gltf_relative = PurePosixPath(decoded_path)
            if gltf_relative.is_absolute() or gltf_relative.suffix.lower() != ".gltf":
                continue
            gltf_path = sdf_path.parent.joinpath(*gltf_relative.parts)
            if gltf_path.is_file():
                continue
            obj_relative = gltf_relative.with_suffix(".obj")
            obj_path = sdf_path.parent.joinpath(*obj_relative.parts)
            if not obj_path.is_file():
                continue
            require_regular_file_within(
                obj_path,
                model_dir,
                f"Artiverse visual OBJ fallback for {sdf_path.name}",
            )
            replacement_uri = parsed.path[:-5] + ".obj"
            element.text = replacement_uri
            rewrites.append((raw_uri, replacement_uri))

    if rewrites:
        serialized = ET.tostring(document.getroot(), encoding="unicode")
        if atomic:
            _atomic_write_text(sdf_path, serialized)
        else:
            # ``sdf_path`` is itself the unpublished transaction. A crash can
            # leave it incomplete, but it never replaces the validated final
            # SDF and the next run deterministically regenerates it.
            sdf_path.write_text(serialized, encoding="utf-8")
    return tuple(rewrites)


def _validate_converted_asset(
    asset: ArtiverseAsset,
    dataset_root: Path,
    sdf_path: Path,
    *,
    publisher_files_before: dict[str, tuple[int, int, int, int, int, int]] | None = None,
    rewrite_missing_visual_gltf: bool = False,
) -> Path:
    """Revalidate converter output and all referenced resources."""

    canonical_sdf = _require_unlinked_regular_file(
        sdf_path,
        asset.model_dir,
        f"Converted Artiverse SDF for {asset.object_id}",
    )
    converted_files = validate_regular_directory_tree(
        asset.model_dir,
        containment_root=dataset_root,
        label=f"Converted Artiverse model tree for {asset.object_id}",
    )
    if publisher_files_before is not None:
        resolved_model_dir = asset.model_dir.resolve(strict=True)
        final_sdf = asset.urdf_path.with_name(DERIVED_SDF_FILENAME)
        derived_relative_paths = {
            Path(os.path.abspath(os.fspath(path))).relative_to(
                Path(os.path.abspath(os.fspath(asset.model_dir)))
            ).as_posix()
            for path in (final_sdf, _sdf_transaction_path(final_sdf))
        }
        canonical_relative_path = canonical_sdf.relative_to(resolved_model_dir).as_posix()
        converted_by_relative_path = {
            path.relative_to(resolved_model_dir).as_posix(): path
            for path in converted_files
            if path.relative_to(resolved_model_dir).as_posix()
            not in derived_relative_paths
        }
        converted_by_relative_path[canonical_relative_path] = canonical_sdf
        converted_relative_paths = set(converted_by_relative_path)
        expected_relative_paths = set(publisher_files_before)
        expected_relative_paths.add(canonical_relative_path)
        mutated = []
        for relative_path, expected_state in publisher_files_before.items():
            current_path = converted_by_relative_path.get(relative_path)
            if current_path is None:
                continue
            file_stat = current_path.stat(follow_symlinks=False)
            current_state = (
                file_stat.st_mode,
                file_stat.st_size,
                file_stat.st_mtime_ns,
                file_stat.st_ctime_ns,
                file_stat.st_dev,
                file_stat.st_ino,
            )
            if current_state != expected_state:
                mutated.append(relative_path)
        if converted_relative_paths != expected_relative_paths or mutated:
            added = sorted(converted_relative_paths - expected_relative_paths)
            removed = sorted(expected_relative_paths - converted_relative_paths)
            raise RuntimeError(
                "Artiverse conversion wrote outside its single derived SDF contract "
                f"for {asset.object_id}: added={added}, removed={removed}, "
                f"mutated={sorted(mutated)}"
            )
    if rewrite_missing_visual_gltf:
        _rewrite_missing_visual_gltf_uris(
            canonical_sdf,
            asset.model_dir,
            atomic=canonical_sdf.name != DERIVED_SDF_TEMP_FILENAME,
        )
    validate_sdf_resource_uris(canonical_sdf, asset.model_dir)
    return canonical_sdf


def _convert_or_recover_sdf(
    asset: ArtiverseAsset,
    dataset_root: Path,
    sdf_path: Path,
    publisher_files_before: dict[str, tuple[int, int, int, int, int, int]],
    physics_contract: PublisherPhysicsContract,
) -> Path:
    """Atomically create an SDF or deterministically recover an orphaned one.

    A valid final SDF with no index row can only be adopted after a fresh
    conversion into the fixed transaction path produces the exact same bytes.
    This makes a crash between final-SDF publication and index checkpointing
    resumable without trusting or deleting an unexplained file.
    """

    transaction_path = _sdf_transaction_path(sdf_path)
    existing_sdf: Path | None = None
    existing_sha256: str | None = None
    if sdf_path.exists() or sdf_path.is_symlink():
        existing_sdf = _validate_converted_asset(
            asset,
            dataset_root,
            sdf_path,
            publisher_files_before=publisher_files_before,
        )
        _validate_sdf_physics_and_collisions(existing_sdf, physics_contract)
        existing_sha256 = sha256_file(existing_sdf)

    if transaction_path.exists() or transaction_path.is_symlink():
        _require_unlinked_regular_file(
            transaction_path,
            asset.model_dir,
            f"Stale Artiverse SDF transaction for {asset.object_id}",
        )
        transaction_path.unlink()

    try:
        convert_urdf_to_sdf(
            urdf_path=asset.urdf_path,
            output_path=transaction_path,
            model_name=f"artiverse_{asset.model_id}",
            link_physics=physics_contract.link_physics,
            repair_missing_meshes=False,
            generate_collision=False,
            merge_visuals=False,
        )
        candidate = _validate_converted_asset(
            asset,
            dataset_root,
            transaction_path,
            publisher_files_before=publisher_files_before,
            rewrite_missing_visual_gltf=True,
        )
        # Rewriting an unpublished transaction in place is safe; validate once
        # more after the rewrite before comparing or publishing it.
        candidate = _validate_converted_asset(
            asset,
            dataset_root,
            candidate,
            publisher_files_before=publisher_files_before,
        )
        _validate_sdf_physics_and_collisions(candidate, physics_contract)
        candidate_sha256 = sha256_file(candidate)
        if existing_sdf is not None and candidate_sha256 != existing_sha256:
            raise RuntimeError(
                "Unindexed Artiverse SDF does not match deterministic regeneration "
                f"for {asset.object_id}: existing={existing_sha256} "
                f"regenerated={candidate_sha256}"
            )
    except Exception:
        if transaction_path.exists() or transaction_path.is_symlink():
            _require_unlinked_regular_file(
                transaction_path,
                asset.model_dir,
                f"Rejected Artiverse SDF transaction for {asset.object_id}",
            )
            transaction_path.unlink()
        raise

    os.replace(candidate, sdf_path)
    final_sdf = _validate_converted_asset(
        asset,
        dataset_root,
        sdf_path,
        publisher_files_before=publisher_files_before,
    )
    _validate_sdf_physics_and_collisions(final_sdf, physics_contract)
    return final_sdf


def _load_existing(output_path: Path) -> tuple[list[np.ndarray], list[str], dict[str, Any]]:
    required = [output_path / name for name in INDEX_FILENAMES]
    if not all(path.exists() for path in required):
        return [], [], {}
    embeddings = np.load(required[0])
    embedding_index = yaml.safe_load(required[1].read_text(encoding="utf-8")) or []
    metadata = yaml.safe_load(required[2].read_text(encoding="utf-8")) or {}
    if len(embeddings) != len(embedding_index):
        raise ValueError("Existing Artiverse embedding array/index length mismatch")
    if set(embedding_index) != set(metadata):
        raise ValueError("Existing Artiverse metadata/index object IDs mismatch")
    return [row.astype(np.float32) for row in embeddings], list(embedding_index), metadata


def _metadata_binds_sdf(
    record: Any, dataset_root: Path, expected_sdf_path: Path
) -> bool:
    """Require an indexed row to bind the exact unchanged derived SDF tree."""

    if not isinstance(record, dict):
        return False
    raw_path = record.get("sdf_path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        return False
    bound_path = Path(
        os.path.abspath(os.fspath(dataset_root / relative_path))
    )
    expected_path = Path(os.path.abspath(os.fspath(expected_sdf_path)))
    if bound_path != expected_path:
        return False
    canonical_sdf = require_regular_file_within(
        expected_sdf_path,
        dataset_root,
        f"Indexed Artiverse SDF bound by {raw_path}",
    )
    actual_sdf_sha256 = sha256_file(canonical_sdf)
    if record.get("sdf_sha256") != actual_sdf_sha256:
        raise RuntimeError(
            f"Indexed Artiverse SDF hash mismatch for {raw_path}"
        )
    actual_tree_sha256 = sha256_directory_tree(canonical_sdf.parent)
    if record.get("sdf_directory_tree_sha256") != actual_tree_sha256:
        raise RuntimeError(
            "Indexed Artiverse SDF directory-tree hash mismatch for "
            f"{raw_path}"
        )
    return True


def _atomic_write_text(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _write_indices(
    output_path: Path,
    embeddings: list[np.ndarray],
    embedding_index: list[str],
    metadata: dict[str, Any],
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    if not embeddings:
        raise ValueError("Cannot write an empty Artiverse retrieval index")
    array = np.stack(embeddings).astype(np.float32)
    temp_npy = output_path / ".clip_embeddings.npy.tmp.npy"
    np.save(temp_npy, array)
    os.replace(temp_npy, output_path / "clip_embeddings.npy")
    _atomic_write_text(
        output_path / "embedding_index.yaml",
        yaml.safe_dump(embedding_index, sort_keys=False),
    )
    _atomic_write_text(
        output_path / "metadata_index.yaml",
        yaml.safe_dump(metadata, sort_keys=True, allow_unicode=True),
    )


def prepare(
    dataset_root: Path,
    output_path: Path,
    categories: set[str] | None,
    max_images: int,
    checkpoint_every: int,
    limit: int | None,
    max_collision_elements: int = 32,
    minimum_indexed: int = 1,
    minimum_indexed_per_category: int = 0,
    max_failure_fraction: float = 0.25,
    source_revision: str = OFFICIAL_REVISION,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()
    source_manifest_path = dataset_root / "dataset_chunks" / "manifest.json"
    if not source_manifest_path.is_file():
        raise RuntimeError(
            "Official Artiverse dataset_chunks/manifest.json is missing; "
            "refusing to prepare an unverified substitute dataset"
        )
    source_manifest_sha256 = sha256_file(source_manifest_path)
    if source_manifest_sha256 != OFFICIAL_SOURCE_MANIFEST_SHA256:
        raise RuntimeError(
            "Official Artiverse dataset_chunks/manifest.json does not match "
            f"the pinned audited digest {OFFICIAL_SOURCE_MANIFEST_SHA256}"
        )
    pack_script_path = dataset_root / "pack_dataset_chunks.py"
    if not pack_script_path.is_file():
        raise RuntimeError("Official Artiverse pack_dataset_chunks.py is missing")
    pack_script_sha256 = sha256_file(pack_script_path)
    if pack_script_sha256 != OFFICIAL_PACK_SCRIPT_SHA256:
        raise RuntimeError(
            "Official Artiverse pack_dataset_chunks.py does not match the "
            f"pinned audited digest {OFFICIAL_PACK_SCRIPT_SHA256}"
        )
    if source_revision != OFFICIAL_REVISION:
        raise RuntimeError(
            f"Artiverse revision must be the pinned official revision {OFFICIAL_REVISION}"
        )
    extraction = validate_extraction_receipt(dataset_root)
    requested_categories = (
        {str(category).lower() for category in categories} if categories else None
    )
    manifest_root_counts = _manifest_root_category_counts(source_manifest_path)
    missing_manifest_categories = sorted(
        (requested_categories or set()) - set(manifest_root_counts)
    )
    if missing_manifest_categories:
        raise RuntimeError(
            "Requested Artiverse categories are absent from the pinned source "
            f"manifest roots: {', '.join(missing_manifest_categories)}"
        )
    readiness_categories = sorted(
        requested_categories if requested_categories is not None else manifest_root_counts
    )

    assets = discover_assets(dataset_root, requested_categories)
    if limit is not None:
        assets = assets[:limit]
    if not assets:
        raise RuntimeError(f"No Artiverse URDF assets found under {dataset_root}")

    embeddings, embedding_index, metadata = _load_existing(output_path)
    completed = set(embedding_index)
    failures: list[dict[str, str]] = []
    added = 0

    LOGGER.info("Discovered %d Artiverse assets; %d already indexed", len(assets), len(completed))
    for ordinal, asset in enumerate(assets, start=1):
        sdf_path = asset.urdf_path.with_name(DERIVED_SDF_FILENAME)
        is_completed = asset.object_id in completed
        transaction_path = _sdf_transaction_path(sdf_path)
        recovering_derived_state = not is_completed and any(
            path.exists() or path.is_symlink()
            for path in (sdf_path, transaction_path)
        )
        try:
            sdf_is_bound = is_completed and _metadata_binds_sdf(
                metadata.get(asset.object_id), dataset_root, sdf_path
            )
            publisher_files = _validate_publisher_asset(
                asset,
                dataset_root,
                sdf_path,
                existing_sdf_is_bound=sdf_is_bound,
                recover_unindexed_sdf=not is_completed,
            )
            physics_contract = _publisher_physics_contract(
                asset.urdf_path,
                max_collision_elements=max_collision_elements,
                require_movable_joint=True,
            )
            if is_completed:
                if not sdf_is_bound:
                    raise RuntimeError(
                        "Indexed Artiverse metadata does not bind its expected SDF: "
                        f"{asset.object_id}"
                    )
                _validate_sdf_physics_and_collisions(sdf_path, physics_contract)
                metadata[asset.object_id].update(
                    _physics_metadata(
                        physics_contract,
                        dataset_root=dataset_root,
                        publisher_urdf=asset.urdf_path,
                        emitted_sdf=sdf_path,
                    )
                )
                continue
            images = _reference_images(asset, max_images=max_images)
            if not images:
                raise ValueError("no reference images found")

            sdf_path = _convert_or_recover_sdf(
                asset,
                dataset_root,
                sdf_path,
                publisher_files,
                physics_contract,
            )
            collision_count = _collision_count(sdf_path)
            if collision_count < 1:
                raise ValueError("converted SDF contains no collision elements")
            if collision_count > max_collision_elements:
                raise ValueError(
                    f"converted SDF has {collision_count} collision elements; "
                    f"cap is {max_collision_elements}"
                )
            movable_joint_count, movable_joint_types = _movable_joint_summary(sdf_path)
            if movable_joint_count < 1:
                allowed_types = ", ".join(sorted(MOVABLE_JOINT_TYPES))
                raise ValueError(
                    "converted SDF contains no usable movable joint; expected "
                    f"at least one of: {allowed_types}"
                )
            combined = combine_sdf_meshes_at_joint_angles(sdf_path, use_max_angles=False)
            bbox_min = [float(value) for value in combined.bounds[0]]
            bbox_max = [float(value) for value in combined.bounds[1]]
            embedding = get_multiview_image_embedding(images).astype(np.float32)

            sdf_relative = sdf_path.resolve().relative_to(dataset_root)
            metadata[asset.object_id] = {
                "category": asset.category,
                "description": _description(asset),
                "is_manipuland": False,
                "placement_type": "floor",
                "placement_options": {"on_floor": True},
                "sdf_path": str(sdf_relative),
                "bounding_box_min": bbox_min,
                "bounding_box_max": bbox_max,
                "artiverse_source": asset.source,
                "artiverse_model_id": asset.model_id,
                "collision_element_count": collision_count,
                **_physics_metadata(
                    physics_contract,
                    dataset_root=dataset_root,
                    publisher_urdf=asset.urdf_path,
                    emitted_sdf=sdf_path,
                ),
                "movable_joint_count": movable_joint_count,
                "movable_joint_types": movable_joint_types,
                "sdf_sha256": sha256_file(sdf_path),
                "sdf_directory_tree_sha256": sha256_directory_tree(sdf_path.parent),
            }
            embeddings.append(embedding)
            embedding_index.append(asset.object_id)
            completed.add(asset.object_id)
            added += 1
            LOGGER.info("[%d/%d] indexed %s", ordinal, len(assets), asset.object_id)
            if added % max(1, checkpoint_every) == 0:
                _write_indices(output_path, embeddings, embedding_index, metadata)
        except Exception as exc:  # keep a long conversion resumable
            LOGGER.exception("[%d/%d] failed %s", ordinal, len(assets), asset.object_id)
            failures.append({"object_id": asset.object_id, "error": str(exc)})
            if asset.object_id in completed or recovering_derived_state:
                raise

    # Revalidate recorded content hashes and backfill derived joint facts when
    # resuming. The index and metadata path/hash binding is authoritative;
    # preexisting unbound SDF output is never adopted.
    for object_id in embedding_index:
        record = metadata.get(object_id)
        if not isinstance(record, dict):
            raise RuntimeError(f"Missing metadata for indexed Artiverse ID {object_id}")
        relative_sdf = Path(str(record.get("sdf_path", "")))
        if relative_sdf.is_absolute():
            raise RuntimeError(f"Absolute Artiverse SDF path in index: {object_id}")
        components = object_id.split("/")
        if (
            len(components) != 4
            or components[0] != "artiverse"
            or any(
                not value or value in {".", ".."} or Path(value).name != value
                for value in components[1:]
            )
        ):
            raise RuntimeError(f"Non-canonical Artiverse ID in index: {object_id}")
        _, category, source, model_id = components
        data_root = _data_root(dataset_root)
        expected_model_dir = data_root / category / source / model_id
        source_sdf = require_regular_file_within(
            dataset_root / relative_sdf,
            expected_model_dir,
            f"Indexed Artiverse SDF for {object_id}",
        )
        validate_regular_directory_tree(
            expected_model_dir,
            containment_root=dataset_root,
            label=f"Indexed Artiverse model tree for {object_id}",
        )
        validate_sdf_resource_uris(source_sdf, expected_model_dir)
        source_urdf = _select_urdf(expected_model_dir)
        if source_urdf is None:
            raise RuntimeError(f"Publisher URDF is missing for indexed Artiverse ID {object_id}")
        physics_contract = _publisher_physics_contract(
            source_urdf,
            max_collision_elements=max_collision_elements,
            require_movable_joint=True,
        )
        _validate_sdf_physics_and_collisions(source_sdf, physics_contract)
        if record.get("collision_element_count") != physics_contract.source_collision_element_count:
            raise RuntimeError(
                f"Indexed Artiverse collision metadata is stale for {object_id}"
            )
        record.update(
            _physics_metadata(
                physics_contract,
                dataset_root=dataset_root,
                publisher_urdf=source_urdf,
                emitted_sdf=source_sdf,
            )
        )
        actual_sdf_sha256 = sha256_file(source_sdf)
        recorded_sdf_sha256 = record.get("sdf_sha256")
        if recorded_sdf_sha256 != actual_sdf_sha256:
            raise RuntimeError(
                f"Indexed Artiverse SDF hash mismatch for {object_id}"
            )
        actual_tree_sha256 = sha256_directory_tree(source_sdf.parent)
        recorded_tree_sha256 = record.get("sdf_directory_tree_sha256")
        if recorded_tree_sha256 != actual_tree_sha256:
            raise RuntimeError(
                f"Indexed Artiverse SDF directory-tree hash mismatch for {object_id}"
            )
        movable_joint_count, movable_joint_types = _movable_joint_summary(source_sdf)
        record["movable_joint_count"] = movable_joint_count
        record["movable_joint_types"] = movable_joint_types

    _write_indices(output_path, embeddings, embedding_index, metadata)
    target_ids = {asset.object_id for asset in assets}
    embedded_target_ids = target_ids.intersection(embedding_index)
    movable_joint_violations = [
        object_id
        for object_id in sorted(embedded_target_ids)
        if int(metadata.get(object_id, {}).get("movable_joint_count", 0)) < 1
        or not set(metadata.get(object_id, {}).get("movable_joint_types", [])).intersection(
            MOVABLE_JOINT_TYPES
        )
    ]
    indexed_target_ids = embedded_target_ids - set(movable_joint_violations)
    unindexed_target_ids = sorted(target_ids - indexed_target_ids)
    unresolved_fraction = len(unindexed_target_ids) / len(target_ids)
    collision_violations = [
        object_id
        for object_id in sorted(embedded_target_ids)
        if int(metadata.get(object_id, {}).get("collision_element_count", max_collision_elements + 1))
        > max_collision_elements
    ]
    discovered_category_counts = {
        category: sum(asset.category.lower() == category for asset in assets)
        for category in readiness_categories
    }
    indexed_category_counts = {
        category: sum(
            object_id.split("/", 3)[1].lower() == category
            for object_id in indexed_target_ids
        )
        for category in readiness_categories
    }
    requested_category_counts = {
        category: {
            "manifest_root_count": manifest_root_counts[category],
            "discovered_count": discovered_category_counts[category],
            "indexed_count": indexed_category_counts[category],
        }
        for category in readiness_categories
    }
    categories_below_minimum_indexed = [
        category
        for category in readiness_categories
        if indexed_category_counts[category] < minimum_indexed_per_category
    ]
    passed = (
        len(indexed_target_ids) >= minimum_indexed
        and unresolved_fraction <= max_failure_fraction
        and not collision_violations
        and not movable_joint_violations
        and not categories_below_minimum_indexed
    )
    physics_bound_ids = sorted(embedding_index)
    manifest = {
        "schema_version": 2,
        "status": "pass" if passed else "fail",
        "physics_policy": dict(PHYSICS_POLICY),
        "physics_bound_indexed_count": len(physics_bound_ids),
        "physics_binding_sha256": physics_binding_sha256(
            metadata,
            physics_bound_ids,
        ),
        "source_repository": OFFICIAL_REPOSITORY,
        "source_revision": source_revision,
        "source_pack_script_sha256": pack_script_sha256,
        "source_manifest": {
            "path": str(source_manifest_path.relative_to(dataset_root)),
            "sha256": source_manifest_sha256,
        },
        "source_extraction_receipt": {
            "path": str(extraction.receipt_path.relative_to(dataset_root)),
            "sha256": extraction.receipt_sha256,
        },
        "safe_extractor_sha256": extraction.safe_extractor_sha256,
        "dataset_root": str(dataset_root),
        "output_path": str(output_path),
        "discovered_count": len(assets),
        "indexed_count": len(embedding_index),
        "embedded_target_count": len(embedded_target_ids),
        "indexed_target_count": len(indexed_target_ids),
        "added_count": added,
        "failure_count": len(failures),
        "unindexed_target_count": len(unindexed_target_ids),
        "unindexed_target_ids": unindexed_target_ids,
        "unresolved_fraction": unresolved_fraction,
        "maximum_failure_fraction": max_failure_fraction,
        "minimum_indexed": minimum_indexed,
        "minimum_indexed_per_category": minimum_indexed_per_category,
        "requested_category_counts": requested_category_counts,
        "categories_below_minimum_indexed": categories_below_minimum_indexed,
        "maximum_collision_elements": max_collision_elements,
        "collision_cap_violations": collision_violations,
        "usable_movable_joint_types": sorted(MOVABLE_JOINT_TYPES),
        "movable_joint_violations": movable_joint_violations,
        "failures": failures,
        "categories": sorted(requested_categories) if requested_categories else "all",
        "index_sha256": {
            filename: sha256_file(output_path / filename)
            for filename in INDEX_FILENAMES
        },
    }
    _atomic_write_text(
        output_path / "artiverse_preparation_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/artiverse"))
    parser.add_argument("--output-path", type=Path)
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional category allow-list. Default: index every released category.",
    )
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int, help="Debug-only maximum asset count.")
    parser.add_argument("--max-collision-elements", type=int, default=32)
    parser.add_argument("--minimum-indexed", type=int, default=1)
    parser.add_argument("--minimum-indexed-per-category", type=int, default=0)
    parser.add_argument("--max-failure-fraction", type=float, default=0.25)
    parser.add_argument(
        "--source-revision",
        default=os.environ.get("ARTIVERSE_REVISION", OFFICIAL_REVISION),
        help="Pinned official Hugging Face dataset revision.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    dataset_root = args.dataset_root.resolve()
    output_path = (
        args.output_path.resolve()
        if args.output_path
        else dataset_root / "embeddings"
    )
    categories = {value.lower() for value in args.categories} if args.categories else None
    if args.max_collision_elements < 1:
        parser.error("--max-collision-elements must be at least 1")
    if args.minimum_indexed < 1:
        parser.error("--minimum-indexed must be at least 1")
    if args.minimum_indexed_per_category < 0:
        parser.error("--minimum-indexed-per-category cannot be negative")
    if not 0.0 <= args.max_failure_fraction <= 1.0:
        parser.error("--max-failure-fraction must be between 0 and 1")
    manifest = prepare(
        dataset_root=dataset_root,
        output_path=output_path,
        categories=categories,
        max_images=max(1, args.max_images),
        checkpoint_every=max(1, args.checkpoint_every),
        limit=args.limit,
        max_collision_elements=args.max_collision_elements,
        minimum_indexed=args.minimum_indexed,
        minimum_indexed_per_category=args.minimum_indexed_per_category,
        max_failure_fraction=args.max_failure_fraction,
        source_revision=args.source_revision,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
