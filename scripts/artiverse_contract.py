#!/usr/bin/env python3
"""Validate the official prepared Artiverse dataset and final usage evidence.

The full-quality pipeline treats Artiverse as an acceptance requirement.  This
module is the shared authority for preflight, router, worker, assembly, and the
last SQZ status transition; filename labels alone are never accepted as proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

import numpy as np
import yaml


OFFICIAL_REPOSITORY = "3dlg-hcvc/artiverse"
OFFICIAL_REVISION = "8c4b120418e7cbdf9ac4c9580c5dbfdbf128a248"
OFFICIAL_SOURCE_MANIFEST_SHA256 = (
    "8fa6468254a1f74c58f0c25699598bf88f622fabdaf74f0cd9268ee5663c5586"
)
OFFICIAL_PACK_SCRIPT_SHA256 = (
    "f438e6fa147514f5260a205bc09d4b6c6ff3c0ce2d3022af424d220a9c933b99"
)
EXTRACTION_RECEIPT_FILENAME = "artiverse_safe_extraction_receipt.json"
SAFE_EXTRACTOR_FILENAME = "safe_extract_artiverse.py"
OFFICIAL_EXTRACTION_ROOT_COUNT = 3_544
OFFICIAL_EXTRACTION_ROOTS_SHA256 = (
    "7bdf1be3acc558df62fcc9077b60c71f860227925d35ea05679b6ebbcc9f9182"
)
OFFICIAL_EXTRACTION_FILE_COUNT = 531_937
OFFICIAL_EXTRACTION_INPUT_BYTES = 86_992_752_890
OFFICIAL_EXTRACTION_ARCHIVES = (
    (
        "artiverse_data-00001-of-00002.tar.gz",
        38_163_580_631,
        "695d2d602faafab922ce66359ea104d81505f5b0fdee8f461d8905f0ccb4ef3b",
        1_561,
        219_064,
        47_866_437_602,
    ),
    (
        "artiverse_data-00002-of-00002.tar.gz",
        27_170_560_473,
        "56dffa50f1c8c20d3b1eef626046805a6c7cd997141e8ab5fac9ebdae8ffab81",
        1_983,
        312_873,
        39_126_315_288,
    ),
)
RECEIPT_ROOT_HASH_ALGORITHM = "sha256-u64be-length-prefixed-utf8-sorted-v1"
RECEIPT_TREE_HASH_ALGORITHM = (
    "sha256-sorted-utf8-path-directory-and-regular-file-content-v1"
)
PREPARATION_MANIFEST = "artiverse_preparation_manifest.json"
INDEX_FILENAMES = (
    "clip_embeddings.npy",
    "embedding_index.yaml",
    "metadata_index.yaml",
)
HASH_CHUNK_SIZE = 1024 * 1024
PHYSICS_POLICY_ID = "publisher_urdf_inertial_v1"
PUBLISHER_PHYSICS_SOURCE = "publisher_urdf_inertial"
PUBLISHER_MASS_SEMANTICS = "publisher_unit_mass_proxy_not_material_density_v1"
JOINT_DYNAMICS_POLICY = "scenesmith_defaults_damping_friction_0.05_v1"
COLLISION_FRICTION_POLICY = "scenesmith_default_0.5_v1"
INERTIAL_FRAME_TRANSFORM_POLICY = "urdf_rpy_RzRyRx_to_link_frame_v1"
PHYSICS_POLICY_KEYS = frozenset(
    {
        "id",
        "required_for_every_link",
        "inertial_frame_transform",
        "require_zero_emitted_inertial_rpy",
        "preserve_source_collision_count",
        "preconversion_collision_cap",
        "publisher_mass_semantics",
        "joint_dynamics_policy",
        "collision_friction_policy",
    }
)
MOVABLE_JOINT_TYPES = frozenset({"continuous", "prismatic", "revolute"})
INERTIA_COMPONENTS = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
MASS_SERIALIZATION_ABS_TOLERANCE_PER_LINK = 5.0e-7
MASS_SERIALIZATION_REL_TOLERANCE = 1.0e-6
INERTIAL_ROTATION_ABS_TOLERANCE = 1.0e-12
INERTIA_TRIANGLE_REL_TOLERANCE = 1.0e-9
INERTIA_TRIANGLE_ABS_TOLERANCE = 1.0e-12
PHYSICS_BINDING_SCHEMA_VERSION = 1
PHYSICS_ROW_BINDING_FIELDS = (
    "publisher_urdf_path",
    "publisher_urdf_sha256",
    "publisher_link_physics_sha256",
    "emitted_sdf_physics_sha256",
    "physics_link_count",
    "physics_geometry_link_count",
    "source_collision_element_count",
)


class ArtiverseContractError(RuntimeError):
    """Raised when prepared or final Artiverse evidence is not authoritative."""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _exact_child(element: ET.Element, name: str, label: str) -> ET.Element:
    children = _direct_children(element, name)
    if len(children) != 1:
        raise ArtiverseContractError(
            f"{label} must contain exactly one <{name}> element; found {len(children)}"
        )
    return children[0]


def _finite_xml_number(element: ET.Element, label: str) -> float:
    try:
        value = float((element.text or "").strip())
    except (TypeError, ValueError) as exc:
        raise ArtiverseContractError(f"{label} is not numeric") from exc
    if not math.isfinite(value):
        raise ArtiverseContractError(f"{label} is not finite")
    return value


def _exact_metadata_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtiverseContractError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _exact_physics_policy(preparation: dict[str, Any]) -> dict[str, Any]:
    policy = preparation.get("physics_policy")
    if not isinstance(policy, dict) or set(policy) != PHYSICS_POLICY_KEYS:
        raise ArtiverseContractError(
            "Artiverse physics_policy must contain exactly the required policy keys"
        )
    if policy.get("id") != PHYSICS_POLICY_ID:
        raise ArtiverseContractError(
            f"Artiverse physics_policy id must be {PHYSICS_POLICY_ID}"
        )
    for key in {
        "required_for_every_link",
        "require_zero_emitted_inertial_rpy",
        "preserve_source_collision_count",
        "preconversion_collision_cap",
    }:
        if policy.get(key) is not True:
            raise ArtiverseContractError(
                f"Artiverse physics_policy {key} must be exactly true"
            )
    expected_transparency = {
        "inertial_frame_transform": INERTIAL_FRAME_TRANSFORM_POLICY,
        "publisher_mass_semantics": PUBLISHER_MASS_SEMANTICS,
        "joint_dynamics_policy": JOINT_DYNAMICS_POLICY,
        "collision_friction_policy": COLLISION_FRICTION_POLICY,
    }
    for key, expected in expected_transparency.items():
        if policy.get(key) != expected:
            raise ArtiverseContractError(
                f"Artiverse physics_policy {key} must be exactly {expected!r}"
            )
    return policy


def _canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtiverseContractError(
            "Artiverse physics binding is not canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _finite_float_hex(value: float, label: str) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ArtiverseContractError(f"{label} is not finite")
    return numeric.hex()


def _finite_attribute_number(element: ET.Element, attribute: str, label: str) -> float:
    raw = element.get(attribute)
    if raw is None:
        raise ArtiverseContractError(f"{label} is missing")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ArtiverseContractError(f"{label} is not numeric") from exc
    if not math.isfinite(value):
        raise ArtiverseContractError(f"{label} is not finite")
    return value


def _finite_vector(raw: str | None, length: int, label: str) -> tuple[float, ...]:
    fields = (raw or "").split()
    if len(fields) != length:
        raise ArtiverseContractError(
            f"{label} must contain exactly {length} values"
        )
    try:
        values = tuple(float(field) for field in fields)
    except ValueError as exc:
        raise ArtiverseContractError(f"{label} is not numeric") from exc
    if not all(math.isfinite(value) for value in values):
        raise ArtiverseContractError(f"{label} is not finite")
    return values


def _validate_inertia_components(
    components: dict[str, float], label: str
) -> tuple[float, ...]:
    tensor = np.asarray(
        [
            [components["ixx"], components["ixy"], components["ixz"]],
            [components["ixy"], components["iyy"], components["iyz"]],
            [components["ixz"], components["iyz"], components["izz"]],
        ],
        dtype=np.float64,
    )
    try:
        principal = np.linalg.eigvalsh(tensor)
    except np.linalg.LinAlgError as exc:
        raise ArtiverseContractError(f"{label} cannot be diagonalized") from exc
    if not np.isfinite(principal).all() or float(principal[0]) <= 0.0:
        raise ArtiverseContractError(
            f"{label} must be symmetric positive-definite"
        )
    largest = float(principal[-1])
    tolerance = max(
        INERTIA_TRIANGLE_ABS_TOLERANCE,
        INERTIA_TRIANGLE_REL_TOLERANCE * largest,
    )
    if largest > float(principal[0] + principal[1]) + tolerance:
        raise ArtiverseContractError(
            f"{label} principal moments violate the triangle inequality"
        )
    return tuple(float(value) for value in principal)


@dataclass(frozen=True)
class PublisherPhysicsEvidence:
    sha256: str
    link_values: dict[str, dict[str, Any]]
    collision_count_by_link: dict[str, int]
    geometry_links: frozenset[str]
    total_mass_kg: float


@dataclass(frozen=True)
class EmittedPhysicsEvidence:
    sha256: str
    link_values: dict[str, dict[str, Any]]
    collision_count_by_link: dict[str, int]
    geometry_links: frozenset[str]


def _urdf_rpy_rotation(rpy: tuple[float, ...]) -> np.ndarray:
    """Return URDF's fixed-axis roll/pitch/yaw rotation ``Rz * Ry * Rx``."""

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def publisher_urdf_physics_evidence(urdf_path: Path) -> PublisherPhysicsEvidence:
    """Reparse publisher URDF physics into an exact, portable digest."""

    try:
        document = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ArtiverseContractError(
            f"Cannot parse publisher Artiverse URDF at {urdf_path}: {exc}"
        ) from exc
    if _local_name(document.tag) != "robot":
        raise ArtiverseContractError("Publisher Artiverse URDF root must be <robot>")
    links = _direct_children(document, "link")
    if not links:
        raise ArtiverseContractError("Publisher Artiverse URDF contains no links")

    values: dict[str, dict[str, Any]] = {}
    canonical_links: dict[str, Any] = {}
    collision_counts: dict[str, int] = {}
    geometry_links: set[str] = set()
    total_mass = 0.0
    for ordinal, link in enumerate(links, start=1):
        name = str(link.get("name", "")).strip()
        if not name or name in values:
            raise ArtiverseContractError(
                f"Publisher Artiverse URDF has missing/duplicate link name at #{ordinal}"
            )
        label = f"Publisher Artiverse URDF link {name}"
        inertial = _exact_child(link, "inertial", label)
        origins = _direct_children(inertial, "origin")
        if len(origins) > 1:
            raise ArtiverseContractError(
                f"{label} inertial must contain at most one <origin>"
            )
        origin = origins[0] if origins else None
        xyz = _finite_vector(
            origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
            3,
            f"{label} inertial xyz",
        )
        rpy = _finite_vector(
            origin.get("rpy", "0 0 0") if origin is not None else "0 0 0",
            3,
            f"{label} inertial rpy",
        )
        mass_element = _exact_child(inertial, "mass", f"{label} inertial")
        mass = _finite_attribute_number(mass_element, "value", f"{label} mass")
        if mass <= 0.0:
            raise ArtiverseContractError(f"{label} mass must be positive")
        inertia = _exact_child(inertial, "inertia", f"{label} inertial")
        terms = {
            component: _finite_attribute_number(
                inertia, component, f"{label} inertia {component}"
            )
            for component in INERTIA_COMPONENTS
        }
        _validate_inertia_components(terms, f"{label} source inertia tensor")
        source_tensor = np.asarray(
            [
                [terms["ixx"], terms["ixy"], terms["ixz"]],
                [terms["ixy"], terms["iyy"], terms["iyz"]],
                [terms["ixz"], terms["iyz"], terms["izz"]],
            ],
            dtype=np.float64,
        )
        rotation = _urdf_rpy_rotation(rpy)
        link_tensor = rotation @ source_tensor @ rotation.T
        link_terms = {
            "ixx": float(link_tensor[0, 0]),
            "iyy": float(link_tensor[1, 1]),
            "izz": float(link_tensor[2, 2]),
            "ixy": float(link_tensor[0, 1]),
            "ixz": float(link_tensor[0, 2]),
            "iyz": float(link_tensor[1, 2]),
        }
        _validate_inertia_components(link_terms, f"{label} link-frame inertia tensor")
        collisions = len(_direct_children(link, "collision"))
        has_geometry = bool(
            _direct_children(link, "visual") or _direct_children(link, "collision")
        )
        if has_geometry:
            geometry_links.add(name)
            if collisions < 1:
                raise ArtiverseContractError(
                    f"{label} has geometry but no source collision"
                )
        collision_counts[name] = collisions
        total_mass += mass
        values[name] = {
            "mass": mass,
            "pose_xyz": xyz,
            "pose_rpy": rpy,
            "source_inertia": terms,
            "inertia": link_terms,
        }
        inertia_hex = {
            key: _finite_float_hex(terms[key], f"{label} inertia {key}")
            for key in INERTIA_COMPONENTS
        }
        xyz_hex = [_finite_float_hex(value, f"{label} xyz") for value in xyz]
        rpy_hex = [_finite_float_hex(value, f"{label} rpy") for value in rpy]
        link_inertia_hex = {
            key: _finite_float_hex(
                link_terms[key], f"{label} link-frame inertia {key}"
            )
            for key in INERTIA_COMPONENTS
        }
        canonical_links[name] = {
            "source_mass_hex": _finite_float_hex(mass, f"{label} mass"),
            "source_origin_xyz_hex": xyz_hex,
            "source_origin_rpy_hex": rpy_hex,
            "source_inertia_hex": inertia_hex,
            "link_frame_com_hex": xyz_hex,
            "link_frame_inertia_hex": link_inertia_hex,
            "projection_applied": False,
        }
    payload = {
        "schema_version": PHYSICS_BINDING_SCHEMA_VERSION,
        "physics_source": PUBLISHER_PHYSICS_SOURCE,
        "publisher_mass_semantics": PUBLISHER_MASS_SEMANTICS,
        "links": dict(sorted(canonical_links.items())),
    }
    return PublisherPhysicsEvidence(
        sha256=_canonical_json_sha256(payload),
        link_values=values,
        collision_count_by_link=collision_counts,
        geometry_links=frozenset(geometry_links),
        total_mass_kg=total_mass,
    )


def emitted_sdf_physics_evidence(sdf_path: Path) -> EmittedPhysicsEvidence:
    """Reparse emitted SDF physics into the canonical serialized-value digest."""

    try:
        document = ET.parse(sdf_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ArtiverseContractError(
            f"Cannot parse emitted Artiverse SDF at {sdf_path}: {exc}"
        ) from exc
    model = _exact_child(document, "model", "Emitted Artiverse SDF")
    links = _direct_children(model, "link")
    if not links:
        raise ArtiverseContractError("Emitted Artiverse SDF contains no links")
    values: dict[str, dict[str, Any]] = {}
    canonical_links: dict[str, Any] = {}
    collision_counts: dict[str, int] = {}
    geometry_links: set[str] = set()
    for ordinal, link in enumerate(links, start=1):
        name = str(link.get("name", "")).strip()
        if not name or name in values:
            raise ArtiverseContractError(
                f"Emitted Artiverse SDF has missing/duplicate link name at #{ordinal}"
            )
        label = f"Emitted Artiverse SDF link {name}"
        inertial = _exact_child(link, "inertial", label)
        pose = _exact_child(inertial, "pose", f"{label} inertial")
        pose_values = _finite_vector(pose.text, 6, f"{label} inertial pose")
        if any(
            abs(value) > INERTIAL_ROTATION_ABS_TOLERANCE
            for value in pose_values[3:]
        ):
            raise ArtiverseContractError(
                f"{label} inertial pose rotation must be zero"
            )
        mass = _finite_xml_number(
            _exact_child(inertial, "mass", f"{label} inertial"),
            f"{label} inertial mass",
        )
        if mass <= 0.0:
            raise ArtiverseContractError(f"{label} inertial mass must be positive")
        inertia = _exact_child(inertial, "inertia", f"{label} inertial")
        terms = {
            component: _finite_xml_number(
                _exact_child(inertia, component, f"{label} inertia"),
                f"{label} inertia {component}",
            )
            for component in INERTIA_COMPONENTS
        }
        _validate_inertia_components(terms, f"{label} inertia tensor")
        collisions = len(_direct_children(link, "collision"))
        has_geometry = bool(
            _direct_children(link, "visual") or _direct_children(link, "collision")
        )
        if has_geometry:
            geometry_links.add(name)
            if collisions < 1:
                raise ArtiverseContractError(
                    f"{label} has geometry but no collision element"
                )
        collision_counts[name] = collisions
        values[name] = {
            "mass": mass,
            "pose_xyz": pose_values[:3],
            "pose_rpy": pose_values[3:],
            "inertia": terms,
        }
        canonical_links[name] = {
            "mass_hex": _finite_float_hex(mass, f"{label} mass"),
            "pose_xyz_hex": [
                _finite_float_hex(value, f"{label} pose")
                for value in pose_values[:3]
            ],
            "inertia_hex": {
                key: _finite_float_hex(terms[key], f"{label} inertia {key}")
                for key in INERTIA_COMPONENTS
            },
        }
    payload = {
        "schema_version": PHYSICS_BINDING_SCHEMA_VERSION,
        "links": dict(sorted(canonical_links.items())),
    }
    return EmittedPhysicsEvidence(
        sha256=_canonical_json_sha256(payload),
        link_values=values,
        collision_count_by_link=collision_counts,
        geometry_links=frozenset(geometry_links),
    )


def physics_binding_sha256(
    metadata: dict[str, dict[str, Any]], object_ids: list[str]
) -> str:
    """Hash the exact sorted prepared-asset physics binding without host paths."""

    assets: list[dict[str, Any]] = []
    for object_id in sorted(object_ids):
        record = metadata.get(object_id)
        if not isinstance(record, dict):
            raise ArtiverseContractError(
                f"Missing Artiverse physics metadata for {object_id}"
            )
        asset = {"object_id": object_id}
        for field in PHYSICS_ROW_BINDING_FIELDS:
            value = record.get(field)
            if field.endswith("_sha256"):
                if not _is_sha256(value):
                    raise ArtiverseContractError(
                        f"Artiverse {field} is invalid for {object_id}"
                    )
            elif field == "publisher_urdf_path":
                if not isinstance(value, str) or not value:
                    raise ArtiverseContractError(
                        f"Artiverse publisher_urdf_path is invalid for {object_id}"
                    )
            else:
                _exact_metadata_int(
                    value, f"Artiverse {field} for {object_id}", minimum=1
                )
            asset[field] = value
        assets.append(asset)
    payload = {
        "schema_version": PHYSICS_BINDING_SCHEMA_VERSION,
        "policies": {
            "inertial_frame_transform": INERTIAL_FRAME_TRANSFORM_POLICY,
            "publisher_mass_semantics": PUBLISHER_MASS_SEMANTICS,
            "joint_dynamics_policy": JOINT_DYNAMICS_POLICY,
            "collision_friction_policy": COLLISION_FRICTION_POLICY,
        },
        "assets": assets,
    }
    return _canonical_json_sha256(payload)


def _physics_values_match(actual: float, expected: float, *, absolute: float) -> bool:
    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=MASS_SERIALIZATION_REL_TOLERANCE,
        abs_tol=absolute,
    )


def _validate_publisher_sdf_preservation(
    publisher: PublisherPhysicsEvidence,
    emitted: EmittedPhysicsEvidence,
    *,
    object_id: str,
) -> None:
    if set(publisher.link_values) != set(emitted.link_values):
        raise ArtiverseContractError(
            f"Artiverse emitted/publisher link sets differ for {object_id}"
        )
    if publisher.geometry_links != emitted.geometry_links:
        raise ArtiverseContractError(
            f"Artiverse emitted/publisher geometry-link sets differ for {object_id}"
        )
    if publisher.collision_count_by_link != emitted.collision_count_by_link:
        raise ArtiverseContractError(
            f"Artiverse emitted/publisher collision inventories differ for {object_id}"
        )
    for name, expected in publisher.link_values.items():
        actual = emitted.link_values[name]
        if not _physics_values_match(
            actual["mass"], expected["mass"], absolute=5.1e-7
        ):
            raise ArtiverseContractError(
                f"Artiverse emitted mass differs from publisher for {object_id}/{name}"
            )
        if any(
            not _physics_values_match(a, b, absolute=5.1e-7)
            for a, b in zip(actual["pose_xyz"], expected["pose_xyz"])
        ):
            raise ArtiverseContractError(
                f"Artiverse emitted COM differs from publisher for {object_id}/{name}"
            )
        if any(
            not _physics_values_match(
                actual["inertia"][component],
                expected["inertia"][component],
                absolute=1.0e-12,
            )
            for component in INERTIA_COMPONENTS
        ):
            raise ArtiverseContractError(
                f"Artiverse emitted inertia differs from publisher for {object_id}/{name}"
            )


def _validate_sdf_physics_and_joints(
    document: ET.Element,
    record: dict[str, Any],
    *,
    object_id: str,
    collision_cap: int,
    physics_policy: dict[str, Any],
) -> int:
    """Independently recompute prepared SDF physics, collision, and joint facts."""

    links = [element for element in document.iter() if _local_name(element.tag) == "link"]
    if not links:
        raise ArtiverseContractError(f"Artiverse SDF contains no links: {object_id}")

    total_mass = 0.0
    geometry_link_count = 0
    emitted_collision_count = 0
    for ordinal, link in enumerate(links, start=1):
        link_name = str(link.get("name", "")).strip() or f"#{ordinal}"
        label = f"Artiverse SDF {object_id} link {link_name}"
        inertial = _exact_child(link, "inertial", label)
        pose = _exact_child(inertial, "pose", f"{label} inertial")
        pose_values = (pose.text or "").split()
        if len(pose_values) != 6:
            raise ArtiverseContractError(
                f"{label} inertial pose must contain exactly six values"
            )
        try:
            normalized_pose = [float(value) for value in pose_values]
        except ValueError as exc:
            raise ArtiverseContractError(f"{label} inertial pose is not numeric") from exc
        if not all(math.isfinite(value) for value in normalized_pose):
            raise ArtiverseContractError(f"{label} inertial pose is not finite")
        if any(
            abs(value) > INERTIAL_ROTATION_ABS_TOLERANCE
            for value in normalized_pose[3:]
        ):
            raise ArtiverseContractError(
                f"{label} inertial pose rotation must be zero"
            )

        mass = _finite_xml_number(
            _exact_child(inertial, "mass", f"{label} inertial"),
            f"{label} inertial mass",
        )
        if mass <= 0.0:
            raise ArtiverseContractError(f"{label} inertial mass must be positive")
        total_mass += mass

        inertia = _exact_child(inertial, "inertia", f"{label} inertial")
        components = {
            component: _finite_xml_number(
                _exact_child(inertia, component, f"{label} inertia"),
                f"{label} inertia {component}",
            )
            for component in INERTIA_COMPONENTS
        }
        tensor = np.asarray(
            [
                [components["ixx"], components["ixy"], components["ixz"]],
                [components["ixy"], components["iyy"], components["iyz"]],
                [components["ixz"], components["iyz"], components["izz"]],
            ],
            dtype=np.float64,
        )
        try:
            principal = np.linalg.eigvalsh(tensor)
        except np.linalg.LinAlgError as exc:
            raise ArtiverseContractError(
                f"{label} inertia tensor cannot be diagonalized"
            ) from exc
        if not np.isfinite(principal).all() or float(principal[0]) <= 0.0:
            raise ArtiverseContractError(
                f"{label} inertia tensor must be symmetric positive-definite"
            )
        largest = float(principal[-1])
        triangle_tolerance = max(
            INERTIA_TRIANGLE_ABS_TOLERANCE,
            INERTIA_TRIANGLE_REL_TOLERANCE * largest,
        )
        if largest > float(principal[0] + principal[1]) + triangle_tolerance:
            raise ArtiverseContractError(
                f"{label} inertia principal moments violate the triangle inequality"
            )

        visuals = _direct_children(link, "visual")
        collisions = _direct_children(link, "collision")
        if visuals or collisions:
            geometry_link_count += 1
            if not collisions:
                raise ArtiverseContractError(
                    f"{label} has geometry but no collision element"
                )
        for collision_ordinal, collision in enumerate(collisions, start=1):
            collision_label = f"{label} collision #{collision_ordinal}"
            surface = _exact_child(collision, "surface", collision_label)
            friction = _exact_child(surface, "friction", f"{collision_label} surface")
            ode = _exact_child(friction, "ode", f"{collision_label} friction")
            for coefficient in ("mu", "mu2"):
                value = _finite_xml_number(
                    _exact_child(ode, coefficient, f"{collision_label} ODE friction"),
                    f"{collision_label} {coefficient}",
                )
                if not math.isclose(value, 0.5, rel_tol=0.0, abs_tol=1.0e-12):
                    raise ArtiverseContractError(
                        f"{collision_label} does not implement the pinned 0.5 friction policy"
                    )
        emitted_collision_count += len(collisions)

    all_collisions = [
        element
        for element in document.iter()
        if _local_name(element.tag) == "collision"
    ]
    if emitted_collision_count != len(all_collisions):
        raise ArtiverseContractError(
            f"Artiverse SDF contains collision elements outside links: {object_id}"
        )
    if not 1 <= emitted_collision_count <= collision_cap:
        raise ArtiverseContractError(
            f"Artiverse SDF collision count is outside 1..{collision_cap}: {object_id}"
        )

    recorded_collision_count = _exact_metadata_int(
        record.get("collision_element_count"),
        f"Artiverse collision_element_count for {object_id}",
        minimum=1,
    )
    source_collision_count = _exact_metadata_int(
        record.get("source_collision_element_count"),
        f"Artiverse source_collision_element_count for {object_id}",
        minimum=1,
    )
    if emitted_collision_count != recorded_collision_count:
        raise ArtiverseContractError(
            f"Artiverse collision metadata is stale for {object_id}"
        )
    if (
        physics_policy["preserve_source_collision_count"]
        and emitted_collision_count != source_collision_count
    ):
        raise ArtiverseContractError(
            f"Artiverse emitted/source collision counts differ for {object_id}"
        )
    if physics_policy["preconversion_collision_cap"] and source_collision_count > collision_cap:
        raise ArtiverseContractError(
            f"Artiverse source collision count exceeds its preconversion cap: {object_id}"
        )

    if record.get("physics_source") != physics_policy["id"]:
        raise ArtiverseContractError(
            f"Artiverse physics_source is missing or stale for {object_id}"
        )
    if _exact_metadata_int(
        record.get("physics_link_count"),
        f"Artiverse physics_link_count for {object_id}",
        minimum=1,
    ) != len(links):
        raise ArtiverseContractError(f"Artiverse physics link count is stale for {object_id}")
    if _exact_metadata_int(
        record.get("physics_geometry_link_count"),
        f"Artiverse physics_geometry_link_count for {object_id}",
        minimum=1,
    ) != geometry_link_count:
        raise ArtiverseContractError(
            f"Artiverse physics geometry-link count is stale for {object_id}"
        )
    recorded_total_mass = record.get("physics_total_mass_kg")
    if (
        isinstance(recorded_total_mass, bool)
        or not isinstance(recorded_total_mass, (int, float))
        or not math.isfinite(float(recorded_total_mass))
        or float(recorded_total_mass) <= 0.0
    ):
        raise ArtiverseContractError(
            f"Artiverse physics_total_mass_kg is invalid for {object_id}"
        )
    mass_abs_tolerance = max(
        1.0e-9,
        len(links) * MASS_SERIALIZATION_ABS_TOLERANCE_PER_LINK,
    )
    if not math.isclose(
        total_mass,
        float(recorded_total_mass),
        rel_tol=MASS_SERIALIZATION_REL_TOLERANCE,
        abs_tol=mass_abs_tolerance,
    ):
        raise ArtiverseContractError(
            f"Artiverse physics total mass is stale for {object_id}"
        )

    movable_types: list[str] = []
    for joint in (
        element for element in document.iter() if _local_name(element.tag) == "joint"
    ):
        joint_type = str(joint.get("type", "")).strip().lower()
        if joint_type == "fixed":
            continue
        if joint_type not in MOVABLE_JOINT_TYPES:
            raise ArtiverseContractError(
                f"Artiverse SDF has unsupported movable joint type {joint_type!r}: {object_id}"
            )
        axis = _exact_child(joint, "axis", f"Artiverse SDF joint in {object_id}")
        dynamics = _exact_child(
            axis, "dynamics", f"Artiverse SDF joint axis in {object_id}"
        )
        for field in ("damping", "friction"):
            value = _finite_xml_number(
                _exact_child(dynamics, field, f"Artiverse SDF joint dynamics in {object_id}"),
                f"Artiverse SDF joint {field} in {object_id}",
            )
            if not math.isclose(value, 0.05, rel_tol=0.0, abs_tol=1.0e-12):
                raise ArtiverseContractError(
                    f"Artiverse SDF joint does not implement the pinned 0.05 {field} policy: {object_id}"
                )
        movable_types.append(joint_type)
    if not movable_types:
        raise ArtiverseContractError(
            f"Artiverse SDF contains no usable movable joint: {object_id}"
        )
    if _exact_metadata_int(
        record.get("movable_joint_count"),
        f"Artiverse movable_joint_count for {object_id}",
        minimum=1,
    ) != len(movable_types):
        raise ArtiverseContractError(
            f"Artiverse movable-joint count is stale for {object_id}"
        )
    recorded_types = record.get("movable_joint_types")
    actual_types = sorted(set(movable_types))
    if recorded_types != actual_types:
        raise ArtiverseContractError(
            f"Artiverse movable-joint types are stale for {object_id}"
        )
    return emitted_collision_count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link_or_junction(path: Path) -> bool:
    """Return true for filesystem indirections that can escape a tree."""

    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _regular_tree_files(root: Path, *, label: str) -> tuple[Path, list[Path]]:
    """Walk ``root`` without following or accepting non-regular entries."""

    raw_root = Path(root)
    if _is_link_or_junction(raw_root):
        raise ArtiverseContractError(f"{label} is a symlink or junction: {raw_root}")
    try:
        root_stat = raw_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtiverseContractError(f"Cannot inspect {label} at {raw_root}: {exc}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ArtiverseContractError(f"{label} is not a directory: {raw_root}")

    resolved_root = raw_root.resolve(strict=True)
    pending = [resolved_root]
    files: list[Path] = []
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ArtiverseContractError(
                f"Cannot inspect {label} directory {directory}: {exc}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink() or _is_link_or_junction(path):
                raise ArtiverseContractError(
                    f"{label} contains a symlink or junction: {path}"
                )
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(path)
                else:
                    raise ArtiverseContractError(
                        f"{label} contains a special filesystem entry: {path}"
                    )
            except OSError as exc:
                raise ArtiverseContractError(
                    f"Cannot inspect {label} entry {path}: {exc}"
                ) from exc

    files.sort(key=lambda path: path.relative_to(resolved_root).as_posix())
    return resolved_root, files


def _lexical_absolute(path: Path) -> Path:
    """Normalize ``.``/``..`` without resolving symlinks."""

    return Path(os.path.abspath(os.fspath(path)))


def _require_lexically_within(path: Path, root: Path, label: str) -> None:
    lexical_path = _lexical_absolute(path)
    lexical_root = _lexical_absolute(root)
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise ArtiverseContractError(
            f"{label} is outside its expected filesystem root: {lexical_path}"
        ) from exc

    current = lexical_root
    if _is_link_or_junction(current):
        raise ArtiverseContractError(
            f"{label} root is a symlink or junction: {current}"
        )
    for component in relative.parts:
        current = current / component
        if _is_link_or_junction(current):
            raise ArtiverseContractError(
                f"{label} uses a symlink or junction: {current}"
            )


def require_regular_file_within(path: Path, root: Path, label: str) -> Path:
    """Resolve a regular file while rejecting traversal and link indirection."""

    _require_lexically_within(path, root, label)
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ArtiverseContractError(f"{label} file is missing: {path}") from exc
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtiverseContractError(
            f"{label} resolves outside its expected filesystem root: {resolved}"
        ) from exc
    try:
        file_stat = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtiverseContractError(f"Cannot inspect {label} at {path}: {exc}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise ArtiverseContractError(f"{label} is not a regular file: {path}")
    return resolved


def validate_regular_directory_tree(
    root: Path, *, containment_root: Path | None = None, label: str
) -> tuple[Path, ...]:
    """Reject links/special files and optionally constrain a complete tree."""

    if containment_root is not None:
        _require_lexically_within(root, containment_root, label)
        try:
            resolved_root = root.resolve(strict=True)
            resolved_container = containment_root.resolve(strict=True)
            resolved_root.relative_to(resolved_container)
        except (OSError, ValueError) as exc:
            raise ArtiverseContractError(
                f"{label} resolves outside its expected filesystem root: {root}"
            ) from exc
    _, files = _regular_tree_files(root, label=label)
    return tuple(files)


def sha256_directory_tree(root: Path) -> str:
    """Hash a regular-file tree without following or accepting symlinks."""

    resolved_root, files = _regular_tree_files(
        root, label="Artiverse asset directory"
    )
    digest = hashlib.sha256()
    if not files:
        raise ArtiverseContractError(f"Artiverse asset directory is empty: {resolved_root}")
    for path in files:
        relative = path.relative_to(resolved_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _resource_path_from_uri(uri: str, *, relative_to: Path, label: str) -> Path:
    if not uri or "\x00" in uri:
        raise ArtiverseContractError(f"{label} is empty or invalid")
    parsed = urlsplit(uri)
    if parsed.query or parsed.fragment:
        raise ArtiverseContractError(f"{label} must not contain a query or fragment: {uri}")

    # A Windows drive letter is parsed as a URI scheme; let pathlib handle it
    # as a normal absolute path on Windows.
    windows_absolute = Path(uri).is_absolute() and len(parsed.scheme) == 1
    if parsed.scheme and parsed.scheme.lower() != "file" and not windows_absolute:
        raise ArtiverseContractError(
            f"{label} uses unsupported non-local URI scheme {parsed.scheme!r}: {uri}"
        )

    if parsed.scheme.lower() == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ArtiverseContractError(
                f"{label} uses a remote file URI authority: {uri}"
            )
        decoded = url2pathname(unquote(parsed.path))
    else:
        decoded = unquote(uri)
    candidate = Path(decoded)
    return candidate if candidate.is_absolute() else relative_to / candidate


def validate_sdf_resource_uris(sdf_path: Path, model_dir: Path) -> tuple[Path, ...]:
    """Require every SDF ``<uri>`` resource to be a local model-tree file."""

    canonical_sdf = require_regular_file_within(
        sdf_path, model_dir, "Artiverse SDF"
    )
    try:
        document = ET.parse(canonical_sdf)
    except (OSError, ET.ParseError) as exc:
        raise ArtiverseContractError(
            f"Cannot parse Artiverse SDF resource URIs at {canonical_sdf}: {exc}"
        ) from exc

    resources: list[Path] = []
    for ordinal, element in enumerate(document.getroot().iter(), start=1):
        if str(element.tag).rsplit("}", 1)[-1] != "uri":
            continue
        raw_uri = (element.text or "").strip()
        label = f"Artiverse SDF resource URI #{ordinal}"
        candidate = _resource_path_from_uri(
            raw_uri, relative_to=canonical_sdf.parent, label=label
        )
        resources.append(require_regular_file_within(candidate, model_dir, label))
    return tuple(resources)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtiverseContractError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtiverseContractError(f"{label} must be a JSON object: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_safe_path_component(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and Path(value).name == value


def _require_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtiverseContractError(
            f"{label} escapes the official Artiverse dataset root: {resolved}"
        ) from exc
    return resolved


def _resolve_manifest_relative_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ArtiverseContractError(f"{label} path is missing")
    relative = Path(value)
    if relative.is_absolute():
        raise ArtiverseContractError(f"{label} path must be relative: {relative}")
    return require_regular_file_within(root / relative, root, label)


def _select_publisher_urdf(model_dir: Path, object_id: str) -> Path:
    urdf_dir = model_dir / "urdf_w_collider"
    try:
        candidates = [
            path
            for path in sorted(urdf_dir.rglob("*.urdf"))
            if "repaired" not in path.name and "scenesmith" not in path.name
        ]
    except OSError as exc:
        raise ArtiverseContractError(
            f"Cannot inspect publisher URDFs for {object_id}: {exc}"
        ) from exc
    if not candidates:
        raise ArtiverseContractError(
            f"Publisher Artiverse model has no canonical URDF: {object_id}"
        )
    preferred = [
        path
        for path in candidates
        if path.stem.lower() in {"mobility", "model", "object"}
    ]
    selected = preferred[0] if preferred else candidates[0]
    return require_regular_file_within(
        selected, model_dir, f"Selected publisher Artiverse URDF for {object_id}"
    )


def _receipt_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtiverseContractError(f"{label} must be a JSON object")
    if set(value) != expected:
        raise ArtiverseContractError(f"{label} has an unexpected schema")
    return value


def _receipt_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtiverseContractError(
            f"{label} must be an integer of at least {minimum}"
        )
    return value


def _load_extraction_receipt(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtiverseContractError(
            f"Cannot read Artiverse extraction receipt at {path}: {exc}"
        ) from exc
    if len(payload) > 1024 * 1024:
        raise ArtiverseContractError("Artiverse extraction receipt exceeds 1 MiB")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtiverseContractError(
                    f"Duplicate JSON key in Artiverse extraction receipt: {key!r}"
                )
            result[key] = value
        return result

    try:
        receipt = json.loads(
            payload.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtiverseContractError(
            f"Cannot parse Artiverse extraction receipt at {path}: {exc}"
        ) from exc
    if not isinstance(receipt, dict):
        raise ArtiverseContractError("Artiverse extraction receipt must be an object")
    return receipt, hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ArtiverseExtractionEvidence:
    receipt_path: Path
    receipt_sha256: str
    safe_extractor_path: Path
    safe_extractor_sha256: str


def validate_extraction_receipt(dataset_root: Path) -> ArtiverseExtractionEvidence:
    """Authenticate the safe-extraction receipt without rehashing the 87 GB tree."""

    try:
        root = dataset_root.resolve(strict=True)
    except OSError as exc:
        raise ArtiverseContractError(
            f"Official Artiverse dataset root is missing: {dataset_root}"
        ) from exc
    receipt_path = require_regular_file_within(
        root / EXTRACTION_RECEIPT_FILENAME,
        root,
        "Artiverse safe-extraction receipt",
    )
    receipt, receipt_sha256 = _load_extraction_receipt(receipt_path)
    _receipt_exact_keys(
        receipt,
        {
            "schema_version",
            "status",
            "manifest",
            "archives",
            "declared_roots",
            "validated_aggregate",
            "data_tree_inventory",
            "safe_extractor",
        },
        "Artiverse extraction receipt",
    )
    if receipt.get("schema_version") != 1:
        raise ArtiverseContractError("Unsupported Artiverse extraction receipt schema")
    if receipt.get("status") != "pass":
        raise ArtiverseContractError("Artiverse extraction receipt did not pass")

    receipt_manifest = _receipt_exact_keys(
        receipt.get("manifest"),
        {"path", "sha256", "format"},
        "Artiverse extraction receipt manifest",
    )
    if receipt_manifest != {
        "path": "dataset_chunks/manifest.json",
        "sha256": OFFICIAL_SOURCE_MANIFEST_SHA256,
        "format": "artiverse-data-tar-gz-chunks-v1",
    }:
        raise ArtiverseContractError(
            "Artiverse extraction receipt is not bound to the official manifest"
        )

    archives = receipt.get("archives")
    if not isinstance(archives, list) or len(archives) != len(
        OFFICIAL_EXTRACTION_ARCHIVES
    ):
        raise ArtiverseContractError(
            "Artiverse extraction receipt archive list is incomplete"
        )
    member_total = 0
    directory_total = 0
    file_total = 0
    byte_total = 0
    for index, (record_value, expected) in enumerate(
        zip(archives, OFFICIAL_EXTRACTION_ARCHIVES)
    ):
        record = _receipt_exact_keys(
            record_value,
            {
                "archive",
                "archive_bytes",
                "sha256",
                "validated_member_count",
                "validated_directory_count",
                "validated_regular_file_count",
                "validated_uncompressed_bytes",
            },
            f"Artiverse extraction receipt archive #{index + 1}",
        )
        archive, archive_bytes, archive_sha256, model_count, file_count, input_bytes = (
            expected
        )
        pinned = {
            "archive": archive,
            "archive_bytes": archive_bytes,
            "sha256": archive_sha256,
            "validated_regular_file_count": file_count,
            "validated_uncompressed_bytes": input_bytes,
        }
        if any(record.get(key) != value for key, value in pinned.items()):
            raise ArtiverseContractError(
                f"Artiverse extraction receipt archive pin mismatch: {archive}"
            )
        member_count = _receipt_integer(
            record.get("validated_member_count"),
            f"Artiverse extraction receipt member count for {archive}",
        )
        directory_count = _receipt_integer(
            record.get("validated_directory_count"),
            f"Artiverse extraction receipt directory count for {archive}",
            minimum=model_count,
        )
        if member_count != directory_count + file_count:
            raise ArtiverseContractError(
                f"Artiverse extraction receipt counts disagree for {archive}"
            )
        member_total += member_count
        directory_total += directory_count
        file_total += file_count
        byte_total += input_bytes

    roots = _receipt_exact_keys(
        receipt.get("declared_roots"),
        {"count", "sha256", "hash_algorithm"},
        "Artiverse extraction receipt declared roots",
    )
    if roots != {
        "count": OFFICIAL_EXTRACTION_ROOT_COUNT,
        "sha256": OFFICIAL_EXTRACTION_ROOTS_SHA256,
        "hash_algorithm": RECEIPT_ROOT_HASH_ALGORITHM,
    }:
        raise ArtiverseContractError(
            "Artiverse extraction receipt declared-root evidence is stale"
        )

    aggregate = _receipt_exact_keys(
        receipt.get("validated_aggregate"),
        {"member_count", "directory_count", "regular_file_count", "uncompressed_bytes"},
        "Artiverse extraction receipt aggregate",
    )
    if aggregate != {
        "member_count": member_total,
        "directory_count": directory_total,
        "regular_file_count": file_total,
        "uncompressed_bytes": byte_total,
    }:
        raise ArtiverseContractError(
            "Artiverse extraction receipt aggregate counts are inconsistent"
        )
    if (
        file_total != OFFICIAL_EXTRACTION_FILE_COUNT
        or byte_total != OFFICIAL_EXTRACTION_INPUT_BYTES
    ):
        raise ArtiverseContractError(
            "Artiverse extraction receipt does not cover the complete official release"
        )

    inventory = _receipt_exact_keys(
        receipt.get("data_tree_inventory"),
        {"directory_count", "regular_file_count", "sha256", "hash_algorithm"},
        "Artiverse extraction receipt data-tree inventory",
    )
    _receipt_integer(
        inventory.get("directory_count"),
        "Artiverse extraction receipt inventory directory count",
        minimum=OFFICIAL_EXTRACTION_ROOT_COUNT,
    )
    if inventory.get("regular_file_count") != OFFICIAL_EXTRACTION_FILE_COUNT:
        raise ArtiverseContractError(
            "Artiverse extraction receipt inventory file count is stale"
        )
    if not _is_sha256(inventory.get("sha256")):
        raise ArtiverseContractError(
            "Artiverse extraction receipt inventory hash is invalid"
        )
    if inventory.get("hash_algorithm") != RECEIPT_TREE_HASH_ALGORITHM:
        raise ArtiverseContractError(
            "Artiverse extraction receipt inventory hash algorithm is invalid"
        )

    extractor_record = _receipt_exact_keys(
        receipt.get("safe_extractor"),
        {"version", "sha256", "filename"},
        "Artiverse extraction receipt safe extractor",
    )
    if (
        not isinstance(extractor_record.get("version"), str)
        or not extractor_record["version"]
        or extractor_record.get("filename") != SAFE_EXTRACTOR_FILENAME
        or not _is_sha256(extractor_record.get("sha256"))
    ):
        raise ArtiverseContractError(
            "Artiverse extraction receipt safe-extractor evidence is invalid"
        )
    scripts_root = Path(__file__).resolve().parent
    extractor_path = require_regular_file_within(
        scripts_root / SAFE_EXTRACTOR_FILENAME,
        scripts_root,
        "Current Artiverse safe extractor",
    )
    extractor_sha256 = sha256_file(extractor_path)
    if extractor_record["sha256"] != extractor_sha256:
        raise ArtiverseContractError(
            "Current Artiverse safe extractor does not match the extraction receipt"
        )
    return ArtiverseExtractionEvidence(
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        safe_extractor_path=extractor_path,
        safe_extractor_sha256=extractor_sha256,
    )


@dataclass(frozen=True)
class ArtiverseAuthority:
    dataset_root: Path
    embeddings_path: Path
    preparation_manifest_path: Path
    source_manifest_path: Path
    source_extraction_receipt_path: Path
    source_manifest_sha256: str
    source_extraction_receipt_sha256: str
    source_pack_script_sha256: str
    safe_extractor_sha256: str
    preparation_manifest_sha256: str
    index_sha256: dict[str, str]
    physics_policy: dict[str, Any]
    physics_binding_sha256: str
    physics_bound_indexed_count: int
    metadata: dict[str, dict[str, Any]]
    embedding_index: list[str]

    def asset(self, object_id: str) -> dict[str, Any]:
        if not object_id.startswith("artiverse/"):
            raise ArtiverseContractError(
                f"Non-canonical Artiverse object ID: {object_id or '<missing>'}"
            )
        metadata = self.metadata.get(object_id)
        if not isinstance(metadata, dict):
            raise ArtiverseContractError(
                f"Artiverse object ID is absent from the prepared metadata index: {object_id}"
            )
        sdf_path = _resolve_manifest_relative_file(
            self.dataset_root,
            metadata.get("sdf_path"),
            f"Artiverse source SDF for {object_id}",
        )
        expected_hash = metadata.get("sdf_sha256")
        if not _is_sha256(expected_hash):
            raise ArtiverseContractError(
                f"Prepared metadata has no valid SDF hash for {object_id}"
            )
        actual_hash = sha256_file(sdf_path)
        if actual_hash != expected_hash:
            raise ArtiverseContractError(
                f"Official prepared SDF hash mismatch for {object_id}: "
                f"index={expected_hash}, current={actual_hash}"
            )
        expected_tree_hash = metadata.get("sdf_directory_tree_sha256")
        actual_tree_hash = sha256_directory_tree(sdf_path.parent)
        if not _is_sha256(expected_tree_hash) or actual_tree_hash != expected_tree_hash:
            raise ArtiverseContractError(
                f"Official prepared asset-tree hash mismatch for {object_id}"
            )
        publisher_urdf = _resolve_manifest_relative_file(
            self.dataset_root,
            metadata.get("publisher_urdf_path"),
            f"Publisher Artiverse URDF for {object_id}",
        )
        publisher_urdf_sha256 = sha256_file(publisher_urdf)
        if metadata.get("publisher_urdf_sha256") != publisher_urdf_sha256:
            raise ArtiverseContractError(
                f"Publisher Artiverse URDF changed for {object_id}"
            )
        publisher_physics_sha256 = publisher_urdf_physics_evidence(
            publisher_urdf
        ).sha256
        if metadata.get("publisher_link_physics_sha256") != publisher_physics_sha256:
            raise ArtiverseContractError(
                f"Publisher Artiverse physics changed for {object_id}"
            )
        emitted_physics_sha256 = emitted_sdf_physics_evidence(sdf_path).sha256
        if metadata.get("emitted_sdf_physics_sha256") != emitted_physics_sha256:
            raise ArtiverseContractError(
                f"Emitted Artiverse physics changed for {object_id}"
            )
        return {
            "object_id": object_id,
            "metadata": metadata,
            "source_sdf_path": sdf_path,
            "source_sdf_sha256": actual_hash,
            "source_tree_sha256": actual_tree_hash,
            "publisher_urdf_path": publisher_urdf,
            "publisher_urdf_sha256": publisher_urdf_sha256,
            "publisher_link_physics_sha256": publisher_physics_sha256,
            "emitted_sdf_physics_sha256": emitted_physics_sha256,
            "physics_binding_sha256": self.physics_binding_sha256,
        }

    def evidence(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "source_repository": OFFICIAL_REPOSITORY,
            "source_revision": OFFICIAL_REVISION,
            "dataset_root": str(self.dataset_root),
            "preparation_manifest": {
                "path": str(self.preparation_manifest_path),
                "sha256": self.preparation_manifest_sha256,
            },
            "source_manifest": {
                "path": str(self.source_manifest_path),
                "sha256": self.source_manifest_sha256,
            },
            "source_extraction_receipt": {
                "path": str(self.source_extraction_receipt_path),
                "sha256": self.source_extraction_receipt_sha256,
            },
            "source_pack_script_sha256": self.source_pack_script_sha256,
            "safe_extractor_sha256": self.safe_extractor_sha256,
            "index_sha256": dict(sorted(self.index_sha256.items())),
            "indexed_count": len(self.embedding_index),
            "physics_policy": dict(self.physics_policy),
            "physics_bound_indexed_count": self.physics_bound_indexed_count,
            "physics_binding_sha256": self.physics_binding_sha256,
        }


def load_artiverse_authority(
    dataset_root: Path, embeddings_path: Path | None = None
) -> ArtiverseAuthority:
    try:
        root = dataset_root.resolve(strict=True)
    except OSError as exc:
        raise ArtiverseContractError(
            f"Official Artiverse dataset root is missing: {dataset_root}"
        ) from exc
    if not root.is_dir():
        raise ArtiverseContractError(f"Artiverse dataset root is not a directory: {root}")

    raw_embeddings = embeddings_path or (root / "embeddings")
    try:
        embeddings = raw_embeddings.resolve(strict=True)
    except OSError as exc:
        raise ArtiverseContractError(
            f"Artiverse embeddings directory is missing: {raw_embeddings}"
        ) from exc
    _require_within(embeddings, root, "Artiverse embeddings directory")

    preparation_path = embeddings / PREPARATION_MANIFEST
    preparation = _load_json(preparation_path, "Artiverse preparation manifest")
    if preparation.get("schema_version") != 2:
        raise ArtiverseContractError("Unsupported Artiverse preparation schema")
    if preparation.get("status") != "pass":
        raise ArtiverseContractError("Artiverse preparation manifest did not pass")
    if preparation.get("source_repository") != OFFICIAL_REPOSITORY:
        raise ArtiverseContractError(
            "Artiverse preparation manifest is not bound to the official repository"
        )
    if preparation.get("source_revision") != OFFICIAL_REVISION:
        raise ArtiverseContractError(
            "Artiverse preparation manifest is not bound to the pinned official revision"
        )
    if Path(str(preparation.get("dataset_root", ""))).resolve() != root:
        raise ArtiverseContractError(
            "Artiverse preparation manifest dataset_root does not match the mounted dataset"
        )
    if Path(str(preparation.get("output_path", ""))).resolve() != embeddings:
        raise ArtiverseContractError(
            "Artiverse preparation manifest output_path does not match the mounted index"
        )
    physics_policy = _exact_physics_policy(preparation)

    source_record = preparation.get("source_manifest")
    if not isinstance(source_record, dict):
        raise ArtiverseContractError("Artiverse source-manifest evidence is missing")
    source_path = _resolve_manifest_relative_file(
        root, source_record.get("path"), "Official Artiverse chunk manifest"
    )
    source_hash = source_record.get("sha256")
    if source_hash != OFFICIAL_SOURCE_MANIFEST_SHA256:
        raise ArtiverseContractError(
            "Artiverse chunk-manifest evidence is not bound to the pinned "
            "official digest"
        )
    if sha256_file(source_path) != OFFICIAL_SOURCE_MANIFEST_SHA256:
        raise ArtiverseContractError(
            "Official Artiverse chunk-manifest content does not match the "
            "pinned digest"
        )
    pack_script_path = root / "pack_dataset_chunks.py"
    pack_script_hash = preparation.get("source_pack_script_sha256")
    if pack_script_hash != OFFICIAL_PACK_SCRIPT_SHA256:
        raise ArtiverseContractError(
            "Artiverse unpack-script evidence is not bound to the pinned "
            "official digest"
        )
    if (
        not pack_script_path.is_file()
        or sha256_file(pack_script_path) != OFFICIAL_PACK_SCRIPT_SHA256
    ):
        raise ArtiverseContractError(
            "Official Artiverse unpack-script content does not match the "
            "pinned digest"
        )

    extraction = validate_extraction_receipt(root)
    extraction_record = preparation.get("source_extraction_receipt")
    if not isinstance(extraction_record, dict) or set(extraction_record) != {
        "path",
        "sha256",
    }:
        raise ArtiverseContractError(
            "Artiverse preparation manifest has no extraction-receipt binding"
        )
    recorded_receipt_path = _resolve_manifest_relative_file(
        root,
        extraction_record.get("path"),
        "Prepared Artiverse safe-extraction receipt",
    )
    if recorded_receipt_path != extraction.receipt_path:
        raise ArtiverseContractError(
            "Artiverse preparation manifest points to the wrong extraction receipt"
        )
    if extraction_record.get("sha256") != extraction.receipt_sha256:
        raise ArtiverseContractError(
            "Artiverse extraction receipt changed after preparation"
        )
    if preparation.get("safe_extractor_sha256") != extraction.safe_extractor_sha256:
        raise ArtiverseContractError(
            "Artiverse safe extractor changed after preparation"
        )

    expected_index_hashes = preparation.get("index_sha256")
    if not isinstance(expected_index_hashes, dict):
        raise ArtiverseContractError("Prepared Artiverse index hashes are missing")
    actual_index_hashes: dict[str, str] = {}
    for filename in INDEX_FILENAMES:
        path = embeddings / filename
        if not path.is_file():
            raise ArtiverseContractError(f"Prepared Artiverse index is missing: {path}")
        expected_hash = expected_index_hashes.get(filename)
        actual_hash = sha256_file(path)
        if not _is_sha256(expected_hash) or actual_hash != expected_hash:
            raise ArtiverseContractError(
                f"Prepared Artiverse index hash mismatch for {filename}"
            )
        actual_index_hashes[filename] = actual_hash

    try:
        embedding_index = yaml.safe_load(
            (embeddings / "embedding_index.yaml").read_text(encoding="utf-8")
        )
        metadata = yaml.safe_load(
            (embeddings / "metadata_index.yaml").read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ArtiverseContractError(f"Cannot parse prepared Artiverse indexes: {exc}") from exc
    if not isinstance(embedding_index, list) or not all(
        isinstance(value, str) for value in embedding_index
    ):
        raise ArtiverseContractError("Artiverse embedding index must be a string list")
    if not isinstance(metadata, dict):
        raise ArtiverseContractError("Artiverse metadata index must be a mapping")
    if len(embedding_index) != len(set(embedding_index)):
        raise ArtiverseContractError("Artiverse embedding index contains duplicate IDs")
    if set(embedding_index) != set(metadata):
        raise ArtiverseContractError("Artiverse embedding and metadata IDs differ")
    if int(preparation.get("indexed_count", -1)) != len(embedding_index):
        raise ArtiverseContractError("Artiverse manifest indexed_count is stale")
    if len(embedding_index) < int(preparation.get("minimum_indexed", 1)):
        raise ArtiverseContractError("Artiverse index is below its required minimum")
    try:
        embeddings_array = np.load(
            embeddings / "clip_embeddings.npy", mmap_mode="r", allow_pickle=False
        )
    except (OSError, ValueError) as exc:
        raise ArtiverseContractError(f"Cannot load Artiverse CLIP embeddings: {exc}") from exc
    if (
        embeddings_array.dtype != np.float32
        or embeddings_array.ndim != 2
        or embeddings_array.shape != (len(embedding_index), 1024)
        or not np.isfinite(embeddings_array).all()
    ):
        raise ArtiverseContractError(
            "Artiverse CLIP embeddings must be finite float32 with shape (N, 1024)"
        )

    collision_cap = int(preparation.get("maximum_collision_elements", 0))
    if collision_cap < 1:
        raise ArtiverseContractError("Artiverse collision cap is missing or invalid")
    if _exact_metadata_int(
        preparation.get("physics_bound_indexed_count"),
        "Artiverse physics_bound_indexed_count",
        minimum=1,
    ) != len(embedding_index):
        raise ArtiverseContractError(
            "Artiverse preparation physics_bound_indexed_count is stale"
        )
    recorded_binding_sha256 = preparation.get("physics_binding_sha256")
    if not _is_sha256(recorded_binding_sha256):
        raise ArtiverseContractError(
            "Artiverse preparation physics_binding_sha256 is invalid"
        )
    data_root = root / "data" if (root / "data").is_dir() else root
    for object_id, record in metadata.items():
        if not isinstance(record, dict):
            raise ArtiverseContractError(
                f"Artiverse metadata record is not a mapping for {object_id}"
            )
        components = object_id.split("/")
        if (
            len(components) != 4
            or components[0] != "artiverse"
            or not all(_is_safe_path_component(value) for value in components[1:])
        ):
            raise ArtiverseContractError(f"Non-canonical Artiverse index ID: {object_id}")
        _, category, source, model_id = components
        if (
            record.get("category") != category
            or record.get("artiverse_source") != source
            or record.get("artiverse_model_id") != model_id
        ):
            raise ArtiverseContractError(
                f"Artiverse ID components disagree with metadata for {object_id}"
            )
        publisher_urdf_path = record.get("publisher_urdf_path")
        if not isinstance(publisher_urdf_path, str) or not publisher_urdf_path:
            raise ArtiverseContractError(
                f"Artiverse publisher_urdf_path is missing or invalid for {object_id}"
            )
        for digest_field in (
            "publisher_urdf_sha256",
            "publisher_link_physics_sha256",
            "emitted_sdf_physics_sha256",
        ):
            if not _is_sha256(record.get(digest_field)):
                raise ArtiverseContractError(
                    f"Artiverse {digest_field} is missing or invalid for {object_id}"
                )
        expected_model_dir = (data_root / category / source / model_id).resolve()
        _require_within(expected_model_dir, data_root, f"Artiverse model {object_id}")
        publisher_urdf = _resolve_manifest_relative_file(
            root,
            publisher_urdf_path,
            f"Publisher Artiverse URDF for {object_id}",
        )
        _require_within(
            publisher_urdf,
            expected_model_dir,
            f"Publisher Artiverse URDF for {object_id}",
        )
        canonical_publisher_path = publisher_urdf.relative_to(root).as_posix()
        if record.get("publisher_urdf_path") != canonical_publisher_path:
            raise ArtiverseContractError(
                f"Artiverse publisher_urdf_path is not canonical for {object_id}"
            )
        selected_publisher_urdf = _select_publisher_urdf(
            expected_model_dir, object_id
        )
        if publisher_urdf != selected_publisher_urdf:
            raise ArtiverseContractError(
                f"Artiverse publisher_urdf_path does not select the canonical URDF for {object_id}"
            )
        publisher_urdf_hash = sha256_file(publisher_urdf)
        if record.get("publisher_urdf_sha256") != publisher_urdf_hash:
            raise ArtiverseContractError(
                f"Artiverse publisher URDF hash is stale for {object_id}"
            )
        publisher_physics = publisher_urdf_physics_evidence(publisher_urdf)
        if record.get("publisher_link_physics_sha256") != publisher_physics.sha256:
            raise ArtiverseContractError(
                f"Artiverse publisher physics digest is stale for {object_id}"
            )
        sdf_path = _resolve_manifest_relative_file(
            root, record.get("sdf_path"), f"Artiverse source SDF for {object_id}"
        )
        _require_within(sdf_path, expected_model_dir, f"Artiverse source SDF for {object_id}")
        try:
            document = ET.parse(sdf_path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise ArtiverseContractError(
                f"Cannot parse prepared Artiverse SDF for {object_id}: {exc}"
            ) from exc
        emitted_physics = emitted_sdf_physics_evidence(sdf_path)
        if record.get("emitted_sdf_physics_sha256") != emitted_physics.sha256:
            raise ArtiverseContractError(
                f"Artiverse emitted SDF physics digest is stale for {object_id}"
            )
        _validate_publisher_sdf_preservation(
            publisher_physics,
            emitted_physics,
            object_id=object_id,
        )
        if record.get("physics_source") != PHYSICS_POLICY_ID:
            raise ArtiverseContractError(
                f"Artiverse physics_source is missing or stale for {object_id}"
            )
        if _exact_metadata_int(
            record.get("physics_link_count"),
            f"Artiverse physics_link_count for {object_id}",
            minimum=1,
        ) != len(publisher_physics.link_values):
            raise ArtiverseContractError(
                f"Artiverse publisher physics link count is stale for {object_id}"
            )
        if _exact_metadata_int(
            record.get("physics_geometry_link_count"),
            f"Artiverse physics_geometry_link_count for {object_id}",
            minimum=1,
        ) != len(publisher_physics.geometry_links):
            raise ArtiverseContractError(
                f"Artiverse publisher physics geometry-link count is stale for {object_id}"
            )
        if _exact_metadata_int(
            record.get("source_collision_element_count"),
            f"Artiverse source_collision_element_count for {object_id}",
            minimum=1,
        ) != sum(publisher_physics.collision_count_by_link.values()):
            raise ArtiverseContractError(
                f"Artiverse publisher collision count is stale for {object_id}"
            )
        recorded_total_mass = record.get("physics_total_mass_kg")
        if (
            isinstance(recorded_total_mass, bool)
            or not isinstance(recorded_total_mass, (int, float))
            or not math.isfinite(float(recorded_total_mass))
            or not math.isclose(
                float(recorded_total_mass),
                publisher_physics.total_mass_kg,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ):
            raise ArtiverseContractError(
                f"Artiverse publisher physics total mass is stale for {object_id}"
            )
        _validate_sdf_physics_and_joints(
            document,
            record,
            object_id=object_id,
            collision_cap=collision_cap,
            physics_policy=physics_policy,
        )
        validate_regular_directory_tree(
            expected_model_dir,
            containment_root=data_root,
            label=f"Artiverse model tree for {object_id}",
        )
        validate_sdf_resource_uris(sdf_path, expected_model_dir)

    actual_binding_sha256 = physics_binding_sha256(metadata, embedding_index)
    if actual_binding_sha256 != recorded_binding_sha256:
        raise ArtiverseContractError(
            "Artiverse preparation physics_binding_sha256 is stale"
        )

    authority = ArtiverseAuthority(
        dataset_root=root,
        embeddings_path=embeddings,
        preparation_manifest_path=preparation_path.resolve(),
        source_manifest_path=source_path,
        source_extraction_receipt_path=extraction.receipt_path,
        source_manifest_sha256=source_hash,
        source_extraction_receipt_sha256=extraction.receipt_sha256,
        source_pack_script_sha256=pack_script_hash,
        safe_extractor_sha256=extraction.safe_extractor_sha256,
        preparation_manifest_sha256=sha256_file(preparation_path),
        index_sha256=actual_index_hashes,
        physics_policy=dict(physics_policy),
        physics_binding_sha256=actual_binding_sha256,
        physics_bound_indexed_count=len(embedding_index),
        metadata=metadata,
        embedding_index=embedding_index,
    )
    for object_id in authority.embedding_index:
        authority.asset(object_id)
    return authority


def validate_usage_manifest(
    usage_manifest_path: Path,
    house_state_path: Path,
    authority: ArtiverseAuthority,
    *,
    scene_dir: Path,
) -> dict[str, Any]:
    usage_path = usage_manifest_path.resolve()
    try:
        canonical_scene = scene_dir.resolve(strict=True)
    except OSError as exc:
        raise ArtiverseContractError(
            f"Artiverse scene directory is missing: {scene_dir}"
        ) from exc
    if not canonical_scene.is_dir():
        raise ArtiverseContractError(
            f"Artiverse scene path is not a directory: {canonical_scene}"
        )
    usage = _load_json(usage_path, "Artiverse usage manifest")
    if usage.get("schema_version") != 2 or usage.get("status") != "pass":
        raise ArtiverseContractError("Artiverse usage manifest is not a passing schema v2 record")
    if usage.get("dataset") != "artiverse":
        raise ArtiverseContractError("Artiverse usage manifest dataset label is invalid")
    if usage.get("authority") != authority.evidence():
        raise ArtiverseContractError("Artiverse usage authority is missing or stale")

    placed = usage.get("placed_assets")
    final = usage.get("final_surviving_assets")
    if not isinstance(placed, list) or not isinstance(final, list):
        raise ArtiverseContractError("Artiverse usage asset lists are malformed")
    if not placed or not final:
        raise ArtiverseContractError("Artiverse usage must contain placed and surviving assets")
    if usage.get("placed_asset_count") != len(placed):
        raise ArtiverseContractError("Artiverse placed-asset count is stale")
    if usage.get("final_surviving_asset_count") != len(final):
        raise ArtiverseContractError("Artiverse surviving-asset count is stale")

    def validate_record(record: Any) -> tuple[tuple[str, str, str], Path]:
        if not isinstance(record, dict):
            raise ArtiverseContractError("Artiverse usage asset record is malformed")
        identity = (
            str(record.get("room_id", "")),
            str(record.get("object_id", "")),
            str(record.get("articulated_id", "")),
        )
        if not all(identity):
            raise ArtiverseContractError("Artiverse usage asset identity is incomplete")
        if not _is_safe_path_component(identity[0]):
            raise ArtiverseContractError(
                f"Artiverse usage has an unsafe room identifier: {identity[0]}"
            )
        expected = authority.asset(identity[2])
        source_value = str(record.get("source_sdf_path", ""))
        source_path = Path(source_value)
        if not source_value or not source_path.is_absolute():
            raise ArtiverseContractError(
                f"Artiverse source path must be absolute for {identity[2]}"
            )
        source_sdf = require_regular_file_within(
            source_path,
            authority.dataset_root,
            f"Artiverse source SDF in usage record for {identity[2]}",
        )
        if source_sdf != expected["source_sdf_path"]:
            raise ArtiverseContractError(f"Artiverse source path is stale for {identity[2]}")
        if record.get("source_sdf_sha256") != expected["source_sdf_sha256"]:
            raise ArtiverseContractError(f"Artiverse source hash is stale for {identity[2]}")
        if record.get("source_tree_sha256") != expected["source_tree_sha256"]:
            raise ArtiverseContractError(
                f"Artiverse source asset-tree hash is stale for {identity[2]}"
            )
        copied_value = str(record.get("sdf_path", ""))
        copied_path = Path(copied_value)
        if not copied_value or not copied_path.is_absolute():
            raise ArtiverseContractError(
                f"Placed Artiverse SDF path must be absolute for {identity[2]}"
            )
        expected_copy_root = (
            canonical_scene
            / f"room_{identity[0]}"
            / "generated_assets"
            / "sdf"
        )
        copied_sdf = require_regular_file_within(
            copied_path,
            expected_copy_root,
            f"Placed Artiverse SDF for {identity[0]}/{identity[1]}",
        )
        validate_regular_directory_tree(
            copied_sdf.parent,
            containment_root=expected_copy_root,
            label=f"Placed Artiverse asset tree for {identity[0]}/{identity[1]}",
        )
        validate_sdf_resource_uris(copied_sdf, copied_sdf.parent)
        if record.get("sdf_sha256") != sha256_file(copied_sdf):
            raise ArtiverseContractError(f"Placed Artiverse SDF hash is stale: {copied_sdf}")
        if record.get("sdf_tree_sha256") != sha256_directory_tree(copied_sdf.parent):
            raise ArtiverseContractError(
                f"Placed Artiverse asset-tree hash is stale: {copied_sdf.parent}"
            )
        return identity, copied_sdf

    placed_entries = [validate_record(record) for record in placed]
    final_entries = [validate_record(record) for record in final]
    placed_ids = {identity for identity, _path in placed_entries}
    final_ids = {identity for identity, _path in final_entries}
    if len(placed_ids) != len(placed_entries):
        raise ArtiverseContractError("Artiverse placed-asset records contain duplicates")
    if len(final_ids) != len(final_entries):
        raise ArtiverseContractError("Artiverse surviving-asset records contain duplicates")
    if not placed_ids.issubset(final_ids):
        raise ArtiverseContractError("Final house lost a placed Artiverse asset")

    placed_paths = {identity: path for identity, path in placed_entries}
    final_paths = {identity: path for identity, path in final_entries}

    def state_artiverse_assets(
        state: Any,
        room_id: str,
        expected_paths: dict[tuple[str, str, str], Path],
        *,
        label: str,
    ) -> dict[tuple[str, str, str], Path]:
        if not isinstance(state, dict):
            raise ArtiverseContractError(f"{label} is not a JSON object")
        objects = state.get("objects")
        if isinstance(objects, dict):
            items = list(objects.items())
        elif isinstance(objects, list):
            items = list(enumerate(objects))
        else:
            raise ArtiverseContractError(f"{label} has no object collection")

        actual: dict[tuple[str, str, str], Path] = {}
        for state_object_id, obj in items:
            if not isinstance(obj, dict):
                continue
            metadata = obj.get("metadata")
            if not isinstance(metadata, dict) or str(
                metadata.get("articulated_source", "")
            ).strip().lower() != "artiverse":
                continue

            identity = (
                room_id,
                str(obj.get("object_id") or state_object_id),
                str(metadata.get("articulated_id", "")).strip(),
            )
            if not all(identity):
                raise ArtiverseContractError(
                    f"{label} contains an Artiverse-labelled object with incomplete identity"
                )
            if identity in actual:
                raise ArtiverseContractError(
                    f"{label} contains duplicate Artiverse object identity {identity}"
                )
            expected_copy = expected_paths.get(identity)
            if expected_copy is None:
                raise ArtiverseContractError(
                    f"{label} contains an Artiverse object omitted from usage evidence: {identity}"
                )
            if (
                str(metadata.get("asset_source", "")).strip().lower() != "articulated"
                or metadata.get("is_articulated") is not True
            ):
                raise ArtiverseContractError(
                    f"{label} contains forged Artiverse runtime labels: {identity}"
                )

            expected_source = authority.asset(identity[2])
            source_value = str(metadata.get("articulated_source_sdf_path", ""))
            if not source_value or Path(source_value).resolve() != expected_source[
                "source_sdf_path"
            ]:
                raise ArtiverseContractError(
                    f"{label} has stale Artiverse source path: {identity}"
                )
            if metadata.get("articulated_source_sdf_sha256") != expected_source[
                "source_sdf_sha256"
            ] or metadata.get("articulated_source_tree_sha256") != expected_source[
                "source_tree_sha256"
            ]:
                raise ArtiverseContractError(
                    f"{label} has stale Artiverse source hashes: {identity}"
                )

            sdf_value = str(obj.get("sdf_path", ""))
            if not sdf_value:
                raise ArtiverseContractError(
                    f"{label} has no placed SDF path for {identity}"
                )
            raw_sdf = Path(sdf_value)
            if not raw_sdf.is_absolute():
                raw_sdf = canonical_scene / f"room_{room_id}" / raw_sdf
            copied_sdf = require_regular_file_within(
                raw_sdf,
                canonical_scene / f"room_{room_id}" / "generated_assets" / "sdf",
                f"{label} Artiverse SDF for {identity}",
            )
            if copied_sdf != expected_copy:
                raise ArtiverseContractError(
                    f"{label} Artiverse SDF differs from usage evidence: {identity}"
                )
            if metadata.get("articulated_copied_sdf_sha256") != sha256_file(
                copied_sdf
            ) or metadata.get("articulated_copied_tree_sha256") != sha256_directory_tree(
                copied_sdf.parent
            ):
                raise ArtiverseContractError(
                    f"{label} has stale placed Artiverse hashes: {identity}"
                )
            actual[identity] = copied_sdf
        return actual

    placed_identifiers = sorted({identity[2] for identity in placed_ids})
    final_identifiers = sorted({identity[2] for identity in final_ids})
    if usage.get("placed_asset_identifiers") != placed_identifiers:
        raise ArtiverseContractError("Artiverse placed identifier summary is stale")
    if usage.get("final_surviving_asset_identifiers") != final_identifiers:
        raise ArtiverseContractError("Artiverse surviving identifier summary is stale")

    canonical_house = house_state_path.resolve()
    if not canonical_house.is_file():
        raise ArtiverseContractError(f"Combined house state is missing: {canonical_house}")
    house_record = usage.get("house_state")
    if not isinstance(house_record, dict):
        raise ArtiverseContractError("Artiverse usage has no combined-house hash binding")
    if Path(str(house_record.get("path", ""))).resolve() != canonical_house:
        raise ArtiverseContractError("Artiverse usage is bound to a different combined house")
    house_hash = sha256_file(canonical_house)
    if house_record.get("sha256") != house_hash:
        raise ArtiverseContractError("Artiverse usage combined-house hash is stale")

    try:
        house_state = json.loads(canonical_house.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtiverseContractError(f"Cannot parse combined house state: {exc}") from exc
    rooms = house_state.get("rooms")
    if not isinstance(rooms, dict) or not rooms:
        raise ArtiverseContractError("Combined house state contains no rooms")
    room_records = usage.get("room_states")
    if not isinstance(room_records, list):
        raise ArtiverseContractError("Artiverse usage room-state evidence is missing")
    recorded_by_id = {
        str(record.get("room_id")): record
        for record in room_records
        if isinstance(record, dict) and record.get("room_id")
    }
    if set(recorded_by_id) != {str(room_id) for room_id in rooms}:
        raise ArtiverseContractError("Artiverse room-state evidence set is stale")
    actual_placed: dict[tuple[str, str, str], Path] = {}
    for room_id, record in recorded_by_id.items():
        expected_path = (
            canonical_scene
            / f"room_{room_id}"
            / "scene_states"
            / "final_scene"
            / "scene_state.json"
        ).resolve()
        if Path(str(record.get("path", ""))).resolve() != expected_path:
            raise ArtiverseContractError(f"Room-state evidence path is stale: {room_id}")
        if not expected_path.is_file() or record.get("sha256") != sha256_file(expected_path):
            raise ArtiverseContractError(f"Room-state evidence hash is stale: {room_id}")

        try:
            room_state = json.loads(expected_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtiverseContractError(
                f"Cannot parse Artiverse-bound room state {room_id}: {exc}"
            ) from exc
        actual_placed.update(
            state_artiverse_assets(
                room_state,
                room_id,
                placed_paths,
                label=f"room state {room_id}",
            )
        )

    if set(actual_placed) != placed_ids:
        raise ArtiverseContractError(
            "Artiverse placed-asset usage does not exactly match the bound room states"
        )

    actual_final: dict[tuple[str, str, str], Path] = {}
    for room_id, room_state in rooms.items():
        actual_final.update(
            state_artiverse_assets(
                room_state,
                str(room_id),
                final_paths,
                label=f"combined house room {room_id}",
            )
        )
    if set(actual_final) != final_ids:
        raise ArtiverseContractError(
            "Artiverse final-survivor usage does not exactly match the combined house state"
        )

    return {
        "status": "pass",
        "schema_version": 1,
        "usage_manifest": {"path": str(usage_path), "sha256": sha256_file(usage_path)},
        "house_state": {"path": str(canonical_house), "sha256": house_hash},
        "authority": authority.evidence(),
        "placed_asset_count": len(placed),
        "final_surviving_asset_count": len(final),
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/artiverse"))
    parser.add_argument("--embeddings-path", type=Path)
    parser.add_argument("--usage-manifest", type=Path)
    parser.add_argument("--house-state", type=Path)
    parser.add_argument("--scene-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        authority = load_artiverse_authority(args.dataset_root, args.embeddings_path)
        if args.usage_manifest:
            if not args.house_state or not args.scene_dir:
                parser.error(
                    "--house-state and --scene-dir are required with --usage-manifest"
                )
            result = validate_usage_manifest(
                args.usage_manifest,
                args.house_state,
                authority,
                scene_dir=args.scene_dir,
            )
        else:
            result = {"status": "pass", "authority": authority.evidence()}
        _atomic_write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ArtiverseContractError as exc:
        result = {"status": "fail", "error": str(exc)}
        _atomic_write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
