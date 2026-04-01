#!/usr/bin/env python3
"""Require an independent stability pass after a manipuland settles."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path


EXPECTED_SHA256 = "eb282cc7e56b3bb4870cde1ff215548ad30672a8f806571f7e2638116f068a91"


OLD = '''    if unstable_ids:
        for manip_id, original_transform in original_transforms.items():
            if full_scene.get_object(manip_id) is not None:
                full_scene.move_object(
                    object_id=manip_id, new_transform=original_transform
                )
        details = ", ".join(unstable_ids)
        console_logger.error(
            "Per-furniture physical validation failed for %s: %s",
            furniture_id,
            details,
        )
        raise RuntimeError(
            "Per-furniture physical validation rejected "
            f"{furniture_id}; manipulands are unstable: {details}"
        )
'''


NEW = '''    if unstable_ids:
        # A thin object can legitimately tip from an imprecise authored pose into
        # a stable resting pose.  Do not accept that first movement by itself:
        # independently simulate the settled pose again and require the exact
        # same strict 2 cm / 5 degree limits.  This preserves the stability gate
        # while avoiding a full room restart for already-stable final geometry.
        settled_transforms: dict[str, RigidTransform] = {}
        revalidation_issues: list[str] = []
        for manip_id in manipuland_ids:
            settled_manip = processed_subset.get_object(manip_id)
            if settled_manip is None:
                revalidation_issues.append(f"{manip_id} (removed before recheck)")
                continue
            settled_transforms[manip_id] = RigidTransform(
                settled_manip.transform.GetAsMatrix4()
            )

        if len(settled_transforms) == len(manipuland_ids):
            revalidated_subset, revalidation_success, _ = (
                apply_physical_feasibility_postprocessing(
                    scene=processed_subset,
                    weld_furniture=True,
                    projection_enabled=projection_cfg.enabled,
                    projection_influence_distance=projection_cfg.influence_distance,
                    projection_solver_name=projection_cfg.solver_name,
                    projection_iteration_limit=projection_cfg.iteration_limit,
                    projection_time_limit_s=projection_cfg.time_limit_s,
                    projection_xy_only=projection_cfg.xy_only,
                    projection_fix_rotation=projection_cfg.fix_rotation,
                    simulation_enabled=simulation_cfg.enabled,
                    simulation_time_s=simulation_cfg.simulation_time_s,
                    simulation_time_step_s=simulation_cfg.time_step_s,
                    simulation_timeout_s=simulation_cfg.timeout_s,
                    simulation_html_path=simulation_html_path,
                    remove_fallen_furniture=False,
                    remove_fallen_manipulands=(
                        simulation_cfg.remove_fallen_manipulands
                    ),
                    fallen_manipuland_floor_z=(
                        simulation_cfg.fallen_manipuland_floor_z
                    ),
                    fallen_manipuland_near_floor_z=(
                        simulation_cfg.fallen_manipuland_near_floor_z
                    ),
                    fallen_manipuland_z_displacement=(
                        simulation_cfg.fallen_manipuland_z_displacement
                    ),
                )
            )
            if not revalidation_success:
                revalidation_issues.append("second physical pass reported issues")
            for manip_id, settled_transform in settled_transforms.items():
                revalidated_manip = revalidated_subset.get_object(manip_id)
                if revalidated_manip is None:
                    revalidation_issues.append(
                        f"{manip_id} (removed during recheck)"
                    )
                    continue
                translation_delta_m = float(
                    np.linalg.norm(
                        revalidated_manip.transform.translation()
                        - settled_transform.translation()
                    )
                )
                relative_rotation = (
                    revalidated_manip.transform.rotation().matrix()
                    @ settled_transform.rotation().matrix().T
                )
                rotation_delta_deg = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                (np.trace(relative_rotation) - 1.0) / 2.0,
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                if translation_delta_m > 0.02 or rotation_delta_deg > 5.0:
                    revalidation_issues.append(
                        f"{manip_id} (second translation="
                        f"{translation_delta_m:.3f}m, second rotation="
                        f"{rotation_delta_deg:.1f} degrees)"
                    )

            if not revalidation_issues:
                processed_subset = revalidated_subset
                unstable_ids = []
                console_logger.info(
                    "Accepted independently revalidated settled poses for %s",
                    furniture_id,
                )

    if unstable_ids:
        for manip_id, original_transform in original_transforms.items():
            if full_scene.get_object(manip_id) is not None:
                full_scene.move_object(
                    object_id=manip_id, new_transform=original_transform
                )
        details = ", ".join(unstable_ids + revalidation_issues)
        console_logger.error(
            "Per-furniture physical validation failed for %s: %s",
            furniture_id,
            details,
        )
        raise RuntimeError(
            "Per-furniture physical validation rejected "
            f"{furniture_id}; manipulands are unstable: {details}"
        )
'''


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    args = parser.parse_args()

    target = args.target.resolve(strict=True)
    before = digest(target)
    if before != EXPECTED_SHA256:
        raise SystemExit(f"unexpected physical-feasibility digest: {before}")
    source = target.read_text(encoding="utf-8")
    if source.count(OLD) != 1:
        raise SystemExit("settling-validation anchor mismatch")

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir / f"{target.name}.{before}.backup"
    if not backup.exists():
        shutil.copy2(target, backup, follow_symlinks=False)
    elif digest(backup) != before:
        raise SystemExit("physical-feasibility backup mismatch")

    updated = source.replace(OLD, NEW, 1).encode("utf-8")
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        target.stat().st_mode & 0o777,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    print("SETTLED_POSE_REVALIDATION_PATCH_PASS", digest(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
