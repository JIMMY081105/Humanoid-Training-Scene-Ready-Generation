#!/usr/bin/env python3
"""Load a final SceneSmith Drake directive and record acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Any

import yaml


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _collision_report(
    scene_dir: Path, *, max_collision_elements: int
) -> dict[str, Any]:
    """Inspect every SDF shipped in the final scene package."""

    records: list[dict[str, Any]] = []
    for sdf_path in sorted(scene_dir.rglob("*.sdf")):
        try:
            count = len(ET.parse(sdf_path).getroot().findall(".//collision"))
        except (OSError, ET.ParseError) as exc:
            records.append(
                {"path": str(sdf_path), "collision_count": None, "error": str(exc)}
            )
            continue
        records.append({"path": str(sdf_path), "collision_count": count})

    numeric = [
        record["collision_count"]
        for record in records
        if isinstance(record["collision_count"], int)
    ]
    maximum = max(numeric, default=0)
    return {
        "sdf_asset_count": len(records),
        "collision_element_cap": max_collision_elements,
        "max_collision_elements_per_sdf": maximum,
        "assets_over_collision_cap": [
            record
            for record in records
            if isinstance(record["collision_count"], int)
            and record["collision_count"] > max_collision_elements
        ],
        # Retained as a diagnostic for compatibility with earlier reports. It is
        # not used for acceptance when a different cap is requested.
        "assets_over_32_collision_elements": [
            record
            for record in records
            if isinstance(record["collision_count"], int)
            and record["collision_count"] > 32
        ],
        "parse_failures": [record for record in records if record.get("error")],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dmd_inventory(dmd_path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(dmd_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return {"status": "fail", "error": str(exc), "model_directive_count": 0}
    directives = document.get("directives") if isinstance(document, dict) else None
    if not isinstance(directives, list):
        return {
            "status": "fail",
            "error": "Drake directives YAML has no directives list",
            "model_directive_count": 0,
        }
    model_directives = [
        item["add_model"]
        for item in directives
        if isinstance(item, dict) and isinstance(item.get("add_model"), dict)
    ]
    names = [str(item.get("name") or "").strip() for item in model_directives]
    files = [str(item.get("file") or "").strip() for item in model_directives]
    malformed = [
        index
        for index, (name, file_value) in enumerate(zip(names, files))
        if not name or not file_value
    ]
    duplicates = sorted(name for name in set(names) if name and names.count(name) > 1)
    status = "pass" if model_directives and not malformed and not duplicates else "fail"
    return {
        "status": status,
        "sha256": _sha256_file(dmd_path),
        "directive_count": len(directives),
        "model_directive_count": len(model_directives),
        "model_names": names,
        "malformed_model_directive_indices": malformed,
        "duplicate_model_names": duplicates,
    }


def _house_state_inventory(package_root: Path) -> dict[str, Any]:
    path = package_root / "combined_house" / "house_state.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "fail", "path": str(path), "error": str(exc)}
    rooms = document.get("rooms") if isinstance(document, dict) else None
    if not isinstance(rooms, dict):
        return {
            "status": "fail",
            "path": str(path),
            "error": "combined house state has no rooms mapping",
        }
    object_count = 0
    malformed_rooms: list[str] = []
    for room_id, room_state in rooms.items():
        if not isinstance(room_state, dict):
            malformed_rooms.append(str(room_id))
            continue
        objects = room_state.get("objects")
        if isinstance(objects, (dict, list)):
            object_count += len(objects)
        else:
            malformed_rooms.append(str(room_id))
    return {
        "status": (
            "pass"
            if rooms and object_count > 0 and not malformed_rooms
            else "fail"
        ),
        "path": str(path),
        "sha256": _sha256_file(path),
        "room_count": len(rooms),
        "object_count": object_count,
        "malformed_room_ids": sorted(malformed_rooms),
    }


GPU_EXERCISE_ELEMENT_COUNT = 256


def _gpu_inventory(required_execution_count: int = 0) -> dict[str, Any]:
    """Inventory CUDA and execute a tiny, synchronized workload per required GPU.

    Device visibility alone is not an execution proof.  The final acceptance
    therefore allocates a tensor, launches a reduction, reads the result, and
    synchronizes each required device independently.  The workload is deliberately
    tiny (1 KiB of float32 payload) so it tests allocation/execution without
    competing materially with the Drake CPU-side load.
    """

    try:
        import torch

        count = int(torch.cuda.device_count())
        names = [torch.cuda.get_device_name(index) for index in range(count)]
        exercises: list[dict[str, Any]] = []
        for index in range(min(count, required_execution_count)):
            record: dict[str, Any] = {
                "index": index,
                "name": names[index],
                "status": "fail",
                "allocated_bytes": GPU_EXERCISE_ELEMENT_COUNT * 4,
                "synchronized": False,
            }
            try:
                properties = torch.cuda.get_device_properties(index)
                free_before, total_before = torch.cuda.mem_get_info(index)
                with torch.cuda.device(index):
                    tensor = torch.full(
                        (GPU_EXERCISE_ELEMENT_COUNT,),
                        float(index + 1),
                        dtype=torch.float32,
                        device=f"cuda:{index}",
                    )
                    observed_sum = float(tensor.sum().item())
                    torch.cuda.synchronize(index)
                expected_sum = float((index + 1) * GPU_EXERCISE_ELEMENT_COUNT)
                if observed_sum != expected_sum:
                    raise RuntimeError(
                        f"CUDA reduction mismatch: expected {expected_sum}, got {observed_sum}"
                    )
                record.update(
                    {
                        "status": "pass",
                        "synchronized": True,
                        "observed_sum": observed_sum,
                        "total_memory_bytes": int(properties.total_memory),
                        "free_memory_before_bytes": int(free_before),
                        "reported_total_memory_bytes": int(total_before),
                    }
                )
                del tensor
            except Exception as exc:
                record.update(
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            exercises.append(record)
        exercised = (
            count >= required_execution_count
            and len(exercises) == required_execution_count
            and all(record.get("status") == "pass" for record in exercises)
        )
        return {
            "count": count,
            "names": names,
            "required_execution_count": required_execution_count,
            "execution_exercises": exercises,
            "all_required_devices_exercised": exercised,
        }
    except Exception as exc:
        return {
            "count": 0,
            "names": [],
            "required_execution_count": required_execution_count,
            "execution_exercises": [],
            "all_required_devices_exercised": required_execution_count == 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _load_with_drake(dmd_path: Path, package_root: Path) -> dict[str, Any]:
    try:
        from pydrake.all import (
            AddMultibodyPlantSceneGraph,
            DiagramBuilder,
            LoadModelDirectives,
            Parser,
            ProcessModelDirectives,
        )

        builder = DiagramBuilder()
        plant, _scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        parser = Parser(plant)
        parser.package_map().Add("scene", str(package_root))
        directives = LoadModelDirectives(str(dmd_path))
        added_models = ProcessModelDirectives(directives=directives, parser=parser)
        plant.Finalize()
        _diagram = builder.Build()
        return {
            "status": "pass",
            "added_model_count": len(added_models),
            "num_model_instances": plant.num_model_instances(),
            "num_bodies": plant.num_bodies(),
            "num_joints": plant.num_joints(),
            "num_positions": plant.num_positions(),
            "num_velocities": plant.num_velocities(),
        }
    except Exception as exc:  # fail closed on every parser/finalization error
        return {
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def validate(
    dmd_path: Path,
    package_root: Path,
    *,
    require_gpus: int = 0,
    max_collision_elements: int = 32,
    minimum_models: int = 1,
    expected_rooms: int = 0,
) -> dict[str, Any]:
    if require_gpus < 0:
        raise ValueError("require_gpus must be non-negative")
    if max_collision_elements < 0:
        raise ValueError("max_collision_elements must be non-negative")
    if minimum_models < 1:
        raise ValueError("minimum_models must be positive")
    if expected_rooms < 0:
        raise ValueError("expected_rooms must be non-negative")

    gpu = _gpu_inventory(require_gpus)
    collision_report = _collision_report(
        package_root, max_collision_elements=max_collision_elements
    )
    dmd_exists = dmd_path.is_file()
    package_root_exists = package_root.is_dir()
    dmd_inventory = (
        _dmd_inventory(dmd_path)
        if dmd_exists
        else {"status": "fail", "model_directive_count": 0}
    )
    house_inventory = (
        _house_state_inventory(package_root)
        if package_root_exists
        else {"status": "fail", "room_count": 0, "object_count": 0}
    )

    result: dict[str, Any] = {
        "status": "fail",
        "dmd_path": str(dmd_path),
        "package_root": str(package_root),
        "acceptance_requirements": {
            "drake_directives_load_successfully": True,
            "all_sdf_files_parse_successfully": True,
            "required_visible_gpu_count": require_gpus,
            "required_gpu_execution_count": require_gpus,
            "max_collision_elements_per_sdf": max_collision_elements,
            "minimum_model_directive_count": minimum_models,
            "expected_room_count": expected_rooms,
        },
        "visible_gpu_count": gpu["count"],
        "visible_gpus": gpu["names"],
        "gpu_inventory": gpu,
        "two_gpu_acceptance_environment": (
            gpu["count"] >= 2
            and gpu.get("all_required_devices_exercised") is True
        ),
        "collision_report": collision_report,
        "dmd_inventory": dmd_inventory,
        "house_state_inventory": house_inventory,
        "checks": {
            "dmd_file_exists": dmd_exists,
            "package_root_exists": package_root_exists,
            "required_gpus_available": gpu["count"] >= require_gpus,
            "required_gpus_exercised": (
                gpu.get("all_required_devices_exercised") is True
            ),
            "all_sdf_files_parsed": not collision_report["parse_failures"],
            "collision_cap_satisfied": not collision_report[
                "assets_over_collision_cap"
            ],
            "sdf_assets_present": collision_report["sdf_asset_count"] > 0,
            "dmd_inventory_nonempty": (
                dmd_inventory.get("status") == "pass"
                and int(dmd_inventory.get("model_directive_count", 0))
                >= minimum_models
            ),
            "house_state_inventory_nonempty": (
                house_inventory.get("status") == "pass"
            ),
            "expected_room_count_satisfied": (
                expected_rooms == 0
                or house_inventory.get("room_count") == expected_rooms
            ),
            "drake_model_count_matches_directives": False,
            "drake_load_succeeded": False,
        },
    }

    if not dmd_exists:
        result["drake_load"] = {
            "status": "fail",
            "error": f"Drake directives file is missing: {dmd_path}",
        }
    elif not package_root_exists:
        result["drake_load"] = {
            "status": "fail",
            "error": f"Scene package root is missing: {package_root}",
        }
    else:
        drake_load = _load_with_drake(dmd_path, package_root)
        result["drake_load"] = drake_load
        result["checks"]["drake_load_succeeded"] = drake_load["status"] == "pass"
        result["checks"]["drake_model_count_matches_directives"] = (
            drake_load.get("status") == "pass"
            and drake_load.get("added_model_count")
            == dmd_inventory.get("model_directive_count")
            and int(drake_load.get("num_bodies", 0)) > 1
        )
        # Preserve the previous top-level metric and error fields for report
        # consumers while keeping the complete load result grouped as well.
        for key, value in drake_load.items():
            if key != "status":
                result[key] = value

    if all(result["checks"].values()):
        result["status"] = "pass"
    else:
        result["failed_checks"] = [
            name for name, passed in result["checks"].items() if not passed
        ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmd", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--require-gpus",
        type=_nonnegative_int,
        default=0,
        metavar="N",
        help=(
            "fail unless at least N CUDA GPUs are visible; defaults to 0 for a "
            "structural Drake-load check"
        ),
    )
    parser.add_argument(
        "--max-collision-elements",
        type=_nonnegative_int,
        default=32,
        metavar="N",
        help="fail if any SDF in the scene package has more than N collisions",
    )
    parser.add_argument(
        "--minimum-models",
        type=_nonnegative_int,
        default=1,
        metavar="N",
        help="fail unless the DMD and Drake load contain at least N add_model directives",
    )
    parser.add_argument(
        "--expected-rooms",
        type=_nonnegative_int,
        default=0,
        metavar="N",
        help="fail unless combined house state has exactly N rooms; zero disables exact count",
    )
    args = parser.parse_args()
    result = validate(
        args.dmd.resolve(),
        args.package_root.resolve(),
        require_gpus=args.require_gpus,
        max_collision_elements=args.max_collision_elements,
        minimum_models=args.minimum_models,
        expected_rooms=args.expected_rooms,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
