#!/usr/bin/env python3
"""Create a hash-bound processing-hall benchmark timing/GPU receipt."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_ID = "scenesmith_processing_hall_full_quality_benchmark_v1"
REQUIRED_EVENTS = (
    "benchmark_start",
    "code_input_preflight_complete",
    "model_asset_router_preflight_complete",
    "floor_layout_gate_complete",
    "room_generation_start",
    "room_generation_complete",
    "blend_refresh_complete",
    "render_complete",
    "deterministic_gate_complete",
    "visual_gate_complete",
    "saved_gate_revalidation_complete",
    "benchmark_complete",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"benchmark evidence is missing/empty: {path}")
    return {"path": str(path), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"malformed benchmark event line {line_number}")
        events.append({"epoch_seconds": int(parts[0]), "utc": parts[1], "event": parts[2]})
    names = [item["event"] for item in events]
    positions = []
    for required in REQUIRED_EVENTS:
        if names.count(required) != 1:
            raise ValueError(f"benchmark event {required!r} must occur exactly once")
        positions.append(names.index(required))
    if positions != sorted(positions):
        raise ValueError("benchmark events are out of contract order")
    if any(events[index]["epoch_seconds"] > events[index + 1]["epoch_seconds"] for index in range(len(events) - 1)):
        raise ValueError("benchmark event timestamps are non-monotonic")
    return events


def _gpu_summary(path: Path) -> dict[str, Any]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or len(row) < 6 or row[0].strip().lower().startswith("timestamp"):
                continue
            try:
                rows.append(
                    {
                        "timestamp": row[0].strip(),
                        "index": int(row[1]),
                        "name": row[2].strip(),
                        "utilization_gpu_percent": float(row[3]),
                        "memory_used_mib": float(row[4]),
                        "memory_total_mib": float(row[5]),
                    }
                )
            except (TypeError, ValueError):
                continue
    if not rows:
        raise ValueError("benchmark contains no parseable nvidia-smi samples")
    if len({row["index"] for row in rows}) != 1:
        raise ValueError("single-room benchmark must bind exactly one visible GPU")
    return {
        "sample_count": len(rows),
        "gpu_index": rows[0]["index"],
        "gpu_name": rows[0]["name"],
        "peak_utilization_gpu_percent": max(row["utilization_gpu_percent"] for row in rows),
        "peak_memory_used_mib": max(row["memory_used_mib"] for row in rows),
        "memory_total_mib": max(row["memory_total_mib"] for row in rows),
        "first_sample": rows[0],
        "last_sample": rows[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--factory-contract", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-attempt-id", required=True)
    args = parser.parse_args()
    events = _load_events(args.events)
    event_map = {item["event"]: item for item in events}
    scene = args.scene_dir.resolve()
    room_id = "processing_hall"
    artifacts = {
        "factory_contract": _entry(args.factory_contract),
        "input_manifest": _entry(args.input_dir / "input_manifest.json"),
        "effective_prompt": _entry(args.input_dir / "prompt.txt"),
        "pipeline_code_contract": _entry(args.run_dir / "pipeline_code_contract.json"),
        "input_validation": _entry(args.run_dir / "factory_input_manifest_validation.json"),
        "layout_gate": _entry(scene / "quality_gates" / "floor_plan_layout.json"),
        "prompt_binding": _entry(scene / "quality_gates" / "room_prompt_binding.json"),
        "final_state": _entry(scene / f"room_{room_id}" / "scene_states" / "final_scene" / "scene_state.json"),
        "final_blend": _entry(scene / f"room_{room_id}" / "scene_states" / "final_scene" / "scene.blend"),
        "deterministic_gate": _entry(scene / "quality_gates" / "room_self_exam_deterministic" / f"{room_id}.json"),
        "visual_gate": _entry(scene / "quality_gates" / "room_self_exam" / f"{room_id}.json"),
        "visual_summary": _entry(scene / "quality_gates" / "room_self_exam" / "summary.json"),
        "events": _entry(args.events),
        "gpu_samples": _entry(args.gpu_samples),
    }
    for key in ("pipeline_code_contract", "input_validation", "layout_gate", "prompt_binding", "deterministic_gate", "visual_gate", "visual_summary"):
        value = json.loads(Path(artifacts[key]["path"]).read_text(encoding="utf-8"))
        if value.get("status") != "pass":
            raise ValueError(f"benchmark bound a non-passing gate: {key}")
    stages = {}
    for first, second in zip(REQUIRED_EVENTS, REQUIRED_EVENTS[1:]):
        stages[f"{first}_to_{second}"] = event_map[second]["epoch_seconds"] - event_map[first]["epoch_seconds"]
    result = {
        "schema_id": SCHEMA_ID,
        "schema_version": 1,
        "status": "pass",
        "terminal_state": "benchmark_complete",
        "run_attempt_id": args.run_attempt_id,
        "room_id": room_id,
        "run_dir": str(args.run_dir.resolve()),
        "scene_dir": str(scene),
        "events": events,
        "stage_elapsed_seconds": stages,
        "total_elapsed_seconds": event_map["benchmark_complete"]["epoch_seconds"] - event_map["benchmark_start"]["epoch_seconds"],
        "gpu": _gpu_summary(args.gpu_samples),
        "artifacts": artifacts,
    }
    unsigned = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["attestation"] = {"algorithm": "sha256", "sha256": hashlib.sha256(unsigned).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
