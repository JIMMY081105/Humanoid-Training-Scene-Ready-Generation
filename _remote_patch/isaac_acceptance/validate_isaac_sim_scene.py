#!/usr/bin/env python3
"""Fail-closed Isaac Sim import and physics-execution validation for a USD export."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "sha256": _sha256_bytes(_canonical_json(payload)),
    }


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise RuntimeError(f"{label} is missing, linked, empty, or not a file: {path}")
    return path.resolve(strict=True)


def _resolve_usd_root(export_dir: Path) -> tuple[Path, Path, str]:
    root = export_dir.resolve(strict=True)
    if export_dir.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Export directory must be a real directory: {export_dir}")
    scene_xml = _require_regular_file(root / "scene.xml", "MJCF scene")
    try:
        model_name = (ET.parse(scene_xml).getroot().get("model") or "").strip()
    except (ET.ParseError, OSError) as exc:
        raise RuntimeError(f"Cannot parse MJCF scene: {exc}") from exc
    if not model_name or any(character in model_name for character in "/\\\0"):
        raise RuntimeError(f"Unsafe or empty MJCF model name: {model_name!r}")
    usd_root = _require_regular_file(root / "usd" / f"{model_name}.usda", "USD root")
    if root not in usd_root.parents:
        raise RuntimeError("USD root resolves outside the export directory")
    return scene_xml, usd_root, model_name


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _build_failure(
    *,
    export_dir: Path,
    usd_root: Path | None,
    started: float,
    error: BaseException,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "fail",
        "validator": "isaac_sim_usd_import_and_physics_execution",
        "export_dir": str(export_dir),
        "usd_root": str(usd_root) if usd_root is not None else None,
        "duration_seconds": round(time.monotonic() - started, 6),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    payload["attestation"] = _attest(payload)
    return payload


def validate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    export_dir = args.export_dir.absolute()
    usd_root: Path | None = None
    simulation_app = None
    failure: BaseException | None = None
    payload: dict[str, Any] | None = None

    try:
        scene_xml, usd_root, model_name = _resolve_usd_root(export_dir)

        # Omniverse modules other than isaacsim must only be imported after the
        # application has initialized its plugin framework.
        from isaacsim import SimulationApp

        simulation_app = SimulationApp(
            {
                "headless": True,
                "hide_ui": True,
                "open_usd": str(usd_root),
                "sync_loads": True,
                "multi_gpu": False,
                "active_gpu": 0,
                "physics_gpu": 0,
                "renderer": "RayTracedLighting",
                "anti_aliasing": 0,
                "width": 640,
                "height": 480,
                "fast_shutdown": False,
            }
        )

        import omni.timeline
        import omni.usd
        import torch
        from omni.isaac.core.utils.stage import is_stage_loading
        from pxr import UsdGeom, UsdPhysics

        visible_gpus = int(torch.cuda.device_count())
        if visible_gpus < args.require_visible_gpus:
            raise RuntimeError(
                f"Isaac runtime sees {visible_gpus} GPUs; "
                f"{args.require_visible_gpus} required"
            )

        loading_updates = 0
        while is_stage_loading() and loading_updates < args.max_loading_updates:
            simulation_app.update()
            loading_updates += 1
        if is_stage_loading():
            raise RuntimeError(
                f"USD stage was still loading after {args.max_loading_updates} updates"
            )

        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim returned no USD stage after import")
        root_layer = stage.GetRootLayer()
        root_real_path = Path(root_layer.realPath).resolve(strict=True)
        if root_real_path != usd_root:
            raise RuntimeError(
                f"Isaac Sim loaded the wrong root layer: {root_real_path} != {usd_root}"
            )
        default_prim = stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            raise RuntimeError("Imported USD has no valid default prim")

        prims = [prim for prim in stage.Traverse() if prim.IsValid()]
        mesh_count = sum(prim.IsA(UsdGeom.Mesh) for prim in prims)
        physics_scene_count = sum(prim.IsA(UsdPhysics.Scene) for prim in prims)
        collision_count = sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in prims)
        rigid_body_count = sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims)
        joint_count = sum(prim.IsA(UsdPhysics.Joint) for prim in prims)
        if len(prims) < args.minimum_prims:
            raise RuntimeError(
                f"Imported USD has only {len(prims)} prims; {args.minimum_prims} required"
            )
        if mesh_count < 1:
            raise RuntimeError("Imported USD contains no mesh prims")
        if physics_scene_count < 1:
            raise RuntimeError("Imported USD contains no UsdPhysics scene")
        if collision_count < 1:
            raise RuntimeError("Imported USD contains no collision-enabled prims")

        used_layers: list[dict[str, Any]] = []
        for layer in stage.GetUsedLayers():
            real_path = layer.realPath
            if not real_path:
                continue
            path = _require_regular_file(Path(real_path), "USD used layer")
            if export_dir.resolve(strict=True) not in path.parents and path != usd_root:
                raise RuntimeError(f"USD used layer escapes the export: {path}")
            used_layers.append(
                {
                    "path": path.relative_to(export_dir.resolve(strict=True)).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        if not used_layers:
            raise RuntimeError("Isaac Sim reported no filesystem-backed USD layers")
        used_layers.sort(key=lambda item: item["path"])

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        simulation_app.update()
        start_time = float(timeline.get_current_time())
        timeline.play()
        for _ in range(args.steps):
            if not simulation_app.is_running():
                raise RuntimeError("Isaac Sim stopped while executing the physics timeline")
            simulation_app.update()
        end_time = float(timeline.get_current_time())
        was_playing = bool(timeline.is_playing())
        timeline.stop()
        simulation_app.update()
        if not was_playing:
            raise RuntimeError("Isaac timeline was not playing after requested steps")
        if end_time <= start_time:
            raise RuntimeError(
                f"Isaac timeline did not advance: {start_time} -> {end_time}"
            )

        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "validator": "isaac_sim_usd_import_and_physics_execution",
            "python_executable": str(Path(sys.executable).resolve(strict=True)),
            "python_version": platform.python_version(),
            "isaacsim_app_version": _distribution_version("isaacsim-app"),
            "isaacsim_core_version": _distribution_version("isaacsim-core"),
            "export_dir": str(export_dir.resolve(strict=True)),
            "mjcf": {
                "path": str(scene_xml),
                "sha256": _sha256_file(scene_xml),
                "model": model_name,
            },
            "usd_root": {
                "path": str(usd_root),
                "sha256": _sha256_file(usd_root),
                "default_prim": default_prim.GetPath().pathString,
            },
            "used_layers": used_layers,
            "stage": {
                "prim_count": len(prims),
                "mesh_count": mesh_count,
                "physics_scene_count": physics_scene_count,
                "collision_prim_count": collision_count,
                "rigid_body_prim_count": rigid_body_count,
                "joint_prim_count": joint_count,
                "loading_updates": loading_updates,
            },
            "execution": {
                "requested_steps": args.steps,
                "timeline_start_seconds": start_time,
                "timeline_end_seconds": end_time,
                "timeline_advanced_seconds": end_time - start_time,
                "status": "pass",
            },
            "hardware": {
                "visible_gpu_count": visible_gpus,
                "required_visible_gpu_count": args.require_visible_gpus,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "duration_seconds": round(time.monotonic() - started, 6),
        }
        payload["attestation"] = _attest(payload)
    except BaseException as exc:  # keep a fail receipt for every non-signal failure
        failure = exc
        payload = _build_failure(
            export_dir=export_dir,
            usd_root=usd_root,
            started=started,
            error=exc,
        )
    finally:
        if simulation_app is not None:
            try:
                simulation_app.close()
            except BaseException as close_error:
                if failure is None:
                    failure = close_error
                    payload = _build_failure(
                        export_dir=export_dir,
                        usd_root=usd_root,
                        started=started,
                        error=close_error,
                    )

    assert payload is not None
    _atomic_write_json(args.output.absolute(), payload)
    if failure is not None:
        raise RuntimeError(str(failure)) from failure
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--max-loading-updates", type=int, default=1200)
    parser.add_argument("--minimum-prims", type=int, default=12)
    parser.add_argument("--require-visible-gpus", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.max_loading_updates < 1 or args.minimum_prims < 1:
        parser.error("step/loading/prim limits must be positive")
    if args.require_visible_gpus < 1:
        parser.error("--require-visible-gpus must be positive")
    try:
        result = validate(args)
    except Exception as exc:
        print(f"ISAAC_SIM_VALIDATION_FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
