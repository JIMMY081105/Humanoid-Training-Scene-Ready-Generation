#!/usr/bin/env python3
"""Atomically replace the two restroom wood/plaster finishes with real tile PBR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED = {
    "boys_toilet/floors/floor.gltf": "f8242c8eeabda64ee8e6fe2b55b7cde947af0fa46ac4fc6661a2da10b4006948",
    "boys_toilet/walls/north_wall/wall.gltf": "2a55a3432cc9f7dd71a28d806961353ffa9319d92ccd1f87367755e46a96d173",
    "boys_toilet/walls/south_wall/wall.gltf": "2a55a3432cc9f7dd71a28d806961353ffa9319d92ccd1f87367755e46a96d173",
    "boys_toilet/walls/east_wall/wall.gltf": "b4951115682c8bef2cedb2b419445d621b1f15eafe5fe51e6e6574e1a7757594",
    "boys_toilet/walls/west_wall/wall.gltf": "2a55a3432cc9f7dd71a28d806961353ffa9319d92ccd1f87367755e46a96d173",
    "girls_toilet/floors/floor.gltf": "f8242c8eeabda64ee8e6fe2b55b7cde947af0fa46ac4fc6661a2da10b4006948",
    "girls_toilet/walls/north_wall/wall.gltf": "2a55a3432cc9f7dd71a28d806961353ffa9319d92ccd1f87367755e46a96d173",
    "girls_toilet/walls/south_wall/wall.gltf": "2a55a3432cc9f7dd71a28d806961353ffa9319d92ccd1f87367755e46a96d173",
    "girls_toilet/walls/east_wall/wall.gltf": "b4951115682c8bef2cedb2b419445d621b1f15eafe5fe51e6e6574e1a7757594",
    "girls_toilet/walls/west_wall/wall.gltf": "b4951115682c8bef2cedb2b419445d621b1f15eafe5fe51e6e6574e1a7757594",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        path.stat().st_mode & 0o777,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plans-dir", required=True, type=Path)
    parser.add_argument("--tile-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    root = args.floor_plans_dir.resolve(strict=True)
    tile_dir = args.tile_dir.resolve(strict=True)
    texture_paths = [
        tile_dir / "Tiles001_2K-JPG_Color.jpg",
        tile_dir / "Tiles001_2K-JPG_NormalGL.jpg",
        tile_dir / "Tiles001_2K-JPG_Roughness.jpg",
    ]
    for texture in texture_paths:
        if not texture.is_file() or texture.is_symlink():
            raise SystemExit(f"missing safe tile texture: {texture}")
    # Hash the large shared textures before touching any live room resource.
    # A slow or unhealthy shared mount therefore cannot leave a half-published
    # material transaction.
    tile_records = [
        {
            "path": str(texture),
            "size_bytes": texture.stat().st_size,
            "sha256": sha(texture),
        }
        for texture in texture_paths
    ]

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    originals: dict[Path, bytes] = {}
    records: list[dict[str, object]] = []
    try:
        for relative, expected in EXPECTED.items():
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise SystemExit(f"unsafe glTF target: {path}")
            before = sha(path)
            if before != expected:
                raise SystemExit(f"unexpected source digest for {relative}: {before}")
            original = path.read_bytes()
            originals[path] = original
            backup = args.backup_dir / f"{relative.replace('/', '__')}.{before}.gltf"
            if not backup.exists():
                shutil.copy2(path, backup, follow_symlinks=False)
            elif sha(backup) != before:
                raise SystemExit(f"backup mismatch: {backup}")

            document = json.loads(original)
            images = document.get("images")
            textures = document.get("textures")
            materials = document.get("materials")
            if not isinstance(images, list) or len(images) != 3:
                raise SystemExit(f"unexpected image contract: {relative}")
            if not isinstance(textures, list) or len(textures) != 3:
                raise SystemExit(f"unexpected texture contract: {relative}")
            if not isinstance(materials, list) or len(materials) != 1:
                raise SystemExit(f"unexpected material contract: {relative}")
            document["images"] = [
                {
                    "uri": os.path.relpath(texture, path.parent).replace(
                        os.sep, "/"
                    )
                }
                for texture in texture_paths
            ]
            payload = json.dumps(
                document, indent=2, sort_keys=False, ensure_ascii=False
            ).encode("utf-8") + b"\n"
            atomic_bytes(path, payload)
            records.append(
                {
                    "path": str(path),
                    "relative_path": relative,
                    "before_sha256": before,
                    "after_sha256": sha(path),
                }
            )

        receipt: dict[str, object] = {
            "schema_version": 1,
            "status": "pass",
            "operation": "replace_restroom_wood_and_plaster_with_tiles001_pbr",
            "geometry_changed": False,
            "collision_changed": False,
            "records": records,
            "tile_textures": tile_records,
        }
        receipt["attestation"] = hashlib.sha256(canonical(receipt)).hexdigest()
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt_payload = json.dumps(
            receipt, indent=2, sort_keys=True, ensure_ascii=False
        ).encode("utf-8") + b"\n"
        if args.receipt.exists():
            raise SystemExit(f"refusing to overwrite receipt: {args.receipt}")
        descriptor = os.open(
            args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(receipt_payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        for path, payload in originals.items():
            if path.exists():
                atomic_bytes(path, payload)
        raise

    print("BATHROOM_TILE_MATERIAL_REPAIR_PASS", len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
