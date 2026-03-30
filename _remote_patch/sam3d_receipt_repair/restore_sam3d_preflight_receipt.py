#!/usr/bin/env python3
"""Restore a valid SAM3D receipt after a failed verify-only path check mutated it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


EXPECTED_FAILED_SHA256 = "4dabe5245a5d9c6c4eee2a6059f261a853ac0ca00b924113b9768a720c2ca64b"
EXPECTED_ATTESTATION = "2dffb41c1fd1b22612de473b23af52a7689adf6490d14fbfe1e05404d59caa92"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def attestation_payload(result: dict[str, object]) -> dict[str, object]:
    keys = (
        "schema_version",
        "status",
        "sam3_checkpoint",
        "pipeline_config",
        "offline",
        "offline_environment",
        "model_loaded",
        "pipeline_loaded",
        "gpu",
        "visible_gpu_count",
        "total_memory_bytes",
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "memory_stats_reset",
        "memory_stats_reset_error",
        "inference_smoke",
        "evidence",
    )
    return {key: result.get(key) for key in keys}


def atomic_json(path: Path, document: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--repair-receipt", required=True, type=Path)
    args = parser.parse_args()

    path = args.receipt.resolve(strict=True)
    before_bytes = path.read_bytes()
    before = sha(before_bytes)
    if before != EXPECTED_FAILED_SHA256:
        raise SystemExit(f"unexpected failed receipt digest: {before}")
    document = json.loads(before_bytes)
    if (
        document.get("schema_version") != 3
        or document.get("status") != "fail"
        or document.get("model_loaded") is not True
        or document.get("pipeline_loaded") is not True
        or document.get("offline") is not True
        or document.get("evidence_verification", {}).get("status") != "pass"
        or document.get("inference_smoke_verification", {}).get("status") != "pass"
        or document.get("attestation", {}).get("sha256") != EXPECTED_ATTESTATION
        or "attestation does not match" not in document.get("verification_error", "")
    ):
        raise SystemExit("failed receipt is not the audited verify-only mutation")

    repaired = dict(document)
    repaired["status"] = "pass"
    repaired.pop("verification_error", None)
    computed = sha(canonical(attestation_payload(repaired)))
    if computed != EXPECTED_ATTESTATION:
        raise SystemExit(f"restored attestation mismatch: {computed}")

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    backup = args.backup_dir / f"{path.name}.failed.{before}"
    if not backup.exists():
        shutil.copy2(path, backup, follow_symlinks=False)
    elif sha(backup.read_bytes()) != before:
        raise SystemExit("existing failed-receipt backup digest mismatch")
    atomic_json(path, repaired)
    after = sha(path.read_bytes())

    repair_receipt = {
        "schema_version": 1,
        "status": "pass",
        "operation": "restore_verify_only_mutated_sam3d_preflight_receipt",
        "target_before_sha256": before,
        "target_after_sha256": after,
        "sam3d_attestation_sha256": EXPECTED_ATTESTATION,
        "model_or_inference_rerun": False,
        "reason": (
            "A verify-only invocation used default cache roots, changed only the saved "
            "receipt status to fail, and preserved the original attested evidence. A "
            "second path-correct verification proved both artifact and inference sections."
        ),
    }
    repair_receipt["attestation"] = sha(canonical(repair_receipt))
    args.repair_receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.repair_receipt, repair_receipt)
    print(f"SAM3D_RECEIPT_RESTORE_PASS {before} {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
