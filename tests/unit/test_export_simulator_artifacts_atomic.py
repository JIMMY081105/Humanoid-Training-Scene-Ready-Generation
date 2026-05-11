from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import export_simulator_artifacts_atomic as atomic_export


class _FakeModel:
    nbody = 2
    ngeom = 3
    njnt = 0
    nq = 0
    nv = 0

    @classmethod
    def from_xml_path(cls, _path):
        return cls()


def _install_fake_mujoco(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(
            MjModel=_FakeModel,
            MjData=lambda _model: object(),
            mj_step=lambda _model, _data: None,
        ),
    )


def _setup_paths(tmp_path):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    exporter = tmp_path / "export_scene_to_mujoco.py"
    exporter.write_text(
        "def export_to_usd():\n"
        "    try:\n"
        "        import bpy\n"
        "        raise RuntimeError('Requested USD export cannot run while bpy is importable')\n"
        "    except ImportError:\n"
        "        pass\n"
        "    try:\n"
        "        import mujoco_usd_converter\n"
        "        if False:\n"
        "            raise RuntimeError('USD converter did not create its requested root layer')\n"
        "        if False:\n"
        "            raise RuntimeError('OpenUSD could not load converted root layer')\n"
        "    except ImportError as exc:\n"
        "        raise RuntimeError('Requested USD export dependencies are unavailable') from exc\n"
        "    except Exception as exc:\n"
        "        raise RuntimeError('Requested USD export failed') from exc\n",
        encoding="utf-8",
    )
    published = scene_dir / "mujoco_export"
    evidence = scene_dir / "quality_gates" / "simulator_exports.json"
    return scene_dir, exporter, published, evidence


def _successful_runner(staging_paths):
    def run(command):
        staging = Path(command[command.index("-o") + 1])
        staging_paths.append(staging)
        assert staging.name.startswith(".mujoco_export.staging.attempt-123.")
        assert json.loads(
            (staging / atomic_export.ATTEMPT_MARKER).read_text(encoding="utf-8")
        )["run_attempt_id"] == "attempt-123"
        (staging / "scene.xml").write_text(
            "<mujoco model='fresh'><worldbody><body name='new'/></worldbody></mujoco>",
            encoding="utf-8",
        )

    return run


def test_unique_attempt_staging_replaces_stale_export_only_after_validation(
    tmp_path, monkeypatch
):
    _install_fake_mujoco(monkeypatch)
    scene_dir, exporter, published, evidence = _setup_paths(tmp_path)
    published.mkdir()
    (published / "scene.xml").write_text(
        "<mujoco model='stale'><worldbody/></mujoco>", encoding="utf-8"
    )
    (published / "stale.txt").write_text("must disappear", encoding="utf-8")
    staging_paths = []

    result = atomic_export.export_validate_and_publish(
        scene_dir=scene_dir,
        published_dir=published,
        validation_output=evidence,
        exporter=exporter,
        run_attempt_id="attempt-123",
        require_usd=False,
        exporter_runner=_successful_runner(staging_paths),
    )

    assert result["status"] == "pass"
    assert result["publication"]["previous_export_replaced"] is True
    assert "fresh" in (published / "scene.xml").read_text(encoding="utf-8")
    assert not (published / "stale.txt").exists()
    assert result["attempt_marker"]["run_attempt_id"] == "attempt-123"
    assert json.loads(evidence.read_text(encoding="utf-8"))["status"] == "pass"
    assert all(not path.exists() for path in staging_paths)
    assert not list(scene_dir.glob(".mujoco_export.previous.*"))


def test_export_failure_never_validates_or_replaces_previous_output(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    scene_dir, exporter, published, evidence = _setup_paths(tmp_path)
    published.mkdir()
    sentinel = published / "previous.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    evidence.parent.mkdir()
    evidence.write_text('{"status":"pass","run_attempt_id":"stale"}\n', encoding="utf-8")

    def fail_export(_command):
        raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        atomic_export.export_validate_and_publish(
            scene_dir=scene_dir,
            published_dir=published,
            validation_output=evidence,
            exporter=exporter,
            run_attempt_id="attempt-123",
            require_usd=False,
            exporter_runner=fail_export,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    current_evidence = json.loads(evidence.read_text(encoding="utf-8"))
    assert current_evidence["status"] == "running"
    assert current_evidence["run_attempt_id"] == "attempt-123"


def test_unpatched_execution_exporter_is_rejected_before_launch(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    scene_dir, exporter, published, evidence = _setup_paths(tmp_path)
    exporter.write_text(
        "def export_to_usd():\n"
        "    try:\n"
        "        import bpy\n"
        "        return\n"
        "    except ImportError:\n"
        "        pass\n",
        encoding="utf-8",
    )
    launched = False

    def unexpected_launch(_command):
        nonlocal launched
        launched = True

    with pytest.raises(RuntimeError, match="not the fail-closed patched version"):
        atomic_export.export_validate_and_publish(
            scene_dir=scene_dir,
            published_dir=published,
            validation_output=evidence,
            exporter=exporter,
            run_attempt_id="attempt-123",
            require_usd=True,
            exporter_runner=unexpected_launch,
        )

    assert launched is False
    assert json.loads(evidence.read_text(encoding="utf-8"))["status"] == "running"


def test_atomic_promotion_failure_restores_previous_export(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    scene_dir, exporter, published, evidence = _setup_paths(tmp_path)
    published.mkdir()
    sentinel = published / "previous.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    staging_paths = []
    real_replace = os.replace

    def fail_staging_promotion(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path in staging_paths and destination_path == published.resolve():
            raise OSError("injected promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(atomic_export.os, "replace", fail_staging_promotion)

    with pytest.raises(RuntimeError, match="promotion failed"):
        atomic_export.export_validate_and_publish(
            scene_dir=scene_dir,
            published_dir=published,
            validation_output=evidence,
            exporter=exporter,
            run_attempt_id="attempt-123",
            require_usd=False,
            exporter_runner=_successful_runner(staging_paths),
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not list(scene_dir.glob(".mujoco_export.previous.*"))


def test_post_promotion_validation_failure_restores_previous_export(
    tmp_path, monkeypatch
):
    _install_fake_mujoco(monkeypatch)
    scene_dir, exporter, published, evidence = _setup_paths(tmp_path)
    published.mkdir()
    sentinel = published / "previous.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    staging_paths = []
    real_validate = atomic_export.validate_exports
    calls = 0

    def fail_second_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected post-promotion validation failure")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(atomic_export, "validate_exports", fail_second_validation)

    with pytest.raises(RuntimeError, match="post-promotion validation failure"):
        atomic_export.export_validate_and_publish(
            scene_dir=scene_dir,
            published_dir=published,
            validation_output=evidence,
            exporter=exporter,
            run_attempt_id="attempt-123",
            require_usd=False,
            exporter_runner=_successful_runner(staging_paths),
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not list(scene_dir.glob(".mujoco_export.previous.*"))


def test_post_commit_backup_cleanup_error_is_nonfatal(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    scene_dir, exporter, published, evidence = _setup_paths(tmp_path)
    published.mkdir()
    (published / "previous.txt").write_text("preserve", encoding="utf-8")
    staging_paths = []

    def fail_cleanup(_path):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(atomic_export, "_remove_tree", fail_cleanup)

    result = atomic_export.export_validate_and_publish(
        scene_dir=scene_dir,
        published_dir=published,
        validation_output=evidence,
        exporter=exporter,
        run_attempt_id="attempt-123",
        require_usd=False,
        exporter_runner=_successful_runner(staging_paths),
    )

    assert result["status"] == "pass"
    assert result["publication"]["previous_backup_cleanup"] == "retained"
    assert "injected cleanup failure" in result["publication"]["cleanup_warning"]
    assert "fresh" in (published / "scene.xml").read_text(encoding="utf-8")
    backups = list(scene_dir.glob(".mujoco_export.previous.*"))
    assert len(backups) == 1
    assert (backups[0] / "previous.txt").read_text(encoding="utf-8") == "preserve"
    persisted = json.loads(evidence.read_text(encoding="utf-8"))
    assert persisted["status"] == "pass"
    assert persisted["publication"]["previous_backup_cleanup"] == "retained"
