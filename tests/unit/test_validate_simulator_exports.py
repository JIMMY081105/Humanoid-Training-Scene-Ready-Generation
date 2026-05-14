from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.validate_simulator_exports import validate_exports


class _FakeModel:
    nbody = 2
    ngeom = 3
    njnt = 1
    nq = 7
    nv = 6

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


def _minimal_export(tmp_path):
    output_dir = tmp_path / "mujoco_export"
    output_dir.mkdir()
    (output_dir / "scene.xml").write_text(
        "<mujoco model='school'><worldbody><body name='school'/></worldbody></mujoco>",
        encoding="utf-8",
    )
    return output_dir


def _install_fake_openusd(monkeypatch):
    class _Layer:
        def __init__(self, real_path):
            self.realPath = str(real_path)

    class _Stage:
        def __init__(self, path):
            self.path = Path(path)

        @classmethod
        def Open(cls, path):
            candidate = Path(path)
            if candidate.read_text(encoding="utf-8") == "corrupt":
                raise RuntimeError("corrupt layer")
            return cls(candidate)

        def TraverseAll(self):
            return iter((object(),))

        def GetUsedLayers(self):
            if self.path.parent.name == "usd":
                return [
                    _Layer(path.resolve())
                    for path in sorted(self.path.parent.rglob("*.usd*"))
                ]
            return [_Layer(self.path.resolve())]

    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(Usd=SimpleNamespace(Stage=_Stage)),
    )


def _minimal_usd_export(output_dir):
    usd_dir = output_dir / "usd"
    payload_dir = usd_dir / "Payload"
    payload_dir.mkdir(parents=True)
    for relative in (
        "school.usda",
        "Payload/Contents.usda",
        "Payload/Geometry.usda",
        "Payload/Physics.usda",
    ):
        (usd_dir / relative).write_text("valid", encoding="utf-8")
    return usd_dir


def test_valid_mujoco_export_is_load_stepped_and_hash_bound(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    output_dir = _minimal_export(tmp_path)

    result = validate_exports(output_dir, require_usd=False)

    assert result["status"] == "pass"
    assert result["mujoco"]["model_counts"]["nbody"] == 2
    assert result["file_count"] == 1
    assert len(result["inventory_sha256"]) == 64


def test_required_usd_cannot_be_silently_omitted(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    output_dir = _minimal_export(tmp_path)

    with pytest.raises(RuntimeError, match="Required USD directory is missing"):
        validate_exports(output_dir, require_usd=True)


def test_validator_never_falls_back_to_a_stale_differently_named_mjcf(
    tmp_path, monkeypatch
):
    _install_fake_mujoco(monkeypatch)
    output_dir = _minimal_export(tmp_path)

    with pytest.raises(RuntimeError, match="Expected MuJoCo MJCF is missing"):
        validate_exports(
            output_dir,
            require_usd=False,
            expected_mjcf="scene_attempt_123.xml",
        )


def test_every_expected_usd_layer_is_opened_and_hash_bound(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    _install_fake_openusd(monkeypatch)
    output_dir = _minimal_export(tmp_path)
    _minimal_usd_export(output_dir)

    result = validate_exports(output_dir, require_usd=True)

    assert result["usd"]["usd_layer_count"] == 4
    assert len(result["usd"]["validated_stages"]) == 4
    assert result["usd"]["candidate_failures"] == []
    assert {
        entry["path"] for entry in result["usd"]["expected_artifacts"]
    } == {
        "usd/school.usda",
        "usd/Payload/Contents.usda",
        "usd/Payload/Geometry.usda",
        "usd/Payload/Physics.usda",
    }
    assert all(
        len(entry["sha256"]) == 64
        for entry in result["usd"]["expected_artifacts"]
    )


def test_missing_exact_expected_usd_payload_fails(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    _install_fake_openusd(monkeypatch)
    output_dir = _minimal_export(tmp_path)
    usd_dir = _minimal_usd_export(output_dir)
    (usd_dir / "Payload" / "Physics.usda").unlink()

    with pytest.raises(RuntimeError, match="Expected USD artifact is missing"):
        validate_exports(output_dir, require_usd=True)


def test_mesh_and_material_layers_are_exact_conditional_requirements(
    tmp_path, monkeypatch
):
    _install_fake_mujoco(monkeypatch)
    _install_fake_openusd(monkeypatch)
    output_dir = _minimal_export(tmp_path)
    meshes = output_dir / "meshes"
    meshes.mkdir()
    (meshes / "desk.obj").write_text("mesh", encoding="utf-8")
    (output_dir / "scene.xml").write_text(
        "<mujoco model='school'><compiler meshdir='meshes'/><asset>"
        "<mesh name='desk' file='desk.obj'/><material name='school_blue'/>"
        "</asset><worldbody/></mujoco>",
        encoding="utf-8",
    )
    _minimal_usd_export(output_dir)

    with pytest.raises(RuntimeError, match="GeometryLibrary.usdc"):
        validate_exports(output_dir, require_usd=True)


def test_any_corrupt_usd_candidate_fails_the_whole_gate(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    _install_fake_openusd(monkeypatch)
    output_dir = _minimal_export(tmp_path)
    usd_dir = _minimal_usd_export(output_dir)
    (usd_dir / "Payload" / "extra.usda").write_text("corrupt", encoding="utf-8")

    with pytest.raises(RuntimeError, match="One or more USD candidate layers failed"):
        validate_exports(output_dir, require_usd=True)


def test_missing_referenced_mujoco_mesh_fails(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    output_dir = _minimal_export(tmp_path)
    (output_dir / "scene.xml").write_text(
        "<mujoco><compiler meshdir='meshes'/><asset>"
        "<mesh name='desk' file='desk.obj'/></asset><worldbody/></mujoco>",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Referenced MuJoCo asset is missing"):
        validate_exports(output_dir, require_usd=False)


def test_referenced_mujoco_assets_are_individually_hash_bound(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    output_dir = _minimal_export(tmp_path)
    meshes = output_dir / "meshes"
    meshes.mkdir()
    mesh = meshes / "desk.obj"
    mesh.write_text("mesh bytes", encoding="utf-8")
    (output_dir / "scene.xml").write_text(
        "<mujoco model='school'><compiler meshdir='meshes'/><asset>"
        "<mesh name='desk' file='desk.obj'/></asset><worldbody/></mujoco>",
        encoding="utf-8",
    )

    result = validate_exports(output_dir, require_usd=False)

    assert result["mujoco"]["referenced_asset_count"] == 1
    assert result["mujoco"]["referenced_assets"][0]["path"] == "meshes/desk.obj"
    assert len(result["mujoco"]["referenced_assets"][0]["sha256"]) == 64


def test_mjcf_include_cannot_load_uninventoried_external_state(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    output_dir = _minimal_export(tmp_path)
    outside = tmp_path / "mutable_external.xml"
    outside.write_text("<worldbody/>", encoding="utf-8")
    (output_dir / "scene.xml").write_text(
        "<mujoco model='school'><include file='../mutable_external.xml'/></mujoco>",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not root/hash-bound"):
        validate_exports(output_dir, require_usd=False)


def test_export_inventory_rejects_symlinks(tmp_path, monkeypatch):
    _install_fake_mujoco(monkeypatch)
    output_dir = _minimal_export(tmp_path)
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    link = output_dir / "linked.bin"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")

    with pytest.raises(RuntimeError, match="contains a symlink"):
        validate_exports(output_dir, require_usd=False)
