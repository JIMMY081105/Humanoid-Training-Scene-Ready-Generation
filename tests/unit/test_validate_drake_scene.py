import contextlib
import sys
import types

from pathlib import Path

import pytest

from scripts import validate_drake_scene as validator


class _FakeCudaTensor:
    def __init__(self, value: float, count: int) -> None:
        self._sum = value * count

    def sum(self):
        return self

    def item(self) -> float:
        return self._sum


def test_gpu_inventory_executes_and_synchronizes_every_required_device(
    monkeypatch,
) -> None:
    synchronized: list[int] = []
    cuda = types.SimpleNamespace(
        device_count=lambda: 2,
        get_device_name=lambda index: f"gpu{index}",
        get_device_properties=lambda _index: types.SimpleNamespace(
            total_memory=24 * 1024**3
        ),
        mem_get_info=lambda _index: (20 * 1024**3, 24 * 1024**3),
        device=lambda _index: contextlib.nullcontext(),
        synchronize=lambda index: synchronized.append(index),
    )

    def fake_full(shape, value, *, dtype, device):
        assert dtype == "float32"
        assert device in {"cuda:0", "cuda:1"}
        return _FakeCudaTensor(value, shape[0])

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=cuda, float32="float32", full=fake_full),
    )

    inventory = validator._gpu_inventory(2)

    assert inventory["count"] == 2
    assert inventory["all_required_devices_exercised"] is True
    assert [item["status"] for item in inventory["execution_exercises"]] == [
        "pass",
        "pass",
    ]
    assert synchronized == [0, 1]


def _write_sdf(path: Path, collision_count: int) -> None:
    collisions = "".join(
        f'<collision name="collision_{index}"/>' for index in range(collision_count)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<sdf version="1.9"><model name="asset"><link name="link">'
        f"{collisions}</link></model></sdf>",
        encoding="utf-8",
    )


def _fake_successful_drake_load(_dmd: Path, _root: Path) -> dict[str, object]:
    return {
        "status": "pass",
        "added_model_count": 1,
        "num_model_instances": 2,
        "num_bodies": 2,
        "num_joints": 0,
        "num_positions": 0,
        "num_velocities": 0,
    }


def _scene(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "scene"
    root.mkdir()
    dmd = root / "scene.dmd.yaml"
    dmd.write_text(
        "directives:\n"
        "- add_model:\n"
        "    name: school_asset\n"
        "    file: package://scene/room_01/generated_assets/asset.sdf\n",
        encoding="utf-8",
    )
    _write_sdf(root / "room_01" / "generated_assets" / "asset.sdf", 0)
    house_state = root / "combined_house" / "house_state.json"
    house_state.parent.mkdir(parents=True)
    house_state.write_text(
        '{"rooms":{"classroom_01":{"objects":{"school_asset":{}}}}}',
        encoding="utf-8",
    )
    return dmd, root


def test_pass_reports_explicit_acceptance_requirements(tmp_path, monkeypatch):
    dmd, root = _scene(tmp_path)
    _write_sdf(root / "room_01" / "generated_assets" / "asset.sdf", 2)
    monkeypatch.setattr(
        validator,
        "_gpu_inventory",
        lambda required: {
            "count": 2,
            "names": ["gpu0", "gpu1"],
            "required_execution_count": required,
            "execution_exercises": [
                {"index": 0, "status": "pass"},
                {"index": 1, "status": "pass"},
            ][:required],
            "all_required_devices_exercised": True,
        },
    )
    monkeypatch.setattr(validator, "_load_with_drake", _fake_successful_drake_load)

    result = validator.validate(
        dmd, root, require_gpus=2, max_collision_elements=2
    )

    assert result["status"] == "pass"
    assert result["acceptance_requirements"] == {
        "drake_directives_load_successfully": True,
        "all_sdf_files_parse_successfully": True,
        "required_visible_gpu_count": 2,
        "required_gpu_execution_count": 2,
        "max_collision_elements_per_sdf": 2,
        "minimum_model_directive_count": 1,
        "expected_room_count": 0,
    }
    assert all(result["checks"].values())


def test_fails_when_fewer_than_required_gpus_are_visible(tmp_path, monkeypatch):
    dmd, root = _scene(tmp_path)
    monkeypatch.setattr(
        validator,
        "_gpu_inventory",
        lambda required: {
            "count": 1,
            "names": ["gpu0"],
            "required_execution_count": required,
            "execution_exercises": [{"index": 0, "status": "pass"}],
            "all_required_devices_exercised": False,
        },
    )
    monkeypatch.setattr(validator, "_load_with_drake", _fake_successful_drake_load)

    result = validator.validate(dmd, root, require_gpus=2)

    assert result["status"] == "fail"
    assert result["checks"]["required_gpus_available"] is False
    assert result["checks"]["required_gpus_exercised"] is False
    assert "required_gpus_available" in result["failed_checks"]
    assert result["checks"]["drake_load_succeeded"] is True


def test_fails_on_any_sdf_parse_error_or_collision_cap_excess(
    tmp_path, monkeypatch
):
    dmd, root = _scene(tmp_path)
    _write_sdf(root / "assets" / "within_cap.sdf", 2)
    _write_sdf(root / "assets" / "over_cap.sdf", 3)
    (root / "assets" / "malformed.sdf").write_text("<sdf>", encoding="utf-8")
    monkeypatch.setattr(
        validator,
        "_gpu_inventory",
        lambda required: {
            "count": 0,
            "names": [],
            "required_execution_count": required,
            "execution_exercises": [],
            "all_required_devices_exercised": required == 0,
        },
    )
    monkeypatch.setattr(validator, "_load_with_drake", _fake_successful_drake_load)

    result = validator.validate(dmd, root, max_collision_elements=2)

    assert result["status"] == "fail"
    assert result["checks"]["all_sdf_files_parsed"] is False
    assert result["checks"]["collision_cap_satisfied"] is False
    assert len(result["collision_report"]["parse_failures"]) == 1
    over = result["collision_report"]["assets_over_collision_cap"]
    assert len(over) == 1
    assert over[0]["path"].endswith("over_cap.sdf")
    assert over[0]["collision_count"] == 3


def test_fails_closed_when_drake_loading_fails(tmp_path, monkeypatch):
    dmd, root = _scene(tmp_path)
    monkeypatch.setattr(
        validator,
        "_gpu_inventory",
        lambda required: {
            "count": 0,
            "names": [],
            "required_execution_count": required,
            "execution_exercises": [],
            "all_required_devices_exercised": required == 0,
        },
    )
    monkeypatch.setattr(
        validator,
        "_load_with_drake",
        lambda _dmd, _root: {
            "status": "fail",
            "error_type": "RuntimeError",
            "error": "SDF parser rejected the model",
        },
    )

    result = validator.validate(dmd, root)

    assert result["status"] == "fail"
    assert result["checks"]["drake_load_succeeded"] is False
    assert result["error_type"] == "RuntimeError"
    assert "drake_load_succeeded" in result["failed_checks"]


def test_empty_dmd_or_house_state_cannot_pass_with_mocked_drake_success(
    tmp_path, monkeypatch
):
    dmd, root = _scene(tmp_path)
    dmd.write_text("directives: []\n", encoding="utf-8")
    (root / "combined_house" / "house_state.json").write_text(
        '{"rooms":{}}', encoding="utf-8"
    )
    monkeypatch.setattr(
        validator,
        "_gpu_inventory",
        lambda required: {
            "count": 2,
            "names": ["g0", "g1"],
            "required_execution_count": required,
            "execution_exercises": [
                {"index": 0, "status": "pass"},
                {"index": 1, "status": "pass"},
            ][:required],
            "all_required_devices_exercised": True,
        },
    )
    monkeypatch.setattr(validator, "_load_with_drake", _fake_successful_drake_load)

    result = validator.validate(
        dmd, root, require_gpus=2, minimum_models=1, expected_rooms=11
    )

    assert result["status"] == "fail"
    assert result["checks"]["dmd_inventory_nonempty"] is False
    assert result["checks"]["house_state_inventory_nonempty"] is False
    assert result["checks"]["expected_room_count_satisfied"] is False
    assert result["checks"]["drake_model_count_matches_directives"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"require_gpus": -1},
        {"max_collision_elements": -1},
        {"minimum_models": 0},
        {"expected_rooms": -1},
    ],
)
def test_rejects_negative_acceptance_thresholds(tmp_path, kwargs):
    dmd, root = _scene(tmp_path)
    with pytest.raises(ValueError, match="non-negative|positive"):
        validator.validate(dmd, root, **kwargs)
