import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from scenesmith.agent_utils.asset_manager import (
    _verify_rigid_sdf_dimension_receipt,
)
from scenesmith.agent_utils.asset_router.dataclasses import (
    AssetItem,
    GeneratedGeometry,
    ValidationResult,
)
from scenesmith.agent_utils.asset_router.router import AssetRouter
from scenesmith.agent_utils.mesh_utils import (
    DIMENSION_CONTRACT_ATOL_METERS,
    gltf_to_scene_dimensions,
    mesh_dimension_candidate_compatibility,
    scale_mesh_uniformly_to_scene_dimension_contract,
    scene_to_gltf_dimensions,
    uniform_dimension_fit_plan,
    validate_dimension_vector,
)
from scenesmith.agent_utils.room import AgentType, ObjectType


LIVE_SHEET_NATIVE = [0.10389152, 0.00073083, 0.05557691]
LIVE_FOLDER_NATIVE = [0.24320523, 0.06487418, 0.16829520]
LIVE_PACKET_NATIVE = [0.16053873, 0.22639443, 0.10109075]


def _box(path: Path, extents: list[float]) -> Path:
    trimesh.creation.box(extents=extents).export(path)
    return path


def _item(dimensions: list[float] | None = None) -> AssetItem:
    return AssetItem(
        description="thin graded quiz sheet",
        short_name="quiz_sheet",
        dimensions=dimensions or [0.24, 0.21, 0.005],
        object_type=ObjectType.FURNITURE,
        strategies=["generated"],
    )


def test_scene_gltf_axis_order_is_explicit_and_round_trips() -> None:
    requested = [0.24, 0.21, 0.005]
    np.testing.assert_allclose(scene_to_gltf_dimensions(requested), [0.24, 0.005, 0.21])
    np.testing.assert_allclose(
        gltf_to_scene_dimensions(scene_to_gltf_dimensions(requested)), requested
    )


@pytest.mark.parametrize(
    "invalid",
    ([1.0, 2.0], [1.0, 0.0, 2.0], [1.0, np.nan, 2.0], [1.0, np.inf, 2.0]),
)
def test_dimension_vectors_reject_wrong_shape_zero_and_nonfinite(invalid) -> None:
    with pytest.raises(ValueError):
        validate_dimension_vector(invalid, label="test dimensions")


def test_orientation_invariant_prefilter_accepts_submillimeter_real_sheet(
    tmp_path: Path,
) -> None:
    mesh = _box(tmp_path / "real_sheet.glb", LIVE_SHEET_NATIVE)
    compatible, plan = mesh_dimension_candidate_compatibility(
        mesh, [0.24, 0.21, 0.005]
    )
    assert compatible
    assert min(plan["major_axis_occupancies"]) >= 0.5


@pytest.mark.parametrize("extents", (LIVE_FOLDER_NATIVE, LIVE_PACKET_NATIVE))
def test_live_wrong_aspect_candidates_are_rejected_before_vlm(
    tmp_path: Path, extents: list[float]
) -> None:
    mesh = _box(tmp_path / f"bad_{len(list(tmp_path.iterdir()))}.glb", extents)
    compatible, plan = mesh_dimension_candidate_compatibility(
        mesh, [0.24, 0.17, 0.004] if extents == LIVE_FOLDER_NATIVE else [0.21, 0.15, 0.018]
    )
    assert not compatible
    assert max(plan["major_axis_occupancies"]) < 0.5


def test_uniform_scaler_preserves_proportions_and_reloads_exactly(
    tmp_path: Path,
) -> None:
    source = _box(tmp_path / "isotropic.glb", [2.0, 1.0, 3.0])
    output = tmp_path / "scaled.glb"
    path, scale, receipt = scale_mesh_uniformly_to_scene_dimension_contract(
        source,
        requested_scene_dimensions=[4.0, 6.0, 2.0],
        output_path=output,
    )
    assert path == output
    assert scale == pytest.approx(2.0)
    measured = trimesh.load(output, force="mesh").extents
    np.testing.assert_allclose(measured, [4.0, 2.0, 6.0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(receipt["measured_final_scene_dimensions_m"]), [4.0, 6.0, 2.0]
    )
    assert receipt["status"] == "pass"


def test_postcanonical_impossible_aspect_fails_without_output(tmp_path: Path) -> None:
    source = _box(tmp_path / "folder.glb", LIVE_FOLDER_NATIVE)
    output = tmp_path / "must_not_publish.glb"
    with pytest.raises(ValueError, match="aspect ratio is incompatible"):
        scale_mesh_uniformly_to_scene_dimension_contract(
            source,
            requested_scene_dimensions=[0.24, 0.17, 0.004],
            output_path=output,
        )
    assert not output.exists()


def test_sdf_receipt_binds_visual_dimensions_and_hashes(tmp_path: Path) -> None:
    source = _box(tmp_path / "source.glb", [1.0, 1.0, 1.0])
    visual, _, receipt = scale_mesh_uniformly_to_scene_dimension_contract(
        source,
        requested_scene_dimensions=[0.5, 0.5, 0.5],
        output_path=tmp_path / "asset.gltf",
    )
    sdf = tmp_path / "asset.sdf"
    sdf.write_text(
        "<?xml version='1.0'?><sdf version='1.7'><model name='a'><link name='l'>"
        "<visual name='v'><geometry><mesh><uri>asset.gltf</uri></mesh></geometry>"
        "</visual></link></model></sdf>",
        encoding="utf-8",
    )
    finalized = _verify_rigid_sdf_dimension_receipt(sdf, visual, receipt)
    assert finalized["receipt_sha256"]
    assert finalized["visual_mesh_sha256"]
    assert finalized["sdf_sha256"]
    assert finalized["asset_directory_sha256"]

    payload = json.loads(json.dumps(finalized))
    payload["measured_final_gltf_dimensions_m"][0] += 0.1
    with pytest.raises(ValueError, match="violates the dimension receipt"):
        _verify_rigid_sdf_dimension_receipt(sdf, visual, payload)


@pytest.mark.parametrize("asset_source", ("generated", "hssd", "objaverse"))
def test_router_retries_incompatible_candidate_before_vlm(
    tmp_path: Path, asset_source: str
) -> None:
    bad = _box(tmp_path / f"{asset_source}_bad.glb", LIVE_FOLDER_NATIVE)
    good = _box(tmp_path / f"{asset_source}_good.glb", LIVE_SHEET_NATIVE)
    item = _item()
    router = object.__new__(AssetRouter)
    router.cfg = SimpleNamespace(
        asset_manager=SimpleNamespace(
            general_asset_source=asset_source,
            hssd=SimpleNamespace(use_lenient_validation=True),
            objaverse=SimpleNamespace(use_lenient_validation=True),
        )
    )
    router._should_use_primitive_manipuland_fallback = lambda _item: False
    router._should_use_primitive_ceiling_fallback = lambda _item: False
    if asset_source == "hssd":
        router._fetch_hssd_candidates = lambda **_kwargs: [object(), object()]
    if asset_source == "objaverse":
        router._fetch_objaverse_candidates = lambda **_kwargs: [object(), object()]

    attempts: list[int] = []

    def acquire(**kwargs):
        attempts.append(kwargs["attempt"])
        path = bad if kwargs["attempt"] == 0 else good
        return GeneratedGeometry(path, item, asset_source)

    router._acquire_generated_candidate = acquire
    validated: list[Path] = []

    def validate_asset(**kwargs):
        validated.append(kwargs["mesh_path"])
        return ValidationResult(True, "compatible")

    router.validate_asset = validate_asset
    result = router._try_generated_strategy(
        item=item,
        max_retries=2,
        geometry_client=None,
        hssd_client=None,
        objaverse_client=None,
        image_generator=None,
        images_dir=tmp_path,
        geometry_dir=tmp_path,
        debug_dir=tmp_path,
    )
    assert result is not None and result.geometry_path == good
    assert attempts == [0, 1]
    assert validated == [good]


def test_router_max_retries_zero_still_fails_closed(tmp_path: Path) -> None:
    bad = _box(tmp_path / "bad.glb", LIVE_PACKET_NATIVE)
    item = _item([0.21, 0.15, 0.018])
    router = object.__new__(AssetRouter)
    router.cfg = SimpleNamespace(
        asset_manager=SimpleNamespace(general_asset_source="generated")
    )
    router._acquire_generated_candidate = lambda **_kwargs: GeneratedGeometry(
        bad, item, "generated"
    )
    router.validate_asset = lambda **_kwargs: pytest.fail("VLM must not run")
    result = router._try_generated_strategy(
        item=item,
        max_retries=0,
        geometry_client=None,
        hssd_client=None,
        objaverse_client=None,
        image_generator=None,
        images_dir=tmp_path,
        geometry_dir=tmp_path,
        debug_dir=tmp_path,
    )
    assert result is None


def test_articulated_router_skips_incompatible_candidate_before_vlm(
    tmp_path: Path,
) -> None:
    item = _item([1.0, 0.5, 1.5])
    bad = SimpleNamespace(
        object_id="bad",
        source="artiverse",
        bounding_box_min=[0.0, 0.0, 0.0],
        bounding_box_max=[0.05, 0.05, 0.05],
        clip_score=1.0,
        bbox_score=0.0,
        sdf_path=str(tmp_path / "bad.sdf"),
    )
    good = SimpleNamespace(
        object_id="good",
        source="artiverse",
        bounding_box_min=[0.0, 0.0, 0.0],
        bounding_box_max=[0.8, 0.4, 1.2],
        clip_score=0.9,
        bbox_score=0.1,
        sdf_path=str(tmp_path / "good.sdf"),
    )
    client = SimpleNamespace(
        retrieve_objects=lambda _requests: [(0, SimpleNamespace(results=[bad, good]))]
    )
    router = object.__new__(AssetRouter)
    router.agent_type = AgentType.FURNITURE
    validated: list[str] = []

    def validate_articulated(**kwargs):
        validated.append(kwargs["result"].object_id)
        return ValidationResult(True, "compatible")

    router._validate_articulated_result = validate_articulated
    result = router._try_articulated_strategy(
        item=item,
        max_retries=2,
        debug_dir=tmp_path,
        articulated_client=client,
    )
    assert result is not None
    assert result.object_id == "good"
    assert validated == ["good"]


def test_articulated_max_retries_zero_is_bounded_and_fails_closed(
    tmp_path: Path,
) -> None:
    item = _item([1.0, 0.5, 1.5])
    bad = SimpleNamespace(
        object_id="bad",
        source="artiverse",
        bounding_box_min=[0.0, 0.0, 0.0],
        bounding_box_max=[0.05, 0.05, 0.05],
        clip_score=1.0,
        bbox_score=0.0,
        sdf_path=str(tmp_path / "bad.sdf"),
    )
    would_pass_but_is_over_budget = SimpleNamespace(
        object_id="over_budget",
        source="artiverse",
        bounding_box_min=[0.0, 0.0, 0.0],
        bounding_box_max=[0.8, 0.4, 1.2],
        clip_score=0.9,
        bbox_score=0.1,
        sdf_path=str(tmp_path / "good.sdf"),
    )
    client = SimpleNamespace(
        retrieve_objects=lambda _requests: [
            (
                0,
                SimpleNamespace(
                    results=[bad, would_pass_but_is_over_budget]
                ),
            )
        ]
    )
    router = object.__new__(AssetRouter)
    router.agent_type = AgentType.FURNITURE
    router._validate_articulated_result = lambda **_kwargs: pytest.fail(
        "max_retries=0 must not call VLM"
    )
    assert (
        router._try_articulated_strategy(
            item=item,
            max_retries=0,
            debug_dir=tmp_path,
            articulated_client=client,
        )
        is None
    )


def test_thin_covering_strategy_advances_after_dimension_rejection(
    tmp_path: Path,
) -> None:
    item = _item([0.24, 0.21, 0.005])
    first = SimpleNamespace(material_id="bad", material_path=str(tmp_path / "bad"), similarity_score=1.0)
    second = SimpleNamespace(material_id="good", material_path=str(tmp_path / "good"), similarity_score=0.9)
    client = SimpleNamespace(
        retrieve_materials=lambda _requests: [
            (0, SimpleNamespace(results=[first, second]))
        ]
    )
    router = object.__new__(AssetRouter)
    router.agent_type = AgentType.MANIPULAND
    router.cfg = SimpleNamespace(
        asset_manager=SimpleNamespace(
            router=SimpleNamespace(
                strategies=SimpleNamespace(
                    thin_covering=SimpleNamespace(
                        thickness_m=0.005,
                        texture_scale=1.0,
                        generator=SimpleNamespace(enabled=False),
                    )
                )
            )
        )
    )
    good_geometry = GeneratedGeometry(
        _box(tmp_path / "good.glb", LIVE_SHEET_NATIVE), item, "thin_covering"
    )
    calls: list[str] = []

    def generate(**kwargs):
        calls.append(kwargs["material_path"].name)
        return None if len(calls) == 1 else good_geometry

    router._generate_thin_covering_geometry = generate
    router._validate_thin_covering = lambda **_kwargs: ValidationResult(True, "ok")
    result = router._try_thin_covering_strategy(
        item=item,
        max_retries=2,
        materials_client=client,
        image_generator=None,
        geometry_dir=tmp_path,
        debug_dir=tmp_path,
    )
    assert result == good_geometry
    assert calls == ["bad", "good"]


def test_uniform_fit_plan_rejects_old_4mm_to_64mm_failure() -> None:
    target = scene_to_gltf_dimensions([0.24, 0.17, 0.004])
    plan = uniform_dimension_fit_plan(LIVE_FOLDER_NATIVE, target)
    assert not plan["compatible"]
    old_result = np.asarray(LIVE_FOLDER_NATIVE) * 0.987
    old_scene = gltf_to_scene_dimensions(old_result)
    assert old_scene[2] > 0.064 - DIMENSION_CONTRACT_ATOL_METERS


def test_uniform_fit_plan_rejects_old_18mm_to_150mm_failure() -> None:
    target = scene_to_gltf_dimensions([0.21, 0.15, 0.018])
    plan = uniform_dimension_fit_plan(LIVE_PACKET_NATIVE, target)
    assert not plan["compatible"]
    old_result = np.asarray(LIVE_PACKET_NATIVE) * 0.663
    old_scene = gltf_to_scene_dimensions(old_result)
    assert old_scene[2] > 0.15 - DIMENSION_CONTRACT_ATOL_METERS
