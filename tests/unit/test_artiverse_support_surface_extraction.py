"""Regression coverage for SDF-authored Artiverse support meshes."""

from __future__ import annotations

import hashlib
import json
import os

from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from scenesmith.agent_utils import support_surface_extraction as extraction


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_artiverse_fixture(
    root: Path,
    *,
    visual_uri: str = "./scenesmith_artiverse_visuals_v2/base.gltf",
    include_collision: bool = True,
    link_pose: str = "",
    visual_pose: str = "",
    mesh_scale: str = "",
) -> Path:
    visual_dir = root / "scenesmith_artiverse_visuals_v2"
    collision_dir = root / "objs"
    publisher_dir = root / "glbs"
    visual_dir.mkdir(parents=True)
    collision_dir.mkdir()
    publisher_dir.mkdir()
    binary = b"derived-bin"
    gltf_document = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "base.bin", "byteLength": len(binary)}],
    }
    gltf = json.dumps(gltf_document, separators=(",", ":")).encode()
    preserved_document = json.loads(json.dumps(gltf_document))
    del preserved_document["buffers"][0]["uri"]
    preserved = (
        json.dumps(
            preserved_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    publisher = b"publisher-glb"
    collision = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    (visual_dir / "base.gltf").write_bytes(gltf)
    (visual_dir / "base.bin").write_bytes(binary)
    (publisher_dir / "base.glb").write_bytes(publisher)
    (collision_dir / "base.obj").write_bytes(collision)
    manifest = {
        "schema_version": 2,
        "status": "pass",
        "policy": "publisher_glb_derived_external_gltf",
        "resource_directory": "scenesmith_artiverse_visuals_v2",
        "resources": [
            {
                "part": "base",
                "publisher_glb_sha256": _sha256(publisher),
                "publisher_glb_size_bytes": len(publisher),
                "publisher_document_sha256": _sha256(b"publisher-document"),
                "preserved_semantics_sha256": _sha256(preserved),
                "derived_gltf": "base.gltf",
                "derived_gltf_size_bytes": len(gltf),
                "derived_gltf_sha256": _sha256(gltf),
                "derived_bin": "base.bin",
                "derived_bin_size_bytes": len(binary),
                "derived_bin_sha256": _sha256(binary),
            }
        ],
    }
    (visual_dir / "_derivation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    collision_scale_xml = f"<scale>{mesh_scale}</scale>" if mesh_scale else ""
    collision_xml = (
        "<collision><geometry><mesh><uri>./objs/base.obj</uri>"
        f"{collision_scale_xml}</mesh></geometry></collision>"
        if include_collision
        else ""
    )
    visual_scale_xml = f"<scale>{mesh_scale}</scale>" if mesh_scale else ""
    link_pose_xml = f"<pose>{link_pose}</pose>" if link_pose else ""
    visual_pose_xml = f"<pose>{visual_pose}</pose>" if visual_pose else ""
    sdf = root / "scenesmith_artiverse.sdf"
    sdf.write_text(
        "<sdf version='1.7'><model name='asset'><link name='publisher_base'>"
        f"{link_pose_xml}<visual>{visual_pose_xml}<geometry><mesh>"
        f"<uri>{visual_uri}</uri>{visual_scale_xml}</mesh></geometry></visual>"
        f"{collision_xml}</link></model></sdf>",
        encoding="utf-8",
    )
    return sdf


def test_artiverse_uses_sdf_visual_and_collision_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdf = _write_artiverse_fixture(tmp_path)
    observed: dict[str, object] = {}

    def fake_collision_unions(records):
        observed["records"] = records
        return None, {}

    def fake_extract(mesh_path: Path, config, **kwargs):
        observed["visual"] = mesh_path
        observed["visual_kwargs"] = kwargs
        return [SimpleNamespace(area=0.5, link_name=None)]

    monkeypatch.setattr(
        extraction, "_load_artiverse_collision_unions", fake_collision_unions
    )
    monkeypatch.setattr(extraction, "extract_support_surfaces_from_mesh", fake_extract)

    surfaces = extraction.extract_support_surfaces_articulated(
        tmp_path, sdf_path=sdf
    )

    assert len(surfaces) == 1
    assert surfaces[0].link_name == "publisher_base"
    assert observed["visual"] == (
        tmp_path / "scenesmith_artiverse_visuals_v2" / "base.gltf"
    ).resolve()
    records = observed["records"]
    assert len(records) == 1
    assert records[0].link_name == "publisher_base"
    assert records[0].collision_meshes[0].path == (
        tmp_path / "objs" / "base.obj"
    ).resolve()


def test_artiverse_missing_collision_fails_closed(tmp_path: Path) -> None:
    sdf = _write_artiverse_fixture(tmp_path, include_collision=False)
    with pytest.raises(ValueError, match="at least one collision"):
        extraction.extract_support_surfaces_articulated(tmp_path, sdf_path=sdf)


def test_artiverse_visual_uri_cannot_escape_asset(tmp_path: Path) -> None:
    sdf = _write_artiverse_fixture(tmp_path, visual_uri="./../outside.gltf")
    with pytest.raises(ValueError, match="unsafe or unsupported"):
        extraction.extract_support_surfaces_articulated(tmp_path, sdf_path=sdf)


def test_artiverse_derivation_hash_is_reverified(tmp_path: Path) -> None:
    sdf = _write_artiverse_fixture(tmp_path)
    (tmp_path / "scenesmith_artiverse_visuals_v2" / "base.gltf").write_bytes(
        b"substituted"
    )
    with pytest.raises(ValueError, match="does not match manifest"):
        extraction.extract_support_surfaces_articulated(tmp_path, sdf_path=sdf)


def test_artiverse_missing_manifest_is_rejected(tmp_path: Path) -> None:
    sdf = _write_artiverse_fixture(tmp_path)
    (
        tmp_path
        / "scenesmith_artiverse_visuals_v2"
        / "_derivation_manifest.json"
    ).unlink()
    with pytest.raises(FileNotFoundError, match="derivation manifest is missing"):
        extraction._parse_artiverse_link_meshes(sdf, tmp_path)


def test_artiverse_required_link_extraction_failure_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdf = _write_artiverse_fixture(tmp_path)
    monkeypatch.setattr(
        extraction, "_load_artiverse_collision_unions", lambda records: (None, {})
    )

    def fail_extract(mesh_path: Path, config, **kwargs):
        raise ValueError("unreadable visual")

    monkeypatch.setattr(extraction, "extract_support_surfaces_from_mesh", fail_extract)
    with pytest.raises(ValueError, match="Failed required Artiverse link extraction"):
        extraction.extract_support_surfaces_articulated(tmp_path, sdf_path=sdf)


def test_artiverse_pose_and_uniform_scale_are_applied(tmp_path: Path) -> None:
    sdf = _write_artiverse_fixture(
        tmp_path,
        link_pose="1 0 0 0 0 0",
        visual_pose="0.5 0 0 0 0 0",
        mesh_scale="0.55 0.55 0.55",
    )
    records = extraction._parse_artiverse_link_meshes(sdf, tmp_path)
    assert records[0].visual_mesh.uniform_scale == pytest.approx(0.55)
    np.testing.assert_allclose(records[0].visual_mesh.transform[:3, 3], [1.5, 0, 0])
    np.testing.assert_allclose(
        records[0].collision_meshes[0].transform[:3, 3], [1.0, 0, 0]
    )


def test_artiverse_nonuniform_scale_fails_closed(tmp_path: Path) -> None:
    sdf = _write_artiverse_fixture(tmp_path, mesh_scale="1 2 1")
    with pytest.raises(ValueError, match="uniform mesh scale"):
        extraction._parse_artiverse_link_meshes(sdf, tmp_path)


def test_artiverse_resource_symlink_is_rejected(tmp_path: Path) -> None:
    sdf = _write_artiverse_fixture(tmp_path)
    collision = tmp_path / "objs" / "base.obj"
    outside = tmp_path.parent / f"{tmp_path.name}-outside.obj"
    outside.write_text("v 0 0 0\n", encoding="utf-8")
    collision.unlink()
    try:
        os.symlink(outside, collision)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink"):
        extraction._parse_artiverse_link_meshes(sdf, tmp_path)


def test_artiverse_manifest_extra_resource_is_rejected(tmp_path: Path) -> None:
    sdf = _write_artiverse_fixture(tmp_path)
    visual_dir = tmp_path / "scenesmith_artiverse_visuals_v2"
    document = json.loads((visual_dir / "_derivation_manifest.json").read_text())
    extra_bin = b"extra"
    extra_document = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "extra.bin", "byteLength": len(extra_bin)}],
    }
    extra_gltf = json.dumps(extra_document, separators=(",", ":")).encode()
    preserved_document = json.loads(json.dumps(extra_document))
    del preserved_document["buffers"][0]["uri"]
    extra_preserved = (
        json.dumps(
            preserved_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    extra_publisher = b"extra-publisher"
    (visual_dir / "extra.bin").write_bytes(extra_bin)
    (visual_dir / "extra.gltf").write_bytes(extra_gltf)
    (tmp_path / "glbs" / "extra.glb").write_bytes(extra_publisher)
    document["resources"].append(
        {
            "part": "extra",
            "publisher_glb_sha256": _sha256(extra_publisher),
            "publisher_glb_size_bytes": len(extra_publisher),
            "publisher_document_sha256": _sha256(b"extra-document"),
            "preserved_semantics_sha256": _sha256(extra_preserved),
            "derived_gltf": "extra.gltf",
            "derived_gltf_size_bytes": len(extra_gltf),
            "derived_gltf_sha256": _sha256(extra_gltf),
            "derived_bin": "extra.bin",
            "derived_bin_size_bytes": len(extra_bin),
            "derived_bin_sha256": _sha256(extra_bin),
        }
    )
    (visual_dir / "_derivation_manifest.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="inventory mismatch"):
        extraction._parse_artiverse_link_meshes(sdf, tmp_path)


def test_artiverse_gltf_transitive_uri_is_rejected(tmp_path: Path) -> None:
    sdf = _write_artiverse_fixture(tmp_path)
    visual_dir = tmp_path / "scenesmith_artiverse_visuals_v2"
    path = visual_dir / "base.gltf"
    document = json.loads(path.read_text())
    document["images"] = [{"uri": "../substituted.png"}]
    payload = json.dumps(document, separators=(",", ":")).encode()
    path.write_bytes(payload)
    manifest_path = visual_dir / "_derivation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["resources"][0]["derived_gltf_size_bytes"] = len(payload)
    manifest["resources"][0]["derived_gltf_sha256"] = _sha256(payload)
    preserved = json.loads(json.dumps(document))
    del preserved["buffers"][0]["uri"]
    manifest["resources"][0]["preserved_semantics_sha256"] = _sha256(
        (
            json.dumps(
                preserved,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unexpected transitive URIs"):
        extraction._parse_artiverse_link_meshes(sdf, tmp_path)


def test_artiverse_unlisted_derived_resource_is_rejected(tmp_path: Path) -> None:
    sdf = _write_artiverse_fixture(tmp_path)
    (tmp_path / "scenesmith_artiverse_visuals_v2" / "substituted.bin").write_bytes(
        b"substituted"
    )
    with pytest.raises(ValueError, match="directory inventory mismatch"):
        extraction._parse_artiverse_link_meshes(sdf, tmp_path)


def test_normalized_artiverse_sdf_cannot_be_bypassed_by_legacy_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdf = _write_artiverse_fixture(tmp_path, include_collision=False)
    (tmp_path / "substituted_combined.gltf").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        extraction,
        "extract_support_surfaces_from_mesh",
        lambda *args, **kwargs: [SimpleNamespace(area=1.0, link_name=None)],
    )
    with pytest.raises(ValueError, match="at least one collision"):
        extraction.extract_support_surfaces_articulated(tmp_path, sdf_path=sdf)


def test_artiverse_nonidentity_model_pose_fails_closed(tmp_path: Path) -> None:
    sdf = _write_artiverse_fixture(tmp_path)
    payload = sdf.read_text(encoding="utf-8").replace(
        "<model name='asset'>", "<model name='asset'><pose>1 0 0 0 0 0</pose>"
    )
    sdf.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="model pose must be identity"):
        extraction._parse_artiverse_link_meshes(sdf, tmp_path)


@pytest.mark.parametrize(
    "attribute",
    ["degrees='true'", "rotation_format='quat_xyzw'", "unexpected='value'"],
)
def test_artiverse_unsupported_pose_attributes_fail_closed(
    tmp_path: Path, attribute: str
) -> None:
    sdf = _write_artiverse_fixture(tmp_path, link_pose="0 0 0 0 0 0")
    sdf.write_text(
        sdf.read_text(encoding="utf-8").replace("<pose>", f"<pose {attribute}>"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported pose attributes"):
        extraction._parse_artiverse_link_meshes(sdf, tmp_path)


class _SyntheticRayIntersector:
    def __init__(self, *, hit_count: int, clearance: float):
        self.hit_count = hit_count
        self.clearance = clearance

    def intersects_location(self, *, ray_origins, ray_directions, multiple_hits):
        assert multiple_hits
        indices = np.arange(self.hit_count, dtype=int)
        locations = ray_origins[indices].copy()
        locations[:, 2] += self.clearance - 0.001
        return locations, indices, np.zeros(self.hit_count, dtype=int)


def _synthetic_surface(*, authored_clearance: float):
    mesh = SimpleNamespace(
        vertices=np.array(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]
        )
    )
    return SimpleNamespace(
        surface_id="surface",
        link_name="link",
        transform=extraction.RigidTransform(p=[0, 0, 0.01]),
        bounding_box_min=np.array([-0.5, -0.5, 0.01]),
        bounding_box_max=np.array([0.5, 0.5, 0.01 + authored_clearance]),
        mesh=mesh,
        contains_point_2d=lambda point: bool(
            -0.5 <= point[0] <= 0.5 and -0.5 <= point[1] <= 0.5
        ),
    )


class _DownwardRayIntersector:
    def __init__(self, *, hit_indices: list[int], gap_m: float):
        self.hit_indices = np.asarray(hit_indices, dtype=int)
        self.gap_m = gap_m

    def intersects_location(self, *, ray_origins, ray_directions, multiple_hits):
        assert multiple_hits
        np.testing.assert_allclose(
            ray_directions, np.tile([0, 0, -1], (len(ray_origins), 1))
        )
        locations = ray_origins[self.hit_indices].copy()
        locations[:, 2] -= self.gap_m + 0.001
        return locations, self.hit_indices, np.zeros(len(self.hit_indices), dtype=int)


def _collision_mesh(*, hit_indices: list[int], gap_m: float):
    return SimpleNamespace(
        ray=_DownwardRayIntersector(
            hit_indices=hit_indices,
            gap_m=gap_m,
        )
    )


def test_artiverse_same_link_collision_support_requires_full_nearby_coverage() -> None:
    config = extraction.SupportSurfaceExtractionConfig()
    surface = _synthetic_surface(authored_clearance=0.5)
    all_samples = list(range(25))

    supported = extraction._measure_artiverse_same_link_collision_support(
        surface,
        {"link": _collision_mesh(hit_indices=all_samples, gap_m=0.01)},
        config,
    )
    assert supported.supported
    assert supported.sample_count == supported.hit_count == 25
    assert supported.max_gap_m == pytest.approx(0.01)
    assert supported.max_allowed_gap_m == pytest.approx(0.03)

    for collision in (
        _collision_mesh(hit_indices=[], gap_m=0.01),
        _collision_mesh(hit_indices=all_samples, gap_m=0.10),
        _collision_mesh(hit_indices=list(range(15)), gap_m=0.01),
    ):
        measurement = extraction._measure_artiverse_same_link_collision_support(
            surface, {"link": collision}, config
        )
        assert not measurement.supported


def test_artiverse_collision_support_is_same_link_not_parent_link() -> None:
    config = extraction.SupportSurfaceExtractionConfig()
    surface = _synthetic_surface(authored_clearance=0.5)
    surface.link_name = "child"
    measurement = extraction._measure_artiverse_same_link_collision_support(
        surface,
        {
            "parent": _collision_mesh(hit_indices=list(range(25)), gap_m=0.01),
            "child": _collision_mesh(hit_indices=[], gap_m=0.01),
        },
        config,
    )
    assert not measurement.supported
    assert measurement.hit_count == 0


def test_clearance_samples_use_inset_actual_mesh_hull_not_centered_bbox() -> None:
    surface = _synthetic_surface(authored_clearance=0.5)
    surface.mesh.vertices[:, 0] += 2.0
    surface.contains_point_2d = lambda point: bool(
        1.5 <= point[0] <= 2.5 and -0.5 <= point[1] <= 0.5
    )
    local_points, _ = extraction._surface_interior_sample_points(surface)
    assert len(local_points) == 25
    assert np.min(local_points[:, 0]) == pytest.approx(1.6)
    assert np.max(local_points[:, 0]) == pytest.approx(2.4)


def test_combined_collision_clearance_tightens_persisted_bbox() -> None:
    config = extraction.SupportSurfaceExtractionConfig()
    surface = _synthetic_surface(authored_clearance=0.5)
    combined = SimpleNamespace(
        ray=_SyntheticRayIntersector(hit_count=5, clearance=0.1)
    )

    clearance = extraction._tighten_surface_clearance_against_combined_mesh(
        surface, combined, config
    )

    assert clearance == pytest.approx(0.1)
    assert (
        surface.bounding_box_max[2] - surface.bounding_box_min[2]
    ) == pytest.approx(0.1)


def test_combined_collision_clearance_never_enlarges_authored_bbox() -> None:
    config = extraction.SupportSurfaceExtractionConfig()
    surface = _synthetic_surface(authored_clearance=0.2)
    combined = SimpleNamespace(
        ray=_SyntheticRayIntersector(hit_count=0, clearance=0.5)
    )

    clearance = extraction._tighten_surface_clearance_against_combined_mesh(
        surface, combined, config
    )

    assert clearance == pytest.approx(0.2)
    assert (
        surface.bounding_box_max[2] - surface.bounding_box_min[2]
    ) == pytest.approx(0.2)


def test_combined_clearance_ignores_near_self_hit_but_keeps_later_overhead() -> None:
    class NearAndOverhead:
        def intersects_location(self, *, ray_origins, ray_directions, multiple_hits):
            assert multiple_hits
            indices = np.repeat(np.arange(5, dtype=int), 2)
            locations = ray_origins[indices].copy()
            locations[0::2, 2] += 0.0005
            locations[1::2, 2] += 0.099
            return locations, indices, np.zeros(len(indices), dtype=int)

    config = extraction.SupportSurfaceExtractionConfig()
    surface = _synthetic_surface(authored_clearance=0.5)
    clearance = extraction._tighten_surface_clearance_against_combined_mesh(
        surface, SimpleNamespace(ray=NearAndOverhead()), config
    )
    assert clearance == pytest.approx(0.1)
    assert (
        surface.bounding_box_max[2] - surface.bounding_box_min[2]
    ) == pytest.approx(0.1)
