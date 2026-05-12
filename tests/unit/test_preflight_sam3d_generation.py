from __future__ import annotations

import hashlib
import json
import os

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from PIL import Image, ImageDraw

from scripts import preflight_sam3d_generation as preflight


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_runtime_identity(root: Path) -> dict[str, Any]:
    runtime_root = (root / "runtime_identity").resolve()
    distributions: list[dict[str, Any]] = []
    for (
        name,
        module_name,
        extension_globs,
        resource_globs,
    ) in preflight.RUNTIME_COMPONENTS:
        slug = name.replace("-", "_")

        def records(patterns: tuple[str, ...], prefix: str) -> list[dict[str, Any]]:
            return [
                {
                    "distribution_relative_path": f"{slug}/{prefix}_{index}",
                    "resolved_path": str(runtime_root / slug / f"{prefix}_{index}"),
                    "size_bytes": 1,
                    "sha256": ("7" if prefix == "extension" else "8") * 64,
                }
                for index, _pattern in enumerate(patterns)
            ]

        distributions.append(
            {
                "requested_name": name,
                "module_name": module_name,
                "canonical_name": name,
                "version": "1.0",
                "record_path": str(runtime_root / slug / "RECORD"),
                "record_size_bytes": 1,
                "record_sha256": "9" * 64,
                "module_origin": str(runtime_root / slug / "__init__.py"),
                "module_origin_size_bytes": 1,
                "module_origin_sha256": "6" * 64,
                "required_extension_globs": list(extension_globs),
                "core_extension_files": records(extension_globs, "extension"),
                "required_resource_globs": list(resource_globs),
                "runtime_resource_files": records(resource_globs, "resource"),
            }
        )
    return {
        "python_implementation": "cpython",
        "python_version": "3.11.0",
        "distribution_count": len(distributions),
        "distributions": distributions,
        "scope_note": "fixture package/runtime identity",
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    input_path = repo / preflight.CANONICAL_INPUT_RELATIVE
    input_path.parent.mkdir(parents=True)
    image = Image.new(
        "RGB",
        (preflight.CANONICAL_INPUT_WIDTH, preflight.CANONICAL_INPUT_HEIGHT),
        "white",
    )
    ImageDraw.Draw(image).rectangle((256, 384, 767, 1151), fill=(40, 90, 160))
    image.save(input_path)
    for relative in preflight.CODE_RELATIVE_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# exact fixture code: {relative}\n", encoding="utf-8")
    checkpoint = repo / "external" / "checkpoints" / "sam3.pt"
    pipeline = repo / "external" / "checkpoints" / "pipeline.yaml"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sam3-checkpoint")
    pipeline.write_text("pipeline: fixture\n", encoding="utf-8")
    model_specs = [
        {"role": "sam3_checkpoint", "path": checkpoint, "kind": "file"},
        {"role": "pipeline_config", "path": pipeline, "kind": "file"},
    ]
    source_tree_specs = []
    for role, relative in preflight.REPOSITORY_SOURCE_TREES:
        root = repo / relative
        root.mkdir(parents=True, exist_ok=True)
        import_origin = root / "__init__.py"
        import_origin.write_text(f"PACKAGE_ROLE = {role!r}\n", encoding="utf-8")
        (root / "module.py").write_text(f"ROLE = {role!r}\n", encoding="utf-8")
        (root / "runtime_config.yaml").write_text(
            f"role: {role}\n", encoding="utf-8"
        )
        extras = preflight.SOURCE_TREE_EXTRA_FILES.get(role, ())
        for extra in extras:
            resource = root / extra
            resource.parent.mkdir(parents=True, exist_ok=True)
            resource.write_bytes(b"runtime-resource")
        source_tree_specs.append(
            {
                "role": role,
                "path": root,
                "recorded_path": relative,
                "import_name": preflight.SOURCE_TREE_MODULES[role],
                "import_origin": import_origin,
                "recorded_import_origin": (
                    Path(relative) / "__init__.py"
                ).as_posix(),
                "extra_relative_files": extras,
            }
        )
    moge_root = tmp_path / "site-packages" / "moge"
    moge_root.mkdir(parents=True)
    (moge_root / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    source_tree_specs.append(
        {
            "role": preflight.MOGE_SOURCE_ROLE,
            "path": moge_root,
            "recorded_path": str(moge_root.resolve()),
            "import_name": "moge",
            "import_origin": moge_root / "__init__.py",
            "recorded_import_origin": str((moge_root / "__init__.py").resolve()),
            "extra_relative_files": (),
        }
    )
    runtime_identity = _fake_runtime_identity(tmp_path)
    calls: list[dict[str, Path]] = []

    def generator(
        source: Path,
        output: Path,
        debug: Path,
        sam3: Path,
        config: Path,
    ) -> None:
        calls.append(
            {
                "source": source,
                "output": output,
                "debug": debug,
                "sam3": sam3,
                "config": config,
            }
        )
        output.write_bytes(b"fixture-glb")
        with Image.open(source) as opened:
            rgb = opened.convert("RGB")
        mask = Image.new("L", rgb.size, 0)
        ImageDraw.Draw(mask).rectangle((256, 384, 767, 1151), fill=255)
        mask.save(debug / preflight.MASK_NAME)
        masked = Image.new("RGB", rgb.size, "black")
        masked.paste(rgb, mask=mask)
        masked.save(debug / preflight.MASKED_IMAGE_NAME)

    def mesh_inspector(path: Path) -> dict[str, Any]:
        assert path.read_bytes() == b"fixture-glb"
        return {
            "loader": "fixture",
            "geometry_count": 1,
            "vertex_count": 8,
            "face_count": 12,
            "bounds": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        }

    return {
        "repo": repo,
        "input": input_path,
        "checkpoint": checkpoint,
        "pipeline": pipeline,
        "preflight_dir": repo / "outputs" / "preflight" / "run",
        "model_specs": model_specs,
        "source_tree_specs": source_tree_specs,
        "runtime_identity": runtime_identity,
        "generator": generator,
        "mesh_inspector": mesh_inspector,
        "calls": calls,
    }


def _run(fixture: dict[str, Any]) -> dict[str, Any]:
    return preflight.run_generation(
        repo_dir=fixture["repo"],
        preflight_dir=fixture["preflight_dir"],
        sam3_checkpoint=fixture["checkpoint"],
        pipeline_config=fixture["pipeline"],
        expected_input_sha256=_hash(fixture["input"]),
        model_specs=fixture["model_specs"],
        source_tree_specs=fixture["source_tree_specs"],
        runtime_identity=fixture["runtime_identity"],
        generator=fixture["generator"],
        mesh_inspector=fixture["mesh_inspector"],
    )


def _verify(fixture: dict[str, Any]) -> dict[str, Any]:
    return preflight.verify_output(
        repo_dir=fixture["repo"],
        preflight_dir=fixture["preflight_dir"],
        sam3_checkpoint=fixture["checkpoint"],
        pipeline_config=fixture["pipeline"],
        expected_input_sha256=_hash(fixture["input"]),
        mesh_inspector=fixture["mesh_inspector"],
        source_tree_specs=fixture["source_tree_specs"],
        runtime_identity=fixture["runtime_identity"],
    )


def _mutate_receipt(fixture: dict[str, Any], mutate) -> dict[str, Any]:
    receipt_path = preflight.canonical_paths(fixture["preflight_dir"])["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutate(receipt)
    receipt["attestation"] = preflight._attestation(receipt)
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    return receipt


def test_generation_atomically_publishes_attested_full_production_artifacts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    result = _run(fixture)
    paths = preflight.canonical_paths(fixture["preflight_dir"])

    assert result["status"] == "pass"
    assert result["offline"] is True
    assert result["paid_api_calls"] == 0
    assert result["generation_parameters"] == {
        "backend": "sam3d",
        "mode": "foreground",
        "object_description": None,
        "threshold": 0.5,
        "use_pipeline_caching": False,
    }
    assert result["outputs"]["glb"]["mesh"]["vertex_count"] == 8
    assert result["outputs"]["glb"]["mesh"]["face_count"] == 12
    assert result["outputs"]["image_validation"]["masked_image_matches_mask"] is True
    assert result["attestation"] == preflight._attestation(result)
    assert paths["receipt"].is_file()
    assert paths["glb"].is_file()
    assert paths["mask"].is_file()
    assert paths["masked_image"].is_file()
    assert len(fixture["calls"]) == 1
    assert not list(paths["preflight_dir"].glob(".sam3d_offline_generation.*.tmp"))


def test_verify_only_rehashes_and_reloads_without_generator_or_receipt_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    receipt = preflight.canonical_paths(fixture["preflight_dir"])["receipt"]
    before = receipt.read_bytes()

    def forbidden_generator(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("verify-only imported/called production inference")

    monkeypatch.setattr(preflight, "_production_generate", forbidden_generator)
    result = _verify(fixture)

    assert result["status"] == "pass"
    assert receipt.read_bytes() == before
    assert len(fixture["calls"]) == 1


def test_portable_acceptance_validation_binds_copied_outputs_without_inference(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _run(fixture)
    artifact_dir = preflight.canonical_paths(fixture["preflight_dir"])["artifact_dir"]

    assert preflight.verify_bound_receipt(
        result,
        input_path=fixture["input"],
        artifact_dir=artifact_dir,
        expected_input_sha256=_hash(fixture["input"]),
        mesh_inspector=fixture["mesh_inspector"],
    ) == []


def test_portable_acceptance_rejects_recomputed_attestation_with_fake_mesh_counts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _run(fixture)
    result["outputs"]["glb"]["mesh"]["vertex_count"] = 0
    result["attestation"] = preflight._attestation(result)

    failures = preflight.verify_bound_receipt(
        result,
        input_path=fixture["input"],
        artifact_dir=preflight.canonical_paths(fixture["preflight_dir"])["artifact_dir"],
        expected_input_sha256=_hash(fixture["input"]),
        mesh_inspector=fixture["mesh_inspector"],
    )

    assert "SAM3D GLB mesh-count evidence is invalid" in failures


@pytest.mark.parametrize("target", ("glb", "mask", "masked_image"))
def test_verify_rejects_mutated_published_outputs(tmp_path: Path, target: str) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    path = preflight.canonical_paths(fixture["preflight_dir"])[target]
    path.write_bytes(path.read_bytes() + b"mutated")

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="cannot reload|cannot independently recompute|output evidence is stale",
    ):
        _verify(fixture)


@pytest.mark.parametrize("target", ("checkpoint", "pipeline"))
def test_verify_rejects_mutated_model_artifacts(tmp_path: Path, target: str) -> None:
    fixture = _fixture(tmp_path)
    expected_input_sha256 = _hash(fixture["input"])
    _run(fixture)
    fixture[target].write_bytes(fixture[target].read_bytes() + b"mutated")

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="SHA-256 mismatch|file size changed|cannot independently rediscover",
    ):
        preflight.verify_output(
            repo_dir=fixture["repo"],
            preflight_dir=fixture["preflight_dir"],
            sam3_checkpoint=fixture["checkpoint"],
            pipeline_config=fixture["pipeline"],
            expected_input_sha256=expected_input_sha256,
            mesh_inspector=fixture["mesh_inspector"],
            source_tree_specs=fixture["source_tree_specs"],
            runtime_identity=fixture["runtime_identity"],
        )


def test_verify_rejects_mutated_bound_generation_code(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    code = fixture["repo"] / preflight.CODE_RELATIVE_PATHS[-1]
    code.write_text("# changed runtime\n", encoding="utf-8")

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="generation code changed",
    ):
        _verify(fixture)


def test_reattested_model_path_redirection_is_rejected_by_independent_discovery(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    redirected = tmp_path / "redirected_pipeline.yaml"
    redirected.write_text("pipeline: redirected\n", encoding="utf-8")

    def mutate(receipt: dict[str, Any]) -> None:
        entry = next(
            item
            for item in receipt["model_artifacts"]["artifacts"]
            if item["role"] == "pipeline_config"
        )
        entry["path"] = str(redirected.resolve())
        entry["size_bytes"] = redirected.stat().st_size
        entry["sha256"] = _hash(redirected)

    _mutate_receipt(fixture, mutate)

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="role/path/kind inventory is not exact|evidence path does not match",
    ):
        _verify(fixture)


def test_reattested_duplicate_model_role_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)

    def mutate(receipt: dict[str, Any]) -> None:
        duplicate = dict(receipt["model_artifacts"]["artifacts"][0])
        receipt["model_artifacts"]["artifacts"].append(duplicate)
        receipt["model_artifacts"]["artifact_count"] += 1

    _mutate_receipt(fixture, mutate)

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="duplicate roles|role/path/kind inventory is not exact",
    ):
        _verify(fixture)


def test_reattested_duplicate_code_entry_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)

    def mutate(receipt: dict[str, Any]) -> None:
        receipt["code_artifacts"].append(dict(receipt["code_artifacts"][0]))

    _mutate_receipt(fixture, mutate)

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="code evidence is malformed",
    ):
        _verify(fixture)


def test_mutated_external_executable_source_tree_invalidates_proof(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    source = Path(fixture["source_tree_specs"][0]["path"]) / "module.py"
    source.write_text("ROLE = 'mutated'\n", encoding="utf-8")

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="source-tree evidence is stale",
    ):
        _verify(fixture)


def test_mutated_non_python_package_resource_invalidates_proof(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    resource = (
        Path(fixture["source_tree_specs"][1]["path"]) / "runtime_config.yaml"
    )
    resource.write_text("role: mutated\n", encoding="utf-8")

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="source-tree evidence is stale",
    ):
        _verify(fixture)


def test_reattested_source_tree_redirection_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)

    def mutate(receipt: dict[str, Any]) -> None:
        receipt["executable_source_trees"][0]["resolved_path"] = "/redirected/source"

    _mutate_receipt(fixture, mutate)

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="source-tree evidence is stale or redirected",
    ):
        _verify(fixture)


def test_reattested_import_origin_redirection_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)

    def mutate(receipt: dict[str, Any]) -> None:
        receipt["executable_source_trees"][0][
            "resolved_import_origin_path"
        ] = str((tmp_path / "redirected" / "__init__.py").resolve())

    _mutate_receipt(fixture, mutate)

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="source-tree evidence is stale or redirected",
    ):
        _verify(fixture)


def test_reattested_runtime_distribution_identity_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)

    def mutate(receipt: dict[str, Any]) -> None:
        receipt["runtime_identity"]["distributions"][0]["version"] = "forged"

    _mutate_receipt(fixture, mutate)

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="runtime distribution identity is stale",
    ):
        _verify(fixture)


def test_reattested_nonexact_validation_object_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)

    def mutate(receipt: dict[str, Any]) -> None:
        receipt["validation"]["unverified_extra"] = True

    _mutate_receipt(fixture, mutate)

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="validation proof is missing",
    ):
        _verify(fixture)


def test_portable_consumer_reloads_glb_instead_of_trusting_reattested_counts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _run(fixture)

    def invalid_glb(_path: Path) -> dict[str, Any]:
        raise ValueError("not a loadable GLB")

    failures = preflight.verify_bound_receipt(
        result,
        input_path=fixture["input"],
        artifact_dir=preflight.canonical_paths(fixture["preflight_dir"])["artifact_dir"],
        expected_input_sha256=_hash(fixture["input"]),
        mesh_inspector=invalid_glb,
    )

    assert any("not a loadable GLB" in failure for failure in failures)


@pytest.mark.parametrize("kind", ("file", "directory"))
def test_verify_rejects_extra_artifact_inventory_entry(
    tmp_path: Path, kind: str
) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    artifact_dir = preflight.canonical_paths(fixture["preflight_dir"])["artifact_dir"]
    extra = artifact_dir / "unexpected"
    if kind == "file":
        extra.write_bytes(b"extra")
    else:
        extra.mkdir()

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="inventory is not exact|directory, special, or empty",
    ):
        _verify(fixture)


def test_verify_rejects_orphan_transaction_sibling(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    orphan = (
        Path(fixture["preflight_dir"])
        / ".sam3d_offline_generation.crashed.tmp"
    )
    orphan.mkdir()

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="orphan SAM3D generation transaction",
    ):
        _verify(fixture)


def test_verify_rejects_special_artifact_inventory_entry_when_supported(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")
    fixture = _fixture(tmp_path)
    _run(fixture)
    artifact_dir = preflight.canonical_paths(fixture["preflight_dir"])["artifact_dir"]
    fifo = artifact_dir / "unexpected_fifo"
    try:
        os.mkfifo(fifo)
    except OSError:
        pytest.skip("FIFO creation is unavailable for this test user")

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="directory, special, or empty",
    ):
        _verify(fixture)


def test_verify_rejects_linked_artifact_inventory_entry_when_supported(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    artifact_dir = preflight.canonical_paths(fixture["preflight_dir"])["artifact_dir"]
    link = artifact_dir / "unexpected_link"
    try:
        link.symlink_to(artifact_dir / preflight.GLB_NAME)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are unavailable for this test user")

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="inventory contains a link",
    ):
        _verify(fixture)


def test_generation_rejects_preexisting_orphan_transaction(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    preflight_dir = Path(fixture["preflight_dir"])
    preflight_dir.mkdir(parents=True)
    (preflight_dir / ".sam3d_offline_generation.crashed.tmp").write_bytes(b"orphan")

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="orphan SAM3D generation transaction",
    ):
        _run(fixture)


def test_portable_consumer_rejects_nonbinary_mask_even_when_reattested(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    paths = preflight.canonical_paths(fixture["preflight_dir"])
    with Image.open(paths["mask"]) as opened:
        mask = opened.copy()
    mask.putpixel((300, 500), 128)
    mask.save(paths["mask"])

    def mutate(receipt: dict[str, Any]) -> None:
        receipt["outputs"]["mask"]["size_bytes"] = paths["mask"].stat().st_size
        receipt["outputs"]["mask"]["sha256"] = _hash(paths["mask"])

    result = _mutate_receipt(fixture, mutate)
    failures = preflight.verify_bound_receipt(
        result,
        input_path=fixture["input"],
        artifact_dir=paths["artifact_dir"],
        expected_input_sha256=_hash(fixture["input"]),
        mesh_inspector=fixture["mesh_inspector"],
    )

    assert any("not exactly binary" in failure for failure in failures)


def test_portable_consumer_rejects_implausibly_tiny_mask_even_when_reattested(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    paths = preflight.canonical_paths(fixture["preflight_dir"])
    with Image.open(fixture["input"]) as opened:
        source = opened.convert("RGB")
    mask = Image.new("L", source.size, 0)
    mask.putpixel((source.width // 2, source.height // 2), 255)
    mask.save(paths["mask"])
    masked = Image.new("RGB", source.size, "black")
    masked.paste(source, mask=mask)
    masked.save(paths["masked_image"])

    def mutate(receipt: dict[str, Any]) -> None:
        for key in ("mask", "masked_image"):
            path = paths[key]
            receipt["outputs"][key]["size_bytes"] = path.stat().st_size
            receipt["outputs"][key]["sha256"] = _hash(path)

    result = _mutate_receipt(fixture, mutate)
    failures = preflight.verify_bound_receipt(
        result,
        input_path=fixture["input"],
        artifact_dir=paths["artifact_dir"],
        expected_input_sha256=_hash(fixture["input"]),
        mesh_inspector=fixture["mesh_inspector"],
    )

    assert any("foreground fraction is implausible" in failure for failure in failures)


def test_verify_rejects_linked_artifact_directory_when_supported(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    artifact_dir = preflight.canonical_paths(fixture["preflight_dir"])["artifact_dir"]
    real_dir = artifact_dir.with_name("real_generation_artifacts")
    artifact_dir.rename(real_dir)
    try:
        artifact_dir.symlink_to(real_dir, target_is_directory=True)
    except (NotImplementedError, OSError):
        real_dir.rename(artifact_dir)
        pytest.skip("directory symlinks are unavailable for this test user")

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="must not be link-like|link-like parent",
    ):
        _verify(fixture)


def test_recomputed_attestation_cannot_hide_missing_mesh_faces(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _run(fixture)
    receipt_path = preflight.canonical_paths(fixture["preflight_dir"])["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["outputs"]["glb"]["mesh"]["face_count"] = 0
    receipt["attestation"] = preflight._attestation(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        preflight.SAM3DGenerationPreflightError,
        match="output evidence is stale",
    ):
        _verify(fixture)


def test_committed_canonical_input_hash_is_fixed_when_present() -> None:
    path = Path(__file__).resolve().parents[2] / preflight.CANONICAL_INPUT_RELATIVE
    if not path.is_file():
        pytest.skip("canonical upstream office_shelf fixture is not in the overlay checkout")
    assert _hash(path) == preflight.CANONICAL_INPUT_SHA256
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.size == (
            preflight.CANONICAL_INPUT_WIDTH,
            preflight.CANONICAL_INPUT_HEIGHT,
        )


def test_external_parent_symlink_is_resolved_and_bound_without_trusting_retarget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    shared_external = tmp_path / "shared_external"
    for role, relative in preflight.REPOSITORY_SOURCE_TREES:
        suffix = Path(relative).relative_to("external")
        root = shared_external / suffix
        root.mkdir(parents=True)
        (root / "__init__.py").write_text(
            f"PACKAGE_ROLE = {role!r}\n", encoding="utf-8"
        )
        (root / "module.py").write_text(f"ROLE = {role!r}\n", encoding="utf-8")
        for extra in preflight.SOURCE_TREE_EXTRA_FILES.get(role, ()):
            resource = root / extra
            resource.parent.mkdir(parents=True, exist_ok=True)
            resource.write_bytes(b"runtime-resource")
    try:
        (repo / "external").symlink_to(shared_external, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable for this test user")
    moge = tmp_path / "moge"
    moge.mkdir()
    (moge / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(
            submodule_search_locations=[str(moge)],
            origin=str(moge / "__init__.py"),
        ),
    )

    specs = preflight._discover_source_tree_specs(repo.resolve())
    evidence = preflight._build_source_tree_evidence(specs)

    by_role = {item["role"]: item for item in evidence}
    sam3 = by_role["external_sam3_python_source"]
    assert sam3["path"] == "external/SAM3/sam3"
    assert sam3["resolved_path"].startswith(str(shared_external.resolve()))
    assert sam3["selection"]["required_runtime_resources"] == [
        "assets/bpe_simple_vocab_16e6.txt.gz"
    ]

    replacement = tmp_path / "replacement_external"
    for role, relative in preflight.REPOSITORY_SOURCE_TREES:
        suffix = Path(relative).relative_to("external")
        root = replacement / suffix
        root.mkdir(parents=True)
        (root / "__init__.py").write_text("RETARGETED_PACKAGE = True\n", encoding="utf-8")
        (root / "module.py").write_text("RETARGETED = True\n", encoding="utf-8")
        for extra in preflight.SOURCE_TREE_EXTRA_FILES.get(role, ()):
            resource = root / extra
            resource.parent.mkdir(parents=True, exist_ok=True)
            resource.write_bytes(b"retargeted-resource")
    (repo / "external").unlink()
    (repo / "external").symlink_to(replacement, target_is_directory=True)

    failures = preflight._verify_source_tree_evidence(repo.resolve(), evidence)
    assert failures == [
        "SAM3D executable source-tree evidence is stale or redirected"
    ]
