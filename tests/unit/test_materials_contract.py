import json
import sys

from pathlib import Path

import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from materials_contract import (  # noqa: E402
    MANIFEST_NAME,
    MaterialsContractError,
    load_materials_authority,
    prepare_contract,
    sha256_file,
)


def _source_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "shared" / "materials"
    source_embeddings = data_root / "embeddings"
    contract_embeddings = tmp_path / "isolated" / "embeddings"
    source_embeddings.mkdir(parents=True)
    for material_id in ("GoodA", "GoodB"):
        directory = data_root / material_id
        directory.mkdir()
        (directory / f"{material_id}_Color.jpg").write_bytes(
            f"color-{material_id}".encode()
        )
        (directory / f"{material_id}_NormalGL.jpg").write_bytes(
            f"normal-{material_id}".encode()
        )
    ids = ["GoodA", "MissingC", "GoodB"]
    np.save(
        source_embeddings / "clip_embeddings.npy",
        np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    (source_embeddings / "embedding_index.yaml").write_text(
        yaml.safe_dump(ids, sort_keys=False), encoding="utf-8"
    )
    (source_embeddings / "metadata_index.yaml").write_text(
        yaml.safe_dump(
            {
                material_id: {"category": material_id, "tags": [material_id.lower()]}
                for material_id in ids
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return data_root, source_embeddings, contract_embeddings


def _prepare(tmp_path: Path):
    data_root, source_embeddings, contract_embeddings = _source_fixture(tmp_path)
    result = prepare_contract(
        data_root=data_root,
        source_embeddings=source_embeddings,
        contract_embeddings=contract_embeddings,
        min_retained=2,
        max_pruned=1,
    )
    return data_root, source_embeddings, contract_embeddings, result


def test_prepare_prunes_only_missing_assets_and_preserves_shared_indexes(
    tmp_path: Path,
) -> None:
    data_root, source_embeddings, contract_embeddings = _source_fixture(tmp_path)
    before = {
        path.name: sha256_file(path)
        for path in source_embeddings.iterdir()
        if path.is_file()
    }

    result = prepare_contract(
        data_root=data_root,
        source_embeddings=source_embeddings,
        contract_embeddings=contract_embeddings,
        min_retained=2,
        max_pruned=1,
    )

    after = {
        path.name: sha256_file(path)
        for path in source_embeddings.iterdir()
        if path.is_file()
    }
    assert before == after
    assert result["status"] == "pass"
    assert result["source_count"] == 3
    assert result["retained_count"] == 2
    assert result["excluded"] == [
        {"material_id": "MissingC", "reason": "missing_material_directory"}
    ]
    assert yaml.safe_load(
        (contract_embeddings / "embedding_index.yaml").read_text(encoding="utf-8")
    ) == ["GoodA", "GoodB"]
    matrix = np.load(contract_embeddings / "clip_embeddings.npy")
    assert np.array_equal(
        matrix,
        np.array([[0, 1, 2, 3], [8, 9, 10, 11]], dtype=np.float32),
    )
    manifest = json.loads(
        (contract_embeddings / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["contract"]["retained_count"] == 2
    assert manifest["contract"]["pruned_count"] == 1


def test_validation_rejects_contract_when_previously_missing_asset_appears(
    tmp_path: Path,
) -> None:
    data_root, source_embeddings, contract_embeddings, _ = _prepare(tmp_path)
    restored = data_root / "MissingC"
    restored.mkdir()
    (restored / "MissingC_Color.jpg").write_bytes(b"now-present")

    with pytest.raises(MaterialsContractError, match="stale"):
        load_materials_authority(
            data_root=data_root,
            source_embeddings=source_embeddings,
            contract_embeddings=contract_embeddings,
            min_retained=2,
            max_pruned=1,
        )


def test_validation_rejects_tampered_contract_index(tmp_path: Path) -> None:
    data_root, source_embeddings, contract_embeddings, _ = _prepare(tmp_path)
    np.save(
        contract_embeddings / "clip_embeddings.npy",
        np.ones((2, 4), dtype=np.float32),
    )

    with pytest.raises(MaterialsContractError, match="hashes"):
        load_materials_authority(
            data_root=data_root,
            source_embeddings=source_embeddings,
            contract_embeddings=contract_embeddings,
            min_retained=2,
            max_pruned=1,
        )


def test_validation_rejects_changed_asset_inventory(tmp_path: Path) -> None:
    data_root, source_embeddings, contract_embeddings, _ = _prepare(tmp_path)
    (data_root / "GoodA" / "GoodA_NormalGL.jpg").write_bytes(
        b"changed-to-a-different-size"
    )

    with pytest.raises(MaterialsContractError, match="inventory"):
        load_materials_authority(
            data_root=data_root,
            source_embeddings=source_embeddings,
            contract_embeddings=contract_embeddings,
            min_retained=2,
            max_pruned=1,
        )


def test_prepare_refuses_to_write_inside_shared_materials_tree(tmp_path: Path) -> None:
    data_root, source_embeddings, _ = _source_fixture(tmp_path)

    with pytest.raises(MaterialsContractError, match="protected shared path"):
        prepare_contract(
            data_root=data_root,
            source_embeddings=source_embeddings,
            contract_embeddings=data_root / "contract",
            min_retained=2,
            max_pruned=1,
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "remote_jobs/run_full_quality_school_sqz.sh",
        "remote_jobs/TEMPLATE_full_quality_room_worker_generated_sam3d.sbatch",
        "remote_jobs/TEMPLATE_full_quality_scene_generate.sbatch",
        "remote_jobs/TEMPLATE_validate_articulated_router.sbatch",
        "remote_jobs/TEMPLATE_gated_assemble_export_render.sbatch",
    ],
)
def test_full_quality_entrypoints_validate_isolated_materials_contract(
    relative_path: str,
) -> None:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert "MATERIALS_SOURCE_EMBEDDINGS" in text
    assert "materials_full_quality_contract/embeddings" in text
    assert "scripts/materials_contract.py validate" in text
    assert "--max-pruned 15" in text


def test_room_worker_template_passes_contract_paths_to_worker() -> None:
    text = (
        REPO_ROOT
        / "remote_jobs"
        / "TEMPLATE_full_quality_room_worker_generated_sam3d.sbatch"
    ).read_text(encoding="utf-8")
    assert text.count('--materials-data "$MATERIALS_DATA"') == 2
    assert (
        text.count('--materials-source-embeddings "$MATERIALS_SOURCE_EMBEDDINGS"') == 2
    )
    assert text.count('--materials-embeddings "$MATERIALS_EMBEDDINGS"') == 2
