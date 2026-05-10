from __future__ import annotations

import copy

import pytest

from scripts.assemble_final_house_and_render import (
    _require_school_articulated_roles,
    _verify_articulated_role_survival,
)
from scripts.school_room_contract import (
    ARTICULATED_ROLE_RULES,
    collect_required_articulated_roles,
)


ROLE_OBJECTS = {
    "library": (
        "library_bookcase",
        "openable library bookcase cabinet with hinged glass doors",
        "artiverse",
        "artiverse/bookcase/fpModel/100",
    ),
    "storage_room": (
        "storage_utility",
        "openable school supply utility cabinet with two hinged doors",
        "artvip",
        "large_furniture/model_storage_utility",
    ),
    "classroom_01": (
        "teacher_filing",
        "teacher filing cabinet with operable drawers",
        "artvip",
        "large_furniture/model_teacher_filing",
    ),
}


def _role_object(
    object_id: str,
    description: str,
    source: str,
    articulated_id: str,
    *,
    asset_source: str = "articulated",
) -> dict:
    return {
        "object_id": object_id,
        "name": description,
        "description": description,
        "object_type": "FURNITURE",
        "sdf_path": f"generated_assets/sdf/{object_id}/asset.sdf",
        "metadata": {
            "asset_source": asset_source,
            "is_articulated": True,
            "articulated_source": source,
            "articulated_id": articulated_id,
        },
    }


def _states() -> dict[str, dict]:
    return {
        room_id: {
            "objects": {
                object_id: _role_object(
                    object_id, description, source, articulated_id
                )
            }
        }
        for room_id, (
            object_id,
            description,
            source,
            articulated_id,
        ) in ROLE_OBJECTS.items()
    }


def _valid_artiverse_record() -> dict:
    object_id, _description, _source, articulated_id = ROLE_OBJECTS["library"]
    return {
        "room_id": "library",
        "object_id": object_id,
        "articulated_id": articulated_id,
    }


def _collect(states: dict[str, dict]) -> dict:
    return collect_required_articulated_roles(
        states, require_runtime_provenance=True
    )


def test_assembly_contract_requires_all_three_roles_even_if_status_is_forged() -> None:
    result = _collect(_states())
    assert result["status"] == "pass"
    assert set(result["roles"]) == set(ARTICULATED_ROLE_RULES)

    forged = copy.deepcopy(result)
    forged["roles"].pop("school_supply_two_door_utility_cabinet")
    forged["status"] = "pass"
    forged["missing_roles"] = []

    with pytest.raises(RuntimeError, match="missing required roles"):
        _require_school_articulated_roles(
            forged, [_valid_artiverse_record()], "passing final room states"
        )


def test_assembly_rejects_semantic_role_with_forged_runtime_provenance() -> None:
    states = _states()
    states["classroom_01"]["objects"]["teacher_filing"]["metadata"][
        "asset_source"
    ] = "generated"

    result = _collect(states)

    assert result["status"] == "fail"
    assert "teacher_filing_drawer_cabinet" in result["missing_roles"]
    assert result["invalid_role_candidates"]
    with pytest.raises(RuntimeError, match="articulated-role contract failed"):
        _require_school_articulated_roles(
            result, [_valid_artiverse_record()], "passing final room states"
        )


def test_assembly_requires_at_least_one_authority_validated_artiverse_role() -> None:
    result = _collect(_states())

    with pytest.raises(RuntimeError, match="lacks validated provenance"):
        _require_school_articulated_roles(
            result, [], "passing final room states"
        )

    unrelated = dict(_valid_artiverse_record())
    unrelated["object_id"] = "unrelated_artiverse_asset"
    with pytest.raises(RuntimeError, match="lacks validated provenance"):
        _require_school_articulated_roles(
            result, [unrelated], "passing final room states"
        )

    _require_school_articulated_roles(
        result, [_valid_artiverse_record()], "passing final room states"
    )


def test_required_role_identity_and_source_must_survive_assembly() -> None:
    placed = _collect(_states())
    final = copy.deepcopy(placed)
    _verify_articulated_role_survival(placed, final)

    missing = copy.deepcopy(final)
    missing["roles"]["teacher_filing_drawer_cabinet"] = []
    with pytest.raises(RuntimeError, match="lost required articulated role"):
        _verify_articulated_role_survival(placed, missing)

    source_swapped = copy.deepcopy(final)
    source_swapped["roles"]["teacher_filing_drawer_cabinet"][0][
        "articulated_source"
    ] = "artiverse"
    with pytest.raises(RuntimeError, match="lost required articulated role"):
        _verify_articulated_role_survival(placed, source_swapped)
