#!/usr/bin/env python3
"""Focused regression for the generated-SDF inertia serialization margin."""

from __future__ import annotations

import importlib.util
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def load_module(path: Path):
    specification = importlib.util.spec_from_file_location("tested_inertia_utils", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load inertia-utils module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def matrix_from_sdf(path: Path) -> np.ndarray:
    inertia = ET.parse(path).getroot().find(".//inertia")
    if inertia is None:
        raise RuntimeError("test SDF lost its inertia element")
    values = {name: float(inertia.findtext(name, "nan")) for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    return np.array(
        [
            [values["ixx"], values["ixy"], values["ixz"]],
            [values["ixy"], values["iyy"], values["iyz"]],
            [values["ixz"], values["iyz"], values["izz"]],
        ]
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, type=Path)
    args = parser.parse_args()
    module = load_module(args.module.resolve(strict=True))
    # Exact tensor Drake rejected in Classroom 6 after the old six-digit
    # serializer rounded away the repair epsilon.
    rejected = np.array(
        [
            [0.002714817, -0.0005679015, -0.0003772943],
            [-0.0005679015, 0.003219098, -0.00001882076],
            [-0.0003772943, -0.00001882076, 0.00584235],
        ]
    )
    original_principal = np.linalg.eigvalsh(rejected)
    if original_principal[0] + original_principal[1] >= original_principal[2]:
        raise AssertionError("regression tensor is no longer invalid")

    with tempfile.TemporaryDirectory() as temporary:
        sdf = Path(temporary) / "regression.sdf"
        sdf.write_text(
            """<?xml version='1.0'?><sdf version='1.7'><model name='x'><link name='base_link'><inertial><mass>1</mass><inertia><ixx>0.002714817</ixx><iyy>0.003219098</iyy><izz>0.00584235</izz><ixy>-0.0005679015</ixy><ixz>-0.0003772943</ixz><iyz>-0.00001882076</iyz></inertia></inertial></link></model></sdf>""",
            encoding="utf-8",
        )
        if not module.fix_sdf_file_inertia(sdf):
            raise AssertionError("invalid inertia was not repaired")
        serialized = matrix_from_sdf(sdf)
        principal = np.linalg.eigvalsh(serialized)
        margin = float(principal[0] + principal[1] - principal[2])
        if principal[0] <= 0.0 or margin <= float(principal[2]) * 5e-8:
            raise AssertionError(
                f"serialized inertia lacks a durable physical margin: {principal}"
            )
        if module.fix_sdf_file_inertia(sdf):
            raise AssertionError("inertia repair is not idempotent")
    print("INERTIA_SERIALIZATION_MARGIN_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
