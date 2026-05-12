from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from omegaconf import OmegaConf

from scenesmith.agent_utils.asset_router.dataclasses import AssetItem
from scenesmith.agent_utils.asset_router.router import AssetRouter
from scenesmith.agent_utils.room import ObjectType
from scenesmith.experiments.base_experiment import BaseExperiment
from scenesmith.experiments.indoor_scene_generation import (
    IndoorSceneGenerationExperiment,
)
from scenesmith.manipuland_agents.stateful_manipuland_agent import (
    StatefulManipulandAgent,
)


class _CaptureAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _agent_config(
    source: str,
    data_path: Path,
    preprocessed_path: Path,
    top_k: int,
):
    return {
        "asset_manager": {
            "general_asset_source": source,
            "objaverse": {
                "data_path": str(data_path),
                "preprocessed_path": str(preprocessed_path),
                "use_top_k": top_k,
            },
        }
    }


def test_build_manipuland_agent_propagates_objathor_endpoint() -> None:
    cfg = OmegaConf.create(
        {
            "manipuland_agent": {"_name": "capture"},
            "experiment": {
                "geometry_generation_server": {"host": "geometry", "port": 7000},
                "hssd_retrieval_server": {"host": "hssd", "port": 7001},
                "objaverse_retrieval_server": {
                    "host": "objectthor.internal",
                    "port": 7109,
                },
                "articulated_retrieval_server": {
                    "host": "articulated",
                    "port": 7002,
                },
                "materials_retrieval_server": {"host": "materials", "port": 7008},
                "num_workers": 1,
            },
        }
    )

    agent = BaseExperiment.build_manipuland_agent(
        cfg_dict=cfg,
        compatible_agents={"capture": _CaptureAgent},
        logger=MagicMock(),
    )

    assert agent.kwargs["objaverse_server_host"] == "objectthor.internal"
    assert agent.kwargs["objaverse_server_port"] == 7109


def test_stateful_manipuland_agent_forwards_objathor_endpoint() -> None:
    agent = object.__new__(StatefulManipulandAgent)
    cfg = MagicMock()
    logger = MagicMock()

    with (
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent."
            "BaseStatefulAgent.__init__",
            return_value=None,
        ) as stateful_init,
        patch(
            "scenesmith.manipuland_agents.stateful_manipuland_agent."
            "BaseManipulandAgent.__init__",
            return_value=None,
        ) as manipuland_init,
    ):
        StatefulManipulandAgent.__init__(
            agent,
            cfg=cfg,
            logger=logger,
            objaverse_server_host="objectthor.internal",
            objaverse_server_port=7109,
        )

    stateful_init.assert_called_once()
    assert "objaverse_server_host" not in stateful_init.call_args.kwargs
    assert "objaverse_server_port" not in stateful_init.call_args.kwargs
    assert manipuland_init.call_args.kwargs["objaverse_server_host"] == (
        "objectthor.internal"
    )
    assert manipuland_init.call_args.kwargs["objaverse_server_port"] == 7109


def test_objathor_server_uses_enabled_agent_paths_and_max_top_k(tmp_path: Path) -> None:
    data_path = tmp_path / "objathor-assets"
    preprocessed_path = data_path / "preprocessed"
    cfg = OmegaConf.create(
        {
            "experiment": {
                "objaverse_retrieval_server": {
                    "host": "127.0.0.1",
                    "port": 7109,
                }
            },
            "furniture_agent": _agent_config(
                "generated", tmp_path / "unused", tmp_path / "unused-index", 1
            ),
            "manipuland_agent": _agent_config(
                "objaverse", data_path, preprocessed_path, 10
            ),
            "wall_agent": _agent_config(
                "objaverse", data_path, preprocessed_path, 6
            ),
            "ceiling_agent": _agent_config(
                "generated", tmp_path / "unused", tmp_path / "unused-index", 1
            ),
        }
    )
    experiment = object.__new__(IndoorSceneGenerationExperiment)
    experiment.cfg = cfg
    experiment.objaverse_server = None

    with (
        patch(
            "scenesmith.experiments.indoor_scene_generation."
            "ObjaverseRetrievalServer"
        ) as server_class,
        patch(
            "scenesmith.experiments.indoor_scene_generation."
            "_get_retrieval_gpu_device",
            return_value="cpu",
        ),
    ):
        experiment._start_objaverse_server()

    server_class.assert_called_once_with(
        host="127.0.0.1",
        port=7109,
        preload_retriever=True,
        objaverse_data_path=str(data_path.resolve()),
        objaverse_preprocessed_path=str(preprocessed_path.resolve()),
        objaverse_top_k=10,
        clip_device="cpu",
    )
    assert experiment.objaverse_server is server_class.return_value
    assert server_class.return_value.mock_calls[-2:] == [
        call.start(),
        call.wait_until_ready(timeout_s=60.0),
    ]


def test_objathor_server_rejects_enabled_agent_path_mismatch(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "experiment": {
                "objaverse_retrieval_server": {
                    "host": "127.0.0.1",
                    "port": 7109,
                }
            },
            "furniture_agent": _agent_config(
                "objaverse", tmp_path / "furniture", tmp_path / "furniture-index", 1
            ),
            "manipuland_agent": _agent_config(
                "objaverse", tmp_path / "manipuland", tmp_path / "manipuland-index", 10
            ),
            "wall_agent": _agent_config(
                "generated", tmp_path / "unused", tmp_path / "unused-index", 1
            ),
            "ceiling_agent": _agent_config(
                "generated", tmp_path / "unused", tmp_path / "unused-index", 1
            ),
        }
    )
    experiment = object.__new__(IndoorSceneGenerationExperiment)
    experiment.cfg = cfg
    experiment.objaverse_server = None

    with pytest.raises(RuntimeError, match="disagree on data paths"):
        experiment._start_objaverse_server()


def _school_supply() -> AssetItem:
    return AssetItem(
        description="blue ballpoint pen",
        short_name="ballpoint_pen",
        dimensions=[0.14, 0.012, 0.012],
        object_type=ObjectType.MANIPULAND,
        strategies=["generated"],
    )


def _try_generated(router: AssetRouter, tmp_path: Path, *, max_retries: int = 0):
    return router._try_generated_strategy(
        item=_school_supply(),
        max_retries=max_retries,
        geometry_client=None,
        hssd_client=MagicMock(name="hssd_client"),
        objaverse_client=MagicMock(name="objaverse_client"),
        image_generator=None,
        images_dir=None,
        geometry_dir=tmp_path / "geometry",
        debug_dir=tmp_path / "debug",
    )


def test_objathor_school_supply_uses_retrieval_not_primitive(tmp_path: Path) -> None:
    router = object.__new__(AssetRouter)
    router.cfg = OmegaConf.create(
        {"asset_manager": {"general_asset_source": "objaverse"}}
    )
    retrieved = MagicMock(name="retrieved")
    acquired = MagicMock(name="acquired")
    router._fetch_objaverse_candidates = MagicMock(return_value=[retrieved])
    router._acquire_generated_candidate = MagicMock(return_value=acquired)
    router._create_primitive_manipuland_fallback_geometry = MagicMock()

    assert _try_generated(router, tmp_path) is acquired
    router._fetch_objaverse_candidates.assert_called_once()
    router._create_primitive_manipuland_fallback_geometry.assert_not_called()


def test_objathor_validation_exhaustion_does_not_use_hssd_primitive(
    tmp_path: Path,
) -> None:
    router = object.__new__(AssetRouter)
    router.cfg = OmegaConf.create(
        {
            "asset_manager": {
                "general_asset_source": "objaverse",
                "objaverse": {"use_lenient_validation": False},
            }
        }
    )
    retrieved = MagicMock(name="retrieved")
    acquired = MagicMock(name="acquired", geometry_path=tmp_path / "candidate.glb")
    router._fetch_objaverse_candidates = MagicMock(return_value=[retrieved])
    router._acquire_generated_candidate = MagicMock(return_value=acquired)
    router.validate_asset = MagicMock(
        return_value=MagicMock(
            is_acceptable=False,
            reason="wrong object",
            suggestions="try another candidate",
        )
    )
    router._create_primitive_manipuland_fallback_geometry = MagicMock()

    assert _try_generated(router, tmp_path, max_retries=1) is None
    router._fetch_objaverse_candidates.assert_called_once()
    router.validate_asset.assert_called_once()
    router._create_primitive_manipuland_fallback_geometry.assert_not_called()


def test_hssd_school_supply_retains_primitive_fallback(tmp_path: Path) -> None:
    router = object.__new__(AssetRouter)
    router.cfg = OmegaConf.create({"asset_manager": {"general_asset_source": "hssd"}})
    primitive = MagicMock(name="primitive")
    router._create_primitive_manipuland_fallback_geometry = MagicMock(
        return_value=primitive
    )

    assert _try_generated(router, tmp_path) is primitive
    router._create_primitive_manipuland_fallback_geometry.assert_called_once()
