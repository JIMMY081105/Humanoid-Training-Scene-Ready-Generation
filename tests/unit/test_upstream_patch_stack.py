from __future__ import annotations

import subprocess

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_DIR = REPO_ROOT / "upstream-patches"
ORDER_FILE = PATCH_DIR / "APPLY_ORDER.txt"


def _ordered_patches() -> list[str]:
    return [
        line.strip()
        for line in ORDER_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_ordered_patch_stack_is_complete_unique_and_parseable() -> None:
    ordered = _ordered_patches()

    assert len(ordered) == 36
    assert len(ordered) == len(set(ordered))
    assert ordered[-1] == "380036e-inprocess-server-shutdown.patch"
    assert "380036e-objathor-offline-payload.patch" in ordered
    assert "380036e-school-double-door-leaves.patch" in ordered
    assert "380036e-navigation-common-zones.patch" in ordered
    assert "380036e-traceable-model-timeout.patch" in ordered
    assert "380036e-objathor-manipuland-routing.patch" in ordered
    assert "380036e-thread-safe-projection-timeout.patch" in ordered
    assert "380036e-action-log-concurrency.patch" in ordered
    assert "380036e-artiverse-support-surface-extraction.patch" in ordered
    assert "380036e-manipuland-furniture-checkpoints.patch" in ordered
    assert "380036e-manipuland-furniture-scope.patch" in ordered
    assert "380036e-inprocess-server-shutdown.patch" in ordered
    assert {path.name for path in PATCH_DIR.glob("*.patch")} == set(ordered)
    for name in ordered:
        result = subprocess.run(
            ["git", "apply", "--numstat", str(PATCH_DIR / name)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"


def test_sam3d_patch_blocks_network_capable_torch_hub_path() -> None:
    patch = (PATCH_DIR / "380036e-sam3d-offline-dino-cache.patch").read_text(
        encoding="utf-8"
    )

    assert "_install_offline_torch_hub_guard" in patch
    assert 'str(source_dir), model, *args, source="local"' in patch
    assert "Offline SAM3D blocked unexpected torch.hub load" in patch
    assert "torch.hub.load = hub_guard[0]" in patch
    assert "exactly one cached weight" in patch
    assert "_release_transient_cuda_memory" in patch
    assert "torch.cuda.empty_cache()" in patch
    assert "output = None" in patch
    assert "with_texture_baking=True" in patch
    assert "with_layout_postprocess=True" in patch


def test_simulator_export_patch_makes_requested_usd_failures_fatal() -> None:
    patch = (PATCH_DIR / "380036e-simulator-export-fail-closed.patch").read_text(
        encoding="utf-8"
    )

    assert "Requested USD export dependencies are unavailable" in patch
    assert "Requested USD export failed" in patch
    assert "USD converter did not create its requested root layer" in patch
    assert "OpenUSD could not load converted root layer" in patch


def test_navigation_common_zone_patch_persists_only_tool_authored_metadata() -> None:
    patch = (PATCH_DIR / "380036e-navigation-common-zones.patch").read_text(
        encoding="utf-8"
    )

    assert "navigation_common_zones: list[dict[str, Any]]" in patch
    assert '"navigation_common_zones": copy.deepcopy' in patch
    assert "raw_navigation_common_zones" in patch
    assert "set_navigation_common_zones" in patch
    assert "backing_door_id" in patch
    assert "backing door has no generated wall opening" in patch
    assert "self.layout.navigation_common_zones.clear()" in patch


def test_traceable_timeout_patch_uses_json_safe_request_budget() -> None:
    patch = (PATCH_DIR / "380036e-traceable-model-timeout.patch").read_text(
        encoding="utf-8"
    )

    assert "class _TraceSafeModelSettings(ModelSettings)" in patch
    assert "traceable_settings = replace(" in patch
    assert "return _TraceSafeModelSettings(**kwargs)" in patch
    assert "model_config_for_trace" in patch


def test_objathor_manipuland_patch_binds_runtime_endpoint_and_source_policy() -> None:
    patch = (PATCH_DIR / "380036e-objathor-manipuland-routing.patch").read_text(
        encoding="utf-8"
    )

    assert "objaverse_server_host=objaverse_server_config.host" in patch
    assert "objaverse_server_port=objaverse_server_config.port" in patch
    assert "Enabled Objaverse/ObjectThor agents disagree on data paths" in patch
    assert "objaverse_top_k = max" in patch
    assert 'asset_source == "hssd"' in patch


def test_projection_timeout_patch_is_worker_thread_safe() -> None:
    patch = (PATCH_DIR / "380036e-thread-safe-projection-timeout.patch").read_text(
        encoding="utf-8"
    )

    assert "threading.current_thread() is threading.main_thread()" in patch
    assert "return solver.Solve(prog, None, options)" in patch
    assert '"Time limit", time_limit_s' in patch
    assert '"max_cpu_time", time_limit_s' in patch
    assert "test_worker_thread_runs_solver_without_installing_signal_handler" in patch


def test_action_log_patch_serializes_and_atomically_publishes() -> None:
    patch = (PATCH_DIR / "380036e-action-log-concurrency.patch").read_text(
        encoding="utf-8"
    )

    assert "_exclusive_action_log_lock" in patch
    assert "fcntl.flock" in patch
    assert "msvcrt.locking" in patch
    assert "os.fsync(temporary_file.fileno())" in patch
    assert "_fsync_parent_directory(log_path.parent)" in patch
    assert "os.replace(temporary_path, log_path)" in patch
    assert "test_concurrent_processes_preserve_every_entry" in patch
    assert "test_failed_atomic_replace_preserves_previous_document" in patch


def test_artiverse_support_patch_uses_authoritative_sdf_resources() -> None:
    patch = (
        PATCH_DIR / "380036e-artiverse-support-surface-extraction.patch"
    ).read_text(encoding="utf-8")

    assert "_parse_artiverse_link_meshes" in patch
    assert "_validated_artiverse_manifest" in patch
    assert "_load_artiverse_collision_union" in patch
    assert "_load_artiverse_collision_unions" in patch
    assert "_measure_artiverse_same_link_collision_support" in patch
    assert "surface_offset_m + 0.02" in patch
    assert "coverage=100%%" in patch
    assert "preserved_semantics_sha256" in patch
    assert "publisher GLB does not match manifest" in patch
    assert "Failed required Artiverse link extraction" in patch
    assert "test_artiverse_uses_sdf_visual_and_collision_inventory" in patch
    assert (
        "test_artiverse_same_link_collision_support_requires_full_nearby_coverage"
        in patch
    )
    assert "test_artiverse_collision_support_is_same_link_not_parent_link" in patch
    assert "test_normalized_artiverse_sdf_cannot_be_bypassed_by_legacy_mesh" in patch


def test_manipuland_checkpoint_patch_binds_legacy_and_durable_states() -> None:
    patch = (
        PATCH_DIR / "380036e-manipuland-furniture-checkpoints.patch"
    ).read_text(encoding="utf-8")

    assert "_restore_legacy_accepted_render" in patch
    assert "_restore_completed_furniture_checkpoint" in patch
    assert "_publish_furniture_checkpoint" in patch
    assert "_publish_furniture_plan" in patch
    assert "_verify_legacy_checkpoint_source" in patch
    assert "_referenced_room_asset_records" in patch
    assert "_validate_legacy_recovery_manifest_document" in patch
    assert "_LEGACY_RECOVERY_SCHEMA = 3" in patch
    assert "Checkpoint restored object delta mismatch" in patch
    assert "unsupported direct asset suffix" in patch
    assert "Referenced SDF has unsupported URI elements" in patch
    assert "added_manipuland_ids" in patch
    assert "referenced_asset_object_ids" in patch
    assert "referenced_asset_files" in patch
    assert "SCENESMITH_LEGACY_MANIPULAND_RECOVERY_MANIFEST" in patch
    assert "SCENESMITH_LEGACY_MANIPULAND_RECOVERY_SHA256" in patch
    assert "input_scene_content_hash" in patch
    assert "Legacy manifest input semantics are not the live base scene" in patch
    assert "_ACCEPTED_IMAGE_NAMES" in patch
    assert "_SCORE_CATEGORIES" in patch
    assert "test_published_checkpoint_restores_only_exact_hash_chain" in patch
    assert "test_legacy_checkpoint_revalidates_added_asset_hashes" in patch
    assert (
        "test_legacy_source_is_revalidated_between_restore_and_checkpoint_publish"
        in patch
    )
    assert "test_required_legacy_recovery_must_be_consumed_before_completion" in patch
    assert "test_referenced_glb_is_rejected_without_an_exact_dependency_audit" in patch
    assert (
        "test_resigned_checkpoint_cannot_substitute_manifest_bound_legacy_source"
        in patch
    )
    assert "test_extra_object_after_legacy_restore_cannot_be_published" in patch
    assert "test_resigned_checkpoint_state_requires_exact_live_input_delta" in patch
    assert (
        "test_target_transitive_asset_changed_after_authorization_cannot_be_promoted"
        in patch
    )
    assert "test_sdf_dae_and_unconsumed_uri_nodes_are_rejected" in patch
    assert "test_legacy_input_accepts_hash_preserving_quaternion_roundtrip_drift" in patch


def test_manipuland_scope_patch_is_fail_closed_per_furniture_and_restart_safe() -> None:
    patch = (
        PATCH_DIR / "380036e-manipuland-furniture-scope.patch"
    ).read_text(encoding="utf-8")

    assert "manipuland_scope_ids" in patch
    assert "protected_object_ids" in patch
    assert "propagate_to_identical=False" in patch
    assert "AssetRegistry.empty_snapshot" in patch
    assert "attestation_history" in patch
    assert "asset_registry_snapshot" in patch
    assert "_serialized_scene_roundtrip_state" in patch
    assert "_run_manipuland_agent_with_cleanup" in patch
    assert "reconcile_registry_with_scene" in patch
    assert "_scene_asset_path" in patch
    assert 'getattr(self, "agent_type", None)' in patch
    assert "manager_is_manipuland != request_is_manipuland" in patch
    assert "register_many_immutable" in patch
    assert "test_articulated_manipuland_registers_one_exclusive_sdf_namespace" in patch
    assert "test_asset_manager_records_visual_normalization_in_artiverse_metadata" in patch
    assert "test_articulated_manager_request_scope_mismatch_has_no_filesystem_effect" in patch
    assert "test_schema3_checkpoint_pins_registry_head" in patch
    assert "test_completed_legacy_checkpoint_normalizes_only_pinned_quaternion_spelling" in patch
    assert "test_manipuland_stage_always_cleans_up_helper_servers" in patch
    assert "test_generic_registry_preserves_session_only_path_compatibility" in patch
