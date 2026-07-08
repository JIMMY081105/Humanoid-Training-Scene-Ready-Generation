# Codex SceneSmith Full-Quality Pipeline

This file is the handoff contract for future SceneSmith runs. It exists to prevent the previous failure mode: continuing a fast HSSD-only run while the real target was a best-quality scene with SAM3D/generated assets, ArtVIP/Artiverse routes, and a room self-exam gate before final export.

## Non-Negotiable Output Rules

- Do not use an `hssd_only_fast` run as the final-quality run unless the user explicitly approves it.
- Do not set `general_asset_source=hssd` for a full-quality scene unless the user explicitly asks for HSSD-only speed.
- Keep SAM3D available by using `general_asset_source=generated` and `backend=sam3d` for furniture, wall, ceiling, and manipuland agents when the target is quality.
- Use Objaverse/ObjectThor retrieval for small manipuland objects in full-quality runs to avoid primitive-shape fallbacks.
- Keep articulated retrieval available. In this repo that means ArtVIP through `asset_manager.articulated.sources.artvip.enabled=true`; Artiverse is the separate `artiverse_articulated` route when that local dataset/config is available.
- Do not assemble `combined_house`, final Drake exports, Isaac/USD exports, or final renders until every required room has passed a room-level quality gate.
- A render-only review folder is not a quality gate. A gate must produce pass/fail records and must block export on failure.
- Cap generated collision decomposition at 32 convex hulls per object per agent. A simulator export that cannot load is not simulation-ready.
- ParaCloud `/data/run01/scvj260/scenesmith` is the canonical execution checkout for production runs. Local edits are overlay-uploaded to ParaCloud; never sync with `--delete` unless explicitly approved.

## Pipeline Overview

1. Prompt and run setup
2. Environment/GPU preflight
3. Compute-node API proxy verification
4. Full-quality asset policy check
5. Articulated router validation
6. Floor plan generation
7. Room-level generation
8. Room render capture
9. Room self-exam gate
10. Room repair/regeneration loop
11. Final assembly
12. Drake/SceneEval/collision export
13. Isaac/USD/MuJoCo export if requested
14. Outlook renders and local transfer
15. Final validation report

## 1. Prompt And Run Setup

Function: define what the scene should be and create a clean run directory.

Required inputs:

- Prompt CSV, for example `inputs/my_scene.csv`.
- Run name that describes the real asset policy, for example `full_quality_school_sam3d_artvip`, not `hssd_only_fast`.

Guardrail:

- If resuming another Codex handoff, inspect the run name and Hydra overrides first. If it says `hssd_only`, `fast`, or sets all agents to HSSD, stop and ask before continuing.

## 2. Environment/GPU Preflight

Function: verify that the job is on a GPU node, proxy/API access works, checkpoints exist, and the selected asset sources can load.

Check these before launching a long run:

```bash
nvidia-smi
test -f external/checkpoints/sam3.pt
test -f external/checkpoints/pipeline.yaml
test -d data/artvip_sdf || true
test -d data/artiverse || true
curl --max-time 60 --proxy "$HTTP_PROXY" -s -o /dev/null -w "%{http_code}" https://api.openai.com/v1/models
```

Expected OpenAI probe codes: `200`, `401`, or `403`. Other codes mean proxy/network is not ready.

## 3. Compute-Node API Proxy Verification

Function: ensure GPU jobs can reach OpenAI for the whole run. A local laptop SSH tunnel bound only to `127.0.0.1` is not enough: compute nodes cannot reach it, and it dies when the laptop disconnects.

Required policy:

- GPU jobs must use a proxy reachable from compute nodes, for example the old cluster-facing login-node endpoint `http://ln08:18092`.
- The proxy/tunnel must be bound on an interface visible to compute nodes, not only login-node localhost.
- The proxy must be managed in a persistent login-node session or service (`tmux`, `screen`, `autossh`, or cluster-managed proxy), not a fragile foreground tunnel from the laptop.
- If the laptop disconnects, the proxy must keep running. Otherwise multi-day runs will stall at the next OpenAI call.
- Every SLURM script must probe OpenAI from inside the GPU allocation and fail fast if unreachable.

Preflight from a GPU allocation:

```bash
curl --max-time 60 --proxy "http://ln08:18092" \
  -s -o /dev/null -w "%{http_code}" \
  https://api.openai.com/v1/models
```

Do not launch scene generation until this works from the compute node that will run the job.

## 4. Full-Quality Asset Policy

Function: make the asset router use generated SAM3D assets where appropriate and keep richer articulated routes available.

Full-quality overrides:

```bash
furniture_agent.asset_manager.general_asset_source=generated
wall_agent.asset_manager.general_asset_source=generated
ceiling_agent.asset_manager.general_asset_source=generated
manipuland_agent.asset_manager.general_asset_source=objaverse
furniture_agent.asset_manager.backend=sam3d
wall_agent.asset_manager.backend=sam3d
ceiling_agent.asset_manager.backend=sam3d
manipuland_agent.asset_manager.backend=sam3d
manipuland_agent.asset_manager.objaverse.use_top_k=10
manipuland_agent.asset_manager.objaverse.use_lenient_validation=true
furniture_agent.asset_manager.router.strategies.generated.enabled=true
wall_agent.asset_manager.router.strategies.generated.enabled=true
ceiling_agent.asset_manager.router.strategies.generated.enabled=true
manipuland_agent.asset_manager.router.strategies.generated.enabled=true
furniture_agent.asset_manager.articulated.sources.artvip.enabled=true
wall_agent.asset_manager.articulated.sources.artvip.enabled=true
ceiling_agent.asset_manager.articulated.sources.artvip.enabled=true
manipuland_agent.asset_manager.articulated.sources.artvip.enabled=true
furniture_agent.collision_geometry.coacd.max_convex_hull=32
wall_agent.collision_geometry.coacd.max_convex_hull=32
ceiling_agent.collision_geometry.coacd.max_convex_hull=32
manipuland_agent.collision_geometry.coacd.max_convex_hull=32
furniture_agent.collision_geometry.vhacd.max_convex_hulls=32
wall_agent.collision_geometry.vhacd.max_convex_hulls=32
ceiling_agent.collision_geometry.vhacd.max_convex_hulls=32
manipuland_agent.collision_geometry.vhacd.max_convex_hulls=32
```

If Artiverse is installed and supported in the active branch, also enable:

```bash
furniture_agent.asset_manager.artiverse_articulated.enabled=true
furniture_agent.asset_manager.artiverse_articulated.data_path=data/artiverse
furniture_agent.asset_manager.router.strategies.artiverse_articulated.enabled=true
```

Important:

- `backend=sam3d` alone is not enough. If `general_asset_source=hssd`, SAM3D is bypassed.
- Before launching room workers, grep `scripts/run_single_room_worker.py` for any forced `general_asset_source` override and run the worker with `--config-only`; the resolved Hydra config printed in the log must show furniture/wall/ceiling as `generated` and manipulands as `objaverse` for `--asset-pipeline generated_sam3d`.

Config-only room-worker check:

```bash
.venv/bin/python scripts/run_single_room_worker.py \
  --repo-dir /data/run01/scvj260/scenesmith \
  --run-dir "outputs/<date>/<run_name>" \
  --csv <prompt_csv> \
  --run-name <run_name>_room_worker_config_check \
  --room-id <room_id> \
  --start-stage furniture \
  --stop-stage manipuland \
  --asset-pipeline generated_sam3d \
  --port-offset <unique_offset> \
  --render-gpu-id 0 \
  --config-only
```

Expected resolved policy:

```json
{
  "furniture_agent": {"general_asset_source": "generated", "backend": "sam3d"},
  "wall_agent": {"general_asset_source": "generated", "backend": "sam3d"},
  "ceiling_agent": {"general_asset_source": "generated", "backend": "sam3d"},
  "manipuland_agent": {"general_asset_source": "objaverse"}
}
```

## 5. Articulated Router Validation

Function: prove that enabling ArtVIP/Artiverse-style paths changes actual router behavior before spending GPU/API budget on a full run.

Required command before a full-quality run:

```bash
.venv/bin/python scripts/validate_articulated_router.py \
  --output outputs/<date>/<run_name>/articulated_router_validation.json \
  --vlm-backend openai \
  --top-k 3
```

ParaCloud note: run this under SLURM, not on a login node. Use:

```bash
sbatch remote_jobs/TEMPLATE_validate_articulated_router.sbatch
```

This script fails unless:

- AssetRouter selects the `articulated` strategy for openable-furniture prompts.
- The articulated retrieval index returns concrete SDF candidates with existing paths.

If this fails:

- Do not launch the full run.
- Inspect/fix the asset router analysis prompt or strategy parsing.
- Do not assume config flags are enough.

## 6. Floor Plan Generation

Function: create the building layout, rooms, doors, wall geometry, and floor-plan assets.

Typical command:

```bash
.venv/bin/python main.py \
  +name=<run_name> \
  experiment.csv_path=<prompt_csv> \
  experiment.num_workers=1 \
  floor_plan_agent.mode=house \
  codex.enabled=true \
  codex.cwd=/data/run01/scvj260/scenesmith \
  hydra.run.dir="outputs/<date>/<run_name>"
```

Output:

- `scene_000/house_layout.json`
- `scene_000/floor_plans/`
- `scene_000/room_geometry/`
- `scene_000/room_<room_id>/`

## 7. Room-Level Generation

Function: generate furniture, wall objects, ceiling objects, and manipulands for each room.

Preferred full-quality command for a single room worker:

```bash
.venv/bin/python scripts/run_single_room_worker.py \
  --repo-dir /data/run01/scvj260/scenesmith \
  --run-dir "outputs/<date>/<run_name>" \
  --csv <prompt_csv> \
  --run-name <run_name>_room_worker \
  --room-id <room_id> \
  --start-stage furniture \
  --stop-stage manipuland \
  --asset-pipeline generated_sam3d \
  --port-offset <unique_offset> \
  --render-gpu-id 0
```

Parallel policy:

- Use one GPU per independent room worker.
- Do not give multiple GPUs to one serial room worker.
- Each worker must write only to its own `scene_000/room_<room_id>/` folder.
- Do not run final assembly while room workers are active.
- Give each worker a unique `--port-offset`.

## 8. Room Render Capture

Function: render each generated room so SAGE/Codex/VLM can examine object placement.

Minimum review images:

- Top-down room image.
- At least two side/oblique views for furniture orientation and wall/door blockage.
- Optional collision/debug overlay if available.

Current helper:

```bash
.venv/bin/python scripts/render_room_review_views.py \
  --blend <combined_or_room_blend> \
  --house-state <house_state_or_room_state_json> \
  --output-dir <review_dir>
```

Limitation: this helper renders review images only. It does not judge them.

## 9. Room Self-Exam Gate

Function: block bad rooms before final assembly/export.

Implemented gate:

```bash
.venv/bin/python scripts/room_self_exam.py \
  --scene-dir outputs/<date>/<run_name>/scene_000 \
  --review-dir outputs/<date>/<run_name>/scene_000/review/room_review_renders \
  --output-dir outputs/<date>/<run_name>/scene_000/quality_gates/room_self_exam \
  --max-collision-hulls 32
```

Current implementation:

- Enforces pass/fail JSON files.
- Blocks missing final room states.
- Blocks missing review images.
- Blocks objects outside room-local bounds.
- Penalizes suspicious object density and collision-hull risk.
- Does not replace a true VLM/SAGE visual judge. If a VLM/SAGE judge is available, it must write the same JSON schema so assembly can enforce it.

Required behavior:

- Input: room review images plus structured room state.
- Output: one JSON result per room.
- Pass/fail must be explicit.
- Failed rooms must not be assembled into final output.

Required JSON shape:

```json
{
  "room_id": "classroom_01",
  "status": "pass",
  "scores": {
    "object_relevance": 8,
    "placement_realism": 8,
    "clearance_and_access": 8,
    "collision_risk": 9,
    "prompt_alignment": 8
  },
  "critical_issues": [],
  "repair_instructions": []
}
```

Failure examples:

- Desks/chairs outside room bounds.
- Furniture floating, intersecting walls, or stacked unnaturally.
- Doorways or corridors blocked.
- Objects bunched in one strip while most room is empty.
- Classroom rows facing the wrong direction or unusable.
- Restroom, closet, office, library, or corridor objects nonsensical for the room type.

Pass threshold:

- No critical issues.
- `placement_realism >= 7`.
- `clearance_and_access >= 7`.
- `collision_risk >= 7`.
- Prompt alignment acceptable for the room type.

If any required room fails, go to the repair loop. Do not combine/export.

## 10. Room Repair/Regeneration Loop

Function: fix failed rooms only, preserving good rooms.

Preferred order:

1. If object placement is bad but asset choices are okay, rerun from the relevant placement stage.
2. If asset choices are wrong, rerun from furniture generation for that room.
3. If geometry/floor plan is wrong, stop and reconsider the floor plan rather than patching final export.

Use the same full-quality asset policy:

```bash
--asset-pipeline generated_sam3d
```

After each repair:

```text
rerender room -> rerun SAGE/Codex room self-exam -> only mark pass if JSON passes
```

## 11. Final Assembly

Function: merge all passed room outputs into one `combined_house`.

Command:

```bash
.venv/bin/python scripts/assemble_final_house_and_render.py \
  --repo-dir /data/run01/scvj260/scenesmith \
  --run-dir "outputs/<date>/<run_name>" \
  --csv <prompt_csv> \
  --run-name <run_name>_final_assemble \
  --gate-dir "outputs/<date>/<run_name>/scene_000/quality_gates/room_self_exam" \
  --render
```

Guardrail:

- Before this command, verify that every expected `room_<room_id>` has a passing gate JSON.
- Back up existing `combined_house` before overwriting.
- `scripts/assemble_final_house_and_render.py` refuses to run if any room gate JSON is missing or non-passing, unless `--allow-ungated` is explicitly passed for debugging.

## 12. Drake/SceneEval/Collision Export

Function: create simulator-ready scene files.

Expected outputs:

- `combined_house/house_state.json`
- `combined_house/sceneeval_state.json`
- `combined_house/house.dmd.yaml`
- `combined_house/house.blend`
- `room_geometry/*.sdf`
- collision meshes under generated asset folders

Validation:

```bash
grep -R "package://scene" scene_000/combined_house/house.dmd.yaml | wc -l
python -m pydrake.visualization.model_visualizer scene_000/combined_house/house.dmd.yaml
```

Full-quality acceptance:

- Full-house Drake load must complete on a 2-GPU ParaCloud allocation without OOM.
- If full-house Drake load OOMs, the export is not simulation-ready even if files exist.
- Collision mesh counts must be reported, including max collision files for a single object.

If Drake package assets are exported to a standalone folder, include `floor_plans/`, `room_geometry/`, and all `room_*/generated_assets/` dependencies.

## 13. Isaac/USD/MuJoCo Export

Function: export to additional simulation formats when requested.

MuJoCo/USD script:

```bash
.venv/bin/python scripts/export_scene_to_mujoco.py \
  outputs/<date>/<run_name>/scene_000 \
  -o outputs/<date>/<run_name>/scene_000/mujoco_export \
  --usd
```

Isaac Sim note:

- Isaac Sim consumes USD best. Drake DMD/SDF is not automatically an Isaac-native export.
- If Isaac is required, run the USD path and then run the Isaac compatibility fixer if needed.

## 14. Outlook Renders And Local Transfer

Function: generate user-facing preview images and copy the final package locally.

Expected renders:

- `overview_top.png`
- `overview_isometric.png`
- `overview_front.png`
- per-room review images
- contact sheet

Transfer rule:

- Use hash-verified archives/chunks for large folders.
- Do not report final success until local files are present and at least one render is visually inspected.

## 15. Final Validation Report

Function: make it clear what was actually run.

The final report must include:

- Run path.
- Exact asset policy used.
- Whether SAM3D was enabled and actually reachable.
- Whether HSSD-only was used anywhere.
- Whether ArtVIP/Artiverse routes were enabled.
- Number of rooms generated.
- Number of rooms passed the gate.
- Failed rooms and repair history.
- Final export files.
- Drake load result.
- Isaac/USD/MuJoCo status if requested.
- Collision hull caps used.
- Full-house Drake load result on a 2-GPU allocation.
- Articulated router validation output path and pass/fail status.
- Worker `--config-only` resolved asset policy output proving no forced HSSD path.
- Compute-node OpenAI proxy endpoint and probe result.
- Whether manipulands used Objaverse/ObjectThor instead of primitive fallback.

## Cost And Schedule Estimate

These are planning estimates, not guarantees. Actual cost depends on object count, SAM3D checkpoint load time, OpenAI usage limits, image generation retries, and repair loops.

Run 3 scale reference:

- Around 18 rooms.
- Around 770 placed objects.
- HSSD-heavy generation still hit OpenAI usage pressure.

Full generated-SAM3D plan:

- GPU: roughly 120-300 GPU-hours for 18 rooms if most assets use SAM3D and validation renders.
- Wall clock on 3 independent GPUs: roughly 2-5 days if API limits do not stall; longer if usage limits force sleep/retry cycles.
- API budget: high. Expect hundreds to low thousands of LLM/VLM/router/agent calls across generation, validation, and repair loops.
- Risk: highest quality but highest stall/cost risk.

Hybrid plan:

- Use `generated_sam3d` for rooms where HSSD failed quality: restrooms, library, storage, corridors, lobby/office, closets, special equipment rooms.
- Use `hssd` for repeated classroom furniture if router/gate validates the classrooms as acceptable.
- GPU: roughly 40-120 GPU-hours depending on how many rooms are regenerated.
- Wall clock on 3 independent GPUs: roughly 1-2 days plus repair loops.
- API budget: medium-high, but substantially lower than all-SAM3D.
- Recommended when deadline or API usage is constrained.

Fast/HSSD plan:

- Use only when the user explicitly chooses speed over quality.
- GPU: roughly 10-40 GPU-hours.
- Wall clock on 3 GPUs: same day is plausible.
- Risk: asset variety and placement realism can be poor; not acceptable as the default full-quality contract.

## Quick Checklist Before Launch

Do not launch until these are answered:

- Output target: full-quality or fast/HSSD-only?
- Asset policy: `generated_sam3d` or `hssd`?
- Are ArtVIP and/or Artiverse expected?
- Are manipulands using Objaverse/ObjectThor rather than primitive fallback?
- Has `run_single_room_worker.py --config-only` proved no forced HSSD path?
- Is the OpenAI proxy reachable from the actual compute node and persistent after laptop disconnect?
- How many GPUs, and which room per GPU?
- What exact files will each job write?
- Where is the SAGE/Codex room gate output stored?
- What condition allows final assembly?
- Which checkout is canonical: ParaCloud production or local dev?
- Are uploads overlay-only with no `--delete`?

## Failure Lesson From Run 3

Run 3 completed a scene, but it was not a valid best-quality result because:

- It continued an existing `furniture_hssd_only_fast` run.
- SAM3D was bypassed by `general_asset_source=hssd`.
- Review images were generated after the fact, but no pass/fail self-exam gate blocked final export.
- Bad placement survived into the final outlook render.

For future scenes, treat this file as the pipeline contract.
