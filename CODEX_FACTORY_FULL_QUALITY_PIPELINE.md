# Codex SceneSmith Full-Quality Pipeline — Food Factory

This file is the handoff contract for the food-factory SceneSmith run. It is a **sibling** of
`CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md` (the reference-school contract) and deliberately keeps
the same mindset: fail closed, hash-bind every decision, run the deterministic gate before the paid
VLM gate, never fall back to HSSD or primitives to satisfy a quality requirement, and never assemble
or export until every zone has passed a real pass/fail gate. The school contract is not edited or
replaced; its prompt hash, `pipeline_code_contract.json`, and `school_reference_20260710` profile
remain the canonical school attestation. This factory contract introduces a new profile,
`factory_reference_<YYYYMMDD>`, and its own factory-specific gate code.

> Implementation status (read first): the generic quality machinery (SAM3D preflight, Artiverse/ArtVIP
> routing, materials contract, ObjectThor retrieval, proxy verification, room worker, assembly,
> Drake/MuJoCo validation, acceptance bundle) already exists and is reused unchanged. The
> **factory-specific gates named below do not exist yet** and must be implemented as factory siblings of
> the school scripts before launch. Each such script is marked `[BUILD]`. Until every `[BUILD]` script
> exists and passes, the run stays fail-closed exactly as the school run would. This is intentional.

## Non-Negotiable Output Rules

These are inherited verbatim from the school contract; the food factory does not relax any of them.

- Do not use an `hssd_only_fast` run as the final-quality run unless the user explicitly approves it.
- Do not set `general_asset_source=hssd` for a full-quality scene unless the user explicitly asks for HSSD-only speed. HSSD is absent and disabled in this environment and may not be introduced as a fallback.
- Keep SAM3D available with `general_asset_source=generated` and `backend=sam3d` for furniture, wall, ceiling, and manipuland agents when the target is quality.
- Use Objaverse/ObjectThor retrieval for small manipuland objects to avoid primitive-shape fallbacks.
- Artiverse is compulsory for every full-quality run. The local `data/artiverse` dataset and the `artiverse_articulated` route must exist, be enabled, pass validation, and contribute at least one asset that survives into the final assembled scene. An enabled flag without proven final asset usage does not satisfy this contract.
- Every copied Artiverse SDF used by a zone must normalize its material-split OBJ visuals through the matching publisher-authored GLB into deterministic zone-local external `.gltf` plus `.bin` resources with explicit normals before Drake/Blender rendering; only the exact derived resource directory and grouped copied visual elements may be added or changed, while publisher sources, collisions, joints, inertials, poses, scales, prompt content, and all quality settings remain unchanged. Direct `.glb` SDF mesh URIs are forbidden because Drake ignores them.
- Keep ArtVIP articulated retrieval enabled alongside Artiverse through `asset_manager.articulated.sources.artvip.enabled=true`; ArtVIP is not a substitute for the compulsory Artiverse route.
- Do not assemble `combined_house`, final Drake exports, Isaac/USD exports, or final renders until every required zone has passed a zone-level quality gate.
- A render-only review folder is not a quality gate. A gate must produce pass/fail records and must block export on failure.
- Cap generated collision decomposition at 32 convex hulls per object per agent. A simulator export that cannot load is not simulation-ready.
- Record the canonical execution checkout for this run and never sync with `--delete` unless explicitly approved. The canonical factory checkout is the authorized `/data/run01/scvj260/scenesmith-factory-codex` on ParaCloud; ParaCloud is also where the final 2-GPU Drake acceptance runs.

### Factory-specific non-negotiables

- The building is a single fixed shell of **44 m x 32 m x 5 m** containing exactly **14 enclosed functional rooms**. Exterior/common areas (loading dock, entrance transition, internal circulation, exterior truck road, landscaping) are explicit non-room zones, not extra rooms.
- The food-safety production **workflow order must be physically satisfied**: ingredient receiving -> dry/cold storage -> washing/preparation -> processing hall -> packaging hall -> finished-goods storage -> loading dock. A layout that violates this adjacency/flow fails closed.
- Circulation minimums are hard gates: **3.0 m forklift routes, 2.5 m production aisles, 1.2-1.5 m worker clearances**, and door/emergency-exit/window clearances. Pedestrian and forklift paths must be separable; the delivery truck stays outside the building shell.
- Wet zones and electrical zones must be separated; machinery must be modular with declared under-machine free space.
- At least one legitimate, authority-validated **Artiverse articulated asset** must survive final assembly, exactly as in the school contract. Factory articulated roles replace the school furniture roles (see Section 10).
- The 7 genuinely distinctive machines are the **only** approved SAM3D generations for the minimum build: forklift, pallet jack, hygienic mixer/hopper, food washing/sorting/inspection unit, filling/dosing unit, combined sealing/carton/label module, and box/delivery truck. Aliases (`packaging machine`, `packing machine`, `heat sealer`, `carton sealer`, ...) reuse the same modular output and must not trigger separate generations. Reference: `CODEX_FOOD_FACTORY_ASSET_AUDIT.md` and `CODEX_FOOD_FACTORY_ASSET_INVENTORY.csv`.
- Structural industrial forms (tanks, pipes, conveyors, racks, guards, cold-room shell, loading platform) are procedural primitives, not SAM3D. Do not spend generation budget on them.

## Pipeline Overview

Identical control flow to the school contract; only the profile and the layout/inventory/navigation/
articulation/acceptance gates change.

1. Read and hash this specification; attest the exact runner, overlay, and ordered patch stack
2. Prompt/reference-image and run setup (factory profile)
3. Environment, model, dataset, index, and GPU preflight
4. Compute-node API proxy verification
5. Full-quality asset policy check
6. Fail-closed ArtVIP/Artiverse router validation
7. Floor-plan-only generation and exact **factory** layout gate `[BUILD]`
8. Sequential zone generation with the **factory** room contract `[BUILD]`
9. Three-view zone render capture
10. Deterministic, itemized VLM, and cross-zone distinctness gates (factory inventories) `[BUILD]`
11. Zone repair/regeneration loop + factory articulated-role motion gate `[BUILD]`
12. Artiverse-provenance-gated final assembly
13. Whole-factory reference-image and **factory navigation** gates `[BUILD]`
14. Drake/SceneEval/collision validation, including final 2-GPU acceptance
15. Atomic Isaac/USD/MuJoCo export if requested
16. Outlook renders and local transfer
17. Hash-complete acceptance bundle and final validation report

A successful single-GPU completion is recorded as `awaiting_2gpu_acceptance`, never final `complete`,
exactly as in the school contract.

## 1. Prompt And Run Setup

Function: define the factory scene and create a clean run directory.

Required inputs:

- Prompt CSV, for example `inputs/full_quality_factory_reference_<YYYYMMDD>/prompt.csv`.
- The complete source prompt and any supplied reference image, preserved under the run input directory with recorded hashes in an input manifest.
- A run name that states the real asset policy, e.g. `full_quality_factory_sam3d_artvip_artiverse`.

Stage `prompt_original.txt`, `scene_contract_appendix.txt`, effective `prompt.txt`, one-row
`prompt.csv`, the reference image, and `input_manifest.json` together. Run
`scripts/validate_input_manifest.py` before any model/service preflight (this generic validator is
reused unchanged). The manifest must contain `pipeline_contract.id=scenesmith_full_quality_v1`,
`final_assembly_policy=external_artiverse_gated`, and `required_articulated_source=artiverse`.

> Placeholder to bind once: the factory effective prompt length and its SHA-256 are **not yet fixed**.
> After the factory prompt is authored and materialized, record `factory_effective_prompt_sha256` here
> and in the input manifest, exactly as the school contract pins `ac8d297c...`. Do not launch against an
> unbound prompt hash.

Before any generation-capable code, the runner must fully read this file and create
`pipeline_code_contract.json` with `scripts/pipeline_code_contract.py`, binding this specification, the
production runner, every pipeline Python helper (including the `[BUILD]` factory gates once they
exist), SAGE checker, remote jobs, and every patch named in `upstream-patches/APPLY_ORDER.txt`.
`--verify-only` must pass again immediately before the acceptance bundle.

## 2. Environment/GPU Preflight

Function: verify the job is on a GPU node, proxy/API access works, checkpoints exist, and the selected
asset sources load. This section is inherited unchanged from the school contract; the factory does not
weaken any preflight. Run the exact same checks:

- `nvidia-smi`; `external/checkpoints/sam3.pt`; `external/checkpoints/pipeline.yaml`.
- SAM3D offline load + full-generation receipts under `outputs/preflight/<run_name>/`.
- Materials contract validate against `data/materials` + `data/materials_full_quality_contract/embeddings` (`scripts/materials_contract.py validate`, `--min-retained 1900 --max-pruned 15`).
- ArtVIP dataset + three index files under `data/artvip_sdf`.
- Artiverse dataset + three index files under `data/artiverse`, plus `scripts/artiverse_contract.py`.
- ObjectThor four prepared files + `scripts/preflight_objathor_retrieval.py --verify-only`.
- OpenAI probe returning `200`/`401`/`403` (production requires `200`).
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `DIFFUSERS_OFFLINE=1`.
- `scripts/preflight_sam3d_offline.py` (schema-v3) and `scripts/preflight_sam3d_generation.py` full-generation GLB/mask/masked-image proof, then their `--verify-only` paths at benchmark and acceptance.
- `scripts/preflight_vlm_vision.py` against the exact factory reference image, requiring hash-bound `status=pass`.
- `scripts/preflight_artiverse_visual_resources.py` audit + `--verify-only`.

Artiverse remains compulsory: missing dataset files or any of the three index files is a hard stop, not
a reason to fall back to ArtVIP, HSSD, generated primitives, or an unproven flag.

> Factory environment note (from the ParaCloud audit): the compulsory **Artiverse prepared index** and
> the **materials-contract derivative** are not yet present in the ParaCloud legacy tree, and the SAM3D
> checkpoint binding must be repaired (the `external` symlink still points at an SQZ-only path). These
> are legitimate hard stops. Prepare the isolated Artiverse index (armoire/bookcase/chest_of_drawers,
> plus any scoped factory categories approved per the audit) and the materials derivative under the
> factory-owned checkout before generation. Never execute the rejected publisher unpacker; use
> `scripts/safe_extract_artiverse.py` and the authority contract.

## 3. Compute-Node API Proxy Verification

Inherited unchanged. GPU jobs must reach OpenAI through a proxy reachable from compute nodes (a
persistent login-node `tmux`/`autossh`/cluster proxy, not a fragile laptop foreground tunnel). Every
SLURM script probes OpenAI from inside the GPU allocation and fails fast if unreachable. Production zone
workers must use the OpenAI Responses API for multimodal turns (do not force Chat Completions: it drops
`ToolOutputImage` and blinds the designer). Preserve the trace-safe `ModelSettings` subclass so provider
timeout budgets survive. Do not launch generation until the compute-node probe works.

## 4. Full-Quality Asset Policy

Inherited unchanged. Apply the same Hydra overrides as the school contract: furniture/wall/ceiling
`general_asset_source=generated` with `backend=sam3d`; manipulands `objaverse`; both ArtVIP and the
`++...artiverse` articulated route enabled; the `artiverse_articulated` strategy enabled with
`required_source=artiverse`; 32-hull caps on CoACD and V-HACD for all four agents; `SCENESMITH_REQUIRE_
GATED_FINAL_ASSEMBLY=1` and `experiment.pipeline.final_assembly_policy=external_artiverse_gated`. On a
single 24 GB GPU set `SCENESMITH_RETRIEVAL_DEVICE=cpu`.

Before launching zone workers, grep `scripts/run_single_room_worker.py` for any forced
`general_asset_source` override and run `--config-only`; the resolved config must show furniture/wall/
ceiling as `generated` and manipulands as `objaverse` for `--asset-pipeline generated_sam3d`. The
expected resolved-policy JSON is identical to the school contract's.

## 5. Articulated Router Validation

Inherited unchanged in mechanism; only the validation prompts become factory-relevant. Run
`scripts/validate_articulated_router.py` (on ParaCloud via
`sbatch remote_jobs/TEMPLATE_validate_articulated_router.sbatch`). It must prove the generic
`articulated` strategy selects an ArtVIP candidate for openable factory furniture (tool cabinet, locker,
filing cabinet, refrigerator), and that `artiverse_articulated` is selected for at least one
Artiverse-eligible factory prompt with a real `source=artiverse` candidate under `data/artiverse`.
Before final assembly, verify at least one placed Artiverse asset remains in a passing zone; assembly
fails if the Artiverse survival count is zero.

## 6. Floor-Plan-Only Generation And Factory Layout Gate `[BUILD]`

Function: create the 44x32x5 building layout first, without loading zone-generation services, then fail
closed unless the layout matches the factory contract.

Materialize the exact factory layout with SceneSmith's native `RoomSpec`, `PlacedRoom`, `Wall`, `Door`,
`Window`, and `RoomGeometry` implementations, mirroring `scripts/seed_reference_school_layout.py`:

- `[BUILD]` `scripts/seed_reference_factory_layout.py --profile factory_reference_<YYYYMMDD>` — builds four genuine walls per room, applies every opening through the real floor-plan tool methods, calls the common-zone tool after backing doors exist, generates structural SDF/GLTF, writes the Drake directive, exports the Blender floor plan, and publishes `house_layout.json` last with the same atomic transaction/rehash discipline as the school seeder. No paid API call.

The **14 required rooms** (immutable IDs, replace the school room IDs):

1. `ingredient_receiving`
2. `dry_storage`
3. `cold_storage`
4. `washing_preparation`
5. `qc_laboratory`
6. `office_administration`
7. `processing_hall`
8. `packaging_hall`
9. `finished_goods_storage`
10. `maintenance`
11. `changing_room`
12. `break_room`
13. `boys_toilet`
14. `girls_toilet`

Explicit exterior/common zones (not rooms): `loading_dock`, `entrance_transition`, `internal_circulation`,
`exterior_truck_road`, `landscaping`.

Enforce the layout with the factory gate immediately afterward:

- `[BUILD]` `scripts/validate_factory_floor_layout.py --layout .../house_layout.json --output .../quality_gates/floor_plan_layout.json` — a deterministic gate that mirrors `validate_school_floor_layout.py` but checks factory facts and computes them itself rather than trusting upstream booleans:
  - The exact 14 room IDs present, none missing/renamed/extra; zero positive-area AABB overlaps; valid connectivity.
  - The overall 44 m x 32 m x 5 m shell dimensions and orientation.
  - **Workflow adjacency**: receiving adjacent to storage; storage to washing/preparation; preparation to processing; processing to packaging; packaging to finished-goods; finished-goods to the loading dock. A path that forces raw ingredients back through a clean zone fails.
  - **Wet/electrical separation**: washing/preparation and cold storage are not co-located with electrical/office zones without a partition.
  - **Loading dock** placed on the correct exterior wall with door + leveller geometry; the truck road is outside the shell.
  - Exterior windows where required (offices, break room, QC lab, changing room); emergency exits; door widths >= interior thresholds.
  - Toilets adjacent, reached through an explicit shared foyer via real backed door openings + `set_navigation_common_zones` (same mechanism as the school restroom foyer).

A missing, renamed, extra, overlapping, disconnected, mis-adjacent, or workflow-violating room stops the
pipeline before any zone generation. Repair/regenerate the floor plan and rerun the gate.

Output: `house_layout.json`, `floor_plans/`, `room_geometry/`, `room_<zone_id>/`, and
`quality_gates/reference_factory_layout_seed.json`.

## 7. Zone-Level Generation `[BUILD] contract`

Function: generate furniture, wall objects, ceiling objects, and manipulands for each zone only after
the factory layout gate passes. The generic `scripts/run_single_room_worker.py` is reused unchanged
(`--asset-pipeline generated_sam3d`, unique `--port-offset`, per-zone output folder).

Bind canonical zone prompts before generation, mirroring `school_room_contract.py bind-layout`:

- `[BUILD]` `scripts/factory_room_contract.py bind-layout --layout .../house_layout.json --input-manifest .../input_manifest.json --output .../quality_gates/room_prompt_binding.json` — replaces each planner-shortened zone prompt with a canonical, immutable, hash-bound zone prompt derived from the factory specification and `CODEX_FOOD_FACTORY_ASSET_INVENTORY.csv`. Both the unique `RoomSpec` and unique `PlacedRoom` for every one of the 14 zone IDs must carry identical canonical prompt bytes bound to the immutable effective prompt; duplicate/ambiguous zone records are fatal.

Each zone gets an **immutable inventory** derived from the CSV (distinct-object counts, not one generic
object satisfying several requirements). Examples of per-zone required inventory the contract must encode:

- `ingredient_receiving`: receiving desk, floor scale, pallets, pallet jack, hand truck, intake bins, wall signage.
- `dry_storage`: rack bays, shelves, cartons/crates/drums, forklift access aisle.
- `cold_storage`: cold-room shell + insulated door, refrigeration unit, shelving, product bins.
- `washing_preparation`: wash sinks, stainless prep tables, floor drains, hoses, wash/sort/inspection unit.
- `processing_hall`: mixer/hopper, conveyor line, tanks/vats, pipes/valves/motors, control panels, guards, emergency stops.
- `packaging_hall`: filling/dosing unit, sealing/carton/label module, conveyor, carton stacks, wrapping station.
- `finished_goods_storage`: pallet racks, wrapped pallets, forklift aisle, dispatch staging.
- `qc_laboratory`: lab bench, microscope, workstation, sample racks, sink.
- `maintenance`: workbench, tool chest/tool cabinet, parts shelving.
- `office_administration`: desks, office chairs, filing cabinets, monitors.
- `changing_room`: lockers, benches, hooks.
- `break_room`: table, chairs, refrigerator, microwave, water dispenser.
- `boys_toilet` / `girls_toilet`: toilets, partitions, sinks, soap/hand-dryer fixtures.

Reuse the same policy as the school run: sequential zones on the available GPU(s), unique ports even when
serial, each worker writes only its own `scene_000/room_<zone_id>/`, no final assembly while any worker is
active, `select_room_resume_stage.py` checkpoint discipline, and `--refresh-final-blend` before any gate
that would use a stale blend.

## 8. Zone Render Capture

Inherited unchanged (`scripts/render_room_review_views.py`): one top-down cutaway plus at least two
oblique cutaway views per zone, ceilings/roof hidden, camera-side envelope walls hidden with recorded
cutaway-evidence proof JSON. The factory's tall machinery makes the cutaway proof more important, not
less; a view whose cutaway cannot be established fails.

## 9. Zone Self-Exam Gate `[BUILD] profile`

Run the deterministic gate first, then the compulsory image-aware gate, mirroring the school flow but
under the factory profile:

- `scripts/room_self_exam.py ... --contract-profile factory_reference_<YYYYMMDD> --max-collision-hulls 32` — the deterministic gate must be taught the factory inventories `[BUILD]`: it blocks missing final states, missing review images, objects outside zone bounds, suspicious density, collision-hull risk, a missing canonical prompt marker, and every missing/over-counted **factory** inventory requirement, with a unique bipartite assignment so one generically named object cannot satisfy several checklist entries.
- `scripts/room_visual_self_exam.py ... --contract-profile factory_reference_<YYYYMMDD> --threshold 7 --minimum-review-images 3` — the visual gate cannot run until deterministic evidence passes; it requires the canonical zone checklist, the complete immutable effective prompt, the supplied reference image, and >=3 distinct views, with one explicit pass/fail record per inventory item, material/daylight/circulation/collision requirement, zone identity, and applicable articulated role. Final scores are the minimum of deterministic and visual scores.

Additional factory deterministic checks the profile must add `[BUILD]`:

- Forklift routes >= 3.0 m; production aisles >= 2.5 m; worker clearances 1.2-1.5 m; door/emergency-exit/window clearances.
- Machinery modularity and declared under-machine free space.
- Wet/electrical separation within the zone.
- Pedestrian vs forklift path separability.

Pass threshold: no critical issues; every combined score (`object_relevance`, `placement_realism`,
`clearance_and_access`, `collision_risk`, `prompt_alignment`) >= 7; deterministic status `pass`.

Run the zones that carry the compulsory articulated roles first (cold storage, maintenance/changing,
office/break) so articulation fails early, then continue sequentially. After all 14 zones pass
individually, rerun the deterministic gate over the complete set and call the visual gate with
`--summarize-existing`. Then run the cross-zone distinctness gate:

- `[BUILD]` `scripts/factory_variation_gate.py` — mirrors `classroom_variation_gate.py` but applies where repetition is a real risk (the two storage rooms, the two toilets, and any duplicated staff rooms): hash-binds inputs, rejects identical semantic/fixture fingerprints deterministically, and requires a schema-valid VLM distinctness score with specific per-zone identities. Where zones are legitimately unique (processing vs packaging), the gate records identity rather than forcing artificial variation.

## 10. Zone Repair/Regeneration Loop + Factory Articulated Roles `[BUILD]`

Repair failed zones only, preserving good zones, using the same order as the school contract (placement
rerun -> furniture rerun -> reconsider floor plan) and the same `--asset-pipeline generated_sam3d`. After
each repair: rerender top + two oblique cutaways -> rerun deterministic gate -> rerun reference-aware
visual gate -> mark pass only if both JSON results pass.

After all zone and distinctness gates pass, exercise the factory articulated roles before assembly,
mirroring `scripts/validate_articulated_motion.py`:

- `[BUILD]` `scripts/validate_articulated_motion.py ... --contract-profile factory_reference_<YYYYMMDD>` (or a factory sibling) must require these three factory roles, replacing the school library/storage/teacher roles:
  1. A **cold-room insulated door** (revolute) in `cold_storage`.
  2. A **maintenance tool cabinet** (hinged doors) OR a **changing-room locker** (hinged door) — one operable-door role in `maintenance`/`changing_room`.
  3. An **office filing cabinet** (prismatic drawers) OR a **break-room refrigerator** (revolute door) — one operable role in `office_administration`/`break_room`.
- At least one of the three roles must be the separately **authority-validated compulsory Artiverse asset**. All three must pass a real `pydrake` joint-motion exercise with a measured child-body transform change (not merely a declared joint), with hash-bound room states and SDF trees. Repeat with `--verify-only` before acceptance.

## 11. Artiverse-Provenance-Gated Final Assembly

Inherited from the school contract with factory role names. Use
`scripts/assemble_final_house_and_render.py ... --contract-profile factory_reference_<YYYYMMDD>
--input-dir inputs/full_quality_factory_reference_<YYYYMMDD> --render` `[BUILD] profile`. Before staging,
the script must find at least one exact `asset_source=articulated`, `articulated_source=artiverse`,
`is_articulated=true` record whose canonical ID exists in the hash-bound prepared indexes and whose
copied SDF/tree hashes match the patched asset manager's provenance. It independently reparses the three
factory articulated roles, recomputes every bound SHA-256, refuses stale/legacy/substituted evidence, and
after assembly requires the compulsory Artiverse usage to survive the merge, renders and verifies three
nonempty overviews, and writes schema-v2 `artiverse_usage.json`. Zero usage, forged provenance, mutation,
missing source, or loss during assembly is fatal. `--allow-ungated` is debug-only and never waives the
Artiverse survival requirement.

## 12. Whole-Factory Reference Gate + Navigation `[BUILD]`

- `scripts/whole_floor_reference_gate.py ... --threshold 7` reused, but taught the factory layout `[BUILD]`: it must require exactly the 14 room IDs in the correct reference-relative arrangement, the three overview renders, and the same hash-binding + rehash-after-VLM discipline. Seven scores >= 7 (`room_count_and_identity`, `room_arrangement`, visual-finish, `circulation_and_access`, `furnishing_completeness`, `simulation_readiness`, `reference_similarity`), no critical issues.
- `[BUILD]` `scripts/validate_factory_navigation.py` — mirrors `validate_school_navigation.py` but proves **factory** circulation: a humanoid route from the entrance/loading transition to every one of the 14 rooms using only entrance/circulation/common zones plus the target (never cheating through another functional room), forklift routes >= 3.0 m and worker aisles at the specified widths validated as free space by inflating final object AABBs and grid-searching, pedestrian/forklift separability, the shared toilet foyer physically explicit in `navigation_common_zones`, and the truck kept outside. Bind layout, combined state, all 14 final states, portals, obstacles, and routes; repeat with `--verify-only`.

## 13. Drake/SceneEval/Collision Validation

Reused with factory counts. Structural precheck locally/on one GPU, then the mandatory 2-GPU ParaCloud
acceptance:

```bash
.venv/bin/python scripts/validate_drake_scene.py \
  --dmd .../combined_house/house.dmd.yaml \
  --package-root .../scene_000 \
  --require-gpus 2 --max-collision-elements 32 \
  --minimum-models <factory_min_models> --expected-rooms 14 \
  --output .../quality_gates/drake_load.json
```

Set `--expected-rooms 14` and a factory `--minimum-models` floor once the inventory is frozen. The 2-GPU
report must show `status=pass`, `visible_gpu_count >= 2`, `two_gpu_acceptance_environment=true`, no
malformed/over-cap SDFs, and a full-house load without OOM. Heavy factory machinery makes the collision
complexity report and the 32-hull cap especially load-bearing.

## 14. Isaac/USD/MuJoCo Export

Inherited unchanged (`.mujoco_venv` + `scripts/export_simulator_artifacts_atomic.py --require-usd`, atomic
staging, schema-v2 validation, `final_acceptance_bundle.py` rewalk). Restore the isolated `.mujoco_venv`
on the factory checkout first; export only when requested.

## 15. Outlook Renders And Local Transfer

Inherited unchanged: `overview_top.png`, `overview_isometric.png`, `overview_front.png`, per-zone review
images, contact sheet; hash-verified transfer; no final-success claim until local files exist and at least
one render is visually inspected.

## 16. Acceptance Bundle And 2-GPU Handoff

Inherited from the school contract with factory inputs/counts. Run
`scripts/final_acceptance_bundle.py ... --input-dir inputs/full_quality_factory_reference_<YYYYMMDD>` to
copy immutable inputs + outside-scene preflight evidence, semantically revalidate every verdict, and
record scene-relative hashes for all 14 deterministic/visual/cutaway sets, the cross-zone distinctness
verdict, factory articulated motion + its saved repeat proof, factory navigation + its recomputation
proof, whole-factory reference, Artiverse preparation/router/usage/final survival, SAM3D schema-v3 +
full-generation proofs, ObjectThor, materials, code attestation, Drake, SAGE, and the rehashed live
simulator export tree. Status is `awaiting_2gpu_acceptance`. Then create the external package manifest
and SQZ/ParaCloud completion receipt with `scripts/two_gpu_drake_acceptance_contract.py`, and run
`remote_jobs/TEMPLATE_2gpu_drake_acceptance.sbatch` for the terminal receipt with a real synchronized
CUDA exercise on devices 0 and 1.

## 17. Final Validation Report

Same required fields as the school contract, with factory substitutions: 14 zones instead of 11; the
factory articulated-role motion verdict (cold-room door, tool-cabinet/locker, filing-cabinet/refrigerator)
with before/after joint positions; the factory navigation/free-space verdict including forklift-route and
worker-aisle widths and pedestrian/forklift separation; the Artiverse survival count (>= 1); the exact
factory asset policy; whether SAM3D was reachable; whether HSSD was used anywhere (must be no); ObjectThor
usage evidence; the resolved `--config-only` policy proving no forced HSSD path; and the 2-GPU Drake
acceptance result.

## Cost And Schedule Estimate

Planning estimates only; no unmeasured figure is treated as a guarantee. Per the audit
(`CODEX_FOOD_FACTORY_ASSET_AUDIT.md`), the minimum build is 70 model families: 40 reused dataset/modular,
23 procedural, 7 SAM3D generations (reused as instances). Measured SAM evidence: ~65 s warm/asset (most
likely), ~8m19s for the 7 unique SAM models on one A10, ~5m04s on two GPUs, ~3m59s on three — throughput
of the 7 generations only, excluding layout, agent critique loops, rendering, and simulator acceptance.

Do not project a whole-factory wall clock before a benchmark. Run one dense benchmark zone through the
exact production path first — use `processing_hall` (the densest machinery zone) as the analogue of the
school `classroom_01` benchmark:

- `[BUILD]` add a `SCENESMITH_EXECUTION_MODE=benchmark_processing_hall` mode to the factory runner that performs the same code/input/model/asset/router preflights, deterministic factory layout materialization + layout gate, generated-SAM3D `processing_hall` worker, blend refresh, three cutaway renders, deterministic gate, paid visual gate, and saved-gate revalidation, then writes `processing_hall_full_quality_benchmark.json` with per-stage timestamps and `nvidia-smi` samples and exits `benchmark_complete`.

Only after that benchmark: estimate total wall time as measured serial preflight/floor-plan + measured
zone sums + serialized assembly/navigation/final-gate/export + a stated repair contingency. Label every
unmeasured term unknown. API cost stays unknown until the benchmark records actual model calls, tokens,
image usage, and retries.

Multi-GPU: ParaCloud allows up to **8 concurrent GPUs** for this account (16 public, node g0609 idle at
audit time). Assign independent zones to independent GPU workers with unique output folders/ports; keep
floor-plan, cross-zone distinctness, assembly, navigation, exports, and acceptance serialized. Do not
claim a fixed speedup before the benchmark identifies the GPU-bound fraction and the serial/API tail.

## Quick Checklist Before Launch

- Output target confirmed full-quality (generated_sam3d), not fast/HSSD.
- Resolved policy prints furniture/wall/ceiling `generated` + manipulands `objaverse`; no forced HSSD.
- SAM3D and every retrieval model proven to load offline.
- ArtVIP + compulsory Artiverse installed, converted/indexed, enabled, source-filtered, validated on the factory checkout (Artiverse prepared index + materials derivative actually present).
- Factory prompt authored, materialized, and its SHA-256 bound in this file and the input manifest.
- Prompt contains a legitimate Artiverse-eligible articulated furniture requirement (cold-room door / cabinet / locker / refrigerator).
- All `[BUILD]` scripts implemented and covered by `pipeline_code_contract.json`: factory layout seed + gate, factory room contract, factory profile in the self-exam gates, factory variation gate, factory navigation, factory articulated-motion roles, factory benchmark mode, whole-factory reference gate teaching.
- Floor-plan-only output passes `validate_factory_floor_layout.py` (14 rooms, workflow adjacency, aisle/forklift/worker clearances, loading dock, wet/electrical separation) before any zone work.
- Each zone has top + two oblique cutaway renders and passing deterministic + VLM gate JSON.
- Every zone inventory satisfies its distinct-count checklist; no generic object double-counts.
- Cold-room door, one cabinet/locker role, and one filing/refrigerator role all pass a real Drake articulation-motion test; at least one is authority-validated Artiverse and survives assembly.
- Assembled factory passes the whole-factory reference gate and the factory navigation/free-space gate (including forklift routes and the exterior truck).
- Internal acceptance record + external package manifest bind without a self-hash loophole.
- 2-GPU ParaCloud Drake acceptance location decided; canonical checkout recorded; uploads overlay-only, no `--delete`.

## Failure Lesson Carried Forward

The school Run 3 failure (continuing an `hssd_only_fast` run, SAM3D bypassed by `general_asset_source=hssd`,
after-the-fact review images with no blocking gate, bad placement surviving into the final render) applies
identically here. The factory adds one more standing risk from the asset audit: **do not let a lexical
match or a raw Artiverse directory stand in for a validated, collision-converted, physically simulated
asset.** Structural machinery is procedural or one of the 7 approved SAM generations; everything is
gate-proven before assembly. Treat this file as the factory pipeline contract.
