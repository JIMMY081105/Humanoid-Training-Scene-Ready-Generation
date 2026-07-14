# Factory 8-GPU Execution Tasks

Canonical checkout: `/data/run01/scvj260/scenesmith-factory-codex`  
Canonical outputs: `/data/run01/scvj260/factory_outputs`  
Account GPU cap observed on 2026-07-13: 8

## Dependency chain

1. **Build readiness (no GPU)**
   - Compile every factory `[BUILD]` helper.
   - Validate the immutable input manifest and factory contract.
   - Create and immediately verify `pipeline_code_contract.json`.
   - Materialize the deterministic native layout and pass the factory layout gate.
   - Do not submit generation while any item fails.

2. **Asset authority preparation (1 GPU, released on completion)**
   - Submit `TEMPLATE_factory_assets_1gpu.sbatch`; ParaCloud rejects compute jobs without a GPU request.
   - Reflink and authenticate the two pinned Artiverse archives into the factory-owned tree, safely extract all 3,544 declared roots, and download the exact pinned OpenCLIP blob into the factory cache.
   - Use the allocation for the 500-item Artiverse CLIP/SDF index, validate its publisher/physics authority, audit visual resources, and revalidate materials plus ObjectThor offline retrieval.
   - Exit as soon as all asset receipts pass; Slurm releases the GPU.

3. **Processing-hall calibration (1 GPU)**
   - Submit `TEMPLATE_factory_benchmark_processing_hall.sbatch`.
   - Run the same code/input/model/asset/router preflights as production.
   - Generate only `processing_hall`, refresh its bound blend, render exactly three cutaways, and pass deterministic plus paid visual gates.
   - Rehash the saved visual decision and write `processing_hall_full_quality_benchmark.json` with ordered timestamps and `nvidia-smi` samples.
   - Exit `benchmark_complete`; Slurm releases the GPU when the job exits.

4. **Calibration decision (no GPU)**
   - Confirm the benchmark receipt is schema-valid and hash-current.
   - Use measured wall-clock/GPU-memory evidence to confirm the seven-day request and storage budget.
   - Do not submit full production if calibration or its saved-gate revalidation failed.

5. **Parallel production allocation (8 GPUs)**
   - Submit `TEMPLATE_factory_full_parallel.sbatch` only after Task 3 passes.
   - The runner creates one background lane per visible GPU; a lane processes its rooms serially so two rooms never share one GPU.
   - Fixed first-wave assignments put all compulsory articulated-role candidates and dense processing work on distinct GPUs:

     | GPU lane | First room | Second room |
     |---|---|---|
     | 0 | `cold_storage` | `packaging_hall` |
     | 1 | `maintenance` | `finished_goods_storage` |
     | 2 | `office_administration` | `washing_preparation` |
     | 3 | `changing_room` | `qc_laboratory` |
     | 4 | `break_room` | `boys_toilet` |
     | 5 | `ingredient_receiving` | `girls_toilet` |
     | 6 | `dry_storage` | none |
     | 7 | `processing_hall` | none |

   - Each lane uses a distinct `CUDA_VISIBLE_DEVICES`, port offset, run name, room directory, and `gpu_lane_<n>.log`.
   - Every room completes generation, exact state-to-blend refresh, three cutaway renders, deterministic gate, paid visual gate, and saved-decision rehash before that lane advances.
   - Any lane failure makes the parent Slurm job fail; assembly is not entered.

6. **Serialized cross-room gates (same allocation, no unsafe parallelism)**
   - Revalidate all 14 deterministic and visual gates together.
   - Run factory variation and real Drake articulated-motion gates.
   - Do not overlap these gates with room mutation.

7. **Serialized assembly and final gates**
   - Assemble only after Task 6 passes and prove Artiverse role survival.
   - Run whole-factory visual review, obstacle-inflated pedestrian/forklift navigation, Drake load, exports, SAGE checks, and repeat verifiers serially.
   - Create the immutable package manifest and pending two-GPU receipt.
   - Exit `awaiting_2gpu_acceptance`; Slurm releases all eight GPUs when the job exits.

8. **Terminal acceptance (2 GPUs)**
   - Submit `TEMPLATE_factory_2gpu_acceptance.sbatch` with the three out-of-band hashes/IDs.
   - Verify the package before and after a real `--require-gpus 2` Drake load.
   - Publish the terminal acceptance receipt and let Slurm release both GPUs immediately on exit.

## Resource and failure rules

- Never request more than the account cap or more GPUs than the runner can isolate into lanes.
- A pending allocation is not a reason to weaken a preflight or gate.
- Do not hold GPUs for input staging, code repair, or manual review. The asset-prep allocation is the narrow exception forced by ParaCloud's GPU-only compute partition and must proceed through GPU-backed Artiverse indexing before exit.
- Cancel a wedged job only after preserving its logs and determining that no atomic publish is in progress.
- Reuse a room only when its hash-bound saved visual decision passes `--summarize-existing`; otherwise regenerate/repair only that room.
- Assembly, navigation, export, acceptance, and evidence publication remain serialized even inside an eight-GPU allocation.
