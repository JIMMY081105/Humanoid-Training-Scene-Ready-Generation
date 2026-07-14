# SceneSmith quality-retention and 48-hour decision

Status date: 2026-07-13. Quality reference: the strict full-quality pipeline in
`CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md`.

## Decision

**NO-GO for full 11-room production today.** This is not a quality waiver. The
optimized design is intended to retain the strict reference acceptance criteria, but
the required representative-room timings, cache-hit measurements, and final-stage
measurements do not yet exist. Therefore neither **≥95% expected quality retention**
nor **≥80% probability of completion within 48 hours** is demonstrated.

Room 1 is the active benchmark, not an authorized full-floor launch. Its durable
state has reached `manipuland_checkpoint_004_storage_cabinet_0`. The current source
repair is subject to renewed offline SAM3D proof before resume.

## Quality-retention contract

Percentages below are planning predictions relative to the strict workflow, not
earned acceptance results. A critical failure is never averaged into any score.

| Dimension | Predicted retained quality | Minimum | Current evidence | Optimization risk | Delivery prevention gate |
| --- | ---: | ---: | --- | --- | --- |
| Visual geometry and materials | 97/100 | 90/100 | Same generated-SAM3D/retrieval sources, material contract, three cutaway views, and reference-aware VLM gate are retained. | Reusing an asset with a stale visual/configuration hash, or skipping a room refresh. | Hash-bound room visual self-exam; all scores ≥7; whole-floor reference gate ≥7. |
| Collision-mesh contour fidelity | 98/100 | 95/100 | Same Artiverse/ArtVIP source collision assets; generated VHACD/COACD cap remains 32; new axis recovery uses only rigid rotations plus uniform scale. | Reusing stale collision evidence, over-simplification, or an axis/scale defect. | Dimension receipt reload, SDF/collision-cap validation, full Drake load, and final two-GPU Drake acceptance. |
| Placement, support, and penetration correctness | 100% pass/fail target | Zero unacceptable penetration | Room 1 caddy, wipes, and paper placements passed local checks; rejected placements were removed. | Replacing exact local validation with broad-phase-only acceptance. | Per-placement exact collision/support checks plus final strict room physical/self-exam gate. |
| Doorway, corridor, and robot clearance | 100% pass/fail target | Zero blocked required doors/corridors | Reference layout gate requires valid doors/connectivity; optimization leaves it intact. | Deferring room-local exclusion checks without restoring final navigation validation. | Layout gate and full-house `validate_school_navigation.py`, including saved recomputation proof. |
| Manipuland accessibility and interaction usability | 100% required-task-object target | 100% required objects accessible | Required cabinet glue/scissors are contained in an accessible caddy on the safe top support. | Cache reuse may preserve a mesh but not a room-specific support/access relationship. | Canonical inventory/self-exam, local support/clearance checks, articulated-motion and navigation gates. |
| Physical stability | 100% pass/fail target | No unstable accepted pose | Room 1 per-furniture simulation discarded its altered solution and restored the critic-approved poses; this preservation is explicit. | Skipping forward simulation or accepting a projected pose without revalidation. | Physical-feasibility post-processing, forward simulation, and room final physical gate. |
| Isaac Sim import and execution | 100% pass/fail target | Mandatory pass | The pipeline requires atomic USD export/validation and an Isaac compatibility path. No final package exists yet. | Treating Drake load or a written USD file as Isaac acceptance. | Atomic simulator-export validation, USD layer load, then final Isaac compatibility/import execution. |
| Reproducibility and checkpoint integrity | 99/100 | 99/100 | Atomic checkpoints and code/input/artifact hashes are retained; stale SAM3D evidence was correctly rejected when `mesh_utils.py` changed. | Broad fingerprints causing needless work, or narrow fingerprints omitting a true dependency. | Input/code contracts, receipt verification, checkpoint-chain validation, final acceptance bundle, external package manifest. |

The expected weighted retention is **not authorized as a single score**. The design
goal is at least **95/100 overall only after every critical pass/fail gate passes**.
Any failed collision, blocked threshold/corridor, inaccessible required task object,
unstable pose, or Isaac failure makes the result ineligible regardless of visual score.

## Measured evidence so far

- One A10 is confirmed available. The reference manifest has 11 rooms.
- The current recovery required about 5.5 minutes of preflight before worker start
  (16:30:03–16:35:34) and 44.36 seconds of SAM3D cold startup.
- The prior 15-view cabinet review took 32.7 seconds; the corrected one-surface
  review took about 2.2 seconds. Local exact physics checks took about 6.3–6.7
  seconds each.
- The storage cabinet reached critic scores 8/9/8/8/10 and durable checkpoint 004.
  Its required glue/scissors caddy passed; unsafe candidate placements were rejected.
- The first normal-desk attempt exposed a generalized post-canonicalization notebook
  axis gap. The repair has 22 focused dimension-contract tests passing; the
  hash-bound SAM3D proof must be republished before it is used in production.

This is useful benchmark evidence, but it is not a completed difficult room, a normal
classroom completion, cache-hit measurement, or final-certification measurement.

## 48-hour deadline model

Reserve 8 hours for failed-room recovery, floor assembly, final gates, export, and
Isaac certification. With one GPU, the usable room budget is therefore:

`(48 h - 8 h) / 11 rooms = 3.64 h per room`.

| Execution mode | 48-hour completion probability | Basis and condition |
| --- | ---: | --- |
| Current restart-heavy one-A10 workflow | **<50% (preliminary 20–35%)** | Repeated whole-run preflight, cold services, restart-driven proof renewal, and unresolved per-room throughput make the 3.64 h/room threshold unproven. |
| Optimized one-A10 workflow | **Unmeasured; do not claim ≥50%** | Becomes Conditional-GO territory only if both a difficult room and a normal classroom complete in ≤3.64 h inclusive of their amortized share of serial work, with ≥95% quality retention evidence. |
| Three independent GPU workers | **Unmeasured; conditional 50–79% only after packing measured durations** | Requires three safely allocated GPUs, isolated state/ports, immutable shared services, and measured serial assembly/certification tail. |
| Maximum presently allocatable GPU count | **1 confirmed; same as one-A10** | No scheduler evidence shows more available GPUs. Do not substitute a hypothetical allocation for an estimate. |

No percentage above is a commitment until duration samples are collected. For multiple
GPUs, estimate wall time by packing measured room durations across workers, adding
measured serial preflight/layout/variation/assembly/certification and the fixed 8-hour
reserve; do not divide a single-GPU total by GPU count.

## Mandatory measurements before production authorization

1. Complete the optimized Room 1 benchmark through its strict deterministic and
   reference-aware visual gates; record every stage timestamp, request usage, GPU
   utilization, peak VRAM, retries, and exact reusable artifacts.
2. Complete one representative difficult room (library or storage) and one normal
   classroom through the same gates.
3. Measure asset-cache hit rate, repeated-generation avoidance, and service cold-start
   versus warm-start time.
4. Break down validation time: asset qualification, local collision/support,
   furniture/room physical feasibility, review render, deterministic gate, VLM gate,
   and final certification.
5. Measure or safely benchmark floor assembly, whole-floor reference/navigation,
   Drake, USD/Isaac export, and two-GPU acceptance duration.
6. Recompute each retention row from actual gate outcomes and each probability from
   measured duration distributions plus the 8-hour reserve.

## Authorization rule

- **GO** only if measured expected retention is ≥95/100, every critical gate remains
  mandatory, and the calculated 48-hour completion probability is ≥80%.
- **CONDITIONAL GO** only if quality is protected but the probability is 50–79%, or if
  the ≥80% result requires explicitly allocated additional GPUs.
- **NO-GO** if retention is <95/100, any critical gate is weakened, or probability is
  <50%.

The current state remains **NO-GO** pending the measurements above. This preserves
strict room collision, robot-clearance, and Isaac acceptance rather than improving the
deadline estimate by weakening them.
