# SceneSmith school-floor: 48-hour engineering decision

## A. Current feasibility

**CONDITIONAL GO.** The target is plausible only if production changes preserve the
validated Room 1 work, avoid restart-wide preflight, keep immutable model services
warm, and run independent rooms whenever a GPU is available. It is not credible to
continue the current recover-by-restart workflow unchanged.

Current evidence:

- The active durable state is Room 1, `manipuland_checkpoint_003_cubby_shelf_unit_0`.
  The storage-cabinet pass is currently in designer/critic review; it has not yet
  published checkpoint 004.
- The current allocation is one NVIDIA A10. The run manifest enumerates 11 rooms,
  so a sequential schedule has a hard average budget of **4.36 hours/room**, including
  retries, final room checks, assembly, and Isaac verification.
- This recovery spent about **5.5 minutes** in mandatory preflight before the Room 1
  worker began (16:30:03–16:35:34), and a cold SAM3D startup took **44.36 seconds**.
- The cabinet designer pass used about **4.5 minutes** before critique. Exact local
  physics checks cost roughly **6.3–6.7 seconds** each; the initial 15-view cabinet
  render previously cost **32.7 seconds**. Designer/critic API turns add tens of
  seconds each.
- Repeated work, not mesh generation, caused the large loss: a thin-scissors axis
  conversion loop, then ten articulated-cabinet support planes embedded in the
  parent collision volume. Both are now repaired; the current pass exposes only the
  verified cabinet-top surface.

The present evidence is insufficient to claim a precise end date. Before freezing
the floor run, record per-room stage timing and the remaining-room inventory. The
go/no-go threshold is simple: with one A10, projected mean completed-room time must
stay below 4.36 hours and leave a final-certification reserve. If it exceeds that,
additional GPUs or approved scope/time are required.

## B. Recommended strategy

Accept these low-risk optimizations after a read-only dependency audit:

1. **Dependency-scoped receipts.** Bind asset/model evidence to asset content,
   converter/configuration versions, and the code that actually consumes it—not to
   unrelated placement-control changes.
2. **Persistent immutable services.** Keep SAM3D, retrieval indexes, and safe
   render workers alive across furniture and room resumes; never share mutable room
   state.
3. **Layered validation.** Use local exact checks for newly placed items, their
   parent furniture, and nearby indexed objects. Keep strict full-room validation
   at room acceptance and full-floor validation at final assembly.
4. **Verified asset reuse and room archetypes.** Reuse content-hash-bound school
   assets and structural archetypes while varying layout, materials, and clutter.
5. **Independent room workers.** Isolate output/state/ports per room; a failed room
   must not halt completed rooms. Use one worker per safely allocated GPU.
6. **Transactional critic edits.** Snapshot accepted state, apply a correction,
   validate the affected relationships, and roll back exactly unless it improves the
   stated failure without breaking prior requirements.

Reject these shortcuts:

- primitive substitutions for robot-relevant assets;
- box-like collision proxies that erase usable concavities;
- trusting cache filenames without content/configuration fingerprints;
- skipping final room/floor/Isaac validation;
- concurrent mutation of a shared scene.

## C. Quality argument

This changes *where* evidence is computed, not the acceptance standard. Immutable
assets retain high-quality visual and contour-aware collision evidence. Each instance
still receives transform, support, local-intersection, clearance, doorway, and
accessibility validation. Completed rooms still receive strict full-room collision,
render, critic, physics, and robot-space checks; the assembled floor still receives
cross-room, connectivity, navigation, integrity, and Isaac Sim verification.

## D. Execution plan

1. Let the active Room 1 pass reach a durable checkpoint or a verified failure; do
   not discard checkpoint 003.
2. Perform a read-only timing/dependency audit of started rooms, receipts, cache
   bindings, GPU capacity, and per-stage throughput.
3. Add compact regressions for checkpoint restore, articulated support extraction,
   thin-object axes, cabinet-top placement, registry resume, critic rollback, and
   local-vs-room collision validation.
4. Implement only the smallest coherent cache/service/validation changes proven by
   those regressions; regenerate only evidence genuinely affected by each change.
5. Freeze production code, start persistent immutable services, and launch isolated
   room workers to match allocated GPUs.
6. Track durable room checkpoints and projected completion time, intervening only on
   measured stalls. Run full certification and final Isaac validation once all
   required room finals are present.

## E. Projection bands

| Strategy | Projection | Assumptions | Risk/confidence |
| --- | --- | --- | --- |
| Current restart-heavy sequential workflow | Cannot responsibly promise 48 h | Repeated full preflight, cold starts, and furniture-specific recovery persist | High risk / low confidence |
| Low-risk workflow optimizations on one A10 | Conditional 48 h | Mean completed room ≤4.36 h; valid assets/checkpoints reused; no recurring systemic failures | Medium risk / medium confidence |
| Persistent services + isolated room workers + additional GPUs | Strongest path | One safe worker/GPU, immutable shared services, final strict certification | Medium engineering risk / highest confidence |

**Immediate decision:** finish Room 1 from checkpoint 003, but do not launch the
complete floor unchanged. Complete the read-only audit and focused regressions first,
then freeze the smallest validated production strategy.
