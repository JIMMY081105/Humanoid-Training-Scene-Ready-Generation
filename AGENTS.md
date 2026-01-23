# Agent instructions

## Multi-agent coordination (Codex + Claude, 2026-07-13)

Two agents work on this project concurrently. Ownership zones — do not write outside yours:

- **Codex**: the in-flight generation-pipeline files (scripts/run_single_room_worker.py,
  scripts/assemble_final_house_and_render.py, scripts/render_room_review_views.py,
  scripts/room_self_exam.py, scripts/validate_articulated_router.py, the four
  remote_jobs/TEMPLATE_*.sbatch files, tools/sage_scene_checker/*, _remote_patch/*,
  CODEX_SCENESMITH_FULL_QUALITY_PIPELINE.md), the sqz-side scenesmith-hts tree and
  generation runs, and SQZ VPN redials.
- **Claude**: the SQZ→ParaCloud migration (`/data/run01/scvj260/` on ParaCloud is
  Claude-managed — do not modify), local_setup/, AGENTS.md, and all other repo files.
  Claude reads sqz logs/evidence but does not write into /root/workspace/scenesmith-hts.
- The single A10 GPU on sqz belongs to Codex's generation runs; Claude's GPU work runs
  on ParaCloud only.
- A verified snapshot of scenesmith-hts + this repo exists on ParaCloud (2026-07-12);
  Claude re-syncs deltas after Codex's goals conclude — Codex does not need to sync
  anything to ParaCloud.

## ParaCloud ownership

Codex is authorized to create, modify, test, and launch files under:

`/data/run01/scvj260/codex_factory/`

Claude must not modify files inside this directory while Codex is working there.
Existing checkpoints and assets outside this directory are read-only unless explicitly
reassigned.

## SQZ cluster / VPN work — read this first

Before touching the SQZ VPN, SSH tunnel, or anything on the remote container, read
`CODEX_SQZ_REMOTE_HANDOFF.md` (local-only, gitignored), especially **§1a "VPN: how to
connect WITHOUT breaking the laptop's internet"**.

Non-negotiable rules (details and exact commands are in that file):

- The Windows VPN connection `SQZ` is **split-tunneled on purpose** — only
  `10.220.0.0/16` goes through it. Never set `SplitTunneling` to `False`, never add a
  broader route, never let it own the default route.
- The user's own internet and personal proxy (Misty / v2rayN) must keep working the
  entire time. After dialing, verify plain internet still works; if it broke,
  disconnect immediately and fix split-tunneling before redialing.
- Dial with `rasdial SQZ` (cached credential). Never write the VPN password into any
  file or guess it — if the cache is lost (error 691), ask the user.
- Announce any UAC/elevated prompt or network change to the user before causing it.
