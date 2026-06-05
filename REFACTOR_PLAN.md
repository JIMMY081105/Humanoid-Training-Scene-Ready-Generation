# Refactor Plan

This repository is an integration layer around SceneSmith. The refactor work
should keep the upstream boundary clear, preserve the HPC workflow, and make
the local tools easier to validate without changing generated assets.

## Goals

- Keep standalone tooling readable and testable.
- Separate policy checks, path handling, and report formatting into small
  helpers where the current code already has natural boundaries.
- Add focused tests for edge cases that matter to scene generation and
  benchmark reproducibility.
- Improve contributor documentation for local validation and safe execution.
- Avoid broad rewrites of generated assets, SLURM templates, and upstream
  patch files unless a specific bug requires it.

## Commit Strategy

- Prefer one meaningful code or documentation change per commit.
- Pair behavior changes with a small unit test in the same area.
- Run the unit suite before pushing batches that touch Python code.
- Keep commit messages descriptive and short.

## Near-Term Refactor Queue

1. Document local development and validation commands.
2. Make no-paid-API environment detection reusable and directly tested.
3. Extract small adapter helpers for normalization and opening placement.
4. Name benchmark cache constants instead of leaving magic numbers inline.
5. Reuse benchmark metadata loading logic in statistics/report code.
6. Clarify config path resolution and preserve existing config behavior.
7. Tighten README navigation so the quality gates, benchmark, and renderer are
   easier to discover.

## Guardrails

- Do not vendor upstream SceneSmith source into this repository.
- Do not commit local output directories, credentials, or machine-specific
  paths.
- Do not weaken the no-paid-API guard or SLURM proxy preflight behavior.
- Do not change rendered reference images unless the render pipeline itself is
  intentionally updated and documented.
