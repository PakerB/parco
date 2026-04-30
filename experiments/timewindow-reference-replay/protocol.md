# Protocol H1/H2: Time Window Reference Replay Verifier

## Hypothesis

If time windows are generated from a no-time-window optimal route, then replaying
that reference route and generating TW around its service times will preserve
hard-cover feasibility while still allowing controllable TW difficulty.

## Prediction

Compared with the current TW generation/checking pipeline, a reference replay
verifier will:

- identify instances where full coverage is impossible due to TW generation;
- separate data infeasibility from GA/RL solver failure;
- enable difficulty profiles with measurable slack margins;
- improve trust in GA and PARCO comparisons.

## Planned Implementation

Add a script:

```text
scripts/validate_pvrpwdp_timewindows.py
```

Inputs:

- `--npz`
- `--offset`
- `--num-batches`
- optional `--reference-actions` or route source if available
- `--target`

Outputs:

- CSV diagnostics under an ignored/generated output directory or user-selected
  output path.
- Console summary for quick checks.

Core diagnostics:

- number of real nodes/agents after `resample_batch_padding`;
- initial serviceable customer ratio;
- per-constraint initial failure counts: capacity, TW, deadline, endurance;
- TW width min/mean/max;
- if reference route exists:
  - reference replay feasible;
  - service time inside TW;
  - freshness/deadline violations;
  - endurance violations;
  - capacity violations;
  - unvisited count.

## Acceptance Criteria

- The verifier runs on a dataset with offset metadata, e.g.
  `data/test_data/test.npz`, with offset 1 and 10 batches.
- It does not modify `.npz` data.
- It uses the same env reset/mask physics as PVRPWDP.
- It clearly reports whether failures are data infeasibility or route quality.

## Suggested First Command

```powershell
uv run python scripts/validate_pvrpwdp_timewindows.py `
  --npz data/test_data/test.npz `
  --offset 1 `
  --num-batches 10 `
  --target makespan
```
