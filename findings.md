# Findings: PVRPWDP Time Window

## Current Understanding

The core problem is not only that time windows make decoding harder. The bigger
issue is that time windows can make full customer coverage impossible unless the
dataset generation process preserves a feasible reference schedule.

For datasets generated from an optimal no-time-window route, the correct
invariant is:

```text
reference_route feasible under no-TW physics
AND generated_time_windows contain service_ref[j]
AND reference_route replay remains feasible after TW is added
```

If this invariant is not enforced, the benchmark becomes optional-customer
PVRPWDP. In that case unserved customers must be part of the objective and
reported explicitly, not treated as a solver bug.

## Patterns And Insights

- Binary action masks are necessary but insufficient for learning TW behavior.
  The model needs continuous slack information to know which valid actions are
  urgent.
- Adaptive big-M penalties are the right reward shape because feasibility must
  dominate makespan or money cost.
- HGS-style giant-tour plus split decoding matches TW routing better than a pure
  greedy permutation decoder or separator round-robin encoding.
- A verifier is needed before training, otherwise training failures can be caused
  by impossible data instead of policy weakness.
- Related VRPTW tools separate route structure from schedule diagnostics:
  wait time, slack, time warp, start time, and end time are first-class route
  statistics. PVRPWDP should log the same diagnostics, plus freshness and
  endurance slack.
- Literature on dropped visits/profitable VRPTW supports the optional-customer
  interpretation, but hard-cover PVRPWDP must instead reject or repair infeasible
  generated instances.
- PDPTW/VRPSPDTW work points toward ALNS/memetic/HGS repair and local search,
  especially destroy-repair and fast insertion evaluation.
- The local files named `test_data/test.npz` and `data/test_data/test.npz` are
  not equivalent. `test_data/test.npz` has 100 customers and 12 agents with no
  `offsets` metadata; `data/test_data/test.npz` has padding metadata and
  `offsets`, and offset 1 strips to 28 customers and 5 agents.

## Initial Verifier Results

Implemented `scripts/validate_pvrpwdp_timewindows.py`.

Results:

- `test_data/test.npz --offset 1` fails by design because the file has no
  `offsets` or `offset` key.
- `data/test_data/test.npz --offset 1 --num-batches 10`:
  - average initial serviceable ratio: `0.814286`;
  - average initial unserviceable count: `5.2` customers;
  - min/max initial serviceable customers: `19 / 27` out of `28`;
  - env/manual initial mask match rate: `1.0`;
  - reference routes unavailable in the `.npz`.
- `test_data/test.npz --num-batches 10`:
  - average initial serviceable ratio: `1.0`;
  - all first 10 instances have `100 / 100` initially serviceable customers;
  - reference routes unavailable in the `.npz`.

## Lessons And Constraints

- Always call `resample_batch_padding` before reset/evaluation when metadata
  exists.
- Keep `_step`, `get_action_mask`, `_compute_operating_time`, and GA decoders
  physically aligned.
- Waiting before leaving depot does not consume endurance; waiting mid-trip
  consumes endurance.
- `target="makespan"` now uses adaptive big-M unvisited penalty plus makespan.
- `target="mincost"` already uses adaptive big-M plus travel/rent cost.
- Torchforge/GRPO is a possible experiment design, not a replacement for the
  current RL4CO/SymNCO stack.

## Open Questions

- Where are the no-time-window optimal routes/actions stored for the current
  `test_data/test.npz` instances?
- Should the primary benchmark be hard-cover PVRPWDP or optional-customer
  PVRPWDP?
- How tight can generated TW be while preserving reference-route feasibility?
- Does `ga_pvrpwdp_hgs_split.py` already outperform the older GA versions on
  offset 1, 10-batch makespan evaluation?
- Which slack feature gives the best improvement: TW slack, freshness/deadline
  slack, endurance slack, or aggregate feasible-customer counts?
- Can we define a PVRPWDP analogue of PyVRP `time_warp` that separately tracks
  TW lateness, freshness deadline excess, endurance excess, and capacity excess
  for GA soft-infeasible search?
- Which dataset should be treated as the main benchmark for the thesis:
  `test_data/test.npz` or `data/test_data/test.npz`?
