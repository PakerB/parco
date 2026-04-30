# PVRPWDP Rules

## Domain

PVRPWDP is a pickup routing problem with heterogeneous trucks and drones. Customers provide perishable goods. Vehicles pick up goods from customers and must return them to the depot before freshness expires.

The active implementation is:

- Env: `parco/envs/pvrpwdp/env.py`
- Generator: `parco/envs/pvrpwdp/generator.py`
- Embeddings: `parco/models/env_embeddings/pvrpwdp.py`
- GA baselines: `ga_pvrpwdp.py`, `ga_pvrpwdp_3.py`, `ga_pvrpwdp_v2.py`

## Node Indexing

- There are `M` depot slots and `N` customers.
- Depot slot for agent `i` has node index `i`.
- Customer `j` has env node index `M + j`.
- Agent `i` may only return to depot slot `i`.
- Do not collapse multiple depot slots into one node inside env/model code.

## Raw Generator Keys

`PVRPWDPGenerator` returns customer-only location data and agent attributes:

- `locs`: `[B, N, 2]`
- `depot`: `[B, 2]`
- `demand`: `[B, N]`
- `time_windows`: `[B, N+1, 2]` in current generator output
- `endurance`: `[B]`
- `freshness`: `[B]`
- `num_trucks`: `[B]`
- `num_drones`: `[B]`
- `capacity_per_vehicle`: `[B, M]`
- `speed_factor`: `[B, M]`

The current env expects loaded or adapted data to expose:

- `agents_capacity`: `[B, M]`
- `agents_speed`: `[B, M]`
- `agents_endurance`: `[B, M]`
- `time_windows`: customer time windows compatible with `_reset`
- `waiting_time`: `[B, N]`

When changing generator output, update env reset, data generation scripts, docs, and baselines together.

## Reset State

`PVRPWDPVEnv._reset` must produce:

- `locs`: `[B, M+N, 2]`
- `demand`: `[B, M+N]`
- `time_window`: `[B, M+N, 2]`
- `waiting_time`: `[B, M+N]`
- `current_node`: `[B, M]`
- `depot_node`: `[B, M]`
- `current_time`: `[B, M]`
- `trip_deadline`: `[B, M]`
- `used_capacity`: `[B, M]`
- `used_endurance`: `[B, M]`
- `action_mask`: `[B, M, M+N]`
- `visited`: `[B, M+N]`
- `done`: `[B]`

Depot time windows should be `[0, max_time]`. Depot waiting time and trip deadline reset should use `max_time`, not a hard-coded infinity.

## Step Physics

`_step`, `get_action_mask`, `_compute_operating_time`, and GA decoders must use identical rules:

- `step_distance = distance(previous_node, action)`.
- `travel_time = step_distance / agents_speed`.
- If `action == current_node`, travel time and state changes must be neutral.
- `current_time = max(arrival_time, selected_earliest)`.
- Waiting before leaving depot does not consume endurance.
- Waiting mid-trip consumes endurance.
- Returning to depot resets:
  - `used_capacity` to `0`
  - `used_endurance` to `0`
  - `trip_deadline` to `max_time`
- Picking up a customer updates:
  - `used_capacity += demand`
  - `used_endurance += travel_time + mid_trip_waiting`
  - `trip_deadline = min(old_deadline, current_time + waiting_time)`

## Action Mask

`get_action_mask` returns `True` for valid actions.

Mask constraints must be applied in this order or with equivalent semantics:

1. Unvisited customer constraint.
2. Capacity constraint.
3. Time-window constraint.
4. Freshness deadline constraint.
5. Endurance constraint.
6. Depot isolation and per-agent depot eye matrix.
7. Safety net for agents with no valid action.

Depot actions are special:

- Depot visits are allowed for agent `i` only at depot slot `i`.
- If all agents are already at depot and customers remain, depot actions should be blocked to force progress.
- If all customers are visited, depot actions should be available for padding or route closure.

## Done And Infeasibility

- `done=True` when all customers are visited.
- `done=True` is also acceptable when no agent can reach any unvisited customer.
- Do not mark depot slots as customers when counting completion.
- Keep infeasibility detection explicit; it protects training loops from infinite decoding.

## Reward

Reward is negative cost.

For `target="makespan"` in the current code:

- Cost uses adaptive big-M unvisited penalty plus makespan.
- Adaptive big-M must exceed any possible makespan for that instance so
  feasibility remains priority 1.

For `target="mincost"`:

- Cost is adaptive big-M unvisited penalty plus travel and rent cost.
- Trucks use distance-based travel cost.
- Drones use operating-time cost.
- Operating time is travel time plus mid-trip waiting, accumulated across the whole mission.
- Big-M must exceed any possible travel plus rent cost for that instance.

## Embedding Rules

`PVRPWDPInitEmbedding` and `PVRPWDPContextEmbedding` must normalize consistently:

- Demand and capacity share `demand_scaler`.
- Speed uses `speed_scaler`.
- Time features use `time_scaler`.
- Endurance should use `time_scaler` unless there is a deliberate experiment using max-endurance scaling.
- Polar features are computed from actual coordinates and then normalized by distance where appropriate.

Context features currently include:

- Current time.
- Remaining capacity.
- Remaining endurance.
- Time to depot.
- Time to deadline.
- Effective time limit: `min(remaining_endurance, time_to_deadline)`.
- Global visited-customer ratio.

## Epoch Data

`EpochDataEnvBase` loads multi-part epoch files lazily.

- Default pattern is `epoch_{epoch:02d}_{part:02d}.npz`.
- `SAMPLES_PER_FILE` is currently hard-coded as `20480`.
- Training scripts update `env.current_epoch`.
- Do not change epoch naming without updating `EpochDataCallback`, configs, docs, and data generation scripts.

## GA Parity Checklist

When editing env physics, update GA logic at the same time:

- Capacity check.
- Time-window service time.
- Mid-trip waiting endurance.
- Return-to-depot deadline check.
- Return-to-depot endurance check.
- Trip close/reset.
- Truck/drone cost split.
- Adaptive unserved penalty.
- Env replay through `evaluate_with_env`.
