# PARCO Code Rules

## Scope

These rules describe the local conventions for this PARCO fork after reading the source tree, including `parco/`, `configs/`, `scripts/`, `train*.py`, and `ga_pvrpwdp*.py`.

## Tooling

- Use `uv` for dependency management.
- Format Python with Black using `pyproject.toml`.
- Lint Python with Ruff using `pyproject.toml`.
- Keep line length compatible with Black's `90` character target.
- CI currently installs with `uv sync --all-extras` and runs `uv run pytest tests/*`.
- The current worktree has deleted `tests/` files; do not assume tests exist until that deletion is resolved.
- Canonical PVRPWDP test data is `data/test_data/test.npz`
  (`D:\k0d3\DATN\parco\data\test_data\test.npz`).
- Do not run pytest directly on `.npz` data files. Pass the test data path to
  evaluation scripts or GA baselines, for example
  `uv run python ga_pvrpwdp.py --npz data/test_data/test.npz --target both`.

## Repository Boundaries

- Source package code lives under `parco/`.
- Hydra config lives under `configs/`.
- One-off runnable scripts live at repo root and under `scripts/`.
- Generated artifacts live under `data/`, `val_data/`, `csv_logs/`, `lightning_logs/`, checkpoints, and caches.
- Do not commit generated `.npz`, `.ckpt`, `.pth`, CSV logs, or notebook execution noise unless the user explicitly asks.

## TensorDict Contract

- Treat TensorDict keys as API contracts between generator, env, embeddings, policy, trainer, and baselines.
- Keep tensors on the same device as the incoming TensorDict.
- Keep batch dimension first and use TensorDict `batch_size` consistently.
- Add new keys in `_reset` only when every downstream consumer either uses them or safely ignores them.
- Preserve these common env keys:
  - `locs`
  - `demand`
  - `current_node`
  - `depot_node`
  - `current_length`
  - `visited`
  - `action_mask`
  - `done`
  - `reward`
- For PVRPWDP also preserve:
  - `time_window`
  - `waiting_time`
  - `current_time`
  - `trip_deadline`
  - `max_time`
  - `used_capacity`
  - `used_endurance`
  - `agents_capacity`
  - `agents_speed`
  - `agents_endurance`

## Shape Rules

- Use `B` for batch size, `M` for agents, `N` for customers, and `M+N` for depot slots plus customers.
- Depot slots always occupy node indices `0..M-1`.
- Customers always occupy node indices `M..M+N-1`.
- Multi-agent action masks are shaped `[B, M, M+N]`.
- Multi-agent actions are shaped `[B, M]` per step and `[B, M, L]` for full decoded sequences.
- If data has padding metadata, call `resample_batch_padding` before environment reset or GA evaluation.

## Environment Rules

- `_reset` creates a compact TensorDict state from raw generator or loaded data.
- `_step` must be pure tensor state transition logic; avoid side effects except TensorDict updates.
- `get_action_mask` must return `True` for valid actions.
- `done` should become true when all real customers are visited or no progress is possible.
- `env.get_reward(td, actions)` must accept the final TensorDict and the full action tensor.
- Reward sign follows RL4CO: maximize reward equals minimize cost.

## Model Rules

- `PARCOPolicy` owns the decode loop and calls `env.step(td)["next"]`.
- `PARCODecoder` consumes `td["action_mask"]` directly.
- `AgentHandler` resolves same-customer conflicts; `highprob` is the current PVRPWDP default.
- `use_init_logp`, `mask_handled`, and replacement behavior affect policy-gradient credit assignment; change them only with a targeted training reason.
- Context embeddings must normalize features using scalers derived from the current batch or explicit config; avoid hard-coded constants unless matching an existing env.

## Baseline Rules

- OR-Tools baseline targets OMDCPDP.
- `ga_pvrpwdp.py` and `ga_pvrpwdp_3.py` use permutation chromosomes and greedy insertion.
- `ga_pvrpwdp_v2.py` uses separator genes (`SEP = -1`) and round-robin vehicle assignment.
- GA solvers must mirror PVRPWDP env physics before comparing their objective to `env.get_reward`.
- When evaluating GA routes in the env, pad each vehicle route by repeating its last action.

## Performance Rules

- Prefer vectorized `torch` operations in env/model hot paths.
- Use `gather_by_index` for node-indexed TensorDict lookup.
- Use `torch.cdist` for all-agent/all-node distance matrices.
- Avoid Python loops over batch or agents inside training-time env/model methods.
- Python loops are acceptable in offline GA baselines and data-generation scripts.

## Safety Rules

- Never replace user-edited dirty files unless explicitly requested.
- Do not silently change objective semantics.
- Do not mask infeasibility by only adding depot fallback without recording why the fallback exists.
- Avoid editing legacy FFSP code unless the request is specifically about `parco/tasks/ffsp_old/`.
- Keep generated-data file naming compatible with `EpochDataEnvBase`.
