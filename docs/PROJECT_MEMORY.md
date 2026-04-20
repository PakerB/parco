# Project Memory

## Architecture Summary

PARCO is a PyTorch/RL4CO project for parallel autoregressive multi-agent combinatorial optimization. The fork adds a PVRPWDP environment with trucks, drones, time windows, perishability, endurance, epoch-based data loading, and GA baselines.

Core flow:

1. Generator or epoch data creates raw TensorDict instances.
2. Env `_reset` builds node/action state.
3. Encoder embeds depot slots, agents, and customers.
4. Decoder uses context embedding plus action mask to produce logits.
5. Decoding strategy samples or greedily selects one action per agent.
6. Agent handler resolves conflicts.
7. Env `_step` updates state.
8. Env `_get_reward` scores final actions.
9. `PARCORLModule` computes SymNCO-style policy-gradient losses.

## Important Decisions

### Depot Slots Are Per Agent

Each agent has its own depot slot even if all depots share coordinates. This supports per-agent return masking and clean route padding.

### Action Mask Truth Convention

`True` means the action is valid. This convention is used by envs, decoders, and RL4CO logits processing.

### PVRPWDP Waiting-Time Semantics

Waiting before leaving depot does not consume endurance because departure can be delayed. Waiting after the trip starts consumes endurance because the vehicle is already committed.

### PVRPWDP Deadline Semantics

Trip deadline is the minimum freshness deadline of the goods currently carried. Returning to depot resets the deadline to `max_time`.

### PVRPWDP Cost Semantics

For money cost, trucks pay by distance and drones pay by operating time. Operating time is travel plus mid-trip waiting across the whole mission.

### Feasibility Priority

Unvisited customers must dominate all secondary cost terms. Adaptive big-M is preferred so this remains true across instance sizes and map scales.

### Epoch Data Loading

Training can load pre-generated epoch parts lazily. This avoids holding all epoch data in RAM. The loader assumes a fixed sample count per part.

## Active Risk Areas

- `tests/` is deleted in the current worktree, so CI's `uv run pytest tests/*` may fail.
- The canonical local PVRPWDP test dataset is `data/test_data/test.npz`
  (`D:\k0d3\DATN\parco\data\test_data\test.npz`); it is input data, not a
  pytest test file.
- Several files contain mojibake Vietnamese comments; edit carefully and keep new docs UTF-8.
- `configs/env/pvrpwdp.yaml` is empty while `configs/env/pvrpwdp_epoch.yaml` references it.
- PVRPWDP generator output keys and env expected keys are not fully aligned in naming.
- Reward comments in `PVRPWDPVEnv._get_reward` do not fully match current implementation.
- `check_solution_validity` for PVRPWDP is still a stub.
- GA versions use different chromosome encodings; compare them carefully before porting changes.

## Extension Checklist

When adding or changing a problem environment:

- Add generator.
- Add env `_reset`, `_step`, `get_action_mask`, `_get_reward`, and specs.
- Register env in `parco/envs/__init__.py`.
- Add env embeddings in `parco/models/env_embeddings/`.
- Update configs.
- Add or update smoke tests.
- Add baseline/evaluation parity checks if applicable.

When changing PVRPWDP:

- Update env.
- Update embeddings if features or normalization changed.
- Update GA baseline parity.
- Update epoch data generation if raw keys changed.
- Update docs in `docs/PVRPWDP_RULES.md`.
- Run formatting and linting on changed files.
