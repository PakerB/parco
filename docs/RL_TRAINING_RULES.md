# RL Training Rules

## Current Training Stack

This repository currently trains PARCO with RL4CO and SymNCO-style losses:

- `PARCORLModule` extends `rl4co.models.zoo.symnco.SymNCO`.
- `PARCOPolicy` performs parallel autoregressive decoding.
- `PARCODecodingStrategy` computes log probabilities and resolves conflicts through `AgentHandler`.
- Main training signal uses reward, log likelihood, solution symmetricity, and optional invariance loss.

This is not a TRL language-model GRPO training project. GRPO and torchforge ideas should be adapted only as experiments unless the training architecture is intentionally changed.

## Metrics To Monitor

For PVRPWDP, always log or inspect:

- Mean reward.
- Feasible rate.
- Unvisited count or unvisited ratio.
- Makespan.
- Total cost when `target="mincost"`.
- Decoder steps.
- Halting ratio.
- Fraction of batches marked impossible or no-progress.
- NaN or inf count in reward, logits, logprobs, and action mask.

Do not evaluate progress from `loss` alone. In policy-gradient and GRPO-style training, loss can move in unintuitive directions while reward improves.

## Safe Training Defaults

For active PVRPWDP scripts:

- Use `agent_handler="highprob"` unless testing conflict behavior.
- Keep `use_padding_mode=True` for padded epoch data.
- Keep `normalize_endurance_by_max=False` unless running an explicit ablation.
- Keep gradient clipping enabled for GPU training.
- Prefer `precision="32-true"` when debugging NaN on consumer GPUs.
- Use `bf16-mixed` only after confirming stable rewards and masks.

## Dataset Rules

- Validate epoch files before long runs.
- Strip virtual nodes and agents before reset if `num_real_nodes` and `num_real_agents` exist.
- Do not let padded agents with speed `0` reach `_reset`; they can produce invalid `max_time`.
- Keep validation/test data format compatible with train data.
- When changing generator keys, update `resample_batch_padding`.

## Reward Design Rules

Reward must remain objective-aligned:

- Feasibility first: visiting all customers must dominate route cost.
- Efficiency second: optimize makespan or money cost only after feasibility.
- Use adaptive big-M rather than manually tuned fixed penalties when possible.
- Keep reward scale stable enough for policy-gradient training.
- Compare GA objective, env reward, and decoded policy reward after reward changes.

## GRPO-Inspired Rules

If adding GRPO-style optimization for routing policies:

- Define a group as multiple decoded rollouts for the same TensorDict instance.
- Reward functions should be composable:
  - feasibility reward
  - route cost reward
  - constraint violation penalty
  - diversity or entropy support if collapse appears
- Test each reward component independently before training.
- Track within-group reward standard deviation; zero variance means no useful GRPO signal.
- Start with small group sizes such as `4`; increase only after memory is stable.
- Keep action masks hard; do not let GRPO sample infeasible actions to "learn" constraints.

## torchforge-Inspired Rules

If adding torchforge experiments:

- Keep algorithm code separate from env/model infrastructure.
- Place experimental entrypoints under `scripts/` or `experiments/`, not inside core env classes.
- Treat PARCO rollout outputs as episodes:
  - prompt/state: initial TensorDict
  - completion/action sequence: `[B, M, L]`
  - reward: env reward plus diagnostic components
  - metadata: feasibility, steps, unvisited count, target
- Keep reference/baseline policy frozen when computing KL-like penalties.
- Document every experiment config and reward weight.

## Debugging Order

When training fails:

1. Check `action_mask.any(dim=-1).all()` after reset and after several steps.
2. Replay a short action sequence through env manually.
3. Compare `get_action_mask` feasibility with GA `_try_append_customer`.
4. Check reward for all-visited and partial-visited routes.
5. Reduce batch size, num augmentations, and max steps.
6. Run greedy decode before sampling decode.
7. Inspect generated data for impossible instances.

## Checkpoint Rules

- Do not save env and policy objects into checkpoints when scripts intentionally ignore them.
- Only `trainer.is_global_zero` should write final `.pth` weights in distributed runs.
- Keep checkpoint output under ignored/generated directories.

