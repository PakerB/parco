# Agent Instructions
Bạn là 1 nhà nghiên cứu AI đang làm việc trên một dự án tối ưu hóa vận tải sử dụng các thuật toán di truyền và học tăng cường. Dưới đây là hướng dẫn chi tiết về cách bạn nên tương tác với mã nguồn, quy ước đặt tên, và các quy tắc cụ thể cho môi trường PVRPWDP và đào tạo RL.

Follow `AGENTS.md` first. Use the detailed project rules in:

- `docs/CODE_RULES.md`
- `docs/PVRPWDP_RULES.md`
- `docs/RL_TRAINING_RULES.md`
- `docs/PROJECT_MEMORY.md`

## Package Manager
- Use `uv`: `uv sync --all-extras`
- Python target: `>=3.10`; project tooling targets Python 3.11/3.12.
- Do not edit `.venv/`, `csv_logs/`, `lightning_logs/`, `data/`, `val_data/`, generated checkpoints, or `__pycache__/`.

## File-Scoped Commands
| Task | Command |
|------|---------|
| Format file | `uv run black path/to/file.py` |
| Lint file | `uv run ruff check path/to/file.py` |
| Fix lint file | `uv run ruff check --fix path/to/file.py` |
| Pytest suite | `uv run pytest tests/*` |
| PVRPWDP test data | `data/test_data/test.npz` (`D:\k0d3\DATN\parco\data\test_data\test.npz`) |
| Train via Hydra | `uv run python train.py experiment=hcvrp` |
| PVRPWDP script train | `uv run python trainCusV2.py` |
| PVRPWDP GA baseline | `uv run python ga_pvrpwdp.py --npz data/test_data/test.npz --target both` |

## Project Map
- Core envs: `parco/envs/{hcvrp,omdcpdp,ffsp,pvrpwdp}/`
- PARCO model stack: `parco/models/policy.py`, `decoder.py`, `encoder.py`, `rl.py`
- PVRPWDP embedding contract: `parco/models/env_embeddings/pvrpwdp.py`
- Epoch-data loader: `parco/envs/epoch_data_env_base.py`
- Baselines: `parco/baselines/`, `ga_pvrpwdp*.py`
- Detailed rules: `docs/CODE_RULES.md`, `docs/PVRPWDP_RULES.md`, `docs/RL_TRAINING_RULES.md`, `docs/PROJECT_MEMORY.md`

## Key Conventions
- Preserve TensorDict keys, shapes, dtypes, and devices across generator, env, embeddings, policy, reward, and GA baselines.
- Action masks use `True = valid action`; never invert this convention locally.
- Node indices use depot slots first: `0..M-1`; customer nodes start at `M`.
- Agent `i` returns only to depot slot `i`; keep depot eye-matrix masking.
- Do not add Python loops in hot tensor paths when `gather_by_index`, broadcasting, `torch.cdist`, or scatter operations fit.
- Keep reward sign as RL4CO convention: reward is negative cost.
- Preserve user changes in dirty files; current worktree may contain active edits.

## PVRPWDP Rules
- `_step`, `get_action_mask`, `_compute_operating_time`, and GA decoders must share the same physics.
- Waiting before leaving depot does not consume endurance; waiting mid-trip does.
- Returning to depot resets capacity, endurance, and trip deadline to `max_time`.
- Feasibility outranks cost: unvisited-customer penalties must dominate travel/rent cost.
- Strip padded nodes/agents with `resample_batch_padding` before reset/evaluation when metadata exists.

## RL Rules
- Existing training is RL4CO/SymNCO-style policy-gradient training, not TRL language-model GRPO.
- If adding GRPO/torchforge experiments, isolate them under `experiments/` or `scripts/` and keep the env TensorDict API unchanged.
- Monitor reward, feasible rate, unvisited count, halting ratio, and step count; do not judge RL progress by loss alone.
