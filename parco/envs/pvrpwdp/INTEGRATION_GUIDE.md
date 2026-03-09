# PVRPWDP Integration Checklist

Để tích hợp PVRPWDP environment vào PARCO, cần tạo/sửa **8 files**.

---

## ✅ CHECKLIST CÁC FILE CẦN TẠO/SỬA

### **Phase 1: Environment Foundation** ⭐⭐⭐⭐⭐

- [ ] **File 1**: `parco/envs/pvrpwdp/__init__.py` [TẠO MỚI]
  ```python
  from .env import PVRPWDPV2Env
  from .generator import PVRPWDPGenerator
  __all__ = ["PVRPWDPV2Env", "PVRPWDPGenerator"]
  ```

- [ ] **File 2**: `parco/envs/pvrpwdp/generator.py` [TẠO MỚI]
  - Tạo class `PVRPWDPGenerator`
  - Generate: `depot`, `locs`, `demand`, `time_window`, `waiting_time`, `capacity`, `speed`, `freshness`
  - Tham khảo: `parco/envs/hcvrp/generator.py`

- [ ] **File 3**: `parco/envs/__init__.py` [SỬA - Line 4]
  ```python
  # Sửa: from .pvrpwdp.env import PVRPWDPVEnv
  # Thành: from .pvrpwdp.env import PVRPWDPV2Env
  ```

---

### **Phase 2: Model Embeddings** ⭐⭐⭐⭐⭐

- [ ] **File 4**: `parco/models/env_embeddings/pvrpwdp.py` [TẠO MỚI]
  - **Class 1**: `PVRPWDPInitEmbedding(nn.Module)`
    - Embed: depot (x,y), agents (x,y,capacity,speed,freshness), customers (x,y,demand,earliest,latest,waiting_time)
    - Output: `[B, m+N, embed_dim]`
  - **Class 2**: `PVRPWDPContextEmbedding(BaseMultiAgentContextEmbedding)`
    - `_agent_state_embedding()`: current_time, remaining_capacity, remaining_endurance, trip_deadline, time_to_depot
    - `_global_state_embedding()`: visited_ratio
  - Tham khảo: `parco/models/env_embeddings/hcvrp.py`

- [ ] **File 5**: `parco/models/env_embeddings/__init__.py` [SỬA - 3 vị trí]
  - **Line ~6**: Thêm import
    ```python
    from .pvrpwdp import PVRPWDPContextEmbedding, PVRPWDPInitEmbedding
    ```
  - **Line ~31**: Thêm vào `env_init_embedding()` registry
    ```python
    "pvrpwdp": PVRPWDPInitEmbedding,
    ```
  - **Line ~43**: Thêm vào `env_context_embedding()` registry
    ```python
    "pvrpwdp": PVRPWDPContextEmbedding,
    ```

---

### **Phase 3: Configuration** ⭐⭐⭐⭐

- [ ] **File 6**: `configs/env/pvrpwdp.yaml` [TẠO MỚI]
  - `_target_: parco.envs.pvrpwdp.PVRPWDPV2Env`
  - `name: pvrpwdp`
  - `generator_params`: num_loc, num_agents, time windows, perishability, capacity/speed/endurance ranges
  - Tham khảo: `configs/env/hcvrp.yaml`

- [ ] **File 7**: `configs/experiment/pvrpwdp.yaml` [TẠO MỚI]
  - Override: model, env, callbacks, trainer, logger
  - `env_name: "${env.name}"` → "pvrpwdp"
  - Embedding configs: `init_embedding_kwargs`, `context_embedding_kwargs`
  - Tham khảo: `configs/experiment/hcvrp.yaml`

---

### **Phase 4: Visualization (Optional)** ⭐⭐⭐

- [ ] **File 8**: `parco/envs/pvrpwdp/render.py` [TẠO MỚI]
  - Function `render(td, actions, ax)` để vẽ routes, time windows, deadlines
  - Tham khảo: `parco/envs/hcvrp/render.py`

---

## 🧪 TEST TỪNG PHASE

### **Test Phase 1**: Generator & Environment
```python
from parco.envs.pvrpwdp import PVRPWDPV2Env, PVRPWDPGenerator
gen = PVRPWDPGenerator(num_loc=10, num_agents=3)
td = gen(batch_size=[2])
assert "time_window" in td.keys()
```

### **Test Phase 2**: Embeddings
```python
from parco.models.env_embeddings import env_init_embedding, env_context_embedding
init_emb = env_init_embedding("pvrpwdp", {"embed_dim": 128})
ctx_emb = env_context_embedding("pvrpwdp", {"embed_dim": 128})
h = init_emb(td)  # [B, m+N, 128]
h_ctx = ctx_emb(h, td)  # [B, m, 128]
```

### **Test Phase 3**: Training
```bash
python train.py experiment=pvrpwdp \
  generator_params.num_loc=10 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10
```

---

## ⚠️ COMMON ISSUES

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pvrpwdp` | Kiểm tra `__init__.py` và restart kernel |
| `KeyError: 'pvrpwdp' not in registry` | Kiểm tra đã thêm vào 3 vị trí trong `env_embeddings/__init__.py` |
| `RuntimeError: shape mismatch` | Kiểm tra feature dimensions (polar=8, no polar=6 cho customers) |
| `KeyError: 'time_window'` | Kiểm tra generator return đủ keys |
| `FileNotFoundError: pvrpwdp.yaml` | Tạo config file trong `configs/env/` |

---

## 📚 REFERENCE FILES

- **Generator**: `parco/envs/hcvrp/generator.py`
- **Init Embedding**: `parco/models/env_embeddings/hcvrp.py` (lines 11-98)
- **Context Embedding**: `parco/models/env_embeddings/hcvrp.py` (lines 101-149)
- **Base Context**: `parco/models/env_embeddings/communication.py`
- **Env Config**: `configs/env/hcvrp.yaml`
- **Exp Config**: `configs/experiment/hcvrp.yaml`

---

## 🎯 NORMALIZATION CONSTANTS

```python
demand_scaler = 40.0       # Demand normalization
speed_scaler = 1.0         # Speed normalization
time_scaler = 100.0        # Time windows & waiting_time normalization
```

---

## ✅ VALIDATION CHECKLIST

- [ ] Environment import works: `from parco.envs.pvrpwdp import PVRPWDPV2Env`
- [ ] Generator produces all keys: `depot`, `locs`, `demand`, `time_window`, `waiting_time`, `capacity`, `speed`, `freshness`
- [ ] Init embedding shape correct: `[B, m+N, 128]`
- [ ] Context embedding shape correct: `[B, m, 128]`
- [ ] Config loads: `python train.py experiment=pvrpwdp --cfg job`
- [ ] Action mask has valid actions: `mask.sum() > 0`
- [ ] Step function works: `td = env.step(td)["next"]`
- [ ] Training runs: End-to-end test with 1 epoch

---

**Version**: 1.0 | **Ngày**: 2026-03-08 | **Tài liệu chi tiết**: `parco/envs/pvrpwdp/README.md`
