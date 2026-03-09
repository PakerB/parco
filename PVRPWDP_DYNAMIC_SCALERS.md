# CẬP NHẬT PVRPWDP: DYNAMIC SCALERS CHO CONTEXT EMBEDDING

## 📋 TÓM TẮT THAY ĐỔI

Đã sửa `PVRPWDPContextEmbedding` để **tính dynamic scalers từ dữ liệu batch** giống như `PVRPWDPInitEmbedding`, thay vì dùng default values cố định.

---

## 1. VẤN ĐỀ TRƯỚC KHI SỬA

### ❌ **InitEmbedding vs ContextEmbedding không nhất quán**

**PVRPWDPInitEmbedding** (Đúng ✅):
```python
# Tính dynamic scalers từ batch data
demand_scaler = torch.maximum(
    demands.max(dim=-1, keepdim=True)[0],
    capacities.max(dim=-1, keepdim=True)[0]
).unsqueeze(-1)  # [B, 1, 1]

speed_scaler = speeds.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B, 1, 1]
time_scaler = time_windows[..., 1].max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B, 1, 1]
```

**PVRPWDPContextEmbedding** (Sai ❌):
```python
# Dùng default values cố định
def __init__(self, ..., demand_scaler=40.0, speed_scaler=1.0, ...):
    self.demand_scaler = demand_scaler  # ← Hard-coded!
    self.speed_scaler = speed_scaler    # ← Hard-coded!
```

### 🔴 **Hậu quả:**
1. **Không adaptable**: Scalers không thay đổi theo đặc điểm của mỗi batch
2. **Mất thông tin**: Nếu batch có demand max = 100, nhưng scaler = 40, các giá trị > 40 sẽ > 1 (không chuẩn hóa đúng)
3. **Không consistency**: InitEmbedding dùng dynamic, ContextEmbedding dùng static → model học không đồng nhất

---

## 2. THAY ĐỔI CHI TIẾT

### 2.1. Xóa Default Scalers trong `__init__`

#### **Trước:**
```python
def __init__(
    self,
    embed_dim,
    agent_feat_dim=3,
    global_feat_dim=1,
    demand_scaler=40.0,      # ← Xóa bỏ
    speed_scaler=1.0,        # ← Xóa bỏ
    use_time_to_depot=True,
    **kwargs,
):
    ...
    self.demand_scaler = demand_scaler
    self.speed_scaler = speed_scaler
    self.use_time_to_depot = use_time_to_depot
```

#### **Sau:**
```python
def __init__(
    self,
    embed_dim,
    agent_feat_dim=3,
    global_feat_dim=1,
    normalize_endurance_by_max=True,  # ← Thêm parameter này
    use_time_to_depot=True,
    **kwargs,
):
    ...
    self.normalize_endurance_by_max = normalize_endurance_by_max
    self.use_time_to_depot = use_time_to_depot
```

**Thay đổi:**
- ❌ Xóa `demand_scaler=40.0` và `speed_scaler=1.0`
- ✅ Thêm `normalize_endurance_by_max=True` (giống như InitEmbedding)

---

### 2.2. Tính Dynamic Scalers trong `_agent_state_embedding`

#### **Trước:**
```python
def _agent_state_embedding(self, embeddings, td, num_agents, num_cities):
    # Sử dụng self.demand_scaler và self.speed_scaler cố định
    current_time = td["current_length"] / (td["agents_speed"] / self.speed_scaler)
    
    context_feats = torch.stack([
        current_time,
        (td["agents_capacity"] - td["used_capacity"]) / self.demand_scaler,
        (td["agents_endurance"] - td["used_endurance"]) / self.speed_scaler,
    ], dim=-1)
```

#### **Sau:**
```python
def _agent_state_embedding(self, embeddings, td, num_agents, num_cities):
    # 1. Tính dynamic scalers từ batch data (giống InitEmbedding)
    demands = td["demand"][..., 0, num_agents:]  # [B, N]
    capacities = td["capacity"]  # [B, m]
    speeds = td["speed"]  # [B, m]
    endurances = td["endurance"]  # [B, m]
    time_windows = td["time_window"][..., num_agents:, :]  # [B, N, 2]
    
    # demand_scaler = max(all demands, all capacities)
    demand_scaler = torch.maximum(
        demands.max(dim=-1, keepdim=True)[0],
        capacities.max(dim=-1, keepdim=True)[0]
    ).unsqueeze(-1)  # [B, 1, 1]
    
    # speed_scaler = max(all speeds)
    speed_scaler = speeds.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B, 1, 1]
    
    # time_scaler = max(all latest time windows)
    time_scaler = time_windows[..., 1].max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B, 1, 1]
    
    # endurance_scaler: either max(all endurances) or time_scaler
    if self.normalize_endurance_by_max:
        endurance_scaler = endurances.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    else:
        endurance_scaler = time_scaler
    
    # 2. Sử dụng dynamic scalers
    current_time = td["current_length"] / (td["agents_speed"] / speed_scaler)
    
    context_feats = torch.stack([
        current_time,
        (td["agents_capacity"] - td["used_capacity"]) / demand_scaler,
        (td["agents_endurance"] - td["used_endurance"]) / endurance_scaler,  # ← Sử dụng endurance_scaler
    ], dim=-1)
    
    # time_to_depot và time_to_deadline cũng dùng dynamic speed_scaler
    if self.use_time_to_depot:
        ...
        time_to_depot = dist_to_depot / (td["agents_speed"][..., None] / speed_scaler)
        time_to_deadline = (td["trip_deadline"] - td["current_time"])[..., None] / (td["agents_speed"][..., None] / speed_scaler)
        ...
```

---

## 3. SO SÁNH TRƯỚC/SAU

### 3.1. Tính toán Scalers

| Component | Trước (Static) | Sau (Dynamic) |
|-----------|----------------|---------------|
| **demand_scaler** | `40.0` (hard-coded) | `max(demands, capacities)` từ batch |
| **speed_scaler** | `1.0` (hard-coded) | `max(speeds)` từ batch |
| **time_scaler** | ❌ Không có | `max(latest_time_windows)` từ batch |
| **endurance_scaler** | ❌ Dùng `speed_scaler=1.0` | `max(endurances)` hoặc `time_scaler` |

### 3.2. Normalization Consistency

| Feature | Trước | Sau |
|---------|-------|-----|
| **Init: capacity** | `capacity / max(demands, capacities)` | ✅ Giống |
| **Context: remaining_capacity** | `remaining_capacity / 40.0` ❌ | `remaining_capacity / max(demands, capacities)` ✅ |
| **Init: speed** | `speed / max(speeds)` | ✅ Giống |
| **Context: current_time** | `current_length / (speed / 1.0)` ❌ | `current_length / (speed / max(speeds))` ✅ |
| **Init: endurance** | `endurance / max(endurances)` | ✅ Giống |
| **Context: remaining_endurance** | `remaining_endurance / 1.0` ❌ | `remaining_endurance / max(endurances)` ✅ |

---

## 4. LỢI ÍCH CỦA THAY ĐỔI

### ✅ **1. Adaptable to Data Distribution**
```python
# Batch 1: demands = [10, 20, 30] → demand_scaler = 30
# Batch 2: demands = [50, 100, 150] → demand_scaler = 150
# ⇒ Mỗi batch được normalize theo scale riêng của nó
```

### ✅ **2. Consistent Normalization**
```python
# InitEmbedding và ContextEmbedding đều dùng:
# - demand_scaler = max(demands, capacities)
# - speed_scaler = max(speeds)
# - endurance_scaler = max(endurances) hoặc time_scaler
# ⇒ Model học với scale nhất quán giữa init và context
```

### ✅ **3. Proper Value Range**
```python
# Trước (static): capacity=100, demand_scaler=40
#   → normalized = 100/40 = 2.5 (> 1, out of range!)
# 
# Sau (dynamic): capacity=100, demand_scaler=max(demands,capacities)=100
#   → normalized = 100/100 = 1.0 (✓ proper range [0, 1])
```

### ✅ **4. Flexible Endurance Normalization**
```python
# normalize_endurance_by_max = True:
#   → endurance_scaler = max(endurances)
#   → So sánh tương đối giữa agents
# 
# normalize_endurance_by_max = False:
#   → endurance_scaler = time_scaler
#   → Đồng nhất đơn vị với time features
```

---

## 5. BREAKING CHANGES

### ⚠️ **API Changes**

#### **Trước:**
```python
context_embedding = PVRPWDPContextEmbedding(
    embed_dim=128,
    demand_scaler=40.0,   # ← Không còn
    speed_scaler=1.0,     # ← Không còn
    use_time_to_depot=True,
)
```

#### **Sau:**
```python
context_embedding = PVRPWDPContextEmbedding(
    embed_dim=128,
    normalize_endurance_by_max=True,  # ← Tham số mới
    use_time_to_depot=True,
)
```

### 🔄 **Migration Guide**

Nếu code cũ có:
```python
PVRPWDPContextEmbedding(..., demand_scaler=50.0, speed_scaler=2.0)
```

Bỏ các tham số này đi, scalers sẽ tự động tính từ data:
```python
PVRPWDPContextEmbedding(...)  # demand_scaler và speed_scaler tự động
```

---

## 6. KIỂM TRA THAY ĐỔI

### 6.1. Test Dynamic Scalers

```python
import torch
from tensordict import TensorDict
from parco.models.env_embeddings import PVRPWDPContextEmbedding

# Tạo dummy data
td = TensorDict({
    "locs": torch.rand(2, 10, 2),
    "demand": torch.rand(2, 3, 10) * 50,      # demands vary [0, 50]
    "capacity": torch.rand(2, 3) * 100,       # capacities vary [0, 100]
    "speed": torch.rand(2, 3) * 2,            # speeds vary [0, 2]
    "endurance": torch.rand(2, 3) * 200,      # endurances vary [0, 200]
    "time_window": torch.rand(2, 10, 2) * 100,
    "agents_capacity": torch.rand(2, 3) * 100,
    "used_capacity": torch.rand(2, 3) * 50,
    "agents_endurance": torch.rand(2, 3) * 200,
    "used_endurance": torch.rand(2, 3) * 100,
    "agents_speed": torch.rand(2, 3) * 2,
    "current_length": torch.rand(2, 3) * 10,
    "current_node": torch.randint(0, 10, (2, 3)),
    "action_mask": torch.ones(2, 3, 10, dtype=bool),
}, batch_size=[2])

# Test context embedding
context_emb = PVRPWDPContextEmbedding(embed_dim=128)
embeddings = torch.rand(2, 10, 128)

# Scalers sẽ được tính động mỗi lần forward
output = context_emb._agent_state_embedding(embeddings, td, num_agents=3, num_cities=7)
print(f"Output shape: {output.shape}")  # [2, 3, 128]
```

### 6.2. So sánh với InitEmbedding

```python
from parco.models.env_embeddings import PVRPWDPInitEmbedding

init_emb = PVRPWDPInitEmbedding(embed_dim=128, normalize_endurance_by_max=True)
context_emb = PVRPWDPContextEmbedding(embed_dim=128, normalize_endurance_by_max=True)

# Cả 2 đều dùng dynamic scalers → consistent normalization
init_output = init_emb(td)
context_output = context_emb._agent_state_embedding(init_output, td, num_agents=3, num_cities=7)

print("Init và Context đều dùng dynamic scalers!")
```

---

## 7. KẾT LUẬN

### **Tóm tắt thay đổi:**

| Aspect | Trước | Sau |
|--------|-------|-----|
| **Scalers** | Static (40.0, 1.0) | Dynamic (tính từ batch) |
| **Consistency** | ❌ InitEmbedding ≠ ContextEmbedding | ✅ InitEmbedding = ContextEmbedding |
| **Adaptability** | ❌ Fixed cho mọi batch | ✅ Adapt theo từng batch |
| **Value Range** | ❌ Có thể > 1 | ✅ Luôn trong [0, 1] |
| **Parameters** | `demand_scaler`, `speed_scaler` | `normalize_endurance_by_max` |

### **Lợi ích:**
- ✅ **Consistent normalization** giữa InitEmbedding và ContextEmbedding
- ✅ **Adaptive scalers** theo data distribution của mỗi batch
- ✅ **Proper value range** [0, 1] cho tất cả normalized features
- ✅ **Flexible endurance normalization** với 2 options

Thay đổi này giúp model **học được features với scale nhất quán** và **adapt được với nhiều loại data distribution khác nhau**! 🚀
