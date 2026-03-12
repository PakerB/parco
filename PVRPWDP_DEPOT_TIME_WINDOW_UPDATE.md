# CẬP NHẬT DEPOT TIME WINDOW - PVRPWDP ENV

## 📋 TÓM TẮT THAY ĐỔI

Đã sửa `_reset()` trong `PVRPWDPVEnv` để **thay đổi time window của depot** từ `[0, 0]` thành `[0, max_time]`, với `max_time` được tính động dựa trên:

```
max_time = max(latest_i + travel_time_to_depot_with_min_speed)
```

---

## 1. VẤN ĐỀ TRƯỚC KHI SỬA

### ❌ **Depot Time Window = [0, 0]**

#### **Trước:**
```python
# Padding depot time_window as [0, 0] for each depot
tw_depot = torch.zeros(
    (*batch_size, num_agents, 2), dtype=torch.float32, device=device
)
time_window = torch.cat((tw_depot, td["time_window"]), -2)  # [B, m+N, 2]
```

**Ý nghĩa cũ:**
- Depot time window = `[0, 0]` nghĩa là depot **chỉ chấp nhận xe quay về tại thời điểm t=0**
- Điều này **không hợp lý** vì:
  - Xe không thể quay về depot tại t=0 (đã đi phục vụ khách hàng)
  - Constraint quá chặt, không cho phép xe quay về sau khi hoàn thành trip

### 🔴 **Hậu quả:**
1. **Không realistic**: Depot phải luôn mở cửa đón xe quay về, không phải chỉ ở t=0
2. **Conflict với action mask**: Action mask kiểm tra time window, nhưng depot [0,0] luôn vi phạm
3. **Khó train**: Model không học được khi nào nên quay depot

---

## 2. GIẢI PHÁP MỚI

### ✅ **Depot Time Window = [0, max_time]**

#### **Sau:**
```python
speeds = td["speed"]
speed_min = speeds.min(dim=-1, keepdim=True)[0]  # [B, 1]

# Calculate max_time for depot time window: max(latest_i + travel_time_to_depot_with_min_speed)
# Get customer locations and depot location
customer_locs = td["locs"]  # [B, N, 2]
depot_loc = td["depot"]  # [B, 2] or [B, 1, 2]
if depot_loc.ndim == 2:
    depot_loc = depot_loc.unsqueeze(-2)  # [B, 1, 2]

# Calculate distance from each customer to depot [B, N]
dist_to_depot = torch.norm(customer_locs - depot_loc, p=2, dim=-1)  # [B, N]

# Calculate travel time with minimum speed [B, N]
travel_time_to_depot = dist_to_depot / speed_min  # [B, N]

# Get latest time windows for customers [B, N]
customer_latest = td["time_window"][..., 1]  # [B, N]

# Calculate max_time = max(latest_i + travel_time_i) [B, 1]
max_time = (customer_latest + travel_time_to_depot).max(dim=-1, keepdim=True)[0]  # [B, 1]

# Padding depot time_window as [0, max_time] for each depot
# Shape: [B, m, 2] for depots, concat with [B, N, 2] for customers -> [B, m+N, 2]
tw_depot = torch.stack([
    torch.zeros((*batch_size, num_agents), dtype=torch.float32, device=device),  # earliest = 0
    max_time.expand(-1, num_agents)  # latest = max_time, expand [B, 1] -> [B, m]
], dim=-1)  # [B, m, 2]
time_window = torch.cat((tw_depot, td["time_window"]), -2)  # [B, m+N, 2]
```

---

## 3. CÔNG THỨC TÍNH `max_time`

### 3.1. **Intuition**

Depot phải **mở cửa đủ lâu** để đón tất cả xe quay về sau khi phục vụ khách hàng xa nhất, chậm nhất.

**Worst case scenario:**
- Xe phục vụ khách hàng `i` có `latest_i` muộn nhất
- Xe di chuyển với tốc độ **chậm nhất** (`speed_min`)
- Xe phải đi từ khách hàng `i` về depot

**Thời gian quay về muộn nhất:**
```
return_time_i = latest_i + travel_time_to_depot_with_min_speed
max_time = max(return_time_i for all customers i)
```

### 3.2. **Ví dụ minh họa**

#### **Setup:**
```python
# Customers
customer_0: location = (0.2, 0.3), time_window = [10, 50]
customer_1: location = (0.8, 0.7), time_window = [20, 80]
customer_2: location = (0.5, 0.9), time_window = [5, 30]

# Depot
depot: location = (0.5, 0.5)

# Agents
agent_0: speed = 2.0
agent_1: speed = 1.0  # slowest
speed_min = 1.0
```

#### **Tính toán:**

**1. Distance to depot:**
```python
dist_0 = norm((0.2, 0.3) - (0.5, 0.5)) = norm(-0.3, -0.2) ≈ 0.36
dist_1 = norm((0.8, 0.7) - (0.5, 0.5)) = norm(0.3, 0.2) ≈ 0.36
dist_2 = norm((0.5, 0.9) - (0.5, 0.5)) = norm(0, 0.4) = 0.40
```

**2. Travel time with min speed:**
```python
travel_time_0 = 0.36 / 1.0 = 0.36
travel_time_1 = 0.36 / 1.0 = 0.36
travel_time_2 = 0.40 / 1.0 = 0.40
```

**3. Return time for each customer:**
```python
return_time_0 = latest_0 + travel_time_0 = 50 + 0.36 = 50.36
return_time_1 = latest_1 + travel_time_1 = 80 + 0.36 = 80.36  ← MAX
return_time_2 = latest_2 + travel_time_2 = 30 + 0.40 = 30.40
```

**4. Max time:**
```python
max_time = max(50.36, 80.36, 30.40) = 80.36
```

**5. Depot time window:**
```
depot_tw = [0, 80.36]
```

**Ý nghĩa:** Depot mở cửa từ t=0 đến t=80.36, đủ để đón xe chậm nhất (xe với speed=1.0 phục vụ customer_1 có latest=80).

---

## 4. TẠI SAO DÙNG `speed_min`?

### 🤔 **Câu hỏi:** Tại sao không dùng speed của từng agent?

### 💡 **Trả lời:**

#### **1. Worst-case guarantee**
```python
# Nếu dùng speed của từng agent:
# - Agent nhanh (speed=2.0) → max_time nhỏ
# - Agent chậm (speed=1.0) không kịp quay về depot (vi phạm time window)

# Nếu dùng speed_min:
# - Tất cả agents đều đảm bảo quay về kịp (vì speed thực >= speed_min)
```

#### **2. Ví dụ minh họa**

**Setup:**
```python
customer_1: location = (0.8, 0.7), latest = 80
depot: location = (0.5, 0.5)
dist = 0.36

agent_0: speed = 2.0
agent_1: speed = 1.0
```

**Nếu dùng speed của agent_0 (fast agent):**
```python
travel_time = 0.36 / 2.0 = 0.18
max_time = 80 + 0.18 = 80.18
depot_tw = [0, 80.18]

# Agent 1 (slow) phục vụ customer_1:
# - Arrive at customer_1 at t=80
# - Travel time to depot = 0.36 / 1.0 = 0.36
# - Arrive at depot at t=80.36
# - Depot closes at t=80.18 → VIOLATED! ❌
```

**Nếu dùng speed_min:**
```python
travel_time = 0.36 / 1.0 = 0.36
max_time = 80 + 0.36 = 80.36
depot_tw = [0, 80.36]

# Agent 1 (slow) phục vụ customer_1:
# - Arrive at depot at t=80.36
# - Depot closes at t=80.36 → OK! ✅

# Agent 0 (fast) phục vụ customer_1:
# - Arrive at depot at t=80.18
# - Depot closes at t=80.36 → OK! ✅
```

**Kết luận:** Dùng `speed_min` đảm bảo **tất cả agents** đều có thể quay về depot kịp time window.

---

## 5. SO SÁNH TRƯỚC/SAU

| Aspect | Trước (TW = [0, 0]) | Sau (TW = [0, max_time]) |
|--------|---------------------|--------------------------|
| **Earliest time** | 0 | 0 (giống) |
| **Latest time** | 0 | `max(latest_i + travel_time_i)` |
| **Có realistic?** | ❌ Không (xe không thể về t=0) | ✅ Có (depot mở cửa đủ lâu) |
| **Action mask** | ❌ Conflict (depot luôn vi phạm TW) | ✅ Consistent (depot trong TW) |
| **Train model** | ❌ Khó học | ✅ Dễ học |
| **Adaptable** | ❌ Fixed [0, 0] | ✅ Dynamic theo data |

---

## 6. ẢNH HƯỞNG ĐÃN ACTION MASK

### 6.1. **Action Mask Logic (trong `get_action_mask`)**

```python
# Time window constraint: service time must <= latest time
latest_times = td["time_window"][..., 1].unsqueeze(-2)  # [B, 1, m+N]
within_time_window = service_time <= latest_times  # [B, m, m+N]
action_mask &= within_time_window
```

### 6.2. **Trước đây (depot TW = [0, 0]):**

```python
# Depot latest = 0
# Service time at depot (when returning) = current_time (e.g., 50)
# Check: 50 <= 0 → False ❌
# → Depot bị mask (không thể chọn) khi current_time > 0
```

**Hậu quả:** Xe không thể quay depot sau khi đã đi, vì vi phạm time window.

### 6.3. **Bây giờ (depot TW = [0, max_time]):**

```python
# Depot latest = max_time (e.g., 80.36)
# Service time at depot (when returning) = current_time (e.g., 50)
# Check: 50 <= 80.36 → True ✅
# → Depot KHÔNG bị mask (có thể chọn)
```

**Lợi ích:** Xe có thể quay depot bất cứ lúc nào trong khoảng [0, max_time].

---

## 7. KIỂM TRA THAY ĐỔI

### 7.1. Test max_time calculation

```python
import torch
from tensordict import TensorDict
from parco.envs.pvrpwdp import PVRPWDPVEnv

# Create dummy data
td = TensorDict({
    "depot": torch.tensor([[0.5, 0.5]]),  # [1, 2]
    "locs": torch.tensor([
        [[0.2, 0.3], [0.8, 0.7], [0.5, 0.9]]  # 3 customers
    ]),  # [1, 3, 2]
    "demand": torch.tensor([[10.0, 20.0, 15.0]]),  # [1, 3]
    "capacity": torch.tensor([[100.0, 80.0]]),  # [1, 2] - 2 agents
    "speed": torch.tensor([[2.0, 1.0]]),  # [1, 2] - speed_min = 1.0
    "endurance": torch.tensor([[100.0, 150.0]]),  # [1, 2]
    "time_window": torch.tensor([
        [[10, 50], [20, 80], [5, 30]]  # [earliest, latest] for each customer
    ]),  # [1, 3, 2]
    "waiting_time": torch.tensor([[5.0, 10.0, 8.0]]),  # [1, 3]
}, batch_size=[1])

env = PVRPWDPVEnv()
td_reset = env._reset(td, batch_size=[1])

# Check depot time window
depot_tw = td_reset["time_window"][:, :2, :]  # [1, 2, 2] - 2 depots (replicated)
print("Depot time windows:")
print(depot_tw)
# Expected: [[[0, max_time], [0, max_time]]]
# where max_time ≈ max(50 + 0.36, 80 + 0.36, 30 + 0.40) = 80.36

# Verify max_time
speed_min = td["speed"].min()  # 1.0
depot_loc = td["depot"].unsqueeze(-2)  # [1, 1, 2]
customer_locs = td["locs"]  # [1, 3, 2]
dist = torch.norm(customer_locs - depot_loc, p=2, dim=-1)  # [1, 3]
travel_time = dist / speed_min  # [1, 3]
customer_latest = td["time_window"][..., 1]  # [1, 3]
max_time_expected = (customer_latest + travel_time).max()
print(f"Expected max_time: {max_time_expected.item():.2f}")
print(f"Actual depot latest: {depot_tw[0, 0, 1].item():.2f}")
```

### 7.2. Test action mask consistency

```python
# After reset, check if depot is maskable
td_reset = env._reset(td, batch_size=[1])

# Simulate some steps
td_step = env._step(td_reset.set("action", torch.tensor([[2, 1]])))  # Go to customers
td_step["current_time"] = torch.tensor([[30.0, 40.0]])  # Simulate current time

# Get action mask
action_mask = env.get_action_mask(td_step)

# Check if depot (node 0, 1) is available
depot_mask = action_mask[:, :, :2]  # [1, 2, 2]
print("Depot mask (should be True if within [0, max_time]):")
print(depot_mask)
```

---

## 8. BREAKING CHANGES

### ⚠️ **Không có Breaking Changes**

Thay đổi này **backward compatible**:
- Không thay đổi API
- Không thay đổi TensorDict structure
- Chỉ thay đổi giá trị của depot time window từ `[0, 0]` → `[0, max_time]`

---

## 9. LỢI ÍCH CỦA THAY ĐỔI

### ✅ **1. Realistic Constraint**
```python
# Depot time window [0, max_time] phản ánh thực tế:
# - Depot mở cửa từ đầu ngày (t=0)
# - Depot đóng cửa sau khi xe chậm nhất quay về
```

### ✅ **2. Consistent with Action Mask**
```python
# Action mask check: service_time <= latest_time
# Với depot latest = max_time:
# - Xe có thể quay depot bất cứ lúc nào trước max_time
# - Không bị conflict với time window constraint
```

### ✅ **3. Easier Training**
```python
# Model học được:
# - Khi nào nên quay depot (trước max_time)
# - Depot luôn available (không bị mask sai)
```

### ✅ **4. Adaptive to Data**
```python
# max_time thay đổi theo:
# - Customer time windows (latest_i)
# - Customer locations (distance to depot)
# - Agent speeds (speed_min)
# → Flexible cho nhiều problem instances
```

---

## 10. KẾT LUẬN

### **Tóm tắt thay đổi:**

| Aspect | Trước | Sau |
|--------|-------|-----|
| **Depot TW** | `[0, 0]` | `[0, max_time]` |
| **max_time formula** | N/A | `max(latest_i + travel_time_i)` |
| **Speed used** | N/A | `speed_min` (worst-case) |
| **Realistic** | ❌ | ✅ |
| **Consistent** | ❌ | ✅ |
| **Adaptive** | ❌ | ✅ |

### **Công thức:**

```python
max_time = max(
    customer_latest[i] + distance(customer_i, depot) / speed_min
    for all customers i
)

depot_time_window = [0, max_time]
```

### **Lợi ích:**
- ✅ **Realistic depot operation** - Depot mở cửa đủ lâu để đón tất cả xe
- ✅ **Consistent action mask** - Depot không bị mask sai do TW constraint
- ✅ **Easier training** - Model học được policy hợp lý
- ✅ **Adaptive to data** - max_time thay đổi theo problem instance

Thay đổi này giúp env **phản ánh đúng thực tế** và **dễ train** hơn! 🚀
