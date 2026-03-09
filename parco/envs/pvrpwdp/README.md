# PVRPWDP Environment - Step và Action Mask Flow

## Tổng quan

**PVRPWDP** (Perishable Vehicle Routing Problem with Drones and Pickup) là bài toán VRP mở rộng với các ràng buộc:
- **Time windows**: Mỗi khách hàng có cửa sổ thời gian [earliest, latest]
- **Perishability**: Hàng hóa có thời gian tươi (waiting_time), phải về depot trước khi hỏng
- **Endurance**: Drone/vehicle có giới hạn pin/năng lượng
- **Heterogeneous agents**: Nhiều loại xe với capacity, speed, endurance khác nhau

---

## 1. Hàm `_step(td)` - Thực thi một bước action

### 📥 Input
- `td["action"]`: Action được chọn (node index) cho mỗi agent `[B, m]`
- `td["current_node"]`: Vị trí hiện tại của mỗi agent `[B, m]`
- `td["current_time"]`: Thời gian hiện tại của mỗi agent `[B, m]`
- `td["used_capacity"]`: Capacity đã sử dụng `[B, m]`
- `td["used_endurance"]`: Endurance đã sử dụng `[B, m]`
- `td["trip_deadline"]`: Deadline để hàng không hỏng `[B, m]`

### 🔄 Các bước xử lý

#### **Bước 1: Xác định Stay Flag**
```python
stay_flag = td["action"] == td["current_node"]  # [B, m]
```
- Kiểm tra agent có đứng yên (không di chuyển) không
- Nếu `stay_flag=True` → không cập nhật gì

#### **Bước 2: Tính toán khoảng cách và thời gian di chuyển**
```python
step_distance = get_distance(previous_loc, current_loc)  # [B, m]
travel_time = (step_distance / agents_speed) * (~stay_flag).float()  # [B, m]
```
- **step_distance**: Khoảng cách từ vị trí trước đến vị trí hiện tại
- **travel_time**: Thời gian di chuyển = distance / speed (0 nếu stay)

#### **Bước 3: Cập nhật Current Time với Time Window**
```python
arrival_time = current_time + travel_time  # [B, m]
selected_earliest = time_window[..., 0][action]  # [B, m]
current_time = torch.maximum(arrival_time, selected_earliest)  # [B, m]
```
- **arrival_time**: Thời gian đến nếu xuất phát ngay
- **current_time**: Thời gian thực tế phục vụ (tôn trọng earliest time)
- Nếu đến sớm (arrival < earliest) → phải chờ đến earliest

#### **Bước 4: Cập nhật Used Endurance (Logic quan trọng!)**
```python
is_departing_from_depot = current_node < num_agents  # [B, m]
waiting_time_at_node = torch.clamp(selected_earliest - arrival_time, min=0.0)

total_endurance_used = torch.where(
    is_departing_from_depot,
    travel_time,  # Từ depot: chỉ travel time
    travel_time + waiting_time_at_node  # Từ customer: travel + waiting
)

used_endurance = (used_endurance + total_endurance_used) * (action >= num_agents).float()
```

**Logic phân biệt:**
- **Xuất phát từ depot**: Chỉ tính `travel_time`
  - Lý do: Có thể chọn thời gian khởi hành để đến đúng lúc `earliest` → không lãng phí pin
- **Đang trong trip**: Tính `travel_time + waiting_time`
  - Lý do: Đã cam kết vị trí, phải chờ nếu đến sớm → tốn pin chờ
- **Về depot**: Reset về 0

**Ví dụ:**
```
Depot → C1 (earliest=10):
  - arrival=8 → waiting=2
  - endurance = travel_time (không tính waiting=2)
  - Vì có thể khởi hành lúc 2 thay vì 0

C1 → C2 (earliest=20):
  - arrival=18 → waiting=2
  - endurance = travel_time + 2
  - Vì đã ở C1, phải chờ 2 đơn vị
```

#### **Bước 5: Cập nhật Trip Deadline**
```python
is_at_depot = current_node < num_agents  # [B, m]
new_deadline = current_time + selected_waiting_time  # [B, m]

trip_deadline_updated = torch.where(
    is_at_depot,
    new_deadline,  # Tại depot: set deadline mới
    torch.min(trip_deadline, new_deadline)  # Tại customer: lấy min
)

# Bảo vệ stay_flag
trip_deadline = torch.where(stay_flag, trip_deadline, trip_deadline_updated)

# Reset khi về depot
trip_deadline = trip_deadline * (action >= num_agents).float() + 1e6 * (action < num_agents).float()
```

**Logic:**
- **Tại depot**: Deadline = current_time + waiting_time (fresh start)
- **Tại customer**: Deadline = min(old_deadline, current_time + waiting_time)
  - Lấy deadline chặt nhất (hàng nào hỏng sớm nhất)
- **Stay**: Giữ nguyên deadline cũ
- **Về depot**: Reset về 1e6 (vô hạn)

#### **Bước 6: Cập nhật Capacity**
```python
selected_demand = demand[action]  # [B, m]
used_capacity = (used_capacity + selected_demand) * (action >= num_agents).float()
```
- Tăng capacity nếu thăm customer
- Reset về 0 nếu về depot

#### **Bước 7: Cập nhật Visited và Done**
```python
visited = visited.scatter(-1, action, 1)  # [B, m+N]
done = visited[..., num_agents:].sum(-1) == (num_customers)  # [B]
```
- Đánh dấu node đã thăm
- Done khi tất cả customers đã được thăm

### 📤 Output
```python
td.update({
    "current_length": current_length,      # Tổng khoảng cách đã đi
    "current_time": current_time,          # Thời gian hiện tại
    "trip_deadline": trip_deadline,        # Deadline hàng không hỏng
    "current_node": action,                # Vị trí mới
    "used_capacity": used_capacity,        # Capacity đã dùng
    "used_endurance": used_endurance,      # Endurance đã dùng
    "visited": visited,                    # Nodes đã thăm
    "done": done                          # Hoàn thành?
})
```

---

## 2. Hàm `get_action_mask(td)` - Tạo mask cho actions hợp lệ

### 📥 Input
- `td["visited"]`: Nodes đã thăm `[B, m+N]`
- `td["current_node"]`: Vị trí hiện tại `[B, m]`
- `td["current_time"]`: Thời gian hiện tại `[B, m]`
- `td["used_capacity"]`: Capacity đã dùng `[B, m]`
- `td["used_endurance"]`: Endurance đã dùng `[B, m]`
- `td["trip_deadline"]`: Deadline `[B, m]`

### 🔄 Các constraints được kiểm tra

#### **Constraint 1: Visited**
```python
action_mask = ~td["visited"][..., None, :]  # [B, m, m+N]
```
- ❌ Không thể thăm lại node đã thăm
- ✅ Chỉ được thăm nodes chưa visited

#### **Constraint 2: Capacity**
```python
remain_capacity = agents_capacity - used_capacity  # [B, m]
within_capacity_flag = demand <= remain_capacity[..., None]  # [B, m, m+N]
action_mask &= within_capacity_flag
```
- ❌ Không thể thăm node nếu demand > remaining capacity
- ✅ Chỉ thăm nodes có demand phù hợp

#### **Constraint 3: Time Window**

**Bước 3.1: Tính arrival time tại mỗi node**
```python
dist_to_nodes = torch.cdist(current_locs, all_locs, p=2)  # [B, m, m+N]
travel_time_to_nodes = dist_to_nodes / agents_speed.unsqueeze(-1)  # [B, m, m+N]
arrival_time = current_time.unsqueeze(-1) + travel_time_to_nodes  # [B, m, m+N]
```

**Bước 3.2: Tính waiting time (nếu đến sớm)**
```python
is_at_depot = current_node < num_agents  # [B, m]
earliest_times = time_window[..., 0].unsqueeze(-2)  # [B, 1, m+N]
waiting_time = torch.clamp(earliest_times - arrival_time, min=0.0)  # [B, m, m+N]

# Chỉ tính waiting nếu KHÔNG ở depot
waiting_time = waiting_time * (~is_at_depot.unsqueeze(-1)).float()
```

**Bước 3.3: Kiểm tra latest time**
```python
service_time = torch.maximum(arrival_time, earliest_times)  # [B, m, m+N]
latest_times = time_window[..., 1].unsqueeze(-2)  # [B, 1, m+N]
within_time_window = service_time <= latest_times  # [B, m, m+N]
action_mask &= within_time_window
```
- ❌ Không thể thăm nếu service_time > latest
- ✅ Phải phục vụ trong time window [earliest, latest]

#### **Constraint 4: Trip Deadline (Freshness)**

**Bước 4.1: Tính thời gian về depot**
```python
depot_locs = locs[..., :num_agents, :]  # [B, m, 2]
dist_to_depot = torch.cdist(depot_locs, all_locs, p=2)  # [B, m, m+N]
travel_time_to_depot = dist_to_depot / agents_speed.unsqueeze(-1)  # [B, m, m+N]
return_time = service_time + travel_time_to_depot  # [B, m, m+N]
```

**Bước 4.2: Kiểm tra deadline**
```python
within_deadline = return_time <= trip_deadline.unsqueeze(-1)  # [B, m, m+N]
action_mask &= within_deadline
```
- ❌ Không thể thăm nếu return_time > trip_deadline
- ✅ Phải đảm bảo về depot trước khi hàng hỏng

**Ví dụ:**
```
Agent có hàng C1 (waiting_time=20)
Nhặt ở time=10 → deadline = 30

Xét C2:
  - Service time tại C2 = 18
  - Travel C2→depot = 15
  - Return time = 18 + 15 = 33
  - Check: 33 <= 30? ❌ → C2 bị mask
```

#### **Constraint 5: Endurance (Battery)**

**Bước 5.1: Tính endurance cần thiết**
```python
endurance_to_node = torch.where(
    is_at_depot.unsqueeze(-1),
    travel_time_to_nodes,  # Từ depot: chỉ travel
    travel_time_to_nodes + waiting_time  # Từ customer: travel + waiting
)

total_endurance_needed = endurance_to_node + travel_time_to_depot  # [B, m, m+N]
```

**Bước 5.2: Kiểm tra tổng endurance**
```python
total_endurance = used_endurance.unsqueeze(-1) + total_endurance_needed  # [B, m, m+N]
within_endurance = total_endurance <= agents_freshness.unsqueeze(-1)  # [B, m, m+N]
action_mask &= within_endurance
```
- ❌ Không thể thăm nếu tổng endurance vượt quá max
- ✅ Phải đủ pin để đến node + về depot

**Ví dụ:**
```
Drone max_endurance = 100
used_endurance = 30

Xét C2:
  - Travel depot→C2 = 25 (từ depot, không tính waiting)
  - Travel C2→depot = 40
  - Total needed = 25 + 40 = 65
  - Total = 30 + 65 = 95
  - Check: 95 <= 100? ✅ OK

Xét C3:
  - Travel mid-trip→C3 = 50
  - Waiting at C3 = 5 (đến sớm, đang mid-trip)
  - Travel C3→depot = 25
  - Total needed = 50 + 5 + 25 = 80
  - Total = 30 + 80 = 110
  - Check: 110 <= 100? ❌ → C3 bị mask
```

#### **Constraint 6: Depot Isolation**

```python
all_back_flag = (current_node >= num_agents).sum(dim=-1) == 0  # [B]
has_finished_early = all_back_flag & ~done  # [B]
depot_mask = ~has_finished_early[..., None]  # [B, 1]

# Nếu tất cả customers đã visited → cho phép về depot
all_visited_flag = (~visited[..., num_agents:]).sum(dim=-1, keepdim=True) == 0
depot_mask |= all_visited_flag

# Eye matrix: agent i chỉ về depot i
eye_matrix = torch.eye(num_agents, device=device)  # [m, m]
eye_matrix = eye_matrix[None, ...].repeat(*batch_size, 1, 1).bool()  # [B, m, m]
eye_matrix &= depot_mask[..., None]
action_mask[..., :num_agents] = eye_matrix
```

**Logic:**
- ❌ Nếu TẤT CẢ agents đều ở depot và chưa done → không cho về depot (phải đi làm việc)
- ✅ Nếu có ít nhất 1 agent đang làm việc → các agent khác có thể về depot
- ✅ Nếu tất cả customers đã thăm → luôn cho phép về depot
- 🔒 Agent i chỉ có thể về depot i (eye matrix)

### 📤 Output
```python
action_mask: [B, m, m+N]
# True = action hợp lệ
# False = action bị mask (không được chọn)
```

---

## 3. Flowchart tổng quan

```
┌─────────────────────────────────────────────────────────────┐
│                    PVRPWDP Environment                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   get_action_mask │
                    │   (tính constraints)│
                    └──────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  ┌─────────┐          ┌─────────┐          ┌─────────┐
  │ Visited │          │Capacity │          │Time Win │
  └─────────┘          └─────────┘          └─────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  ┌─────────┐          ┌─────────┐          ┌─────────┐
  │Deadline │          │Endurance│          │ Depot   │
  └─────────┘          └─────────┘          └─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Action Mask     │
                    │  [B, m, m+N]     │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Model chọn      │
                    │  action          │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   _step(action)  │
                    └──────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  ┌──────────┐          ┌──────────┐         ┌──────────┐
  │Stay Flag │          │Travel    │         │Current   │
  │Check     │          │Time      │         │Time      │
  └──────────┘          └──────────┘         └──────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  ┌──────────┐          ┌──────────┐         ┌──────────┐
  │Endurance │          │Trip      │         │Capacity  │
  │Update    │          │Deadline  │         │Update    │
  └──────────┘          └──────────┘         └──────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Update State    │
                    │  (visited, done) │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Return td       │
                    └──────────────────┘
```

---

## 4. Key Design Decisions

### 🔑 **Decision 1: Waiting Time từ Depot vs Mid-trip**

**Vấn đề**: Khi agent đến sớm (trước earliest time window), có nên tính waiting time vào endurance không?

**Giải pháp**:
- **Từ depot**: KHÔNG tính waiting time vào endurance
  - Agent có thể TỰ CHỌN thời gian khởi hành
  - Đợi lúc depot không tốn pin (vehicle tắt máy)
  
- **Mid-trip**: CÓ tính waiting time vào endurance
  - Agent đã cam kết vị trí, không thể "quay về" để đợi
  - Phải đợi tại chỗ → hover/idle → tốn pin

**Implementation**:
```python
is_departing_from_depot = current_node < num_agents

endurance = torch.where(
    is_departing_from_depot,
    travel_time,                      # Từ depot
    travel_time + waiting_time        # Mid-trip
)
```

### 🔑 **Decision 2: Trip Deadline với Min Strategy**

**Vấn đề**: Agent nhặt nhiều hàng với waiting_time khác nhau, deadline nào quan trọng?

**Giải pháp**: Luôn lấy **deadline chặt nhất** (min)
```python
trip_deadline = torch.min(old_deadline, current_time + waiting_time)
```

**Lý do**: Hàng nào hỏng sớm nhất sẽ quyết định deadline cho cả trip

**Ví dụ**:
```
Nhặt C1 (waiting=20) ở t=10 → deadline1 = 30
Nhặt C2 (waiting=30) ở t=15 → deadline2 = 45
Trip deadline = min(30, 45) = 30  ← C1 quyết định
```

### 🔑 **Decision 3: Instant Reset tại Depot**

**Vấn đề**: Khi về depot, capacity và endurance có reset ngay không?

**Giải pháp**: Reset **NGAY LẬP TỨC** (thời gian reset = 0)
```python
used_capacity = used_capacity * (action >= num_agents).float()
used_endurance = used_endurance * (action >= num_agents).float()
trip_deadline = 1e6  # Reset về vô hạn
```

**Lý do**: 
- Giả định unload/recharge rất nhanh
- Đơn giản hóa model
- Khuyến khích về depot thường xuyên nếu cần

---

## 5. Shape Reference

| Variable | Shape | Description |
|----------|-------|-------------|
| `td["locs"]` | `[B, m+N, 2]` | Vị trí m depots + N customers |
| `td["demand"]` | `[B, m, m+N]` | Demand (repeat cho mỗi agent) |
| `td["time_window"]` | `[B, m+N, 2]` | [earliest, latest] |
| `td["waiting_time"]` | `[B, m+N]` | Freshness duration |
| `td["current_node"]` | `[B, m]` | Vị trí hiện tại mỗi agent |
| `td["current_time"]` | `[B, m]` | Thời gian hiện tại |
| `td["used_capacity"]` | `[B, m]` | Capacity đã dùng |
| `td["used_endurance"]` | `[B, m]` | Endurance đã dùng |
| `td["trip_deadline"]` | `[B, m]` | Deadline freshness |
| `td["visited"]` | `[B, m+N]` | Nodes đã thăm (shared) |
| `td["action_mask"]` | `[B, m, m+N]` | Mask cho mỗi agent |
| `td["agents_capacity"]` | `[B, m]` | Max capacity |
| `td["agents_speed"]` | `[B, m]` | Tốc độ |
| `td["agents_freshness"]` | `[B, m]` | Max endurance |

**Chú thích**:
- `B`: Batch size
- `m`: Số lượng agents/vehicles
- `N`: Số lượng customers
- `m+N`: Tổng số nodes (depots + customers)

---

## 6. Testing Checklist

### ✅ Các test cases cần kiểm tra:

1. **Basic Flow**
   - [ ] Agent đi từ depot → customer → depot
   - [ ] Capacity và endurance reset đúng khi về depot
   - [ ] Visited được cập nhật đúng

2. **Time Window**
   - [ ] Agent đến sớm phải chờ đến earliest
   - [ ] Agent không thể đến sau latest
   - [ ] Waiting time từ depot = 0 (endurance)
   - [ ] Waiting time mid-trip > 0 (endurance)

3. **Trip Deadline**
   - [ ] Deadline = min(các waiting_times của hàng đã nhặt)
   - [ ] Không thể thăm customer nếu return_time > deadline
   - [ ] Deadline reset về 1e6 khi về depot

4. **Endurance**
   - [ ] Endurance từ depot = travel time only
   - [ ] Endurance mid-trip = travel + waiting
   - [ ] Không thể thăm nếu vượt quá max endurance
   - [ ] Endurance reset về 0 khi về depot

5. **Stay Flag**
   - [ ] Nếu action == current_node → không cập nhật gì
   - [ ] Travel time = 0
   - [ ] Deadline giữ nguyên

6. **Depot Isolation**
   - [ ] Tất cả agents ở depot + chưa done → không cho về depot
   - [ ] Có agent đang làm → cho về depot
   - [ ] Agent i chỉ về depot i (eye matrix)

---

## 7. Common Pitfalls & Solutions

### ⚠️ Pitfall 1: Quên kiểm tra stay_flag
**Vấn đề**: Agent stay nhưng vẫn cập nhật state
**Giải pháp**: Đặt stay_flag check đầu tiên trong `_step`

### ⚠️ Pitfall 2: Dùng current_length thay vì step_distance
**Vấn đề**: Travel time = cumulative distance / speed (SAI!)
**Giải pháp**: Travel time = step_distance / speed

### ⚠️ Pitfall 3: Quên reset trip_deadline khi về depot
**Vấn đề**: Deadline cũ còn carry over sang trip mới
**Giải pháp**: Reset về 1e6 khi `action < num_agents`

### ⚠️ Pitfall 4: Waiting time logic inconsistent
**Vấn đề**: `_step` và `get_action_mask` dùng logic khác nhau
**Giải pháp**: Cả 2 đều phải dùng `is_at_depot` để phân biệt

### ⚠️ Pitfall 5: Customer_mask logic ngược
**Vấn đề**: Mask depot thay vì mask customer
**Giải pháp**: 
```python
customer_mask[..., :num_agents] = True  # Always allow depot
within_constraint = within_constraint | ~customer_mask
```

---

## 8. Performance Optimization Tips

### 🚀 Optimization 1: Batch cdist thay vì loop
```python
# ❌ Slow
for i in range(num_agents):
    dist = get_distance(current_loc[i], all_locs)

# ✅ Fast
dist_to_nodes = torch.cdist(current_locs, all_locs, p=2)
```

### 🚀 Optimization 2: Broadcast thay vì repeat
```python
# ❌ Memory heavy
demand_repeated = demand.unsqueeze(-2).repeat(1, num_agents, 1)

# ✅ Better (nếu có thể)
# Sử dụng broadcasting trực tiếp trong comparison
```

### 🚀 Optimization 3: In-place operations
```python
# ❌ Creates new tensor
action_mask = action_mask & within_capacity_flag

# ✅ In-place
action_mask &= within_capacity_flag
```

---

## 9. Debugging Guide

### 🐛 Debug Step 1: Print shapes
```python
print(f"current_locs: {current_locs.shape}")  # Should be [B, m, 2]
print(f"dist_to_nodes: {dist_to_nodes.shape}")  # Should be [B, m, m+N]
print(f"action_mask: {action_mask.shape}")  # Should be [B, m, m+N]
```

### 🐛 Debug Step 2: Check constraint coverage
```python
# Mỗi agent phải có ít nhất 1 action hợp lệ
valid_actions = action_mask.sum(dim=-1)  # [B, m]
assert (valid_actions > 0).all(), "Some agents have no valid actions!"
```

### 🐛 Debug Step 3: Visualize mask
```python
# Agent 0, batch 0
mask = action_mask[0, 0]  # [m+N]
print(f"Valid depots: {mask[:num_agents]}")
print(f"Valid customers: {mask[num_agents:]}")
```

### 🐛 Debug Step 4: Trace endurance
```python
print(f"used_endurance: {td['used_endurance']}")
print(f"total_endurance_needed: {total_endurance_needed}")
print(f"agents_freshness: {td['agents_freshness']}")
print(f"within_endurance: {within_endurance.sum(dim=-1)}")  # Count valid
```

---

## 10. References

- **Base Environment**: `rl4co.envs.common.base.RL4COEnvBase`
- **HCVRP Baseline**: `parco/envs/hcvrp/env.py`
- **Generator**: `parco/envs/pvrpwdp/generator.py`
- **Model Embeddings**: `parco/models/env_embeddings/`

---

**Tạo bởi**: PVRPWDP Environment Documentation
**Ngày**: 2026-03-06
**Version**: 1.0
