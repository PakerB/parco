# HƯỚNG DẪN CHI TIẾT VỀ GENERATOR VÀ ENVIRONMENT HCVRP

## 📋 MỤC LỤC
1. [Tổng quan về HCVRP](#1-tổng-quan-về-hcvrp)
2. [Generator - HCVRPGenerator](#2-generator---hcvrpgenerator)
3. [Environment - HCVRPEnv](#3-environment---hcvrpenv)
4. [Luồng dữ liệu TensorDict](#4-luồng-dữ-liệu-tensordict)
5. [Ví dụ minh họa](#5-ví-dụ-minh-họa)

---

## 1. TỔNG QUAN VỀ HCVRP

**HCVRP (Heterogeneous Capacitated Vehicle Routing Problem)** là bài toán định tuyến xe với các đặc điểm:

- **Heterogeneous (Không đồng nhất)**: Các xe có **capacity** và **speed** khác nhau
- **Capacitated (Có giới hạn)**: Mỗi xe có sức chứa giới hạn
- **Mục tiêu**: Phục vụ tất cả khách hàng, tối thiểu hóa thời gian hoàn thành lớn nhất (makespan)

### Ràng buộc:
1. Mỗi khách hàng được thăm **đúng 1 lần** bởi 1 xe
2. Xe không được vượt quá sức chứa
3. Mỗi xe xuất phát và kết thúc tại depot
4. Xe có thể quay lại depot để refill capacity

---

## 2. GENERATOR - HCVRPGenerator

Generator chịu trách nhiệm **tạo dữ liệu đầu vào** cho bài toán.

### 2.1. Tham số khởi tạo

```python
HCVRPGenerator(
    num_loc: int = 40,                    # Số khách hàng
    min_loc: float = 0.0,                 # Tọa độ tối thiểu
    max_loc: float = 1.0,                 # Tọa độ tối đa
    loc_distribution = Uniform,           # Phân phối vị trí khách hàng
    depot_distribution = None,            # Phân phối vị trí depot (None = lấy từ locs)
    min_demand: int = 1,                  # Nhu cầu tối thiểu
    max_demand: int = 10,                 # Nhu cầu tối đa
    demand_distribution = Uniform,        # Phân phối nhu cầu
    min_capacity: float = 20,             # Sức chứa tối thiểu của xe
    max_capacity: float = 41,             # Sức chứa tối đa của xe
    capacity_distribution = Uniform,      # Phân phối sức chứa
    min_speed: float = 0.5,               # Tốc độ tối thiểu
    max_speed: float = 1.0,               # Tốc độ tối đa
    speed_distribution = Uniform,         # Phân phối tốc độ
    num_agents: int = 3,                  # Số xe (agents)
    scale_data: bool = False,             # Chuẩn hóa dữ liệu (khuyến nghị: False)
)
```

### 2.2. Output của Generator - TensorDict

Method `_generate(batch_size)` trả về TensorDict với các trường:

| Key | Shape | Dtype | Mô tả | Giá trị |
|-----|-------|-------|-------|---------|
| **locs** | `[B, N, 2]` | float32 | Tọa độ khách hàng (x, y) | [0.0, 1.0] |
| **depot** | `[B, 2]` | float32 | Tọa độ depot (x, y) | [0.0, 1.0] |
| **num_agents** | `[B]` | int64 | Số xe | num_agents |
| **demand** | `[B, N]` | float32 | Nhu cầu của khách hàng | [1, 10] |
| **capacity** | `[B, m]` | float32 | Sức chứa của từng xe | [20, 41] |
| **speed** | `[B, m]` | float32 | Tốc độ của từng xe | [0.5, 1.0] |

**Ký hiệu:**
- `B`: batch_size
- `N`: num_loc (số khách hàng)
- `m`: num_agents (số xe)

### 2.3. Chi tiết cách sinh dữ liệu

#### a) Sinh tọa độ (locs và depot)

```python
# Trường hợp 1: depot_sampler được định nghĩa
if self.depot_sampler is not None:
    depot = self.depot_sampler.sample((*batch_size, 2))        # [B, 2]
    locs = self.loc_sampler.sample((*batch_size, num_loc, 2))  # [B, N, 2]

# Trường hợp 2: depot_sampler = None (depot lấy từ locs[0])
else:
    locs_all = self.loc_sampler.sample((*batch_size, num_loc + 1, 2))  # [B, N+1, 2]
    depot = locs_all[..., 0, :]    # [B, 2] - Phần tử đầu tiên
    locs = locs_all[..., 1:, :]    # [B, N, 2] - Phần còn lại
```

#### b) Sinh nhu cầu (demand)

```python
# Sample từ [min_demand - 1, max_demand - 1], sau đó +1
demand = self.demand_sampler.sample((*batch_size, num_loc))  # [B, N]
demand = (demand.int() + 1).float()  # Đảm bảo demand >= 1

# Nếu scale_data = True
demand = demand / self.max_capacity  # Chuẩn hóa về [0, 1]
```

**Lưu ý:** Demand **KHÔNG được normalize mặc định** (scale_data=False)

#### c) Sinh sức chứa (capacity)

```python
# Sample từ [0, max_capacity - min_capacity], sau đó + min_capacity
capacity = self.capacity_sampler.sample((*batch_size, num_agents))  # [B, m]
capacity = (capacity.int() + self.min_capacity).float()  # [20, 41]

# Nếu scale_data = True
capacity = capacity / self.max_capacity  # Chuẩn hóa
```

**Mỗi xe có capacity KHÁC NHAU** → Heterogeneous!

#### d) Sinh tốc độ (speed)

```python
# Sample trực tiếp từ [min_speed, max_speed]
speed = self.speed_sampler.sample((*batch_size, num_agents))  # [B, m]

# Nếu scale_data = True
speed = speed / self.max_speed  # Chuẩn hóa
```

**Mỗi xe có speed KHÁC NHAU** → Heterogeneous!

---

## 3. ENVIRONMENT - HCVRPEnv

Environment quản lý **trạng thái** và **động lực học** của bài toán.

### 3.1. Khởi tạo Environment

```python
env = HCVRPEnv(
    generator: HCVRPGenerator = None,      # Generator (nếu None sẽ tạo mặc định)
    generator_params: dict = {},           # Params cho generator
    check_solution: bool = False,          # Kiểm tra tính hợp lệ (chỉ debug)
)
```

### 3.2. Reset Environment - `_reset()`

Method `_reset()` nhận TensorDict từ Generator và **khởi tạo trạng thái ban đầu**.

#### Input (từ Generator):
```python
td_input = {
    "locs": [B, N, 2],
    "depot": [B, 2],
    "demand": [B, N],
    "capacity": [B, m],
    "speed": [B, m],
}
```

#### Output (TensorDict sau reset):

| Key | Shape | Dtype | Mô tả | Giá trị ban đầu |
|-----|-------|-------|-------|-----------------|
| **locs** | `[B, m+N, 2]` | float32 | Tọa độ depot (x m) + khách hàng | depot lặp m lần + locs |
| **demand** | `[B, m, m+N]` | float32 | Nhu cầu (lặp cho mỗi xe) | [0...0, demand] lặp m lần |
| **current_length** | `[B, m]` | float32 | Quãng đường đã đi | 0.0 |
| **current_node** | `[B, m]` | int64 | Vị trí hiện tại của xe | [0, 1, 2, ..., m-1] |
| **depot_node** | `[B, m]` | int64 | Vị trí depot (cố định) | [0, 1, 2, ..., m-1] |
| **used_capacity** | `[B, m]` | float32 | Sức chứa đã sử dụng | 0.0 |
| **agents_capacity** | `[B, m]` | float32 | Tổng sức chứa của xe | từ Generator |
| **agents_speed** | `[B, m]` | float32 | Tốc độ của xe | từ Generator |
| **i** | `[B, 1]` | int64 | Số bước đã thực hiện | 0 |
| **visited** | `[B, m+N]` | bool | Đánh dấu đã thăm | False (chưa thăm) |
| **action_mask** | `[B, m, m+N]` | bool | Mask hành động hợp lệ | Tính từ get_action_mask() |
| **done** | `[B]` | bool | Episode kết thúc? | False |

#### Chi tiết các bước xử lý:

**Bước 1: Lặp lại depot cho mỗi xe**
```python
# Input: depot [B, 2]
# Output: depots [B, m, 2]

depots = td["depot"]
if depots.ndim == 2:
    depots = depots.unsqueeze(-2)         # [B, 1, 2]
depots = depots.repeat(1, num_agents, 1)  # [B, m, 2]

# Ghép depot + customers
locs = torch.cat((depots, td["locs"]), dim=-2)  # [B, m+N, 2]
```

**Giải thích:** Mỗi xe có depot riêng (dù cùng vị trí) → index 0, 1, 2, ..., m-1

**Bước 2: Thêm demand của depot và lặp lại**
```python
# Depot có demand = 0
demand_depot = torch.zeros((*batch_size, num_agents))  # [B, m]
demand = torch.cat((demand_depot, td["demand"]), -1)   # [B, m+N]

# Lặp lại cho mỗi xe (để tính action mask dễ dàng)
demand = demand.unsqueeze(-2).repeat(1, num_agents, 1)  # [B, m, m+N]
```

**Bước 3: Khởi tạo vị trí xe**
```python
# Mỗi xe bắt đầu tại depot của nó
depot_node = torch.arange(num_agents)  # [0, 1, 2, ..., m-1]
depot_node = depot_node.repeat(*batch_size, 1)  # [B, m]

current_node = depot_node.clone()  # [B, m] - Xe đang ở depot
```

**Bước 4: Khởi tạo các trạng thái khác**
```python
current_length = torch.zeros((*batch_size, num_agents))  # [B, m]
used_capacity = torch.zeros((*batch_size, num_agents))   # [B, m]
visited = torch.zeros((*batch_size, num_loc_all), dtype=bool)  # [B, m+N]
i = torch.zeros((*batch_size, 1), dtype=int64)  # [B, 1]
done = torch.zeros((*batch_size,), dtype=bool)  # [B]
```

**Bước 5: Tính action mask**
```python
action_mask = self.get_action_mask(td_reset)  # [B, m, m+N]
```

### 3.3. Step Environment - `_step(td)`

Method `_step()` thực hiện một bước action và **cập nhật trạng thái**.

#### Input:
```python
td["action"]: [B, m]  # Action của mỗi xe (index của node cần đi)
```

#### Các bước cập nhật:

**Bước 1: Cập nhật quãng đường**
```python
current_loc = gather_by_index(td["locs"], td["action"])      # [B, m, 2]
previous_loc = gather_by_index(td["locs"], td["current_node"])  # [B, m, 2]
distance = get_distance(previous_loc, current_loc)           # [B, m]

current_length = td["current_length"] + distance  # [B, m]
```

**Bước 2: Cập nhật sức chứa**
```python
selected_demand = gather_by_index(td["demand"], td["action"], dim=-1)  # [B, m]

# Nếu xe ở lại chỗ cũ → không tính demand
stay_flag = (td["action"] == td["current_node"])  # [B, m]
selected_demand = selected_demand * (~stay_flag).float()

# Nếu về depot (action < num_agents) → reset capacity về 0
# Ngược lại → cộng thêm demand
used_capacity = (td["used_capacity"] + selected_demand) * (td["action"] >= num_agents).float()
```

**Bước 3: Cập nhật visited**
```python
visited = td["visited"].scatter(-1, td["action"], 1)  # [B, m+N]
# Đánh dấu node vừa thăm = 1
```

**Bước 4: Kiểm tra done**
```python
# Đếm số khách hàng đã thăm (bỏ qua depot)
num_visited = visited[..., num_agents:].sum(-1)  # [B]
num_customers = visited.size(-1) - num_agents

done = (num_visited == num_customers)  # [B] - True nếu thăm hết
```

**Bước 5: Cập nhật TensorDict**
```python
td.update({
    "current_length": current_length,
    "current_node": td["action"],  # Vị trí mới
    "used_capacity": used_capacity,
    "i": td["i"] + 1,
    "visited": visited,
    "reward": torch.zeros_like(done),  # Reward tính ở cuối
    "done": done,
})
td.set("action_mask", self.get_action_mask(td))  # Cập nhật mask
```

### 3.4. Action Mask - `get_action_mask()`

Action mask xác định **hành động hợp lệ** cho mỗi xe.

#### Quy tắc tính action_mask:

**1. Chỉ thăm điểm chưa visited**
```python
# Lặp lại visited cho mỗi xe
action_mask = torch.repeat_interleave(
    ~td["visited"][..., None, :],  # [B, 1, m+N]
    dim=-2, 
    repeats=num_agents
)  # [B, m, m+N]

# False nếu đã visited, True nếu chưa
```

**2. Kiểm tra sức chứa còn lại**
```python
remain_capacity = td["agents_capacity"] - td["used_capacity"]  # [B, m]
within_capacity = td["demand"] <= remain_capacity[..., None]   # [B, m, m+N]

action_mask &= within_capacity  # AND với điều kiện capacity
```

**3. Depot có điều kiện đặc biệt**
```python
# Đếm số xe đang ở depot
all_back_flag = torch.sum(td["current_node"] >= num_agents, dim=-1) == 0  # [B]

# Nếu tất cả xe về depot NHƯNG chưa hoàn thành → KHÔNG được ở depot
has_finished_early = all_back_flag & ~td["done"]  # [B]
depot_mask = ~has_finished_early[..., None]  # [B, 1] - True nếu được vào depot

# Nếu đã thăm hết khách hàng → depot luôn available
all_visited = torch.sum(~td["visited"][..., num_agents:], dim=-1, keepdim=True) == 0
depot_mask |= all_visited  # [B, 1]
```

**4. Cập nhật depot mask vào action_mask**
```python
# Ma trận đơn vị: mỗi xe chỉ về depot của nó
eye_matrix = torch.eye(num_agents, device=td.device)  # [m, m]
eye_matrix = eye_matrix[None, ...].repeat(*batch_size, 1, 1).bool()  # [B, m, m]

# Áp dụng depot_mask
eye_matrix &= depot_mask[..., None]  # [B, m, m]

# Gán vào cột depot của action_mask
action_mask[..., :num_agents] = eye_matrix  # [B, m, m+N]
```

**Kết quả:** `action_mask[b, i, j]` = True nếu xe `i` có thể đi đến node `j`

### 3.5. Tính Reward - `_get_reward()`

Reward được tính **khi episode kết thúc**.

```python
def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
    # 1. Tính quãng đường còn lại về depot
    current_loc = gather_by_index(td["locs"], td["current_node"])  # [B, m, 2]
    depot_loc = gather_by_index(td["locs"], td["depot_node"])      # [B, m, 2]
    final_distance = get_distance(depot_loc, current_loc)          # [B, m]
    
    total_length = td["current_length"] + final_distance           # [B, m]
    
    # 2. Tính thời gian = quãng đường / tốc độ
    current_time = total_length / td["agents_speed"]               # [B, m]
    
    # 3. Lấy thời gian lớn nhất (makespan)
    max_time = current_time.max(dim=1)[0]                          # [B]
    
    # 4. Reward = -makespan (maximize reward = minimize makespan)
    return -max_time
```

**Mục tiêu:** Minimize makespan (thời gian xe chậm nhất hoàn thành)

---

## 4. LUỒNG DỮ LIỆU TENSORDICT

### 4.1. Sơ đồ tổng quan

```
┌──────────────────────────┐
│   HCVRPGenerator         │
│  _generate(batch_size)   │
└────────────┬─────────────┘
             │
             ▼
    ┌────────────────────┐
    │   TensorDict       │
    │ - locs: [B, N, 2]  │
    │ - depot: [B, 2]    │
    │ - demand: [B, N]   │
    │ - capacity: [B, m] │
    │ - speed: [B, m]    │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │  HCVRPEnv._reset() │
    └────────┬───────────┘
             │
             ▼
    ┌─────────────────────────┐
    │   TensorDict (State)    │
    │ - locs: [B, m+N, 2]     │
    │ - demand: [B, m, m+N]   │
    │ - current_node: [B, m]  │
    │ - depot_node: [B, m]    │
    │ - current_length: [B,m] │
    │ - used_capacity: [B, m] │
    │ - agents_capacity:[B,m] │
    │ - agents_speed: [B, m]  │
    │ - visited: [B, m+N]     │
    │ - action_mask:[B,m,m+N] │
    │ - done: [B]             │
    └────────┬────────────────┘
             │
             ▼
    ┌────────────────────┐
    │  HCVRPEnv._step()  │
    │  + action: [B, m]  │
    └────────┬───────────┘
             │
             ▼
    ┌─────────────────────────┐
    │ TensorDict (Updated)    │
    │ (các trường được update)│
    └────────┬────────────────┘
             │
             ▼ (lặp lại cho đến done)
    ┌────────────────────────┐
    │ HCVRPEnv._get_reward() │
    └────────┬───────────────┘
             │
             ▼
        reward: [B]
```

### 4.2. Biến đổi kích thước chi tiết

| Giai đoạn | Trường | Shape Ban đầu | Shape Sau | Giải thích |
|-----------|--------|---------------|-----------|------------|
| **Generator** | depot | `[B, 2]` | - | Tọa độ depot |
| **Reset** | depot → locs | `[B, 2]` | `[B, m, 2]` | Lặp lại m lần |
| **Reset** | locs | `[B, N, 2]` | `[B, m+N, 2]` | Concat depot + customers |
| **Generator** | demand | `[B, N]` | - | Nhu cầu khách hàng |
| **Reset** | demand | `[B, N]` | `[B, m+N]` | Thêm 0 cho depot |
| **Reset** | demand | `[B, m+N]` | `[B, m, m+N]` | Lặp lại cho mỗi xe |
| **Generator** | capacity | `[B, m]` | `[B, m]` | Không đổi |
| **Generator** | speed | `[B, m]` | `[B, m]` | Không đổi |
| **Reset** | - | - | `[B, m]` | Tạo current_node |
| **Reset** | - | - | `[B, m]` | Tạo depot_node |
| **Reset** | - | - | `[B, m]` | Tạo current_length |
| **Reset** | - | - | `[B, m]` | Tạo used_capacity |
| **Reset** | - | - | `[B, m+N]` | Tạo visited |
| **Reset** | - | - | `[B, m, m+N]` | Tạo action_mask |
| **Reset** | - | - | `[B]` | Tạo done |

---

## 5. VÍ DỤ MINH HỌA

### 5.1. Tạo dữ liệu với Generator

```python
import torch
from parco.envs.hcvrp import HCVRPGenerator

# Khởi tạo generator
generator = HCVRPGenerator(
    num_loc=5,        # 5 khách hàng
    num_agents=2,     # 2 xe
    min_demand=1,
    max_demand=10,
    min_capacity=20,
    max_capacity=41,
    min_speed=0.5,
    max_speed=1.0,
)

# Sinh dữ liệu batch_size=2
batch_size = [2]
td = generator._generate(batch_size)

print("TensorDict từ Generator:")
print(f"locs: {td['locs'].shape}")           # [2, 5, 2]
print(f"depot: {td['depot'].shape}")         # [2, 2]
print(f"demand: {td['demand'].shape}")       # [2, 5]
print(f"capacity: {td['capacity'].shape}")   # [2, 2]
print(f"speed: {td['speed'].shape}")         # [2, 2]

print("\nGiá trị ví dụ (batch 0):")
print(f"depot: {td['depot'][0]}")            # tensor([0.5, 0.5])
print(f"locs[0]: {td['locs'][0, 0]}")        # tensor([0.2, 0.3])
print(f"demand: {td['demand'][0]}")          # tensor([3., 5., 2., 8., 4.])
print(f"capacity: {td['capacity'][0]}")      # tensor([30., 25.])
print(f"speed: {td['speed'][0]}")            # tensor([0.8, 0.6])
```

### 5.2. Reset Environment

```python
from parco.envs.hcvrp import HCVRPEnv

# Khởi tạo environment
env = HCVRPEnv(generator=generator)

# Reset với dữ liệu từ generator
td_reset = env.reset(td)

print("\nTensorDict sau reset:")
print(f"locs: {td_reset['locs'].shape}")                 # [2, 7, 2] = 2 depot + 5 customers
print(f"demand: {td_reset['demand'].shape}")             # [2, 2, 7]
print(f"current_node: {td_reset['current_node'].shape}") # [2, 2]
print(f"depot_node: {td_reset['depot_node'].shape}")     # [2, 2]
print(f"current_length: {td_reset['current_length'].shape}") # [2, 2]
print(f"used_capacity: {td_reset['used_capacity'].shape}")   # [2, 2]
print(f"agents_capacity: {td_reset['agents_capacity'].shape}") # [2, 2]
print(f"agents_speed: {td_reset['agents_speed'].shape}")     # [2, 2]
print(f"visited: {td_reset['visited'].shape}")           # [2, 7]
print(f"action_mask: {td_reset['action_mask'].shape}")   # [2, 2, 7]
print(f"done: {td_reset['done'].shape}")                 # [2]

print("\nGiá trị ban đầu (batch 0):")
print(f"locs[0]: {td_reset['locs'][0]}")
# tensor([[0.5, 0.5],  <- depot xe 0
#         [0.5, 0.5],  <- depot xe 1
#         [0.2, 0.3],  <- customer 0
#         [0.7, 0.8],  <- customer 1
#         ...])

print(f"current_node[0]: {td_reset['current_node'][0]}")  # tensor([0, 1]) - Xe 0 ở node 0, xe 1 ở node 1
print(f"depot_node[0]: {td_reset['depot_node'][0]}")      # tensor([0, 1])
print(f"current_length[0]: {td_reset['current_length'][0]}") # tensor([0., 0.])
print(f"used_capacity[0]: {td_reset['used_capacity'][0]}")   # tensor([0., 0.])
print(f"visited[0]: {td_reset['visited'][0]}")            # tensor([False, False, ..., False])
print(f"action_mask[0, 0]: {td_reset['action_mask'][0, 0]}") 
# tensor([True, False, True, True, ...]) - Xe 0 có thể về depot 0 hoặc đi customer
```

### 5.3. Thực hiện Step

```python
# Giả sử xe 0 đi customer 2 (index=2), xe 1 đi customer 3 (index=3)
actions = torch.tensor([[2, 3]])  # [1, 2]

# Step
td_next = env.step(td_reset.set("action", actions))["next"]

print("\nSau khi step:")
print(f"current_node: {td_next['current_node'][0]}")     # tensor([2, 3])
print(f"current_length: {td_next['current_length'][0]}") # tensor([0.xxx, 0.yyy])
print(f"used_capacity: {td_next['used_capacity'][0]}")   # tensor([3., 5.]) - demand của customer 2, 3
print(f"visited: {td_next['visited'][0]}")
# tensor([False, False, True, True, False, False, False])
#        depot0   depot1   cust2  cust3  cust4  cust5  cust6

print(f"done: {td_next['done'][0]}")  # False - chưa xong
```

### 5.4. Tính Reward

```python
# Giả sử đã hoàn thành tất cả khách hàng
# actions là sequence của tất cả bước: [B, m, steps]

reward = env.get_reward(td_final, actions)
print(f"Reward: {reward}")  # tensor([-15.5, -18.2]) - Negative makespan
```

---

## 6. LƯU Ý QUAN TRỌNG

### 6.1. Về Depot

- Depot được **lặp lại m lần** trong `locs`: `[B, m+N, 2]`
- Mỗi xe có depot riêng ở **index khác nhau**: xe 0 → node 0, xe 1 → node 1, ...
- Tất cả depot có **cùng tọa độ** nhưng **khác index**

### 6.2. Về Demand

- Demand được **lặp lại m lần**: `[B, m, m+N]`
- Mỗi xe có bản sao demand riêng → tiện tính action_mask
- Depot có demand = 0

### 6.3. Về Action Mask

- Shape: `[B, m, m+N]`
- `action_mask[b, i, j]` = True → xe `i` có thể đi node `j`
- Depot chỉ available cho xe của nó: `action_mask[b, 0, 0]=True`, `action_mask[b, 0, 1]=False`

### 6.4. Về Heterogeneous

- **Capacity khác nhau**: `agents_capacity[b, 0]` ≠ `agents_capacity[b, 1]`
- **Speed khác nhau**: `agents_speed[b, 0]` ≠ `agents_speed[b, 1]`
- → Cần tối ưu hóa phân công xe phù hợp

### 6.5. Về Reward

- Reward = **-makespan** (thời gian hoàn thành lớn nhất)
- Makespan = `max(current_length[i] / agents_speed[i])`
- Maximize reward = Minimize makespan

---

## 7. TỔNG KẾT

### Luồng dữ liệu chính:

1. **Generator** tạo dữ liệu: `locs, depot, demand, capacity, speed`
2. **Reset** xử lý và tạo state: `locs (m+N), demand (m,m+N), current_node, ...`
3. **Step** cập nhật state: `current_node, current_length, used_capacity, visited, ...`
4. **Action Mask** xác định hành động hợp lệ: visited, capacity, depot rules
5. **Reward** tính makespan: `-max(length / speed)`

### Các điểm chính:

- ✅ **Heterogeneous**: Mỗi xe có capacity và speed khác nhau
- ✅ **Multi-agent**: Nhiều xe hành động song song
- ✅ **Depot per agent**: Mỗi xe có depot riêng (cùng tọa độ)
- ✅ **Action mask**: Ràng buộc capacity và visited
- ✅ **Makespan objective**: Tối ưu thời gian lớn nhất

File này cung cấp tất cả thông tin cần thiết về cấu trúc TensorDict trong HCVRP! 🚀
