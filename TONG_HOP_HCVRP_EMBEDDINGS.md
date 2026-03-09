# TỔNG HỢP CÁC TRƯỜNG DỮ LIỆU CHO ENCODER, DECODER VÀ MODEL CỦA MÔI TRƯỜNG HCVRP

## 1. TỔNG QUAN VỀ HCVRP

**HCVRP (Heterogeneous Capacitated Vehicle Routing Problem)** là bài toán định tuyến xe với nhiều loại xe khác nhau (có khả năng chở hàng và tốc độ khác nhau). Mục tiêu là phục vụ tất cả khách hàng với chi phí tối thiểu, tối ưu hóa thời gian hoàn thành lớn nhất (makespan).

---

## 2. DỮ LIỆU ĐẦU VÀO TỪ GENERATOR (HCVRPGenerator)

### 2.1. Tham số cấu hình Generator
```python
num_loc: int = 40                    # Số lượng khách hàng
min_loc: float = 0.0                 # Tọa độ tối thiểu
max_loc: float = 1.0                 # Tọa độ tối đa
min_demand: int = 1                  # Nhu cầu tối thiểu của khách hàng
max_demand: int = 10                 # Nhu cầu tối đa của khách hàng
min_capacity: float = 20             # Sức chứa tối thiểu của xe
max_capacity: float = 41             # Sức chứa tối đa của xe
min_speed: float = 0.5               # Tốc độ tối thiểu của xe
max_speed: float = 1.0               # Tốc độ tối đa của xe
num_agents: int = 3                  # Số lượng xe (agents)
scale_data: bool = False             # Chuẩn hóa dữ liệu (mặc định False)
```

### 2.2. Dữ liệu được tạo ra (TensorDict)
```python
{
    "locs": [batch_size, num_loc, 2],           # Tọa độ các khách hàng (x, y)
    "depot": [batch_size, 2],                   # Tọa độ kho depot (x, y)
    "num_agents": [batch_size],                 # Số lượng xe
    "demand": [batch_size, num_loc],            # Nhu cầu của từng khách hàng
    "capacity": [batch_size, num_agents],       # Sức chứa của từng xe
    "speed": [batch_size, num_agents],          # Tốc độ của từng xe
}
```

---

## 3. DỮ LIỆU TRONG MÔI TRƯỜNG (HCVRPEnv)

### 3.1. Dữ liệu sau khi Reset (_reset)
Khi môi trường được reset, các trường sau được khởi tạo:

```python
{
    # Vị trí
    "locs": [batch_size, num_agents + num_loc, 2],  
    # Tọa độ của depot (lặp lại cho mỗi xe) + khách hàng
    
    # Nhu cầu
    "demand": [batch_size, num_agents, num_agents + num_loc],  
    # Nhu cầu của khách hàng (depot = 0), lặp lại cho mỗi xe
    
    # Trạng thái di chuyển
    "current_length": [batch_size, num_agents],      # Quãng đường đã đi của mỗi xe
    "current_node": [batch_size, num_agents],        # Vị trí hiện tại của mỗi xe
    "depot_node": [batch_size, num_agents],          # Vị trí depot của mỗi xe (cố định)
    
    # Trạng thái sức chứa
    "used_capacity": [batch_size, num_agents],       # Sức chứa đã sử dụng của mỗi xe
    "agents_capacity": [batch_size, num_agents],     # Tổng sức chứa của mỗi xe
    "agents_speed": [batch_size, num_agents],        # Tốc độ của mỗi xe
    
    # Trạng thái thăm viếng
    "i": [batch_size, 1],                            # Bước thời gian hiện tại
    "visited": [batch_size, num_agents + num_loc],   # Đánh dấu điểm đã thăm (1=đã thăm)
    
    # Action mask
    "action_mask": [batch_size, num_agents, num_agents + num_loc],  
    # Mask cho hành động hợp lệ (1=có thể đi)
    
    # Trạng thái hoàn thành
    "done": [batch_size],                            # Đánh dấu episode kết thúc
}
```

### 3.2. Các ràng buộc trong Action Mask
Action mask được tính toán dựa trên:
1. **Điểm chưa thăm**: Chỉ có thể thăm các điểm chưa được thăm
2. **Sức chứa còn lại**: `demand <= (capacity - used_capacity)`
3. **Về depot**: Chỉ cho phép về depot khi:
   - Chưa hoàn thành tất cả khách hàng
   - Hoặc không còn điểm nào có thể thăm (đã thăm hết hoặc vượt sức chứa)

---

## 4. ENCODER - PARCOENCODER

### 4.1. Cấu hình Encoder
```python
env_name: str = "hcvrp"              # Tên môi trường
num_heads: int = 8                   # Số attention heads
embed_dim: int = 128                 # Kích thước embedding
num_layers: int = 3                  # Số lớp Transformer
normalization: str = "instance"      # Loại normalization
use_final_norm: bool = False         # Sử dụng normalization cuối
norm_after: bool = False             # Áp dụng norm sau hay trước
use_pos_token: bool = False          # Sử dụng POS token
trainable_pos_token: bool = True     # POS token có trainable không
```

### 4.2. Initial Embedding (HCVRPInitEmbedding)

#### Đầu vào từ TensorDict:
```python
# Từ môi trường
td["locs"]: [B, m+N, 2]              # Vị trí depot + khách hàng
td["capacity"]: [B, m]               # Sức chứa các xe
td["speed"]: [B, m]                  # Tốc độ các xe
td["demand"]: [B, m, m+N]            # Nhu cầu (lặp lại cho mỗi xe)
td["action_mask"]: [B, m, m+N]       # Để lấy num_agents
```

#### Các thành phần được embedding:

**a) Depot Embedding với Positional Encoding:**
```python
depot_locs: [B, m, 2]                           # Tọa độ depot (lặp lại m lần)
depots_embedding: [B, m, embed_dim]             # Sau Linear(2, embed_dim)
pos_embedding: [B, m, embed_dim]                # Positional encoding
depot_embedding: [B, m, embed_dim]              # = depots_embedding + α * pos_embedding
```

**b) Agents (Vehicles) Embedding:**
```python
agents_locs: [B, m, 2]                          # Tọa độ xe (ban đầu = depot)
capacity: [B, m, 1]                             # Sức chứa / demand_scaler (40.0)
speed: [B, m, 1]                                # Tốc độ / speed_scaler (1.0)
agents_feats: [B, m, 4]                         # Concat [locs, capacity, speed]
agents_embedding: [B, m, embed_dim]             # Sau Linear(4, embed_dim)
```

**c) Kết hợp Depot + Agents:**
```python
depot_agents_feats: [B, m, 2*embed_dim]         # Concat [depot_emb, agents_emb]
depot_agents_embedding: [B, m, embed_dim]       # Sau Linear(2*embed_dim, embed_dim)
```

**d) Clients (Customers) Embedding:**
```python
clients_locs: [B, N, 2]                         # Tọa độ khách hàng
demands: [B, N, 1]                              # Nhu cầu / demand_scaler (40.0)
clients_feats: [B, N, 3]                        # Concat [locs, demand]

# Nếu use_polar_feats = True (mặc định):
dist_to_depot: [B, N, 1]                        # Khoảng cách đến depot
angle_to_depot: [B, N, 1]                       # Góc so với depot
clients_feats: [B, N, 5]                        # Concat [locs, demand, dist, angle]

clients_embedding: [B, N, embed_dim]            # Sau Linear(5, embed_dim)
```

#### Đầu ra Initial Embedding:
```python
init_h: [B, m+N, embed_dim]                     # Concat [depot_agents_emb, clients_emb]
```

### 4.3. Encoder Layers (Transformer)
```python
# Qua num_layers lớp TransformerBlock
for layer in layers:
    h = layer(h, mask)                          # Self-attention + FFN

# Nếu use_final_norm = True
h = norm(h)                                     # RMS Normalization hoặc Instance Norm
```

### 4.4. Đầu ra Encoder
```python
h: [B, m+N, embed_dim]                          # Latent representation
init_h: [B, m+N, embed_dim]                     # Initial embedding (để sử dụng sau)
```

**Lưu ý về POS Token:**
Nếu `use_pos_token = True`, một token đặc biệt "pause-of-sequence" được thêm vào:
```python
init_h: [B, m+N+1, embed_dim]                   # Thêm 1 POS token
h: [B, m+N+1, embed_dim]                        # Sau encoder
```

---

## 5. DECODER - PARCODECODER

### 5.1. Cấu hình Decoder
```python
embed_dim: int = 128                 # Kích thước embedding
num_heads: int = 8                   # Số attention heads
env_name: str = "hcvrp"              # Tên môi trường
use_graph_context: bool = False      # PARCO không dùng graph context
use_pos_token: bool = False          # Sử dụng POS token
```

### 5.2. Context Embedding (HCVRPContextEmbedding)

Context embedding tạo ra thông tin về trạng thái hiện tại của mỗi agent.

#### Đầu vào từ TensorDict:
```python
td["current_node"]: [B, m]           # Vị trí hiện tại của mỗi xe
td["depot_node"]: [B, m]             # Vị trí depot
td["current_length"]: [B, m]         # Quãng đường đã đi
td["agents_speed"]: [B, m]           # Tốc độ xe
td["agents_capacity"]: [B, m]        # Tổng sức chứa
td["used_capacity"]: [B, m]          # Đã sử dụng
td["visited"]: [B, m+N]              # Đã thăm
td["locs"]: [B, m+N, 2]              # Vị trí
embeddings: [B, m+N, embed_dim]      # Từ encoder
```

#### Các thành phần Context:

**a) Current Node Embedding:**
```python
cur_node_embedding: [B, m, embed_dim]   # gather_by_index(embeddings, current_node)
```

**b) Depot Embedding:**
```python
depot_embedding: [B, m, embed_dim]      # gather_by_index(embeddings, depot_node)
```

**c) Agent State Features:**
```python
current_time: [B, m]                    # current_length / speed
remaining_capacity: [B, m]              # (capacity - used_capacity) / demand_scaler

# Nếu use_time_to_depot = True (mặc định):
dist_to_depot: [B, m, 1]                # ||cur_loc - depot||
time_to_depot: [B, m, 1]                # dist_to_depot / speed

agent_state_feats: [B, m, 3]            # [current_time, remaining_capacity, time_to_depot]
agent_state_embed: [B, m, embed_dim]    # Sau Linear(3, embed_dim)
```

**d) Global State Features:**
```python
visited_ratio: [B, 1]                   # visited[m:].sum() / num_cities
global_feats: [B, 1]                    # [visited_ratio]
global_embed: [B, m, embed_dim]         # Sau Linear(1, embed_dim), repeat cho m agents
```

**e) Kết hợp Context:**
```python
context_embed: [B, m, 4*embed_dim]      # Concat [cur_node, depot, agent_state, global]
context_embed: [B, m, embed_dim]        # Sau Linear(4*embed_dim, embed_dim)
```

### 5.3. Communication Layers
```python
# Qua num_communication_layers lớp (mặc định = 1)
for layer in communication_layers:
    h_comm = layer(context_embed)       # TransformerBlock với self-attention

# Nếu use_final_norm = True
h_comm = norm(h_comm)                   # Normalization
```

### 5.4. Glimpse Query (Q)
```python
glimpse_q: [B, m, embed_dim]            # step_context (từ communication)
```

### 5.5. Glimpse Key, Value, Logit Key (K, V, L)
```python
# Từ encoder embeddings
embeddings: [B, m+N, embed_dim]         # (hoặc [B, m+N+1, embed_dim] nếu có POS token)

# Project qua 3 heads
glimpse_key: [B, m+N, embed_dim]        # Key cho attention
glimpse_val: [B, m+N, embed_dim]        # Value cho attention  
logit_key: [B, m+N, embed_dim]          # Key cho logits
```

### 5.6. Pointer Attention & Logits
```python
# Multi-head attention
logits: [B, m, m+N]                     # Điểm số cho mỗi action
# (hoặc [B, m, m+N+1] nếu có POS token)

# Apply mask
mask: [B, m, m+N]                       # Action mask từ environment
logits = logits.masked_fill(~mask, -inf)
```

### 5.7. Đầu ra Decoder
```python
logits: [B, m, m+N]                     # Điểm số cho mỗi action của mỗi agent
mask: [B, m, m+N]                       # Action mask
```

---

## 6. MODEL - PARCOPOLICY

### 6.1. Cấu hình Model
```python
encoder: PARCOEncoder                # Encoder (xem mục 4)
decoder: PARCODecoder                # Decoder (xem mục 5)
embed_dim: int = 128                 # Kích thước embedding
num_encoder_layers: int = 3          # Số lớp encoder
num_heads: int = 8                   # Số attention heads
train_decode_type: str = "sampling"  # Cách decode khi train
val_decode_type: str = "greedy"      # Cách decode khi validation
test_decode_type: str = "greedy"     # Cách decode khi test
agent_handler: str = "highprob"      # Cách xử lý conflict giữa agents
use_init_logp: bool = True           # Sử dụng log prob ban đầu
mask_handled: bool = False           # Mask các action đã xử lý
replacement_value_key: str = "current_node"  # Key để replace khi conflict
use_pos_token: bool = False          # Sử dụng POS token
```

### 6.2. Forward Pass

#### Đầu vào:
```python
td: TensorDict                       # State từ environment
env: HCVRPEnv                        # Environment
phase: str = "train"                 # train/val/test
```

#### Quy trình:

**Bước 1: Encoder**
```python
hidden, init_embeds = encoder(td)
# hidden: [B, m+N, embed_dim]
# init_embeds: [B, m+N, embed_dim]
```

**Bước 2: Setup Decoding Strategy**
```python
num_agents = td["action_mask"].shape[-2]  # m
decode_strategy = PARCODecodingStrategy(
    decode_type=decode_type,          # sampling/greedy/evaluate
    num_agents=num_agents,
    agent_handler=agent_handler,      # highprob/random/sequential
    num_samples=num_samples,          # Số lượng sample (cho sampling)
)
```

**Bước 3: Pre-decoder Hook**
```python
td, env, num_samples = decode_strategy.pre_decoder_hook(td, env)
td, env, hidden = decoder.pre_decoder_hook(td, env, hidden, num_samples)
```

**Bước 4: Main Decoding Loop**
```python
while not td["done"].all():
    # Tính logits
    logits, mask = decoder(td, hidden, num_samples)
    # logits: [B, m, m+N]
    # mask: [B, m, m+N]
    
    # Chọn action
    td = decode_strategy.step(logits, mask, td)
    # td["action"]: [B, m] - action cho mỗi agent
    
    # Thực hiện action trong environment
    td = env.step(td)["next"]
    
    step += 1
```

**Bước 5: Post-decoder Hook**
```python
logprobs, actions, td, env, halting_ratio = decode_strategy.post_decoder_hook(td, env)
# logprobs: [B, m, steps] - Log probability của actions
# actions: [B, m, steps] - Sequence of actions
# halting_ratio: Tỷ lệ sử dụng POS token (nếu có)
```

**Bước 6: Tính Reward**
```python
td["reward"] = env.get_reward(td, actions)
# reward: [B] - Negative của max time (makespan)

# Reward calculation:
current_length = td["current_length"]  # [B, m]
current_time = current_length / td["agents_speed"]  # [B, m]
max_time = current_time.max(dim=1)[0]  # [B]
reward = -max_time  # Maximize reward = minimize makespan
```

### 6.3. Đầu ra Model
```python
{
    "reward": [B],                   # Reward (negative makespan)
    "log_likelihood": [B] or [B, m, steps],  # Log probability
    "actions": [B, m, steps],        # Sequence of actions
    "init_embeds": [B, m+N, embed_dim],  # Initial embeddings
    "halting_ratio": scalar,         # Tỷ lệ POS token
    "steps": int,                    # Số bước decode
}
```

---

## 7. TỔNG HỢP CÁC SCALERS

Các hệ số chuẩn hóa được sử dụng trong embedding:

```python
demand_scaler: float = 40.0          # Chuẩn hóa demand
speed_scaler: float = 1.0            # Chuẩn hóa speed
```

**Lưu ý:** Capacity không được normalize theo default trong Generator (scale_data=False), nhưng được normalize trong embedding với cùng giá trị `demand_scaler = 40.0`.

---

## 8. CHIỀU CỦA CÁC TENSOR QUAN TRỌNG

### Ký hiệu:
- `B`: batch_size
- `m`: num_agents (số xe)
- `N`: num_loc (số khách hàng)
- `H`: embed_dim

### Bảng tổng hợp:

| Tên Tensor | Shape | Mô tả |
|-----------|-------|-------|
| **Input từ Generator** | | |
| locs | [B, N, 2] | Tọa độ khách hàng |
| depot | [B, 2] | Tọa độ depot |
| demand | [B, N] | Nhu cầu khách hàng |
| capacity | [B, m] | Sức chứa các xe |
| speed | [B, m] | Tốc độ các xe |
| **State từ Environment** | | |
| locs | [B, m+N, 2] | Depot (x m) + Khách hàng |
| demand | [B, m, m+N] | Demand (lặp cho mỗi xe) |
| current_node | [B, m] | Vị trí hiện tại |
| depot_node | [B, m] | Vị trí depot |
| current_length | [B, m] | Quãng đường đã đi |
| used_capacity | [B, m] | Sức chứa đã dùng |
| visited | [B, m+N] | Đánh dấu đã thăm |
| action_mask | [B, m, m+N] | Mask hành động |
| **Encoder** | | |
| init_h | [B, m+N, H] | Initial embedding |
| hidden | [B, m+N, H] | Encoder output |
| **Decoder** | | |
| context_embed | [B, m, H] | Context cho mỗi agent |
| glimpse_q | [B, m, H] | Query |
| glimpse_k, glimpse_v | [B, m+N, H] | Key, Value |
| logits | [B, m, m+N] | Điểm số action |
| **Output** | | |
| actions | [B, m, steps] | Sequence actions |
| logprobs | [B, m, steps] | Log probabilities |
| reward | [B] | Reward (negative makespan) |

---

## 9. CÁC THAM SỐ QUAN TRỌNG THEO CONFIG

Từ file `configs/experiment/hcvrp.yaml`:

```yaml
# Environment
num_loc: 100                         # Số khách hàng
num_agents: 7                        # Số xe

# Model
embed_dim: 128                       # Kích thước embedding
num_encoder_layers: 3                # Số lớp encoder (mặc định)
num_heads: 8                         # Số attention heads (mặc định)
normalization: "rms"                 # RMS normalization
use_final_norm: true                 # Normalize cuối cùng
norm_after: false                    # Normalize trước (pre-norm)

# Context Embedding
use_communication: true              # Sử dụng communication layers
num_communication_layers: 1          # Số lớp communication
use_final_norm: true                 # Normalize cuối communication

# Training
batch_size: 128
train_min_agents: 3                  # Min số xe khi train
train_max_agents: 7                  # Max số xe khi train
train_min_size: 60                   # Min số khách hàng
train_max_size: 100                  # Max số khách hàng
num_augment: 8                       # Data augmentation

# Agent Handler
agent_handler: "highprob"            # Xử lý conflict: chọn agent có prob cao nhất
```

---

## 10. KẾT LUẬN

Model PARCO cho HCVRP sử dụng kiến trúc Transformer với các đặc điểm chính:

### Đầu vào chính:
1. **Vị trí**: Depot và khách hàng (tọa độ 2D)
2. **Nhu cầu**: Demand của mỗi khách hàng
3. **Đặc tính xe**: Capacity và Speed khác nhau cho mỗi xe
4. **Trạng thái động**: Vị trí hiện tại, sức chứa còn lại, đã thăm

### Kiến trúc:
1. **Encoder**: Tạo embedding từ vị trí, nhu cầu, capacity, speed với positional encoding và polar features
2. **Decoder**: Sử dụng context embedding (current state + global state) với communication layers cho multi-agent coordination
3. **Pointer Network**: Multi-head attention để tính logits cho việc chọn action

### Đặc điểm nổi bật:
- **Multi-agent**: Xử lý nhiều xe cùng lúc với communication layer
- **Heterogeneous**: Mỗi xe có capacity và speed khác nhau
- **Dynamic masking**: Action mask dựa trên visited và capacity constraints
- **Makespan optimization**: Minimize thời gian hoàn thành lớn nhất

File này tổng hợp đầy đủ các trường dữ liệu đi qua encoder, decoder và model trong HCVRP environment.
