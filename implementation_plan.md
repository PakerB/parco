# PVRPWDP Dynamic Embedding — Implementation Plan

## Mục tiêu

Giúp model PVRPWDP:
- Biết node nào khẩn cấp, khó phục vụ
- Agents trao đổi "claim" để phân chia node hợp lý
- Pointer attention nhận trực tiếp tín hiệu urgency


## Quyết định thiết kế

| # | Câu hỏi | Chốt |
|---|---------|------|
| Q1 | node_emb cho claim_embed | Encoder output (đã có sẵn) |
| Q2 | Attention injection | Cài cả MLP + Projected additive, config switch |
| Q3 | Cache strategy | Tính trong env, cache vào td |
| Q4 | Temperature | Không cần (Linear scale implicit) |


## Lưu ý bài toán

Service duration = 0 trong PVRPWDP.
Agent đến node → pickup ngay → rời đi ngay.

Các mốc thời gian per (agent k, node i):
- arrival[k,i] = current_time[k] + travel_time(k to i)
- ready_time[k,i] = max(arrival[k,i], earliest_i)
  Nếu đến sớm hơn earliest thì phải chờ.
  Vì service_duration = 0, ready_time CŨNG LÀ departure_time.


------------------------------------------------------------------------
## PHẦN 1: ENV — Cache matrices vào td
------------------------------------------------------------------------

File: parco/envs/pvrpwdp/env.py

get_action_mask() hiện đã tính sẵn (dùng lại, không tính lại):
- travel_time_to_nodes [B, M, M+N]
- arrival_time [B, M, M+N]
- ready_time (trong code gọi là service_time) [B, M, M+N]
- travel_time_to_depot [B, M, M+N]

Cần tính thêm và cache vào td:


### 1.1 slack_matrix [B, M, M+N]

Công thức:
    slack[k,i] = latest_i - ready_time[k,i]

Ý nghĩa: agent k đến node i còn dư bao nhiêu thời gian.

Ví dụ:
    Node i: earliest=3, latest=10
    Agent k: arrival = 5 → ready_time = max(5, 3) = 5
    slack = 10 - 5 = 5 → "dư 5 đơn vị, thoải mái"

    Agent k: arrival = 9 → ready_time = max(9, 3) = 9
    slack = 10 - 9 = 1 → "gần deadline, rất gấp"

Node bị mask (unreachable): set slack = 0.


### 1.2 future_slack [B, M, N] (chỉ customer reachable)

Phạm vi tính:
    i ∈ {customer mà action_mask[k,i] = True}  (agent k có thể đến)
    j ∈ {customer chưa visited, j ≠ i}           (unvisited)
    KHÔNG tính depot.
    Nếu action_mask[k,i] = False → bỏ qua, set future_slack[k,i] = 0.

Công thức:
    future_slack[k,i,j] = latest_j - (ready_time[k,i] + travel_time(i to j) / speed_k)

Ý nghĩa: nếu agent k phục vụ node i trước, liệu còn kịp đến node j không?

CÓ THỂ ÂM. Giá trị âm là thông tin quan trọng nhất:
- future_slack >= 0: sau khi đi i, vẫn kịp j → an toàn
- future_slack < 0:  sau khi đi i, KHÔNG kịp j → j bị miss

Ví dụ chi tiết:
    Agent A có action_mask[A] = [True, True, False, True, False] cho 5 customer
    Chỉ tính future_slack cho node 0, 1, 3 (bỏ node 2, 4)

    Agent A phục vụ node 1 xong, ready_time[A,1] = 7
    Đi từ node 1 tới node 3 mất 4 phút → đến node 3 lúc t=11
    Node 3 có latest = 9

    future_slack[A, 1, 3] = 9 - 11 = -2 → ÂM
    "Nếu A đi node 1 trước, sẽ trễ node 3"

Aggregate:
    min_future_slack[k,i] = min over j (unvisited customers only)
    Nếu action_mask[k,i] = False → min_future_slack[k,i] = 0

Complexity: O(B x M x R x N) trong đó R = số reachable nodes (R <= N).
Với R << N: tiết kiệm đáng kể so với tính đủ N.


### 1.3 reachability_loss [B, M, N] (chỉ customer reachable)

Phạm vi tính:
    i ∈ {customer mà action_mask[k,i] = True}  (agent k có thể đến)
    j ∈ {customer mà action_mask[k,j] = True}  (agent k hiện đang reach)
    KHÔNG tính depot.
    Nếu action_mask[k,i] = False → reachability_loss[k,i] = 0.

Agent k reach được node j SAU KHI đi node i, cần thỏa CẢ 3 điều kiện:

    TIME:
        ready_time[k,i] + travel(i→j)/speed_k <= latest_j
        (kịp đến j trước deadline)

    ENDURANCE:
        used_endurance[k]
        + endurance_to_i             (travel + wait nếu mid-trip)
        + travel(i→j)/speed_k
        + wait_at_j                  (nếu đến sớm hơn earliest_j)
        + travel(j→depot)/speed_k
        <= max_endurance[k]
        (đủ pin cho cả chuyến)

    CAPACITY:
        remaining_capacity[k] - demand[i] >= demand[j]
        (pickup i rồi vẫn đủ chỗ cho j)

    Ví dụ CAPACITY:
        Agent k capacity còn 8, demand[i]=5, demand[j]=4
        Sau pickup i: 8-5 = 3 < 4 → KHÔNG đủ chỗ cho j
        Dù time và endurance OK → vẫn mất j

    Ví dụ ENDURANCE:
        Agent k endurance còn 10, đi tới i mất 3, chờ ở i mất 1
        Đi tiếp j mất 4, về depot mất 5 → tổng = 3+1+4+5 = 13 > 10
        Dù time OK → hết pin, mất j

Tính feasibility và loss:

    k_reaches_j_now     = action_mask[k, j]     (bool)
    k_feasible_j_after_i = TIME ok AND ENDURANCE ok AND CAPACITY ok
    k_loses_j           = k_reaches_j_now AND (NOT k_feasible_j_after_i)

    loss_weight[j] = 1 / num_reachable[j]

    reachability_loss[k,i] = sum_j (k_loses_j × loss_weight[j]) / total_weight_k

    total_weight_k = sum over reachable j of (1 / num_reachable[j])
    Chia total_weight_k để normalize về [0, 1].
    Mất hết → 1.0, không mất → 0.0.

Ví dụ:
    Agent A reach 4 nodes:
        node 3 (num_reachable=2) → weight = 1/2
        node 5 (num_reachable=2) → weight = 1/2
        node 7 (num_reachable=1) → weight = 1/1
        node 9 (num_reachable=3) → weight = 1/3
        total_weight_A = 1/2 + 1/2 + 1/1 + 1/3 = 2.333

    Agent A đi node 3, mất node 5 và node 7:
        loss = (1/2 + 1/1) / 2.333 = 1.5 / 2.333 = 0.643
        → "mất 64% weighted coverage"

    Cuối episode, Agent A reach 2 nodes:
        node 7 (num_reachable=1) → weight = 1/1
        node 9 (num_reachable=1) → weight = 1/1
        total_weight_A = 2.0

        Đi node 9, mất node 7:
        loss = 1.0 / 2.0 = 0.5
        → "mất 50% weighted coverage, nghiêm trọng"


### 1.4 num_reachable [B, N]

    num_reachable[i] = sum over agents of action_mask[k, i]
    Chỉ customer nodes.

Ví dụ:
    3 agents, node 5 chỉ agent B reach → num_reachable[5] = 1
    "Node 5 kén chọn — chỉ 1 agent phục vụ được"


### 1.5 Tổ chức code

Thêm method mới, KHÔNG sửa get_action_mask():

    def _cache_heuristic_features(self, td):
        """Tính và cache các matrices vào td."""
        ...
        td.update({
            "slack_matrix":       ...,   # [B, M, M+N]
            "future_slack_matrix": ...,  # [B, M, N]
            "reachability_loss":  ...,   # [B, M, N]
            "num_reachable":      ...,   # [B, N]
        })

Gọi trong _reset() và _step() sau khi set action_mask.


### 1.6 Tối ưu VRAM: Loop over agents

future_slack và reachability_loss cần tensor [B, M, N, N].
Với B=64, M=12, N=100: 64×12×100×100×4 = 30.7 MB chỉ cho 1 tensor.
Cộng thêm intermediate (feasibility checks): peak ~120 MB.
Training (autograd giữ gradient): peak ~240 MB.
→ KHÔNG chấp nhận được.

Giải pháp: Loop từng agent, giảm peak từ [B, M, N, N] → [B, N, N].

    # Pre-allocate outputs
    min_future_slack = zeros(B, M, N)
    reachability_loss = zeros(B, M, N)

    # Tính 1 lần, dùng chung (node locs không đổi)
    inter_customer_dist = cdist(customer_locs, customer_locs)   # [B, N, N]

    for k in range(M):
        # Chỉ [B, N, N] thay vì [B, M, N, N]
        inter_travel_time = inter_dist / speed_k               # [B, N, N]
        arrival_at_j = departure_from_i + inter_travel_time     # [B, N, N]

        # 3 feasibility checks, mỗi cái [B, N, N]
        time_ok     = arrival_at_j <= latest_j
        endurance_ok = total_endurance <= max_endurance_k
        capacity_ok  = remaining_cap_k - demand_i >= demand_j

        feasible = time_ok & endurance_ok & capacity_ok         # [B, N, N]

        # Aggregate ngay, không lưu full tensor
        min_future_slack[:, k] = fs_k.min(dim=-1)[0]           # [B, N]
        reachability_loss[:, k] = (k_loses * weight).sum(-1)/N # [B, N]

        # fs_k, feasible, arrival_at_j → freed khi loop tiếp

Peak memory sau tối ưu:

    Constant:
        inter_customer_dist  [B, N, N] = 2.56 MB

    Per-agent (loop, freed mỗi vòng):
        inter_travel_time    [B, N, N] = 2.56 MB
        arrival_at_j         [B, N, N] = 2.56 MB
        feasible checks      [B, N, N] = 0.64 MB (bool)
        fs_k                 [B, N, N] = 2.56 MB
        endurance calc       [B, N, N] = 2.56 MB
        → Peak per-agent: ~13 MB

    Outputs (giữ lại trong td):
        slack_matrix         [B, M, M+N] = 0.34 MB
        min_future_slack     [B, M, N]   = 0.31 MB
        reachability_loss    [B, M, N]   = 0.31 MB
        num_reachable        [B, N]      = 0.025 MB
        → Total trong td: ~1 MB

    TỔNG PEAK: ~16 MB (giảm ~8x so với không loop)
    Training (×2 gradient): ~32 MB → chấp nhận được

Trade-off: chậm hơn do loop M=12 lần thay vì vectorized.
Nhưng mỗi vòng vẫn vectorized trên B và N.
Với M=12: overhead không đáng kể so với tiết kiệm VRAM.


### 1.7 Cache tensor tĩnh trong _reset()

Một số tensor không đổi giữa các steps, chỉ tính 1 lần:

    # Trong _reset(), tính và lưu vào td:
    inter_customer_dist = cdist(customer_locs, customer_locs)  # [B, N, N]
    dist_to_depot_static = cdist(depot_locs, all_locs)         # [B, M, M+N]
    eye_mask = eye(M).expand(B, M, M).bool()                   # [B, M, M]

    td.update({
        "inter_customer_dist": inter_customer_dist,  # dùng cho future_slack
        "dist_to_depot_static": dist_to_depot_static, # dùng cho get_action_mask
        "depot_eye_mask": eye_mask,                    # dùng cho get_action_mask
    })

Lợi ích:
    - get_action_mask() bỏ 2 lần cdist mỗi step
    - _cache_heuristic_features() dùng chung inter_customer_dist
    - eye_matrix không tạo lại mỗi step


------------------------------------------------------------------------
## PHẦN 2: Dynamic Embedding — Thay StaticEmbedding
------------------------------------------------------------------------

File: parco/models/env_embeddings/pvrpwdp.py (thêm class mới)

### 2.1 PVRPWDPDynamicEmbedding

Hiện tại PVRPWDP dùng StaticEmbedding (return 0 → K/V không đổi).
Thay bằng embedding 5 features per node:

    1. min_slack[i] / time_scaler
       = min over agents of slack[k,i]
       Ý nghĩa: node i gấp cỡ nào (best case từ bất kỳ agent)

    2. mean_slack[i] / time_scaler
       = mean over agents of slack[k,i]
       Ý nghĩa: node i khó phục vụ cỡ nào (trung bình)

    3. num_reachable[i] / num_agents
       Ý nghĩa: bao nhiêu % agents reach được (0~1)

    4. min_future_slack[i] / time_scaler
       = min over agents of future_slack[k,i]
       Ý nghĩa: node i có gây miss node khác không

    5. is_visited[i]
       = 0.0 hoặc 1.0

Output: Linear(5 → 3*D) → chunk → (dyn_k, dyn_v, dyn_l)
Cộng vào K, V, logit_key tĩnh từ encoder.


### 2.2 Node indexing

Encoder output: [B, M+N, D] (M depots + N customers).
Dynamic embedding: [B, N, D] (chỉ N customers).
→ Pad M zeros phía trước để match shape.


### 2.3 Đăng ký

File: parco/models/env_embeddings/__init__.py
    "pvrpwdp": PVRPWDPDynamicEmbedding   (thay StaticEmbedding)


------------------------------------------------------------------------
## PHẦN 3: Context Embedding — Claim embed + Agent features
------------------------------------------------------------------------

File: parco/models/env_embeddings/pvrpwdp.py (sửa existing)
File: parco/models/env_embeddings/communication.py (sửa base)


### 3.1 Thêm claim_embed vào context

Hiện tại:
    context = concat(cur_node_emb, depot_emb, agent_state_emb, global_emb)
    → Linear(4D → D) → communication

Thêm:
    context = concat(cur_node_emb, depot_emb, agent_state_emb, global_emb, claim_emb)
    → Linear(5D → D) → communication


### 3.2 Claim embed tính như sau

6 features per (agent k, customer node i):

    1. -slack[k,i] / time_scaler
       Urgency. Slack thấp → score cao.

    2. 1 / (num_reachable[i] + 1)
       Exclusivity. Ít agent reach → score cao.

    3. travel_time_to_depot[k,i] / time_scaler
       Xa depot. Node xa depot khó phục vụ sau.

    4. -min_future_slack[k,i] / time_scaler
       Forward risk. Đi node i gây miss nhiều → score cao.

    5. demand[i] / remaining_capacity[k]
       Capacity fit. Node chiếm nhiều capacity → lưu ý.

    6. -reachability_loss[k,i]     (đã normalize / R_k)
       Reachability risk. Đi node i mất nhiều coverage → score cao.

Kết hợp:
    score = Linear(6 → 1)(features)         [B, M, N]
    weights = softmax(score, dim=-1)         [B, M, N]
    claim_embed = bmm(weights, node_emb)     [B, M, D]

    node_emb = encoder output (customer phần) = embeddings[:, M:, :]

Mask: softmax chỉ trên reachable nodes (unreachable → -inf trước softmax).


### 3.3 Thêm 2 agent-level scalar features

Hiện tại agent_feat_dim = 6.
Thêm 2 → tổng = 8:

    7. min_slack_of_agent[k] / time_scaler
       = min over reachable nodes of slack[k,i]
       "Agent k gấp nhất ở mức nào"

    8. num_reachable_nodes[k] / N
       = count reachable customers / total customers
       "Agent k reach được bao nhiêu % nodes"


### 3.4 Sửa communication.py

    project_context: Linear(4D → D) thành Linear(5D → D)
    Thêm: claim_score_proj = Linear(6, 1)
    Conditional: use_claim_embed=True/False


------------------------------------------------------------------------
## PHẦN 4: Pointer Attention — Injection
------------------------------------------------------------------------

File: parco/models/decoder.py (sửa PARCODecoder)
File: parco/models/nn/attention_injection.py (NEW)


### 4.1 Config

    attention_injection_mode = "none" | "mlp" | "projected"
    "none" = backward compatible.


### 4.2 Mode "mlp" — MLP per-head (MatNet-style)

Trộn dot-product score với 3 heuristic features, per head:

    dot_score:          [B, H, M, M+N]
    3 features (expand to H heads):
        slack / time_scaler
        future_slack / time_scaler
        reachability_loss           (đã normalize / R_k)

    combined = stack([
        dot_score,
        slack_expanded,
        future_slack_expanded,
        reachability_expanded,
    ], dim=-1)                       [B, H, M, M+N, 4]
                                                     ↑ 1 dot + 3 heuristic

    hidden = relu(Linear(4 → 16)(combined))    [B, H, M, M+N, 16]
    score = Linear(16 → 1)(hidden).squeeze(-1)  [B, H, M, M+N]

Ví dụ:
    Head 0 học: "khi dot_score thấp + slack thấp → tăng score (cứu node gấp)"
    Head 1 học: "chỉ dùng dot_score, bỏ hết heuristic"
    Mỗi head tự quyết cách trộn.


### 4.3 Mode "projected" — Linear additive

3 features cộng thẳng vào score:

    features = stack([
        slack / time_scaler,
        future_slack / time_scaler,
        reachability_loss,              (đã normalize / R_k)
    ], dim=-1)                       [B, M, N, 3]

    bias = Linear(3 → 1)(features)   [B, M, N]
    score = dot_score + bias          broadcast over heads

Ví dụ:
    Node i có slack=0.1, future_slack=-0.2, reachability_loss=0.075
    → bias = w1*0.1 + w2*(-0.2) + w3*0.075 + b
    → Nếu model học w2 < 0: future_slack âm → bias giảm → ít chọn node i


### 4.4 Nơi inject

Trong decoder.forward(), SAU pointer TRƯỚC tanh clipping:

    logits = self.pointer(...)

    if self.attention_injection is not None:
        bias = self.attention_injection(td, ...)
        logits = logits + bias


------------------------------------------------------------------------
## PHẦN 5: Files cần sửa
------------------------------------------------------------------------

[MODIFY] parco/envs/pvrpwdp/env.py
    - Thêm _cache_heuristic_features(td) với loop-over-agents
    - Sửa _reset(): cache tensor tĩnh + gọi heuristic cache
    - Sửa _step(): gọi heuristic cache
    - get_action_mask(): dùng cached dist_to_depot, eye_mask
    - Bỏ demand repeat M lần (dùng broadcast thay thế)
    - Bỏ endurance_to_node.clone() không cần
    - Cache scalers (demand_scaler, time_scaler) vào td

[MODIFY] parco/models/env_embeddings/pvrpwdp.py
    - Thêm class PVRPWDPDynamicEmbedding
    - Sửa PVRPWDPContextEmbedding: agent_feat_dim 6→8, thêm claim_embed
    - Dùng cached scalers từ td thay vì tính lại mỗi step

[MODIFY] parco/models/env_embeddings/communication.py
    - project_context dim: 4D → 5D
    - forward(): thêm claim_embed
    - Thêm claim_score_proj layer

[MODIFY] parco/models/env_embeddings/__init__.py
    - Đăng ký PVRPWDPDynamicEmbedding

[MODIFY] parco/models/decoder.py
    - Thêm attention_injection_mode param
    - Thêm injection logic

[NEW] parco/models/nn/attention_injection.py
    - MLPAttentionInjection
    - ProjectedAdditiveInjection


------------------------------------------------------------------------
## PHẦN 6: Verification
------------------------------------------------------------------------

### Smoke test
- Forward 1 batch → check shapes, no NaN/Inf
- Backward → gradients flow

### Ablation
1. Baseline (StaticEmbedding, no injection)
2. +Dynamic embedding only
3. +Dynamic +Context (claim_embed + agent features)
4. +Dynamic +Context +Inject(projected)
5. +Dynamic +Context +Inject(mlp)

### Sanity checks
- claim_embed thay đổi mỗi step
- slack_matrix >= 0 cho reachable nodes
- inject_bias magnitude vs dot_score magnitude


------------------------------------------------------------------------
## PHẦN 7: Rủi ro & Performance
------------------------------------------------------------------------

Memory (với B=64, M=12, N=100):
    Cache trong td mỗi step: ~1 MB (slack + future_slack + reachability + num_reachable)
    Cache tĩnh (1 lần): ~2.6 MB (inter_customer_dist) + 0.34 MB (dist_to_depot)
    Peak compute: ~16 MB (loop over agents) → ~32 MB khi training
    → Chấp nhận được.

Compute:
    future_slack: O(B*M*N^2). Loop over agents: 12 vòng × O(B*N^2).
    B=64, N=100: 64×100×100 = 640K ops/vòng × 12 = 7.7M ops/step.
    → Chấp nhận được. N=200+: cần top-k thay all-pairs.

Tiết kiệm từ optimization code cũ:
    - Bỏ cdist trong get_action_mask(): ~172K distance ops/step
    - Bỏ demand repeat: 344KB → 29KB trong td
    - Bỏ scaler recompute: ~6 max() ops/step trong embedding
    - Bỏ eye_matrix recreate: nhỏ nhưng sạch hơn

Backward compatible:
    attention_injection_mode="none" giữ behavior cũ.
    Dynamic embedding fallback return 0 nếu td thiếu cached tensors.

Checkpoint:
    Model cũ không load weights mới. Dùng strict=False.
