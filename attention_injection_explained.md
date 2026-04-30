# Attention Injection — Giải thích chi tiết & So sánh với MatNet

## 1. Tổng quan

File `attention_injection.py` cung cấp 2 cách inject heuristic features vào **pointer attention** (bước cuối cùng chọn node tiếp theo). Cả 2 đều nhận `dot_score` (attention score gốc từ Q·K) và 3 heuristic features, rồi trả về score đã được điều chỉnh.

### 3 Heuristic Features

| Feature | Shape | Ý nghĩa |
|---------|-------|---------|
| `slack` | `[B, M, M+N]` | Thời gian dư = latest_tw - (arrival_time + waiting_time). Slack nhỏ → node khẩn cấp |
| `future_slack` | `[B, M, N]` | Min slack của các node lân cận nếu đi node i. Âm → sẽ miss node lân cận |
| `reachability_loss` | `[B, M, N]` | Tỉ lệ node mất khi đi node i (do capacity/endurance hết). Cao → rủi ro |

---

## 2. MLPAttentionInjection — Giải thích từng khối

### 2.1 Khởi tạo

```python
def __init__(self, num_heads: int = 8, hidden_dim: int = 16):
    self.fc1 = nn.Linear(4, hidden_dim)    # 4 input: dot_score + 3 features
    self.fc2 = nn.Linear(hidden_dim, 1)    # output: 1 mixed score
```

- **Input dim = 4**: gồm `[dot_score, slack, future_slack, reachability_loss]`
- **hidden_dim = 16**: MLP nhỏ vì mỗi position `(b, h, k, n)` chỉ có 4 features
- **Output dim = 1**: trả về 1 score thay thế `dot_score` gốc
- **Weights chia sẻ** across positions: cùng 1 bộ weight `fc1, fc2` cho mọi `(agent, node)` pair

### 2.2 Chuẩn bị features

```python
# Normalize slack bằng time_scaler
slack = td["slack_matrix"] / ts  # [B, M, M+N]

# future_slack và reachability_loss chỉ cho customers [B, M, N]
# → pad depot columns với 0 để khớp shape [B, M, M+N]
future_slack_full = torch.zeros(B, M, N_total, ...)
future_slack_full[..., num_agents:] = td["min_future_slack"] / ts
```

> **Tại sao pad depot = 0?** Vì depot không có future_slack hay reachability_loss. Giá trị 0 = trung tính, không ảnh hưởng score depot.

### 2.3 Expand cho multi-head

```python
slack_exp = slack.unsqueeze(1).expand(-1, H, -1, -1)  # [B, M, M+N] → [B, H, M, M+N]
```

3 features giống nhau cho mọi head — expand (không copy memory) để khớp shape với `dot_score [B, H, M, N]`.

### 2.4 Stack + MLP

```python
combined = torch.stack([dot_score, slack_exp, fs_exp, rl_exp], dim=-1)  # [B, H, M, N, 4]
hidden = torch.relu(self.fc1(combined))   # [B, H, M, N, 16]
mixed_score = self.fc2(hidden).squeeze(-1)  # [B, H, M, N]
```

MLP tính **song song** cho mọi `(batch, head, agent, node)`:
```
score_final(b,h,k,n) = W2 · relu(W1 · [dot(b,h,k,n), slack(b,k,n), fs(b,k,n), rl(b,k,n)] + b1) + b2
```

> **Điểm quan trọng**: MLP **hoàn toàn thay thế** `dot_score` — output không cộng thêm vào `dot_score` mà tạo score mới từ 4 inputs. Model tự học cách phối hợp.

---

## 3. ProjectedAdditiveInjection — Giải thích từng khối

### 3.1 Khởi tạo

```python
def __init__(self):
    self.proj = nn.Linear(3, 1, bias=True)  # 3 features → 1 bias
```

- Chỉ **3 parameters** (3 weights + 1 bias)
- Không lấy `dot_score` làm input — chỉ dùng 3 heuristic features

### 3.2 Project features → bias

```python
features = torch.stack([slack, future_slack_full, reachability_full], dim=-1)  # [B, M, N, 3]
bias = self.proj(features).squeeze(-1)  # [B, M, N]
```

```
bias(b,k,n) = w1·slack + w2·future_slack + w3·reachability + b
```

### 3.3 Cộng bias vào dot_score

```python
score = dot_score + bias.unsqueeze(1)  # bias [B, 1, M, N] broadcast → [B, H, M, N]
```

> **Khác biệt quan trọng**: bias **giống nhau cho mọi head** và chỉ **cộng thêm** vào dot_score gốc, không thay thế.

---

## 4. So sánh với MatNet

### 4.1 MatNet MixedScoreFF (gốc)

```python
class MixedScoreFF(nn.Module):
    def __init__(self, num_heads, ms_hidden_dim):
        self.lin1 = nn.Linear(2 * num_heads, num_heads * ms_hidden_dim, bias=False)
        self.lin2 = nn.Linear(num_heads * ms_hidden_dim, 2 * num_heads, bias=False)

    def forward(self, dot_product_score, cost_mat_score):
        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=-1)  # [B, H, R, C, 2]
        two_scores = rearrange(two_scores, "b h r c s -> b r c (h s)")        # [B, R, C, 2H]
        ms = self.lin2(F.relu(self.lin1(two_scores)))                         # [B, R, C, 2H]
        mixed_scores = rearrange(ms, "b r c (h two) -> b h r c two", two=2)   # [B, H, R, C, 2]
        ms1, ms2 = mixed_scores.chunk(2, dim=-1)
        return ms1.squeeze(-1), ms2.squeeze(-1)                                # 2× [B, H, R, C]
```

### 4.2 So sánh chi tiết

| Tiêu chí | MatNet MixedScoreFF | MLPAttentionInjection | ProjectedAdditiveInjection |
|----------|-------------------|-----------------------|---------------------------|
| **Input** | 2 scores: dot + cost_mat | 4 values: dot + 3 heuristics | 3 heuristics (không dùng dot) |
| **Output** | 2 mixed scores (ms1, ms2) dùng cho V1 và V2 | 1 score thay thế dot_score | dot_score + bias |
| **Per-head?** | Có — rearrange `(h s)` trộn cross-head | Không — weights share across heads | Không — bias broadcast all heads |
| **Hidden dim** | `num_heads × ms_hidden_dim` (vd: 16×32 = 512) | `hidden_dim` (vd: 16) | Không có hidden |
| **Parameters** | 2×(2H)×(H×ms) ≈ 33K (H=16, ms=32) | 4×16 + 16×1 = 80 | 3×1 + 1 = 4 |
| **Cách mix** | Cross-head mixing (heads trao đổi thông tin) | Independent mixing per position | Simple additive bias |
| **Vị trí** | Encoder (attention giữa row/col embeddings) | Decoder (pointer attention) | Decoder (pointer attention) |

### 4.3 Sơ đồ data flow

**MatNet:**
```
dot_score [B,H,R,C] ─┐
                      ├→ stack [B,H,R,C,2] → rearrange [B,R,C,2H] → MLP → rearrange → ms1, ms2
cost_mat  [B,H,R,C] ─┘                         ↑
                                          cross-head mixing!
```

**MLPAttentionInjection (của ta):**
```
dot_score [B,H,M,N] ─┐
slack     [B,H,M,N] ─┤
future_sl [B,H,M,N] ─┼→ stack [B,H,M,N,4] → MLP(4→16→1) → mixed_score [B,H,M,N]
reach_los [B,H,M,N] ─┘     ↑
                       per-position, shared weights
```

**ProjectedAdditiveInjection (của ta):**
```
dot_score [B,H,M,N] ───────────────────────────────────┐
                                                        + → output [B,H,M,N]
slack     [B,M,N] ─┐                                   │
future_sl [B,M,N] ─┼→ stack [B,M,N,3] → Linear(3→1) → bias [B,1,M,N] broadcast
reach_los [B,M,N] ─┘
```

### 4.4 Khác biệt cốt lõi

1. **MatNet mix cross-head**: `rearrange "b h r c s -> b r c (h s)"` trộn scores của tất cả heads lại → cho mỗi head "nhìn thấy" scores của head khác. Đây là thiết kế cho **encoder** nơi cần diễn đạt cross-modal relationships (rows × columns).

2. **Của ta KHÔNG mix cross-head**: 3 features giống nhau cho mọi head, chỉ mix với dot_score per-head. Đây là thiết kế cho **pointer/decoder** — mỗi head giữ "chuyên môn" riêng (head 1 chuyên về khoảng cách, head 2 chuyên về time window...), heuristics chỉ "gợi ý" thêm.

3. **MatNet dùng cost_mat tĩnh**: cost matrix (processing time) không đổi trong episode → tính 1 lần. **Của ta dùng features động**: slack, future_slack, reachability_loss thay đổi mỗi step → cần tính lại mỗi decode step.

---

## 5. Khi nào dùng mode nào?

| Mode | Khi nào dùng | Lý do |
|------|-------------|-------|
| `"none"` | Baseline / so sánh | Không injection, model dựa hoàn toàn vào learned embeddings |
| `"projected"` | Thử nghiệm đầu tiên | Chỉ 4 params, ít risk overfit, nhanh hội tụ. Heuristics chỉ "nudge" attention |
| `"mlp"` | Khi projected không đủ | 80 params, có khả năng phi tuyến. Nếu heuristics conflict nhau, MLP có thể resolve |
