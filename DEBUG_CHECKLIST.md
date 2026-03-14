# DEBUG OUTPUT - Checklist Để Phân Tích

## 1️⃣ Bước 1: Kiểm Tra Hành Động Lặp Lại (Action Repeat)

Trong output, tìm:
```
Current node (all agents):
  [5 5 0 2]

Previous action (all agents):
  [5 5 0 2]
```

**Checklist:**
- [ ] Current node == Previous action cho tất cả agent?
  - **CÓ** → Agent **bị kẹt lặp lại hành động** ❌
  - **KHÔNG** → Agent đang di chuyển bình thường ✅

---

## 2️⃣ Bước 2: Kiểm Tra Action Mask (Node Có Thể Đi)

Tìm:
```
Action mask for first agent (valid actions per node):
  Agent 0: 3 valid nodes -> [0, 5, 20]...
  Agent 1: 2 valid nodes -> [0, 10]...
```

**Checklist:**
- [ ] Valid nodes của tất cả agent > 0?
  - **CÓ** → Còn node có thể đi, vậy **model sai lựa chọn** 🤖
  - **KHÔNG** → Constraint quá chặt, **environment lỗi** 🔧

**Nếu valid nodes > 0:**
- [ ] Có node 0 (depot) trong danh sách?
  - **CÓ** → Có thể quay về depot ✅
  - **KHÔNG** → **Bị block depot** - lỗi nghiêm trọng ❌

---

## 3️⃣ Bước 3: Phân Tích Constraint

Tìm:
```
Detailed constraint analysis for agent 0:
  Visited nodes: 8 out of 21
  Capacity left: 45.30
  Time left: 240.50
  Slack time: 125.30
  Trip deadline reached: False
  Done agents: [False False False False]
```

**Checklist - Tìm giá trị bất thường:**

| Constraint | Bình thường | Bất thường | Ý nghĩa |
|-----------|-----------|-----------|---------|
| **Visited nodes** | < 21 | = 21 và done=False | Đã phục vụ hết nhưng chưa xong |
| **Capacity left** | > 0 | ≤ 0 | Quá tải, không thể đi thêm |
| **Time left** | > 0 | ≤ 0 | Hết thời gian, không thể đi |
| **Slack time** | > 0 | ≤ 0 | Hết time buffer, quá sát deadline |
| **Trip deadline** | False | True | Hàng hư (nếu là perishable) |

---

## 4️⃣ Bước 4: Lập Kế Hoạch Debug

### Trường Hợp A: Action Lặp + Valid > 0
```
Current node == Previous action
Capacity left: 45.30, Time left: 200, etc. OK
Action mask: 3 valid nodes
```
**Kết luận:** ❌ **Lỗi model - không phải environment**

**Hành động:**
1. Stuck detection có làm việc không? (kiểm tra code)
2. Model chọn sai action? (cần debug model)
3. Conflict handler có bug?

---

### Trường Hợp B: Action Không Lặp + Valid > 0
```
Current node [2 3 1 4] != Previous action [1 2 0 3]
Action mask: 2 valid nodes
```
**Kết luận:** ✅ **Environment OK, model chọn đúng**

**Hành động:**
1. Tăng max_steps lên 1000 để cho model thêm thời gian
2. Model cần nhiều bước hơn để giải quyết

---

### Trường Hợp C: Valid = 0 + Capacity thấp
```
Action mask: 0 valid nodes
Capacity left: 5.5
Visited: 15 out of 21
```
**Kết luận:** ❌ **Constraint sức chứa quá chặt**

**Hành động:**
1. Kiểm tra constraint capacity trong env.py
2. Có lỗi trong tính toán capacity?
3. Instance data có sai không?

---

### Trường Hợp D: Valid = 0 + Time thấp
```
Action mask: 0 valid nodes
Time left: -5.2
Slack time: -10.1
```
**Kết luận:** ❌ **Constraint thời gian quá chặt**

**Hành động:**
1. Kiểm tra constraint time_window trong env.py
2. Kiểm tra time slack calculation
3. Model quá chậm chọn action?

---

### Trường Hợp E: Valid = 0 + Không có Depot
```
Action mask: 0 valid nodes
Valid actions: [2, 5, 15]  <- không có 0 (depot)
Capacity left: 45.30
Time left: 200
```
**Kết luận:** ❌ **Depot bị block bởi stuck detection**

**Hành động:**
1. Stuck detection logic lỗi
2. Kiểm tra điều kiện block trong env.py L565-591

---

## 5️⃣ Tóm Tắt Để Báo Cáo

Khi báo cáo, ghi rõ:

```
Lần chạy: [lần thứ mấy]
Instance: [số lượng khách, số agent]
Step vượt: 501 (max=500)
Hành động lặp: [YES/NO] - Current=[...] vs Previous=[...]
Valid actions: [bao nhiêu node]
Node có thể đi: [liệt kê]
Capacity left: [con số]
Time left: [con số]
Kết luận: [Model / Environment / Cả hai]
```

**Ví dụ:**
```
Lần chạy: Lần 1 sau fix stuck detection
Instance: 20 customers, 4 agents
Step vượt: 501 (max=500)
Hành động lặp: YES - Current=[5,5,0,2] vs Previous=[5,5,0,2]
Valid actions: Agent0=3, Agent1=2
Node có thể đi: [0,5,20]
Capacity left: 45.30
Time left: 240.50
Kết luận: LỖI STUCK DETECTION - Agent bị kẹt lặp dù còn node và constraint OK
```
