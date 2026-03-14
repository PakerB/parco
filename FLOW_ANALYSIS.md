# FLOW PHÂN TÍCH: Action Selection & Conflict Handling

## FLOW HOẠT ĐỘNG (Chi tiết từng bước)

```
MAIN LOOP: policy.py line 168-180
  while not td["done"].all():
    
    STEP 1: Decoder tính logits
    ===============================
    logits, mask = self.decoder(td, hidden, ...)  
    → Output: [B, m, m+N] logits (không xử lý constraint)
             [B, m, m+N] mask (action_mask từ env)
    
    ⚠️  LƯU Ý: mask = action_mask đã được update bởi env.step() của step trước!
    
    STEP 2: Decode Strategy xử lý logits
    ===============================
    td = decode_strategy.step(logits, mask, td, ...)
         → decoding_strategies.py line 155
    
      a) process_logits(): Normalize logits theo mask
         logprobs = apply_mask(logits, mask)
         
      b) _step(): Select action (Greedy/Sampling)
         actions = argmax(logprobs) or sample(logprobs)
         → actions: [B, m] - Initial action TRƯỚC xử lý conflict
         
      c) CONFLICT RESOLUTION (Agent Handler)
         ✅ ĐÂY LÀ XỬ LÝ CONFLICT!
         
         replacement_value = td[self.replacement_value_key]  # = current_node
         
         actions, handling_mask = self.agent_handler(
             actions,           # [B, m] - initial action
             replacement_value,  # [B, m] - current_node (fallback)
             td,
             probs=logprobs,
         )
         → actions AFTER conflict handling: [B, m]
         
         Ví dụ: Nếu 2 agents chọn same node → handler thay 1 agent = current_node
    
    STEP 3: Set action và return
    ===============================
    td.set("action", actions)  # actions đã qua conflict handling!
    return td
    
    STEP 4: Environment step (env.py)
    ===============================
    td = env.step(td)["next"]
         → env._step(td):
             - Tính constraint violations
             - Update state (current_node, visited, etc)
             - Gọi get_action_mask(td) ← UPDATE ACTION MASK
             → ACTION MASK ĐÃ QUA CONFLICT HANDLING
    
    step += 1
```

## TIMELINE ACTION SELECTION

```
Step t (t-th iteration):
├─ decoder(td[t-1])
│  └─ Logits từ neural network (không biết constraint)
│
├─ decode_strategy.step(logits, mask[t-1])
│  ├─ Select action[t] (greedy/sampling)
│  ├─ Conflict resolution: action[t] → action[t]'
│  └─ Set td["action"] = action[t]'
│
└─ env.step(td[t])
   ├─ Thực thi action[t]' từ decoder
   ├─ Update state (current_node[t+1], visited[t+1], etc)
   ├─ get_action_mask(td[t+1])
   │  └─ Tính mask[t+1] dựa trên state[t+1] UPDATED
   └─ Return td[t+1]
```

## LƯU Ý QUAN TRỌNG

### ❌ HIỂU SAI:
- "get_action_mask được gọi TRƯỚC conflict handling"
- "action mask được dùng để select action MỚI"

### ✅ HIỂU ĐÚNG:
1. **Decoder step t**:
   - Input: mask[t-1] (từ env.step(t-1))
   - Output: action[t] TRƯỚC conflict handling
   
2. **Conflict handling**:
   - XỬ LÝ action[t] → action[t]'
   - Đây là nơi giải quyết collision (2 agents same node)
   
3. **Env.step(t)**:
   - Input: action[t]' (đã conflict handling)
   - Process: Tính constraint violations, update state
   - Output: mask[t] → dùng cho step t+1

### ⚠️ MASKING TIMING:

```
t=0: reset
   ├─ action_mask[0] = get_action_mask(td_reset)
   └─ Tất cả agent ở depot
   
t=1: decoder
   ├─ logits[1] (từ neural net)
   ├─ mask[1] = action_mask[0] (từ reset)
   ├─ Select action[1] → conflict handling → action[1]'
   └─ env.step() với action[1]'
      └─ Update state[1] → action_mask[1]
      
t=2: decoder
   ├─ logits[2]
   ├─ mask[2] = action_mask[1] (từ env.step(t=1))
   ├─ Select action[2] → conflict handling → action[2]'
   └─ env.step() với action[2]'
      └─ Update state[2] → action_mask[2]
```

## CÂUHỎI VỀ ACTION MASK TIMING:

### Q1: "Action được tính trong get_action_mask khi nào?"
**A**: get_action_mask được gọi TRONG env.step(), TRƯỚC action thực thi:
- t: env.step(td) gọi get_action_mask(td[t])
- Mask[t+1] được tính từ state[t] TRƯỚC update

❌ KHÔNG, get_action_mask tính mask dùng cho **step t+1**, không phải **step t**
✅ action_mask[t] được sinh ra từ state[t-1] trong env.step(t-1)

### Q2: "Nếu action repeat (stuck), ta phát hiện ở đâu?"
**Hiện tại**: 
- Phát hiện ở get_action_mask (CASE 5: depot isolation + CASE 6: safety net)
- Nhưng QUÁ MUỘN - action đã được select

**Bạn đề xuất**:
- So sánh action[t] vs action[t-1]
- Nếu repeat → force logic: agent ở depot → block depot, agent ngoài depot → force về

## KẾT LUẬN:

**Action selection flow**: 
1. decoder: logits → action[t]
2. agent_handler: action[t] → action[t]' (resolve conflict)
3. env.step: xử lý action[t]', update state, tính mask[t]

**Mask timing**:
- mask được dùng ở step t+1 decoder
- mask được tính từ state[t] trong env.step(t)

**Stuck detection**:
- Hiện tại: ở get_action_mask (CASE 5, 6)
- Bạn đề xuất: thêm logic comparison action[t] vs action[t-1]
