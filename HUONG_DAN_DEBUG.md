# Hướng Dẫn: Debug Khi Max Steps Bị Vượt Quá

## Những Thay Đổi

Mã code đã được sửa trong file `parco/models/policy.py` (dòng ~195-262).

Khi training chạy quá `max_steps` (hiện tại = 500), thay vì chỉ in ra 1 dòng lỗi:
```
ERROR: Exceeded maximum number of steps (500) during decoding
```

Bây giờ nó sẽ in ra **TOÀN BỘ DEBUG INFO** bao gồm:

## Output Chi Tiết Sẽ In Ra

```
================================================================================
DEBUG INFO AT MAX_STEPS EXCEEDED
================================================================================

Batch index: 0                           ← Index của batch đang train
Number of agents: 4                      ← Số agent (2 xe tải + 2 drone)
Current done status: [False False False False]  ← Agent nào hoàn thành (toàn False = bị kẹt)
Step reached: 501                        ← Đã bước bao nhiêu bước

Current node (all agents):               ← Agent hiện tại ở node nào
  [5 5 0 2]                              (0 = depot, 1-20 = khách hàng)

Previous action (all agents):            ← Bước trước agent ở node nào
  [5 5 0 2]

Action mask shape: torch.Size([4, 21])
Action mask for first agent (valid actions per node):
  Agent 0: 3 valid nodes -> [0, 5, 20]...  ← Agent 0 còn 3 node có thể đến
  Agent 1: 2 valid nodes -> [0, 10]...     ← Agent 1 còn 2 node có thể đến

Detailed constraint analysis for agent 0:
  Visited nodes: 8 out of 21              ← Đã phục vụ 8 khách hàng
  Capacity left: 45.30                    ← Sức chứa còn lại
  Time left: 240.50                       ← Thời gian còn lại
  Slack time: 125.30                      ← Thời gian dự phòng
  Trip deadline reached: False             ← Sản phẩm hư chưa
  Done agents: [False False False False]  ← Agent nào xong

Full TensorDict keys available in td
================================================================================
```

## Cách Đọc Output Để Debug

### 1. Kiểm Tra Hành Động Lặp Lại
```
Current node: [5 5 0 2]
Previous action: [5 5 0 2]
```
**Nếu giống nhau → Agent bị kẹt trong vòng lặp!**
- Agent 0 ở node 5, bước trước cũng ở node 5 → đang lặp lại hành động
- Agent 1 giống vậy
- Điều này có nghĩa **stuck detection có thể không hoạt động**

### 2. Kiểm Tra Có Node Nào Có Thể Đi
```
Agent 0: 3 valid nodes -> [0, 5, 20]
Agent 1: 2 valid nodes -> [0, 10]
```
- Nếu valid nodes = 0 → **tất cả constraint đều block** → bài toán không giải được
- Nếu valid nodes > 0 nhưng mô hình vẫn không đi → **lỗi của mô hình, không phải environment**

### 3. Kiểm Tra Constraint Nào Là Vấn Đề
Xem các giá trị:
- **Capacity left = âm hoặc rất thấp** → lỗi ở constraint sức chứa
- **Time left = âm hoặc rất thấp** → lỗi ở constraint thời gian
- **Trip deadline reached = True** → hàng hư, không thể đi
- **Visited nodes = 20** → đã phục vụ hết, nên xong (nhưng done status = False?)

### 4. Kiểm Tra Node 0 (Depot)
Trong `valid nodes`, kiểm tra xem node 0 (depot) có trong danh sách không:
- Nếu **không có node 0** → agent **không thể về depot** → bị kẹt vĩnh viễn
- Nếu **có node 0** → agent có thể quay lại depot để reset

## Cách Dùng Khi Training

1. **Chạy training như bình thường**:
   ```powershell
   python train.py
   ```

2. **Khi thấy DEBUG OUTPUT in ra** → dừng lại
   - Nó sẽ tự động `break` ra khỏi vòng lặp sau khi in xong
   - Nhưng bạn cũng có thể bấm **Ctrl+C** để dừng ngay lập tức

3. **Đọc output:**
   - Ghi chép lại những số liệu quan trọng
   - Chụp ảnh/copy text nếu cần

4. **Báo cáo cho tôi:**
   - Chia sẻ **toàn bộ debug output**
   - Nêu rõ:
     - Current node vs Previous action có lặp không?
     - Valid actions là bao nhiêu?
     - Constraint nào là vấn đề (capacity/time/deadline)?

## Mục Đích

- **Xác định chính xác** environment bị kẹt hay model chọn sai action
- **Biết constraint nào** gây vấn đề nhất
- **Debug được chính xác** thay vì cứ sửa code mà không biết gốc rễ

## Tệp Được Sửa

- `parco/models/policy.py` - Thêm debug output tại dòng ~195-262
- `DEBUG_OUTPUT_INFO.md` - Hướng dẫn tiếng Anh (file này)
