# Related Work: xử lý Time Window cho PVRPWDP

## 1. PVRPWDP nằm gần nhóm bài toán nào?

PVRPWDP trong repo này có thể xem là tổ hợp của nhiều lớp bài toán:

1. **VRPTW** - Vehicle Routing Problem with Time Windows.
   - Mỗi customer có `[earliest, latest]`.
   - Xe đến sớm thì chờ, đến muộn thì infeasible nếu là hard TW.
   - Đây là nền tảng của phần `time_window` trong env.

2. **PDPTW / VRPSPDTW** - Pickup and Delivery hoặc Simultaneous Pickup-Delivery
   with Time Windows.
   - Có pickup/delivery load, capacity thay đổi theo route.
   - Gần PVRPWDP vì customer là điểm pickup hàng dễ hỏng.
   - Khác biệt: PVRPWDP đưa hàng về depot, không phải pickup-delivery theo cặp
     tùy ý giữa customer.

3. **Perishable VRPTW**.
   - Hàng có giới hạn tươi/shelf-life.
   - Gần `waiting_time` và `trip_deadline` trong PVRPWDP.
   - Literature ghi nhận có trường hợp không thể phục vụ toàn bộ customer nếu
     time window và thời gian làm việc quá chặt.

4. **Truck-drone VRP with Time Windows**.
   - Có dị thể truck/drone, tốc độ/capacity/endurance khác nhau.
   - PVRPWDP khác nhiều paper truck-drone ở chỗ drone/truck đều là agent độc lập
     quay về depot slot riêng, không phải luôn launch/recover từ truck.

5. **Profitable / Optional / Orienteering VRPTW**.
   - Không bắt buộc phục vụ mọi customer.
   - Solver chọn customer để tối đa profit hoặc tối thiểu cost + penalty.
   - Đây là fallback semantics phù hợp nếu generated TW làm full coverage bất
     khả thi.

## 2. Các pattern xử lý Time Window trong literature

### 2.1. Hard time window bằng schedule state

OR-Tools mô hình VRPTW bằng một **time dimension**. Mỗi node có time window, xe
phải có thời điểm đến/phục vụ nằm trong cửa sổ. Solution còn có "solution window"
cho biết khoảng arrival khả thi bên trong constraint window.

Áp dụng cho PVRPWDP:

- `current_time` trong env là time dimension.
- `service_time = max(arrival, earliest)`.
- `service_time <= latest` là hard constraint.
- Nên log thêm `solution/service window margin`, không chỉ valid/invalid.

### 2.2. Waiting/slack là thông tin tối ưu, không chỉ constraint

PyVRP expose các thống kê route như `wait_duration`, `slack`, `time_warp`,
`start_time`, `end_time`, `schedule`. Đặc biệt, `slack` là thời gian có thể trì
hoãn xuất phát mà không làm tăng time-warp hoặc duration.

Áp dụng cho PVRPWDP:

- Rule "waiting before leaving depot does not consume endurance" là hợp lý.
- Nhưng policy cần thấy slack, ví dụ:

```text
tw_slack_ij = latest_j - service_ij
deadline_slack_ij = trip_deadline_i - (service_ij + time_to_depot_ij)
endurance_slack_ij = endurance_i - used_endurance_i - edge_operating_ij - return_time_ij
```

Binary action mask chỉ loại action infeasible; nó không nói action nào sắp làm
route mất tính khả thi.

### 2.3. Soft infeasible search bằng time-warp penalty

HGS/PyVRP dùng `penalised_cost` và `tw_penalty(time_warp)` cho lời giải tạm thời
infeasible. `time_warp` là lượng vi phạm time window; non-zero time warp cho biết
route infeasible.

Áp dụng cho PVRPWDP:

- Env/RL vẫn nên dùng hard mask để không sample action infeasible.
- GA có thể dùng soft-infeasible intermediate solution để tìm hướng sửa route:

```text
penalized_cost =
    route_cost
    + w_unserved * unserved
    + w_timewarp * lateness
    + w_deadline * freshness_violation
    + w_endurance * endurance_excess
    + w_capacity * capacity_excess
```

- Trước khi so sánh với env reward, solution cuối phải được repair/replay thành
  hard-feasible route.

### 2.4. Dropped visits / optional customers bằng penalty

OR-Tools dùng disjunction penalty để cho phép bỏ node. Penalty càng lớn thì solver
càng ưu tiên giữ node trong route. PyVRP cũng có prize/uncollected-prize logic
cho biến thể prize-collecting.

Áp dụng cho PVRPWDP:

- Nếu dataset có customer không thể phục vụ do TW, cần coi bài toán là optional.
- Reward hiện tại nên là:

```text
cost = adaptive_big_m * unvisited_count + secondary_cost
```

- Với hard-cover benchmark, không nên dùng dropped visits để che lỗi dữ liệu;
  cần reject/repair instance.

### 2.5. ALNS/LNS cho PDPTW

PDPTW literature thường dùng Adaptive Large Neighborhood Search:

- remove/destroy một phần route;
- repair bằng insertion heuristic;
- chọn operator theo performance;
- chấp nhận một số deteriorating moves để tránh local optimum.

Áp dụng cho PVRPWDP:

- Đây là hướng tốt cho repair unserved customer sau GA decode.
- Operator nên ưu tiên:
  - remove customer có slack thấp gây nghẽn;
  - reinsert customer vào trip/vehicle có minimum slack loss;
  - split/merge trip;
  - relocate/swap giữa vehicle;
  - ejection chain cho customer cô lập.

### 2.6. Memetic / HGS cho VRPSPDTW

Memetic Search for VRPSPDTW nhấn mạnh:

- initialization tốt;
- crossover phù hợp route;
- local exploitation hiệu quả;
- move evaluation nhanh.

Áp dụng cho PVRPWDP:

- `ga_pvrpwdp.py` greedy insertion là baseline tốt nhưng dễ bị myopic.
- `ga_pvrpwdp_hgs_split.py` đi đúng hướng hơn:
  - giant tour biểu diễn thứ tự customer;
  - split DP quyết định trip/vehicle boundary;
  - repair xử lý unserved.
- Nên phát triển HGS/split thành baseline chính cho TW thay vì tiếp tục dựa vào
  separator round-robin.

### 2.7. Truck-drone VRPTW

Truck-drone TW papers thường có thêm synchronization: drone launch/recover từ
truck, truck chờ drone, drone battery/load. PVRPWDP đơn giản hơn ở synchronization
truck-drone nhưng khó ở điểm:

- nhiều agent độc lập;
- depot slot riêng;
- pickup hàng dễ hỏng;
- freshness deadline reset khi quay về depot;
- waiting mid-trip ảnh hưởng endurance.

Áp dụng:

- Không bê nguyên launch/recover constraints.
- Chỉ lấy các pattern:
  - heterogeneous vehicle assignment;
  - drone endurance/load feasibility;
  - time-window-aware route construction;
  - dynamic demand hoặc optional service nếu cần.

## 3. Cách sinh Time Window nên dùng cho dataset của bạn

Vì instance ban đầu sinh từ bộ test tối ưu chưa có TW, cách đúng nhất là
**reference-schedule-preserving time window generation**.

Pipeline:

1. Lấy route tối ưu gốc không TW.
2. Replay route bằng đúng PVRPWDP physics hiện tại:
   - speed từng agent;
   - capacity;
   - endurance;
   - waiting before depot không tính endurance;
   - waiting mid-trip tính endurance;
   - freshness deadline.
3. Ghi `service_ref[j]` cho từng customer.
4. Sinh TW quanh `service_ref[j]`:

```text
earliest_j = max(0, service_ref[j] - left_slack_j)
latest_j = service_ref[j] + right_slack_j
```

5. Replay lại route sau khi gắn TW.
6. Nếu fail:
   - hard-cover: sửa TW hoặc reject instance;
   - optional-customer: giữ instance nhưng label optional và dùng unserved penalty.

## 4. Phân loại benchmark nên có

### Hard-cover PVRPWDP-TW

Mục tiêu: phục vụ toàn bộ customer.

Yêu cầu data:

- reference route feasible after TW;
- verifier pass 100% trên train/val/test;
- reward big-M chỉ dùng để bảo vệ training khi policy fail, không phải để hợp
  thức hóa data infeasible.

Metric chính:

- full-cover feasible rate;
- makespan/cost trên các solution full-cover;
- no-progress rate;
- average slack margin.

### Optional PVRPWDP-TW

Mục tiêu: phục vụ tốt nhất có thể khi full coverage có thể bất khả thi.

Yêu cầu:

- report unserved/customer served ratio;
- adaptive penalty hoặc prize;
- có lower-bound/diagnostic cho unavoidable unserved nếu tính được.

Metric chính:

- served ratio;
- unserved count;
- cost/makespan;
- penalty calibration sensitivity.

## 5. Đề xuất nghiên cứu cho repo

### Ưu tiên 1: verifier

Viết `scripts/validate_pvrpwdp_timewindows.py`:

- load `.npz`;
- chọn `offset` giống GA;
- strip padding;
- reset env;
- tính customer serviceability từ depot;
- tính failure count theo capacity/TW/deadline/endurance;
- nếu có reference route/actions thì replay.

### Ưu tiên 2: HGS/split baseline

Phát triển `ga_pvrpwdp_hgs_split.py`:

- split DP dùng exact PVRPWDP physics;
- repair insertion cho unserved;
- local search TW-aware;
- optional soft infeasible search với penalty.

### Ưu tiên 3: slack-aware PARCO

Thêm context/candidate feature:

- number of feasible customers per agent;
- min/mean TW slack;
- min deadline slack;
- min endurance slack;
- urgency ratio.

Sau đó mới thử time-aware conflict handler và GRPO-style group rollout.

## 6. Nguồn chính

- OR-Tools VRPTW: https://developers.google.com/optimization/routing/vrptw
- OR-Tools penalties/dropped visits: https://developers.google.com/optimization/routing/penalties
- PyVRP VRPTW example: https://pyvrp.github.io/v0.2.1/examples/vrptw.html
- PyVRP API route diagnostics/time warp/slack:
  https://pyvrp.readthedocs.io/en/stable/api/pyvrp.html
- Memetic Search for VRPSPDTW:
  https://arxiv.org/abs/2011.06331
- ALNS for PDPTW:
  https://pubsonline.informs.org/doi/fpi/10.1287/trsc.1050.0135
- Solomon VRPTW:
  https://pubsonline.informs.org/doi/10.1287/opre.35.2.254
- Perishable VRPTW / possible unserved customers:
  https://www.sciencedirect.com/science/article/abs/pii/S0360835225005741
- VRP with drones and TW:
  https://www.sciencedirect.com/science/article/pii/S0957417421015736
- Truck-drone dynamic demand and TW:
  https://www.mdpi.com/2076-3417/13/24/13086
