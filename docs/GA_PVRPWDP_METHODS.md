# Tài liệu 3 phương pháp GA cho PVRPWDP

Tài liệu này mô tả ba baseline di truyền đang nằm ở repo root:

- `ga_pvrpwdp.py`: GA permutation + greedy insertion.
- `ga_pvrpwdp_v2.py`: GA separator encoding + round-robin vehicle assignment.
- `ga_pvrpwdp_hgs_split.py`: Hybrid genetic search + dynamic-programming split decoder.

Các phương pháp này đều chạy offline trên dữ liệu `.npz`, dùng `PVRPWDPVEnv` để reset instance và có bước replay lời giải qua env bằng `evaluate_with_env`.

## Quy ước chung

PVRPWDP dùng `M` depot slot và `N` customer:

- depot của agent `i` có node index `i`;
- customer `j` có env node index `M + j`;
- agent `i` chỉ được quay về depot slot `i`;
- route khi replay qua env được padding bằng cách lặp lại action cuối cùng.

Các decoder GA phải giữ cùng physics với env:

- `travel_time = distance / agent_speed`;
- đến sớm hơn time window thì `service_time = earliest`;
- chờ trước khi rời depot không tính vào endurance;
- chờ giữa chuyến tính vào endurance;
- khi quay về depot thì capacity, endurance và trip deadline được reset;
- hàng dễ hỏng tạo `trip_deadline = min(deadline_cũ, service_time + waiting_time)`;
- feasibility phải ưu tiên hơn cost, nên customer không phục vụ được bị phạt bằng adaptive big-M.

Hai objective chính:

- `target="mincost"`: tối thiểu hóa `unserved_penalty + total_cost`.
- `target="makespan"`: tối thiểu hóa `unserved_penalty + makespan`.

`total_cost` theo quy ước PVRPWDP:

- truck tính travel cost theo distance;
- drone tính travel cost theo operating time;
- xe có chạy thì cộng rent cost.

## Phương pháp 1: `ga_pvrpwdp.py`

### Ý tưởng

Đây là baseline GA cơ bản. Chromosome là một permutation của toàn bộ customer id `0..N-1`. Chromosome không mã hóa xe hay ranh giới chuyến; decoder tự quyết định customer tiếp theo nên được chèn vào xe nào.

Ví dụ:

```text
[5, 2, 7, 1, 8, 4, 6, 3]
```

Nghĩa là GA chỉ tối ưu thứ tự ưu tiên xét customer. Khi decode, từng customer được thử trên mọi vehicle và chọn phương án khả thi có rank tốt nhất.

### Decoder

Với mỗi customer trong chromosome, solver thử hai kiểu chèn:

1. Chèn customer vào chuyến hiện tại của từng xe.
2. Nếu xe đang ngoài depot, đóng chuyến hiện tại, quay về depot rồi mở chuyến mới để phục vụ customer.

Một candidate chỉ được nhận nếu thỏa:

- capacity;
- time window;
- freshness deadline khi quay về depot;
- endurance bao gồm travel time và mid-trip waiting;
- return-to-depot còn khả thi sau khi lấy customer.

Mặc định, rank chọn xe là bộ cũ:

```text
(return_time, travel_distance, service_time)
```

Trong đó `service_time` là thời điểm bắt đầu phục vụ/lấy hàng tại customer, tức `max(arrival, earliest)`.

Script cũng hỗ trợ bộ rank mới qua `--rank-strategy serve-max`. Bộ này ưu tiên phục vụ được nhiều khách nhất trước:

```text
(-reachable_after, -min_slack, target_secondary, return_time, distance_delta, service_time)
```

Ý nghĩa:

- `reachable_after`: số customer còn lại mà ít nhất một vehicle vẫn có thể phục vụ sau candidate hiện tại; dấu âm để số lớn hơn được ưu tiên.
- `min_slack`: biên an toàn nhỏ nhất giữa freshness deadline, endurance và time window; dấu âm để slack lớn hơn được ưu tiên.
- `target_secondary`: với `mincost` là chi phí tăng thêm, với `makespan` là finish time dự kiến.
- `return_time`, `distance_delta`, `service_time`: tie-breaker sau khi đã giữ khả năng phục vụ khách.

Nếu không vehicle nào phục vụ được customer, customer đó được đưa vào `unserved_customers`.

### Genetic operators

Solver dùng các operator:

- seed theo customer id, earliest time window, latest time window, freshness, demand giảm dần, khoảng cách depot;
- order crossover;
- swap mutation;
- reverse segment;
- relocate mutation;
- local search trên elite và một phần offspring;
- tournament selection;
- elitism, culling và immigrant random permutation.

### Cost và replay

`ga_pvrpwdp.py` có tracking `total_op_time` nên khi tính `mincost`:

- truck dùng `current_length`;
- drone dùng `total_op_time`;
- rent chỉ tính cho vehicle có chạy.

Sau khi tìm lời giải, `solution_to_actions` chuyển route thành tensor `[1, M, L]`, padding bằng action cuối, rồi `evaluate_with_env` replay qua `env.step`.

### Khi nên dùng

Dùng phương pháp này làm baseline ổn định, dễ hiểu và dễ debug. Nó phù hợp để kiểm tra nhanh feasibility/cost parity với env vì decoder viết gần trực tiếp theo state transition của PVRPWDP.

Điểm yếu chính là chromosome không trực tiếp học assignment xe hoặc trip boundary. Decoder tham lam quyết định toàn bộ việc phân xe, nên GA chỉ điều khiển thứ tự xét customer.

### Lệnh chạy

```bash
uv run python ga_pvrpwdp.py --npz data/test_data/test.npz --target both
```

Chọn bộ rank mới:

```bash
uv run python ga_pvrpwdp.py --npz data/test_data/test.npz --target both --rank-strategy serve-max
```

Chạy một offset hoặc một batch:

```bash
uv run python ga_pvrpwdp.py --npz data/test_data/test.npz --offset 1 --num-batches 10 --target makespan
uv run python ga_pvrpwdp.py --npz data/test_data/test.npz --batch-idx 0 --target mincost --show-routes
```

## Phương pháp 2: `ga_pvrpwdp_v2.py`

### Ý tưởng

Phương pháp v2 đưa separator gene vào chromosome. Customer vẫn là `0..N-1`, còn `SEP = -1` nghĩa là đóng chuyến hiện tại và chuyển sang vehicle tiếp theo theo round-robin.

Ví dụ với 3 xe:

```text
[5, 2, -1, 7, 1, -1, 8, -1, 4, 6, -1, 3]
```

Diễn giải:

```text
vehicle 0 trip 1: [5, 2]
vehicle 1 trip 1: [7, 1]
vehicle 2 trip 1: [8]
vehicle 0 trip 2: [4, 6]
vehicle 1 trip 2: [3]
```

Vì vậy GA không chỉ tối ưu thứ tự customer mà còn tối ưu tương đối ranh giới chuyến.

### Decoder

Decoder giữ `VehicleState` cho từng vehicle:

- `current_node`;
- `current_time`;
- `used_capacity`;
- `used_endurance`;
- `trip_deadline`;
- `current_length`;
- `route_actions`.

Khi gặp customer gene, decoder thử append vào current vehicle. Nếu không append được, nó thử fallback sang các vehicle khác theo thứ tự vòng tròn. Nếu vẫn thất bại, customer được đánh dấu unserved.

Khi gặp `SEP`, decoder:

1. đóng trip hiện tại của current vehicle nếu vehicle đang ngoài depot;
2. append depot action vào route;
3. reset capacity/endurance/deadline;
4. chuyển current vehicle sang `(current_vehicle + 1) % M`.

Cuối decode, mọi vehicle chưa về depot sẽ được đóng trip.

### Genetic operators

Các operator được thiết kế để bảo toàn tập customer và điều chỉnh separator:

- seed bằng các permutation heuristic rồi chèn separator;
- random chromosome bằng permutation random + separator random;
- crossover bảo toàn thứ tự customer bằng order crossover, sau đó copy separator layout từ một parent;
- swap customer;
- move separator;
- add/remove separator;
- reverse customer segment nhưng giữ separator position;
- relocate customer;
- local search bằng mutation lặp.

### Khi nên dùng

V2 phù hợp để thử nghiệm ảnh hưởng của việc mã hóa trip boundary trực tiếp trong chromosome. So với v1, search space lớn hơn vì separator cũng tiến hóa, nhưng decoder ít tham lam hơn ở tầng ranh giới chuyến.

Điểm yếu là assignment xe vẫn bị ràng buộc bởi round-robin và fallback cục bộ. Nếu separator đặt chưa tốt, nhiều customer có thể bị đẩy sang fallback hoặc unserved.

### Lưu ý hiện trạng

Trong code hiện tại, `ga_pvrpwdp_v2.py` dùng `current_length * travel_price` cho mọi vehicle khi tính `total_cost`. Như vậy phần `mincost` chưa tách drone theo operating time như quy ước PVRPWDP. Nếu dùng v2 để so sánh nghiêm túc với env hoặc với hai phương pháp còn lại, cần cập nhật tracking operating time cho drone.

Runner v2 cũng tái sử dụng `InstanceResult` từ `ga_pvrpwdp.py`; khi chỉnh tiếp cần bảo đảm các field `offset` và `cost` được truyền thống nhất như v1/HGS.

### Lệnh chạy

```bash
uv run python ga_pvrpwdp_v2.py --npz data/test_data/test.npz --target both
```

Chạy một batch và in route:

```bash
uv run python ga_pvrpwdp_v2.py --npz data/test_data/test.npz --batch-idx 0 --target makespan --show-routes
```

## Phương pháp 3: `ga_pvrpwdp_hgs_split.py`

### Ý tưởng

Đây là biến thể mạnh hơn theo hướng Hybrid Genetic Search. Chromosome là giant tour thuần customer, không có separator:

```text
[5, 2, 7, 1, 8, 4, 6, 3]
```

Điểm khác biệt nằm ở decoder: thay vì tham lam chọn xe từng customer, solver dùng dynamic programming split để chia giant tour thành các block liên tiếp, gán block cho vehicle và tách mỗi block thành một hoặc nhiều trip khả thi.

Nó giữ cấu trúc "giant tour + split" của HGS/SPA, nhưng thay time-dependent travel của paper gốc bằng physics PVRPWDP trong repo.

### Split decoder

Decoder gồm hai tầng:

1. `_best_vehicle_block(tour, vehicle_id, start, end)` tìm cách phục vụ đoạn `tour[start:end]` bằng một vehicle.
2. `_decode_split(tour)` dùng DP qua vehicle để chọn các block tốt nhất.

Trong một block, `_simulate_trip` thử các trip liên tiếp:

- bắt đầu ở depot của vehicle;
- phục vụ một đoạn customer liên tiếp;
- kiểm tra capacity, time window, freshness và endurance;
- quay về depot cuối trip;
- trả về finish time, distance, operating time và travel cost.

`_best_vehicle_block` giữ nhiều label không bị dominate theo hai chiều:

```text
(travel_cost, finish_time)
```

`max_labels` giới hạn số label để tránh nổ tổ hợp. Với `target="makespan"`, label tốt ưu tiên finish time; với `target="mincost"`, label tốt ưu tiên travel cost.

DP cấp vehicle có state:

```text
dp[vehicle_count][served_prefix_length]
```

Với mỗi vehicle, solver có thể:

- không giao thêm customer cho vehicle đó;
- giao một block liên tiếp `tour[served:end]` nếu block khả thi.

Sau DP, nếu không phục vụ hết tour thì phần suffix còn lại được xem là unserved và bị phạt adaptive big-M.

### Repair stage

Sau split, `_repair_and_score` replay các route đã chọn để dựng lại state theo từng vehicle. Sau đó nó thử chèn lại các customer unserved bằng heuristic tương tự v1:

- thử append vào chuyến hiện tại;
- thử đóng chuyến rồi mở chuyến mới;
- chọn candidate theo `(return_time, added_cost, service_time)`.

Các customer vẫn không chèn được mới bị giữ lại trong `remaining_unserved`.

### Genetic operators và survivor selection

HGS-Split dùng population nhỏ hơn nhưng có cơ chế chọn lọc có tính diversity:

- seed theo id, earliest, latest, freshness, demand, depot distance, polar angle;
- tùy chọn seed OPTICS nếu `sklearn` có sẵn;
- order crossover sinh hai con và chọn con tốt hơn;
- mutation đa dạng: relocate một node, relocate pair, relocate pair đảo, swap một node, swap pair/single, swap pair/pair, reverse segment, exchange tails;
- local search thử nhiều neighbor và nhận best improvement;
- biased fitness = cost rank + diversity rank có trọng số;
- survivor selection giữ cá thể có biased fitness tốt;
- diversify khi quá lâu không cải thiện.

### Cost và replay

HGS-Split tính cost theo đúng metric truck/drone:

- `_vehicle_metric_cost` dùng distance cho truck;
- dùng operating time cho drone;
- block có customer thì cộng rent của vehicle;
- score cuối là `base_score + penalty * len(remaining_unserved)`.

`evaluate_with_env` replay route giống hai script còn lại, sau đó trả về raw env reward để đối chiếu.

### Khi nên dùng

Dùng HGS-Split khi cần baseline mạnh hơn và có khả năng phân chia xe/trip tốt hơn v1/v2. DP split giúp assignment xe và trip boundary được quyết định có cấu trúc thay vì chỉ bằng greedy insertion hoặc separator round-robin.

Đổi lại, phương pháp này tốn CPU hơn. `max_labels`, `mu`, `lambda_size`, `max_iters`, `local_search_rate` và `diversify_after` ảnh hưởng mạnh đến thời gian chạy.

### Lệnh chạy

```bash
uv run python ga_pvrpwdp_hgs_split.py --npz data/test_data/test.npz --target both
```

Chạy một batch với log chi tiết:

```bash
uv run python ga_pvrpwdp_hgs_split.py --npz data/test_data/test.npz --batch-idx 1500 --target makespan --mu 14 --lambda-size 16 --max-iters 500 --max-workers 1 --show-routes --show-generation-progress
```

## So sánh nhanh

| Tiêu chí | `ga_pvrpwdp.py` | `ga_pvrpwdp_v2.py` | `ga_pvrpwdp_hgs_split.py` |
|---|---|---|---|
| Chromosome | Permutation customer | Customer + `SEP=-1` | Giant tour customer |
| Vehicle assignment | Greedy best vehicle | Round-robin + fallback | DP split qua vehicle |
| Trip boundary | Decoder quyết định tham lam | Gene separator điều khiển | DP split + trip simulation |
| Search space | Nhỏ nhất | Lớn hơn v1 | Giant tour, decoder mạnh |
| Local search | Có | Có | Có, đa operator hơn |
| Diversity control | Immigrant + culling | Immigrant + culling | Biased fitness + diversify |
| Cost truck/drone | Đúng theo distance/op-time | Cần kiểm tra/sửa drone op-time | Đúng theo distance/op-time |
| Phù hợp nhất cho | Baseline/debug | Thử nghiệm encoding separator | Baseline mạnh để báo cáo |

## Quy trình đánh giá khuyến nghị

1. Chạy từng script trên cùng `npz`, cùng `target`, cùng `batch_idx` hoặc `offset`.
2. Ghi lại `objective_cost`, `raw_env_reward`, `unserved`, `visited_customers`, `done`, `makespan`, `total_distance`, `total_cost`.
3. Ưu tiên lời giải có `unserved=0` và `done=1`.
4. Chỉ so sánh cost/makespan khi feasibility giống nhau.
5. Nếu `objective_reward` và `raw_env_reward` lệch nhiều, kiểm tra lại parity giữa decoder và `PVRPWDPVEnv`.

Ví dụ chạy cùng một batch:

```bash
uv run python ga_pvrpwdp.py --npz data/test_data/test.npz --batch-idx 0 --target makespan --show-routes
uv run python ga_pvrpwdp_v2.py --npz data/test_data/test.npz --batch-idx 0 --target makespan --show-routes
uv run python ga_pvrpwdp_hgs_split.py --npz data/test_data/test.npz --batch-idx 0 --target makespan --show-routes
```

## Gợi ý chọn phương pháp

- Cần baseline đơn giản, dễ giải thích: dùng `ga_pvrpwdp.py`.
- Cần nghiên cứu tác động của separator/trip-boundary encoding: dùng `ga_pvrpwdp_v2.py`, nhưng sửa cost drone trước nếu dùng `mincost`.
- Cần kết quả tốt hơn để đối chiếu RL hoặc viết báo cáo: dùng `ga_pvrpwdp_hgs_split.py`.

Khi thay đổi physics trong env, phải cập nhật cả ba decoder hoặc ít nhất đánh dấu rõ script nào chưa parity. Các điểm dễ lệch nhất là waiting mid-trip, deadline khi quay depot, endurance khi đóng chuyến, và cách tính cost drone.
