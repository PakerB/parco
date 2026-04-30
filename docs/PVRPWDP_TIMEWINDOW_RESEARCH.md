# Ghi chú nghiên cứu Time Window cho PVRPWDP

## Chẩn đoán hiện tại

PVRPWDP khó hơn các môi trường benchmark gốc của PARCO vì feasibility không chỉ
còn là capacity và coverage. Một khách hàng có thể vĩnh viễn không phục vụ được
do nhiều ràng buộc kết hợp:

- thời điểm phục vụ phải nằm trong time window của khách;
- xe phải còn khả năng quay về đúng depot của nó trước freshness deadline đang
  mang theo;
- xe phải thỏa endurance, trong đó chờ tại depot trước khi xuất phát không tốn
  endurance, nhưng chờ giữa chuyến thì có tốn endurance;
- capacity, tốc độ và chi phí dị thể của từng loại xe vẫn phải thỏa.

Sau khi strip padding bằng `resample_batch_padding`, mình kiểm tra nhanh 32
instance đầu của `data/test_data/test.npz`. Mỗi instance có 5 agent thật và 28
customer thật. Số customer có thể được phục vụ trực tiếp bởi ít nhất một agent
ở trạng thái ban đầu là:

- nhỏ nhất: 19 / 28 customer;
- trung bình: 22.5 / 28 customer;
- lớn nhất: 27 / 28 customer.

Tất cả 32 instance được kiểm tra đều có ít nhất một customer không phục vụ trực
tiếp được ngay từ depot dưới physics hiện tại. Nếu PVRPWDP được định nghĩa là
bài toán hard all-customer, generator/data pipeline phải reject hoặc repair các
instance như vậy. Nếu bài toán cho phép bỏ khách, objective phải được định nghĩa
rõ thành biến thể optional-customer, gần với prize-collecting/orienteering VRPTW,
và dùng penalty lớn cho khách không được phục vụ.

## Hàm ý với PARCO

Đóng góp cốt lõi của PARCO là xây dựng lời giải song song cho nhiều agent:
communication layers, nhiều pointer, và conflict handler theo độ ưu tiên. Các
bài routing được test trong PARCO là HCVRP và OMDCPDP, chưa phải VRPTW-style.
Vì vậy PVRPWDP cần thêm cấu trúc thời gian lên trên PARCO:

- dynamic time features trong context embedding;
- thông tin slack/deadline/endurance theo từng candidate gần decoder logits;
- conflict handler xử lý nhiều agent chọn cùng customer theo độ khẩn cấp thời
  gian, không chỉ theo xác suất model;
- ngữ nghĩa early termination rõ ràng khi không còn customer action hợp lệ.

`highprob` hiện tại vẫn đúng tinh thần PARCO là để model học priority, nhưng
với time window chặt, nên cân nhắc handler kiểu lowest lateness, minimum slack
loss, hoặc smallest future-feasibility damage.

## Quyết định mô hình hóa cần chốt

Cần chọn một trong hai semantics dưới đây và giữ nhất quán giữa env, reward, GA
và báo cáo kết quả.

### Option A: Hard All-Customer PVRPWDP

Dùng nếu bài toán luận văn yêu cầu mọi customer đều phải được phục vụ.

- Reject/resample mọi instance có customer không thể phục vụ.
- Thêm feasibility precheck sau khi strip padding và trước train/eval.
- Nếu `done=True` nhưng còn unvisited customer, coi đó là episode infeasible,
  không phải một lời giải hợp lệ nhưng kém.
- Dùng constructive solver hoặc GA repair để xác nhận validation/test data có
  feasible full coverage.

### Option B: Optional-Customer PVRPWDP

Dùng nếu time window có thể khiến full coverage bất khả thi và mục tiêu là tìm
kế hoạch phục vụ tốt nhất có thể.

- Giữ hard action mask cho physics.
- Cho phép episode kết thúc khi không còn tiến triển hợp lệ.
- Score bằng lexicographic/adaptive big-M penalty:
  `cost = big_m * unserved_count + secondary_cost`.
- Report served ratio, unserved count, feasible/full-coverage rate, makespan,
  total cost, và no-progress/halting rate.
- Có thể trừ một unavoidable lower bound theo từng instance khỏi reward khi
  train để ổn định scale, nhưng report vẫn phải dùng raw unserved customer.

## Phương án cho bộ instance sinh từ route tối ưu chưa có time window

Bộ test hiện tại được sinh từ một lời giải tối ưu/chuẩn khi chưa có time window,
sau đó time window được tạo dựa trên arrival time của khách. Cách này hợp lý nếu
mục tiêu là tạo instance có "lời giải tham chiếu" khả thi, nhưng cần giữ một
invariant quan trọng: route gốc phải vẫn feasible sau khi gắn time window.

Đề xuất pipeline:

1. Replay lời giải gốc không time window bằng đúng physics PVRPWDP hiện tại:
   depot slot theo agent, tốc độ từng agent, capacity, endurance, freshness, và
   rule waiting-at-depot.
2. Ghi lại `arrival_ref[j]`, `service_ref[j]`, agent phục vụ customer `j`, trip
   id, và thời điểm quay về depot của trip chứa `j`.
3. Sinh time window quanh `service_ref[j]`, không chỉ quanh khoảng cách depot:

```text
earliest_j = max(0, service_ref[j] - left_slack_j)
latest_j = service_ref[j] + right_slack_j
```

4. Với customer được route gốc phục vụ bằng drone, width có thể chặt hơn; với
   customer chỉ truck phục vụ được, width nên rộng hơn. Không nên ép cùng một
   width cho mọi loại customer nếu muốn full coverage vẫn khả thi.
5. Sau khi sinh time window, replay lại route gốc. Nếu route gốc infeasible thì
   repair time window hoặc reject instance.
6. Tạo thêm difficulty levels:
   - easy: `left_slack/right_slack` rộng, route gốc feasible với margin lớn;
   - medium: một phần customer có tight window nhưng vẫn feasible theo route gốc;
   - hard: tight window nhiều hơn, nhưng phải pass replay feasibility nếu bài toán
     là hard all-customer.

Nếu muốn optional-customer benchmark, vẫn có thể giữ các instance mà route gốc
không feasible sau time window, nhưng phải đổi nhãn bài toán và metric: lúc đó
unserved customer là một phần objective, không phải lỗi dữ liệu.

## Reward nên sửa

`target="mincost"` đang đi đúng hướng: adaptive big-M penalty cho unvisited
customer cộng với travel/rent cost. Nhưng `target="makespan"` hiện chỉ dùng
`unvisited_ratio`, nên các lời giải có cùng số customer bị bỏ sẽ nhận cùng reward,
dù makespan khác nhau.

Objective makespan nên là:

```text
cost = lambda_unserved * unvisited_count + makespan_or_operating_makespan
reward = -cost
```

Nếu dùng absolute schedule time, thời gian chờ trước khi rời depot có thể làm
tăng makespan. Nếu dùng operating time, chờ trước depot không nên tính, khớp với
quy ước endurance/operating-time hiện tại.

## Action mask và state feature

Giữ hard mask với quy ước `True = valid`. Mask hiện tại đã có physics chính:

- chưa visited;
- capacity;
- service time <= latest;
- quay về trước trip deadline;
- endurance gồm đi tới node, chờ giữa chuyến, và quay về depot;
- depot isolation theo từng agent.

Vấn đề học nằm ở chỗ model chỉ thấy valid/invalid, không thấy mức độ "nguy cấp"
của từng action hợp lệ. Nên thêm candidate-level slack feature hoặc logit bias:

```text
arrival_ij = current_time_i + travel_time_ij
service_ij = max(arrival_ij, earliest_j)
tw_slack_ij = latest_j - service_ij
deadline_slack_ij = trip_deadline_i - (service_ij + return_time_ji)
endurance_slack_ij = endurance_i - used_endurance_i - edge_operating_ij - return_time_ji
```

Các đại lượng này đã ngầm nằm trong mask, nhưng binary mask không giúp model
phân biệt hai action đều hợp lệ nhưng một action sẽ làm mất rất nhiều khả năng
phục vụ các customer sau đó.

## Hướng cho GA và decoder

GA v2 dùng separator encoding là baseline hữu ích, nhưng separator + round-robin
assignment yếu với time window chặt vì thứ tự route, gán vehicle, và vị trí quay
về depot bị coupling rất mạnh.

Nâng cấp GA nên làm:

- seed chromosome theo urgency: earliest deadline, minimum time-window slack,
  return-deadline slack, endurance slack;
- sau decode, chạy repair pass thử cheapest feasible insertion cho mọi unserved
  customer trên tất cả vehicle/trip;
- thêm local search phù hợp TW: relocate, swap, 2-opt*, trip split, trip merge,
  customer reinsert;
- cân nhắc HGS-style giant-tour + split decoder, trong đó split stage quyết định
  vehicle/trip boundary bằng đúng physics của PVRPWDP;
- có thể cho phép GA search qua route tạm thời infeasible với adaptive penalty
  cho lateness/time-warp, sau đó repair hoặc loại trước khi replay bằng env.

## Pattern từ literature

- OR-Tools mô hình VRPTW bằng time dimension và slack variables cho waiting.
  Với trường hợp không thể phục vụ hết khách, OR-Tools dùng node dropping penalty
  qua disjunction; penalty lớn hơn sẽ ưu tiên phục vụ nhiều khách hơn.
- PyVRP/HGS xử lý feasibility của VRPTW qua route schedule và các diagnostics
  như wait time, time warp. Cách này hữu ích cho GA vì soft infeasible search có
  thể dẫn đường cho repair.
- Các formulation soft time window trong RL thường thêm penalty sớm/trễ dạng
  piecewise vào route distance. Cách này khác hard PVRPWDP, nhưng hữu ích nếu
  luận văn cho phép phục vụ trễ/sớm với penalty.
- Các bài DRL VRPTW gần đây dùng feasibility mask ở mỗi decoding step: unvisited,
  capacity, và arrival-before-deadline được mask trước softmax bằng cách đặt
  infeasible logits về âm vô cùng.
- Các bài time-dependent VRPTW nhấn mạnh depot departure time có thể là decision
  variable. Điều này ủng hộ rule local hiện tại: waiting trước khi rời depot
  không tiêu hao endurance.

## Nguồn tham khảo

- PARCO paper: https://arxiv.org/abs/2409.03811
- PARCO OpenReview PDF: https://openreview.net/pdf/672bb4ede45dba28f133a8ebb2cfe40fb1f395c8.pdf
- OR-Tools VRPTW: https://developers.google.com/optimization/routing/vrptw
- OR-Tools dropped visits penalties: https://developers.google.com/optimization/routing/penalties
- OR-Tools dimensions/slack: https://developers.google.com/optimization/routing/dimensions
- PyVRP VRPTW example: https://pyvrp.github.io/v0.2.1/examples/vrptw.html
- Multi-vehicle routing with soft time windows and multi-agent RL:
  https://arxiv.org/abs/2002.05513
- Hybrid Genetic Search for Dynamic VRPTW:
  https://arxiv.org/abs/2307.11800
- Time-dependent VRPTW hybrid ALNS/TS:
  https://www.sciencedirect.com/science/article/pii/S0305054820303105
- Hybrid DRL framework for VRPTW:
  https://www.mdpi.com/1999-4893/19/2/149
