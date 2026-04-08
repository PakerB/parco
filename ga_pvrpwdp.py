from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from rl4co.data.utils import load_npz_to_tensordict

from parco.envs.pvrpwdp import PVRPWDPVEnv

"""Baseline Genetic Algorithm cho bài toán PVRPWDP.

Ý tưởng chính của file:
1. Mỗi cá thể GA là một hoán vị của danh sách customer.
2. Hàm `decode()` biến hoán vị này thành route thật cho từng vehicle bằng
   greedy insertion, đồng thời kiểm tra toàn bộ ràng buộc khả thi.
3. GA chỉ tối ưu trên không gian hoán vị; phần phân xe và chia chuyến được
   decoder quyết định.

Nhờ tách "chromosome" và "decoder", phần tiến hóa giữ được đơn giản, còn
logic nghiệp vụ của bài toán được gom vào một nơi duy nhất.
"""


@dataclass
class VehicleState:
    """Trạng thái tích lũy của một vehicle trong quá trình decode.

    Mỗi vehicle luôn được theo dõi ở "điểm hiện tại" của route đang xây dựng:
    node đang đứng, thời gian đã trôi qua, tải đã dùng, endurance đã dùng,
    deadline hiệu lực của chuyến hiện tại và danh sách action đã sinh ra.
    """

    vehicle_id: int
    depot_idx: int
    speed: float
    capacity: float
    endurance: float
    max_time: float
    current_node: int = 0
    current_time: float = 0.0
    current_length: float = 0.0
    used_capacity: float = 0.0
    used_endurance: float = 0.0
    trip_deadline: float = 0.0
    route_actions: list[int] = field(default_factory=list)

    def clone(self) -> "VehicleState":
        """Tạo bản sao để thử chèn customer mà không phá hỏng state gốc."""
        return VehicleState(
            vehicle_id=self.vehicle_id,
            depot_idx=self.depot_idx,
            speed=self.speed,
            capacity=self.capacity,
            endurance=self.endurance,
            max_time=self.max_time,
            current_node=self.current_node,
            current_time=self.current_time,
            current_length=self.current_length,
            used_capacity=self.used_capacity,
            used_endurance=self.used_endurance,
            trip_deadline=self.trip_deadline,
            route_actions=list(self.route_actions),
        )


@dataclass
class DecodedSolution:
    """Kết quả sau khi decoder biến chromosome thành lời giải hoàn chỉnh."""

    states: list[VehicleState]
    unserved_customers: list[int]
    score: float
    makespan: float
    total_distance: float
    total_cost: float


@dataclass
class InstanceResult:
    """Kết quả cuối cùng của một instance sau khi GA chạy xong."""

    target: str
    batch_idx: int
    score: float
    objective_cost: float
    objective_reward: float
    raw_env_reward: float
    unserved: int
    visited_customers: int
    done: int
    makespan: float
    total_distance: float
    total_cost: float


class BasicPVRPWDPSolver:
    """GA baseline: permutation chromosome + greedy feasible decoder.

    GA chịu trách nhiệm khám phá thứ tự phục vụ customer.
    Decoder chịu trách nhiệm:
    - gán customer cho vehicle nào,
    - quyết định có cần đóng chuyến hiện tại và mở chuyến mới hay không,
    - loại bỏ các phương án vi phạm capacity / time window / endurance / deadline.
    """

    def __init__(self, env: PVRPWDPVEnv, td_raw: torch.Tensor, td_reset: torch.Tensor):
        self.env = env
        self.td_raw = td_raw
        self.td_reset = td_reset

        # Số agent = số depot ở đầu tensor; customer nằm sau cụm depot.
        self.num_agents = int(td_reset["current_node"].shape[-1])
        self.num_customers = int(td_reset["locs"].shape[-2] - self.num_agents)
        self.max_time = float(td_reset["max_time"][0].item())

        # Cache toàn bộ dữ liệu instance ra numpy để decode nhanh hơn.
        self.locs = td_reset["locs"][0].cpu().numpy()
        self.customer_demands = td_reset["demand"][0, 0, self.num_agents :].cpu().numpy()
        self.time_windows = td_reset["time_window"][0, self.num_agents :, :].cpu().numpy()
        self.waiting_time = td_reset["waiting_time"][0, self.num_agents :].cpu().numpy()
        self.agents_speed = td_reset["agents_speed"][0].cpu().numpy()
        self.agents_capacity = td_reset["agents_capacity"][0].cpu().numpy()
        self.agents_endurance = td_reset["agents_endurance"][0].cpu().numpy()
        self.customer_env_indices = np.arange(self.num_agents, self.num_agents + self.num_customers)

        # Suy ra loại phương tiện để áp đúng cost của truck/drone.
        self.is_truck = self._truck_mask_from_capacity(self.agents_capacity)
        self.travel_price = np.where(
            self.is_truck,
            float(env.travel_price_truck),
            float(env.travel_price_drone),
        )
        self.rent_price = np.where(
            self.is_truck,
            float(env.rent_price_truck),
            float(env.rent_price_drone),
        )

        # Penalty cho unserved phải đủ lớn để GA ưu tiên lời giải khả thi.
        # Ở đây dùng một upper-bound thô dựa trên bounding box của instance.
        bbox_diag = np.linalg.norm(self.locs.max(axis=0) - self.locs.min(axis=0))
        travel_max = 2.0 * bbox_diag * self.num_customers * float(self.travel_price.max())
        self.lambda_unserved = travel_max + float(self.rent_price.sum()) + 1.0
        latest_max = float(self.time_windows[:, 1].max())
        min_speed = float(np.min(self.agents_speed))
        self.lambda_makespan_unserved = (2.0 * bbox_diag * self.num_customers / max(min_speed, 1e-9)) + latest_max + 1.0

        # Memoization để không decode lại các chromosome đã gặp.
        self._score_cache: dict[tuple[int, ...], tuple[float, DecodedSolution]] = {}

    @staticmethod
    def _truck_mask_from_capacity(capacity: np.ndarray) -> np.ndarray:
        """Phân biệt truck/drone bằng ngưỡng capacity trung gian."""
        thresh = 0.5 * (capacity.max() + capacity.min())
        return capacity > (thresh + 1e-6)

    def _distance(self, node_a: int, node_b: int) -> float:
        """Khoảng cách Euclid giữa hai node trong instance."""
        return float(np.linalg.norm(self.locs[node_a] - self.locs[node_b]))

    def _initial_states(self) -> list[VehicleState]:
        """Khởi tạo mỗi vehicle ở chính depot của nó, chưa phục vụ ai."""
        states: list[VehicleState] = []
        for agent_id in range(self.num_agents):
            states.append(
                VehicleState(
                    vehicle_id=agent_id,
                    depot_idx=agent_id,
                    speed=float(self.agents_speed[agent_id]),
                    capacity=float(self.agents_capacity[agent_id]),
                    endurance=float(self.agents_endurance[agent_id]),
                    max_time=self.max_time,
                    current_node=agent_id,
                    trip_deadline=self.max_time,
                )
            )
        return states

    def _close_trip(self, state: VehicleState) -> VehicleState:
        """Đưa vehicle quay về depot để kết thúc chuyến hiện tại.

        Hàm này được gọi khi:
        - decoder quyết định mở chuyến mới trước khi thăm customer tiếp theo,
        - hoặc khi hoàn tất decode để đóng tất cả route còn dang dở.
        """
        if state.current_node == state.depot_idx:
            return state

        state = state.clone()
        back_dist = self._distance(state.current_node, state.depot_idx)
        state.current_length += back_dist
        state.current_time += back_dist / state.speed
        state.current_node = state.depot_idx
        state.used_capacity = 0.0
        state.used_endurance = 0.0
        state.trip_deadline = state.max_time
        state.route_actions.append(state.depot_idx)
        return state

    def _try_insert(
        self,
        state: VehicleState,
        customer_id: int,
        force_new_trip: bool,
    ) -> tuple[VehicleState | None, tuple[float, float, float]]:
        """Thử chèn một customer vào route của `state`.

        Nếu `force_new_trip=True`, vehicle sẽ bị ép quay depot trước rồi mới
        xét customer này như điểm đầu của chuyến mới.

        Kết quả trả về:
        - `trial_state`: trạng thái mới nếu chèn hợp lệ, ngược lại là `None`.
        - `heuristic_key`: khóa so sánh để chọn phương án tốt hơn trong decoder.
          Thứ tự ưu tiên hiện tại là: quay về kịp sớm hơn, đi ít hơn, phục vụ sớm hơn.
        """
        customer_idx = int(self.customer_env_indices[customer_id])
        trial_state = state.clone()
        if force_new_trip:
            trial_state = self._close_trip(trial_state)

        demand = float(self.customer_demands[customer_id])
        # Capacity là ràng buộc cứng, vi phạm thì loại ngay.
        if trial_state.used_capacity + demand > trial_state.capacity + 1e-9:
            return None, (math.inf, math.inf, math.inf)

        travel_dist = self._distance(trial_state.current_node, customer_idx)
        travel_time = travel_dist / trial_state.speed
        arrival = trial_state.current_time + travel_time
        earliest, latest = self.time_windows[customer_id]
        service_time = max(arrival, float(earliest))
        # Không kịp vào time window của customer.
        if service_time > float(latest) + 1e-9:
            return None, (math.inf, math.inf, math.inf)

        waiting_mid_trip = 0.0
        # Chỉ tính waiting vào endurance nếu vehicle đang ở giữa chuyến.
        # Khi đã ở depot và khởi động chuyến mới thì chờ ở depot không bị tính.
        if trial_state.current_node != trial_state.depot_idx:
            waiting_mid_trip = max(float(earliest) - arrival, 0.0)

        used_endurance = trial_state.used_endurance + travel_time + waiting_mid_trip
        back_time = self._distance(customer_idx, trial_state.depot_idx) / trial_state.speed
        return_time = service_time + back_time
        new_deadline = service_time + float(self.waiting_time[customer_id])
        effective_deadline = min(trial_state.trip_deadline, new_deadline)

        # Sau khi phục vụ customer hiện tại, vehicle vẫn phải còn đường về depot
        # trước deadline của chuyến đang hoạt động.
        if return_time > effective_deadline + 1e-9:
            return None, (math.inf, math.inf, math.inf)
        # Endurance cũng được kiểm tra theo kiểu "phục vụ xong rồi vẫn quay về được".
        if used_endurance + back_time > trial_state.endurance + 1e-9:
            return None, (math.inf, math.inf, math.inf)

        trial_state.current_length += travel_dist
        trial_state.current_time = service_time
        trial_state.current_node = customer_idx
        trial_state.used_capacity += demand
        trial_state.used_endurance = used_endurance
        trial_state.trip_deadline = effective_deadline
        trial_state.route_actions.append(customer_idx)

        # Trả về các chỉ số cơ sở để bước xếp hạng bên ngoài quyết định theo target.
        # metrics = (return_time, travel_dist, service_time)
        return trial_state, (return_time, travel_dist, service_time)

    def _candidate_rank(
        self,
        vehicle_id: int,
        original_state: VehicleState,
        trial_state: VehicleState,
        metrics: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Xếp hạng ứng viên chèn theo mục tiêu tối ưu hiện hành.

        Hai target dùng hai heuristic khác nhau:
        - `makespan`: ưu tiên phương án giúp route còn nhiều slack thời gian.
        - `mincost`: ưu tiên phương án làm tăng chi phí nhỏ nhất.
        """
        return_time, travel_dist, service_time = metrics

        if self.env.target == "makespan":
            # Với makespan, tiêu chí chính vẫn là thời điểm có thể quay về depot.
            # Tie-break bằng quãng đường tăng thêm rồi đến thời điểm phục vụ.
            return (return_time, travel_dist, service_time)

        if self.env.target == "mincost":
            # Chi phí tăng thêm được xấp xỉ bằng phần quãng đường tăng thêm nhân đơn giá
            # cộng với phí thuê nếu đây là lần đầu vehicle thực sự được sử dụng.
            delta_distance = max(0.0, trial_state.current_length - original_state.current_length)
            delta_cost = delta_distance * float(self.travel_price[vehicle_id])

            vehicle_was_unused = original_state.current_length <= 1e-8
            vehicle_is_used_now = trial_state.current_length > 1e-8
            if vehicle_was_unused and vehicle_is_used_now:
                delta_cost += float(self.rent_price[vehicle_id])

            # Vẫn giữ return_time làm tie-break để tránh route quá căng thời gian.
            return (delta_cost, return_time, service_time)

        raise NotImplementedError(f"Unsupported target: {self.env.target}")

    def decode(self, chromosome: np.ndarray) -> DecodedSolution:
        """Biến một chromosome thành lời giải route hoàn chỉnh.

        Decoder duyệt customer theo thứ tự xuất hiện trong chromosome.
        Với mỗi customer, nó thử gán vào tất cả vehicle theo hai cách:
        - nối tiếp vào chuyến hiện tại;
        - đóng chuyến hiện tại rồi mở chuyến mới.

        Phương án hợp lệ có heuristic tốt nhất sẽ được chọn.
        Heuristic này phụ thuộc vào `target`:
        - `makespan`: ưu tiên phương án quay về depot sớm hơn.
        - `mincost`: ưu tiên phương án làm tăng chi phí nhỏ hơn.
        Nếu không vehicle nào phục vụ được thì customer bị đưa vào `unserved`.
        """
        key = tuple(int(x) for x in chromosome.tolist())
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached[1]

        states = self._initial_states()
        unserved: list[int] = []

        for customer_id in chromosome.tolist():
            best_state: VehicleState | None = None
            best_vehicle: int | None = None
            best_rank = (math.inf, math.inf, math.inf)

            for vehicle_id, state in enumerate(states):
                # Ưu tiên thử nối trực tiếp vào route hiện tại.
                trial_same, metrics_same = self._try_insert(state, customer_id, force_new_trip=False)
                if trial_same is not None:
                    rank_same = self._candidate_rank(vehicle_id, state, trial_same, metrics_same)
                else:
                    rank_same = (math.inf, math.inf, math.inf)
                if trial_same is not None and rank_same < best_rank:
                    best_state = trial_same
                    best_vehicle = vehicle_id
                    best_rank = rank_same

                # Nếu vehicle đang ở giữa chuyến, cho phép đóng chuyến rồi thử lại.
                if state.current_node != state.depot_idx:
                    trial_new, metrics_new = self._try_insert(state, customer_id, force_new_trip=True)
                    if trial_new is not None:
                        rank_new = self._candidate_rank(vehicle_id, state, trial_new, metrics_new)
                    else:
                        rank_new = (math.inf, math.inf, math.inf)
                    if trial_new is not None and rank_new < best_rank:
                        best_state = trial_new
                        best_vehicle = vehicle_id
                        best_rank = rank_new

            if best_state is None or best_vehicle is None:
                unserved.append(int(customer_id))
            else:
                states[best_vehicle] = best_state

        # Đóng toàn bộ route để tính các chỉ số cuối cùng một cách nhất quán.
        closed_states = [self._close_trip(state) for state in states]
        total_distance = float(sum(state.current_length for state in closed_states))
        makespan = float(max(state.current_time for state in closed_states))

        travel_part = float(sum(state.current_length * self.travel_price[i] for i, state in enumerate(closed_states)))
        used_vehicle = np.array([state.current_length > 1e-8 for state in closed_states], dtype=np.float32)
        rent_part = float(np.sum(used_vehicle * self.rent_price))
        total_cost = travel_part + rent_part

        score = self.objective_cost_from_parts(
            target=self.env.target,
            unserved_count=len(unserved),
            total_cost=total_cost,
            makespan=makespan,
        )

        solution = DecodedSolution(
            states=closed_states,
            unserved_customers=unserved,
            score=float(score),
            makespan=makespan,
            total_distance=total_distance,
            total_cost=total_cost,
        )
        self._score_cache[key] = (solution.score, solution)
        return solution

    def evaluate(self, chromosome: np.ndarray) -> float:
        """Trả về fitness scalar để GA tối ưu."""
        return self.decode(chromosome).score

    def objective_cost_from_parts(
        self,
        target: str,
        unserved_count: int,
        total_cost: float,
        makespan: float,
    ) -> float:
        """Gom các thành phần objective về một scalar duy nhất.

        `score` là giá trị GA tối thiểu hóa.
        - `mincost`: cost thực + penalty unserved.
        - `makespan`: makespan + penalty unserved.
        """
        if target == "mincost":
            return self.lambda_unserved * unserved_count + total_cost
        if target == "makespan":
            return self.lambda_makespan_unserved * unserved_count + makespan
        raise NotImplementedError(f"Unsupported target: {target}")

    def objective_cost(self, solution: DecodedSolution) -> float:
        return self.objective_cost_from_parts(
            target=self.env.target,
            unserved_count=len(solution.unserved_customers),
            total_cost=solution.total_cost,
            makespan=solution.makespan,
        )

    def chromosome_seeds(self) -> list[np.ndarray]:
        """Sinh một vài permutation heuristic để khởi tạo population tốt hơn random.

        Các seed này phản ánh nhiều trực giác khác nhau:
        - customer mở sớm,
        - customer đóng sớm,
        - customer có freshness gắt hơn,
        - demand lớn trước,
        - customer gần depot trước.
        """
        customer_ids = np.arange(self.num_customers, dtype=np.int64)
        earliest = np.argsort(self.time_windows[:, 0])
        latest = np.argsort(self.time_windows[:, 1])
        freshness = np.argsort(self.waiting_time)
        demand_desc = np.argsort(-self.customer_demands)
        depot_dist = np.argsort(
            [self._distance(0, int(env_idx)) for env_idx in self.customer_env_indices]
        )
        seeds = [customer_ids, earliest, latest, freshness, demand_desc, depot_dist]
        unique: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        for seed in seeds:
            key = tuple(int(x) for x in seed.tolist())
            if key not in seen:
                seen.add(key)
                unique.append(seed.astype(np.int64, copy=True))
        return unique

    def order_crossover(self, parent_a: np.ndarray, parent_b: np.ndarray, rng: random.Random) -> np.ndarray:
        """Order Crossover (OX) cho permutation chromosome.

        Giữ nguyên một đoạn liên tiếp của `parent_a`, sau đó điền phần còn lại
        theo thứ tự xuất hiện trong `parent_b`.
        """
        size = len(parent_a)
        left, right = sorted(rng.sample(range(size), 2))
        child = np.full(size, -1, dtype=np.int64)
        child[left : right + 1] = parent_a[left : right + 1]
        used = set(int(x) for x in child[left : right + 1].tolist())
        insert_pos = (right + 1) % size
        for gene in parent_b.tolist():
            if gene in used:
                continue
            child[insert_pos] = gene
            insert_pos = (insert_pos + 1) % size
        return child

    def mutate(self, chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        """Mutation cơ bản: swap hoặc đảo ngược một đoạn."""
        child = chromosome.copy()
        size = len(child)
        op = rng.random()

        if op < 0.5:
            i, j = rng.sample(range(size), 2)
            child[i], child[j] = child[j], child[i]
        else:
            i, j = sorted(rng.sample(range(size), 2))
            child[i : j + 1] = child[i : j + 1][::-1]

        return child

    def relocate_mutation(self, chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        """Lấy một gene ra rồi chèn vào vị trí khác để đổi thứ tự phục vụ."""
        child = chromosome.copy()
        size = len(child)
        i, j = rng.sample(range(size), 2)
        gene = int(child[i])
        child = np.delete(child, i)
        child = np.insert(child, j, gene)
        return child.astype(np.int64, copy=False)

    def reverse_segment(self, chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        """Đảo chiều một đoạn con; hữu ích khi route tốt nằm ở thứ tự ngược."""
        child = chromosome.copy()
        i, j = sorted(rng.sample(range(len(child)), 2))
        child[i : j + 1] = child[i : j + 1][::-1]
        return child

    def local_search(
        self,
        chromosome: np.ndarray,
        rng: random.Random,
        max_iters: int = 20,
        neighborhood_trials: int = 12,
    ) -> np.ndarray:
        """Hill-climbing cục bộ trên một chromosome.

        Mỗi vòng thử vài operator lân cận; gặp lời giải tốt hơn thì nhận luôn.
        Đây là cách rẻ để "mài" thêm cá thể tốt mà không cần tăng generations.
        """
        current = chromosome.copy()
        current_score = self.evaluate(current)
        operators = (
            self.mutate,
            self.relocate_mutation,
            self.reverse_segment,
        )

        for _ in range(max_iters):
            improved = False
            for _ in range(neighborhood_trials):
                operator = operators[rng.randrange(len(operators))]
                candidate = operator(current, rng)
                candidate_score = self.evaluate(candidate)
                if candidate_score + 1e-9 < current_score:
                    current = candidate
                    current_score = candidate_score
                    improved = True
                    break
            if not improved:
                break

        return current

    def tournament_select(
        self,
        population: list[np.ndarray],
        fitness: list[float],
        tournament_size: int,
        rng: random.Random,
    ) -> np.ndarray:
        """Chọn parent bằng tournament selection để cân bằng tốt/đa dạng."""
        tournament_size = min(tournament_size, len(population))
        idxs = rng.sample(range(len(population)), tournament_size)
        best_idx = min(idxs, key=lambda idx: fitness[idx])
        return population[best_idx]

    def run(
        self,
        population_size: int = 80,
        generations: int = 200,
        mutation_rate: float = 0.2,
        elite_size: int = 4,
        tournament_size: int = 5,
        cull_ratio: float = 0.5,
        immigrant_ratio: float = 0.05,
        local_search_rate: float = 0.2,
        local_search_elites: int = 2,
        local_search_iters: int = 20,
        seed: int = 42,
        verbose: bool = True,
        progress_label: str | None = None,
        progress_every: int = 10,
    ) -> DecodedSolution:
        """Chạy vòng lặp GA chính và trả về lời giải tốt nhất tìm được."""
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        run_start_time = time.time()

        # Khởi tạo population bằng seed heuristic trước, sau đó bù random.
        seeds = self.chromosome_seeds()
        population: list[np.ndarray] = [seed_perm.copy() for seed_perm in seeds]

        base_perm = np.arange(self.num_customers, dtype=np.int64)
        while len(population) < population_size:
            population.append(np_rng.permutation(base_perm))

        best_solution: DecodedSolution | None = None

        for generation in range(generations):
            # Decode toàn bộ population để lấy fitness thực tế.
            decoded = [self.decode(chromosome) for chromosome in population]
            fitness = [item.score for item in decoded]
            order = sorted(range(len(population)), key=lambda idx: fitness[idx])
            population = [population[idx] for idx in order]
            decoded = [decoded[idx] for idx in order]
            fitness = [fitness[idx] for idx in order]

            if local_search_elites > 0:
                # Mài thêm một số elite ngay sau khi đã sort population.
                elite_ls_count = min(local_search_elites, len(population))
                for idx in range(elite_ls_count):
                    improved = self.local_search(
                        population[idx],
                        rng,
                        max_iters=local_search_iters,
                    )
                    improved_score = self.evaluate(improved)
                    if improved_score + 1e-9 < fitness[idx]:
                        population[idx] = improved
                        decoded[idx] = self.decode(improved)
                        fitness[idx] = improved_score

                order = sorted(range(len(population)), key=lambda idx: fitness[idx])
                population = [population[idx] for idx in order]
                decoded = [decoded[idx] for idx in order]
                fitness = [fitness[idx] for idx in order]

            # Luôn giữ lại global best, không chỉ best của generation hiện tại.
            if best_solution is None or decoded[0].score < best_solution.score:
                best_solution = decoded[0]

            if verbose and (
                generation == 0
                or (generation + 1) % max(1, progress_every) == 0
                or generation == generations - 1
            ):
                prefix = f"[{progress_label}] " if progress_label else "[GA] "
                print(
                    f"{prefix}{format_progress(generation + 1, generations, run_start_time, width=20)} "
                    f"gen={generation + 1:03d} "
                    f"best_score={decoded[0].score:.4f} "
                    f"unserved={len(decoded[0].unserved_customers)} "
                    f"makespan={decoded[0].makespan:.4f} "
                    f"distance={decoded[0].total_distance:.4f}"
                )

            # Elitism: copy nguyên một số cá thể tốt nhất sang thế hệ mới.
            next_population: list[np.ndarray] = [population[idx].copy() for idx in range(min(elite_size, len(population)))]
            survivor_count = max(elite_size + 2, int(round(population_size * cull_ratio)))
            survivor_count = min(survivor_count, len(population))
            breeding_pool = population[:survivor_count]
            breeding_fitness = fitness[:survivor_count]

            # "Immigrant" giúp giữ đa dạng quần thể, tránh kẹt local optimum.
            immigrant_count = int(round(population_size * immigrant_ratio))
            immigrant_count = min(
                max(0, immigrant_count),
                max(0, population_size - len(next_population)),
            )

            while len(next_population) < population_size:
                if immigrant_count > 0 and len(next_population) >= population_size - immigrant_count:
                    next_population.append(np_rng.permutation(base_perm))
                    continue

                # Pipeline sinh con: chọn parent -> crossover -> mutation -> local search.
                parent_a = self.tournament_select(breeding_pool, breeding_fitness, tournament_size, rng)
                parent_b = self.tournament_select(breeding_pool, breeding_fitness, tournament_size, rng)
                child = self.order_crossover(parent_a, parent_b, rng)
                if rng.random() < mutation_rate:
                    child = self.mutate(child, rng)
                if rng.random() < 0.35:
                    child = self.relocate_mutation(child, rng)
                if rng.random() < local_search_rate:
                    child = self.local_search(
                        child,
                        rng,
                        max_iters=max(1, local_search_iters // 2),
                        neighborhood_trials=8,
                    )
                next_population.append(child)

            population = next_population

        assert best_solution is not None
        return best_solution

    def solution_to_actions(self, solution: DecodedSolution) -> torch.Tensor:
        """Chuyển lời giải decode về tensor action để replay trong env."""
        route_lists: list[list[int]] = []
        for state in solution.states:
            if state.route_actions:
                route_lists.append(list(state.route_actions))
            else:
                route_lists.append([state.depot_idx])

        horizon = max(len(route) for route in route_lists)
        action_rows = []
        for route in route_lists:
            # Pad bằng node cuối để tất cả agent có cùng horizon step.
            last = route[-1]
            padded = route + [last] * (horizon - len(route))
            action_rows.append(padded)

        actions = torch.tensor(action_rows, dtype=torch.int64).unsqueeze(0)
        return actions

    def evaluate_with_env(self, solution: DecodedSolution) -> dict[str, float]:
        """Replay lời giải trong env để đối chiếu với objective nội bộ của solver."""
        actions = self.solution_to_actions(solution)
        td = self.td_reset.clone()
        for step in range(actions.shape[-1]):
            td.set("action", actions[:, :, step])
            td = self.env.step(td)["next"]

        objective_cost = self.objective_cost(solution)
        objective_reward = -objective_cost
        raw_env_reward = float(self.env.get_reward(td.clone(), actions).item())
        return {
            "objective_reward": objective_reward,
            "objective_cost": objective_cost,
            "raw_env_reward": raw_env_reward,
            "done": float(td["done"].float().item()),
            "visited_customers": float(td["visited"][0, self.num_agents :].sum().item()),
        }

    def print_solution(self, solution: DecodedSolution) -> None:
        """In lời giải ở dạng dễ đọc cho người dùng."""
        print("\nBest solution summary")
        print(f"  score          : {solution.score:.4f}")
        print(f"  unserved       : {len(solution.unserved_customers)}")
        print(f"  makespan       : {solution.makespan:.4f}")
        print(f"  total_distance : {solution.total_distance:.4f}")
        print(f"  total_cost     : {solution.total_cost:.4f}")
        if solution.unserved_customers:
            print(f"  unserved_ids   : {solution.unserved_customers}")

        for state in solution.states:
            display_route = []
            for node in state.route_actions:
                if node < self.num_agents:
                    display_route.append(f"D{node}")
                else:
                    display_route.append(f"C{node - self.num_agents}")
            print(
                f"  vehicle {state.vehicle_id}: "
                f"{' -> '.join(display_route) if display_route else f'D{state.depot_idx}'}"
            )


def load_instance_from_td(
    td_loaded: torch.Tensor,
    batch_idx: int,
    target: str,
) -> tuple[PVRPWDPVEnv, torch.Tensor, torch.Tensor]:
    """Lấy một instance cụ thể từ dataset đã load sẵn."""
    if batch_idx >= td_loaded.batch_size[0]:
        raise IndexError(f"batch_idx={batch_idx} out of range for batch size {td_loaded.batch_size[0]}")

    td_raw = td_loaded[batch_idx].unsqueeze(0)
    env = PVRPWDPVEnv(
        use_epoch_data=False,
        fallback_to_generator=False,
        target=target,
    )
    td_reset = env.reset(td_raw.clone())
    return env, td_raw, td_reset


def load_instance(npz_path: str, batch_idx: int, target: str) -> tuple[PVRPWDPVEnv, torch.Tensor, torch.Tensor]:
    """Load NPZ rồi rút ra một instance duy nhất."""
    td_loaded = load_npz_to_tensordict(npz_path)
    return load_instance_from_td(td_loaded, batch_idx, target)


def summarize_results(results: list[InstanceResult], target: str) -> dict[str, float]:
    """Tổng hợp thống kê trung bình cho một tập instance."""
    count = len(results)
    if count == 0:
        return {
            "count": 0.0,
            "avg_objective_cost": math.nan,
            "avg_score": math.nan,
            "avg_unserved": math.nan,
            "avg_makespan": math.nan,
            "avg_distance": math.nan,
            "feasible_rate": math.nan,
        }

    return {
        "count": float(count),
        "avg_objective_cost": float(np.mean([item.objective_cost for item in results])),
        "avg_score": float(np.mean([item.score for item in results])),
        "avg_unserved": float(np.mean([item.unserved for item in results])),
        "avg_makespan": float(np.mean([item.makespan for item in results])),
        "avg_distance": float(np.mean([item.total_distance for item in results])),
        "feasible_rate": float(np.mean([1.0 if item.done and item.unserved == 0 else 0.0 for item in results])),
    }


def save_results_csv(output_path: Path, results: list[InstanceResult]) -> None:
    """Lưu toàn bộ kết quả chi tiết của từng instance ra CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "batch_idx",
                "score",
                "objective_cost",
                "objective_reward",
                "raw_env_reward",
                "unserved",
                "visited_customers",
                "done",
                "makespan",
                "total_distance",
                "total_cost",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(asdict(item))


def format_progress(
    current: int,
    total: int,
    start_time: float,
    width: int = 28,
) -> str:
    """Tạo progress bar text kèm elapsed time và ETA."""
    ratio = 1.0 if total <= 0 else max(0.0, min(1.0, current / total))
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(0.0, time.time() - start_time)
    avg = elapsed / max(current, 1)
    eta = avg * max(total - current, 0)
    return (
        f"[{bar}] {current}/{total} "
        f"({ratio * 100:5.1f}%) "
        f"elapsed={elapsed:7.1f}s "
        f"eta={eta:7.1f}s"
    )


def run_single_instance(
    td_loaded: torch.Tensor,
    batch_idx: int,
    target: str,
    population: int,
    generations: int,
    mutation_rate: float,
    elite_size: int,
    tournament_size: int,
    cull_ratio: float,
    immigrant_ratio: float,
    local_search_rate: float,
    local_search_elites: int,
    local_search_iters: int,
    seed: int,
    show_routes: bool,
    show_generation_progress: bool,
    generation_progress_every: int,
) -> InstanceResult:
    """Chạy GA cho đúng một batch/instance và trả về record kết quả."""
    env, td_raw, td_reset = load_instance_from_td(td_loaded, batch_idx, target)
    solver = BasicPVRPWDPSolver(env=env, td_raw=td_raw, td_reset=td_reset)
    best = solver.run(
        population_size=population,
        generations=generations,
        mutation_rate=mutation_rate,
        elite_size=elite_size,
        tournament_size=tournament_size,
        cull_ratio=cull_ratio,
        immigrant_ratio=immigrant_ratio,
        local_search_rate=local_search_rate,
        local_search_elites=local_search_elites,
        local_search_iters=local_search_iters,
        seed=seed,
        verbose=show_generation_progress,
        progress_label=f"{target}|batch={batch_idx:03d}",
        progress_every=generation_progress_every,
    )
    env_eval = solver.evaluate_with_env(best)

    if show_routes:
        print(f"\n[target={target}] batch={batch_idx}")
        solver.print_solution(best)
        print("Environment evaluation")
        for key, value in env_eval.items():
            print(f"  {key:16s}: {value}")

    return InstanceResult(
        target=target,
        batch_idx=batch_idx,
        score=float(best.score),
        objective_cost=float(env_eval["objective_cost"]),
        objective_reward=float(env_eval["objective_reward"]),
        raw_env_reward=float(env_eval["raw_env_reward"]),
        unserved=len(best.unserved_customers),
        visited_customers=int(env_eval["visited_customers"]),
        done=int(env_eval["done"]),
        makespan=float(best.makespan),
        total_distance=float(best.total_distance),
        total_cost=float(best.total_cost),
    )


def run_target_over_dataset(
    npz_path: str,
    target: str,
    population: int,
    generations: int,
    mutation_rate: float,
    elite_size: int,
    tournament_size: int,
    cull_ratio: float,
    immigrant_ratio: float,
    local_search_rate: float,
    local_search_elites: int,
    local_search_iters: int,
    seed: int,
    output_dir: str | None,
    batch_idx: int | None,
    num_batches: int | None,
    show_routes: bool,
    show_generation_progress: bool,
    generation_progress_every: int,
) -> list[InstanceResult]:
    """Chạy một objective (`mincost` hoặc `makespan`) trên nhiều instance."""
    td_loaded = load_npz_to_tensordict(npz_path)
    total_batches = int(td_loaded.batch_size[0])

    if batch_idx is not None:
        batch_indices = [batch_idx]
    else:
        batch_indices = list(range(total_batches))
        if num_batches is not None:
            batch_indices = batch_indices[:num_batches]
    results: list[InstanceResult] = []
    start_time = time.time()

    print(f"\n=== Running target={target} on {len(batch_indices)}/{total_batches} instances ===")
    for order_idx, idx in enumerate(batch_indices, start=1):
        instance_seed = seed + idx + (0 if target == "mincost" else 10_000)
        result = run_single_instance(
            td_loaded=td_loaded,
            batch_idx=idx,
            target=target,
            population=population,
            generations=generations,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            tournament_size=tournament_size,
            cull_ratio=cull_ratio,
            immigrant_ratio=immigrant_ratio,
            local_search_rate=local_search_rate,
            local_search_elites=local_search_elites,
            local_search_iters=local_search_iters,
            seed=instance_seed,
            show_routes=show_routes and len(batch_indices) == 1,
            show_generation_progress=show_generation_progress,
            generation_progress_every=generation_progress_every,
        )
        results.append(result)
        print(
            f"[{target}] {format_progress(order_idx, len(batch_indices), start_time)} "
            f"batch={idx:03d} "
            f"objective_cost={result.objective_cost:.4f} "
            f"score={result.score:.4f} "
            f"unserved={result.unserved} "
            f"makespan={result.makespan:.4f} "
            f"done={result.done}"
        )

    summary = summarize_results(results, target)
    print(f"\nSummary for target={target}")
    print(f"  instances    : {int(summary['count'])}")
    print(f"  avg_obj_cost : {summary['avg_objective_cost']:.4f}")
    print(f"  avg_score    : {summary['avg_score']:.4f}")
    print(f"  avg_unserved : {summary['avg_unserved']:.4f}")
    print(f"  avg_makespan : {summary['avg_makespan']:.4f}")
    print(f"  avg_distance : {summary['avg_distance']:.4f}")
    print(f"  feasible_rate: {summary['feasible_rate']:.4f}")

    if output_dir is not None:
        output_root = Path(output_dir)
    else:
        output_root = Path(npz_path).resolve().parent / "ga_results"
    output_root.mkdir(parents=True, exist_ok=True)
    output_file = output_root / f"{Path(npz_path).stem}_{target}_results.csv"
    save_results_csv(output_file, results)
    print(f"  saved_csv    : {output_file}")

    return results


def parse_args() -> argparse.Namespace:
    """Khai báo CLI để tinh chỉnh hyper-parameter và phạm vi chạy."""
    parser = argparse.ArgumentParser(description="Basic GA baseline for PVRPWDP.")
    parser.add_argument("--npz", required=True, help="Path to NPZ file containing PVRPWDP instances.")
    parser.add_argument("--batch-idx", type=int, default=None, help="Nếu truyền thì chỉ chạy một batch.")
    parser.add_argument("--num-batches", type=int, default=1, help="Chỉ chạy N batch đầu tiên của file NPZ.")
    parser.add_argument("--target", choices=["makespan", "mincost", "both"], default="both")
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--elite-size", type=int, default=4)
    parser.add_argument("--tournament-size", type=int, default=5)
    parser.add_argument("--cull-ratio", type=float, default=0.5, help="Tỷ lệ top population được giữ làm breeding pool.")
    parser.add_argument("--immigrant-ratio", type=float, default=0.05, help="Tỷ lệ cá thể mới random bơm vào mỗi thế hệ.")
    parser.add_argument("--local-search-rate", type=float, default=0.2, help="Xác suất áp local search cho child.")
    parser.add_argument("--local-search-elites", type=int, default=2, help="Số elite được local search mỗi thế hệ.")
    parser.add_argument("--local-search-iters", type=int, default=20, help="Số vòng local search tối đa.")
    parser.add_argument("--show-generation-progress", action="store_true", help="Hiển thị progress theo generation trong từng batch.")
    parser.add_argument("--generation-progress-every", type=int, default=10, help="In progress generation mỗi N thế hệ.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục lưu CSV kết quả.")
    parser.add_argument("--show-routes", action="store_true", help="In route chi tiết khi chạy một batch.")
    return parser.parse_args()


def main() -> None:
    """Điểm vào của script khi chạy từ command line."""
    args = parse_args()
    targets = ["mincost", "makespan"] if args.target ==  "both" else [args.target]
    all_results: dict[str, list[InstanceResult]] = {}

    for target in targets:
        all_results[target] = run_target_over_dataset(
            npz_path=args.npz,
            target=target,
            population=args.population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            elite_size=args.elite_size,
            tournament_size=args.tournament_size,
            cull_ratio=args.cull_ratio,
            immigrant_ratio=args.immigrant_ratio,
            local_search_rate=args.local_search_rate,
            local_search_elites=args.local_search_elites,
            local_search_iters=args.local_search_iters,
            seed=args.seed,
            output_dir=args.output_dir,
            batch_idx=args.batch_idx,
            num_batches=args.num_batches,
            show_routes=args.show_routes,
            show_generation_progress=args.show_generation_progress,
            generation_progress_every=args.generation_progress_every,
        )

    if len(targets) == 2:
        mincost_summary = summarize_results(all_results["mincost"], "mincost")
        makespan_summary = summarize_results(all_results["makespan"], "makespan")
        print("\n=== Combined overview ===")
        print(f"  mincost_avg_obj_cost  : {mincost_summary['avg_objective_cost']:.4f}")
        print(f"  mincost_feasible_rate : {mincost_summary['feasible_rate']:.4f}")
        print(f"  makespan_avg_obj_cost : {makespan_summary['avg_objective_cost']:.4f}")
        print(f"  makespan_avg_makespan : {makespan_summary['avg_makespan']:.4f}")
        print(f"  makespan_feasible_rate: {makespan_summary['feasible_rate']:.4f}")


if __name__ == "__main__":
    main()
