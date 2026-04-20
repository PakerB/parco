from __future__ import annotations

import argparse
import concurrent.futures
import csv
import functools  # Đã thêm thư viện functools cho lru_cache
import math
import random
import time

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rl4co.data.utils import load_npz_to_tensordict

from parco.envs.pvrpwdp import PVRPWDPVEnv
from parco.models.utils import resample_batch_padding


@dataclass
class DecodedSolution:
    routes: list[list[int]]
    unserved_customers: list[int]
    score: float
    makespan: float
    total_distance: float
    total_cost: float


@dataclass
class InstanceResult:
    target: str
    batch_idx: int
    offset: int | None
    cost: float
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
    """GA baseline: Optimized with 1D Arrays and Distance Matrix."""

    def __init__(self, env: PVRPWDPVEnv, td_raw: torch.Tensor, td_reset: torch.Tensor):
        self.env = env
        self.td_raw = td_raw
        self.td_reset = td_reset

        self.num_agents = int(td_reset["current_node"].shape[-1])
        self.num_customers = int(td_reset["locs"].shape[-2] - self.num_agents)
        self.max_time = float(td_reset["max_time"][0].item())

        self.locs = td_reset["locs"][0].cpu().numpy()
        self.customer_demands = td_reset["demand"][0, self.num_agents :].cpu().numpy()
        self.time_windows = td_reset["time_window"][0, self.num_agents :, :].cpu().numpy()
        self.waiting_time = td_reset["waiting_time"][0, self.num_agents :].cpu().numpy()
        self.agents_speed = np.maximum(td_reset["agents_speed"][0].cpu().numpy(), 1e-9)
        self.agents_capacity = td_reset["agents_capacity"][0].cpu().numpy()
        self.agents_endurance = td_reset["agents_endurance"][0].cpu().numpy()
        self.customer_env_indices = np.arange(
            self.num_agents, self.num_agents + self.num_customers
        )

        # Pre-compute distance matrix
        diff = self.locs[:, np.newaxis, :] - self.locs[np.newaxis, :, :]
        self.dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        self.is_truck = self._truck_mask_from_capacity(self.agents_capacity)
        self.travel_price = np.where(
            self.is_truck,
            float(getattr(env, "travel_price_truck", 0.35 / 1000)),
            float(getattr(env, "travel_price_drone", 1.5 / 3600)),
        )
        self.rent_price = np.where(
            self.is_truck,
            float(getattr(env, "rent_price_truck", 40.0)),
            float(getattr(env, "rent_price_drone", 10.0)),
        )

        bbox_diag = np.max(self.dist_matrix)

        # Đã cập nhật lambda_unserved để cover cả chi phí thời gian của Drone
        travel_max = (
            2.0
            * bbox_diag
            * self.num_customers
            * float(np.where(self.is_truck, self.travel_price, 0).max())
        )
        time_max = (
            self.max_time
            * self.num_customers
            * float(np.where(~self.is_truck, self.travel_price, 0).max())
        )
        self.lambda_unserved = travel_max + time_max + float(self.rent_price.sum()) + 1.0

        latest_max = float(self.time_windows[:, 1].max())
        min_speed = float(np.min(self.agents_speed))
        self.lambda_makespan_unserved = (
            (2.0 * bbox_diag * self.num_customers / max(min_speed, 1e-9))
            + latest_max
            + 1.0
        )

    @staticmethod
    def _truck_mask_from_capacity(capacity: np.ndarray) -> np.ndarray:
        thresh = 0.5 * (capacity.max() + capacity.min())
        return capacity > (thresh + 1e-6)

    def decode(self, chromosome: np.ndarray) -> DecodedSolution:
        """Hàm bọc ngoài để gọi _decode_cached với input có thể băm (hash) được."""
        key = tuple(int(x) for x in chromosome.tolist())
        return self._decode_cached(key)

    @functools.lru_cache(maxsize=15000)
    def _decode_cached(self, chromosome_tuple: tuple) -> DecodedSolution:
        """Hàm decode lõi, có LRU cache chống tràn RAM."""
        c_node = np.arange(self.num_agents, dtype=np.int64)
        c_time = np.zeros(self.num_agents, dtype=np.float64)
        c_length = np.zeros(self.num_agents, dtype=np.float64)
        total_op_time = np.zeros(
            self.num_agents, dtype=np.float64
        )  # Track thời gian hoạt động của Drone
        u_cap = np.zeros(self.num_agents, dtype=np.float64)
        u_end = np.zeros(self.num_agents, dtype=np.float64)
        t_dead = np.full(self.num_agents, self.max_time, dtype=np.float64)
        routes = [[] for _ in range(self.num_agents)]
        unserved: list[int] = []

        for customer_id in chromosome_tuple:
            cust_idx = int(self.customer_env_indices[customer_id])
            demand = float(self.customer_demands[customer_id])
            earliest, latest = self.time_windows[customer_id]
            wait_limit = float(self.waiting_time[customer_id])

            best_rank = (math.inf, math.inf, math.inf)
            best_update = None
            best_vehicle = -1

            # Thử chèn vào từng xe
            for v in range(self.num_agents):
                speed = float(self.agents_speed[v])
                capacity = float(self.agents_capacity[v])
                endurance = float(self.agents_endurance[v])

                # 1. Thử chèn vào tuyến hiện tại
                if u_cap[v] + demand <= capacity + 1e-9:
                    travel_dist = float(self.dist_matrix[c_node[v], cust_idx])
                    travel_time = travel_dist / speed
                    arrival = c_time[v] + travel_time
                    service_time = max(arrival, float(earliest))

                    if service_time <= float(latest) + 1e-9:
                        wait_mid = (
                            max(float(earliest) - arrival, 0.0) if c_node[v] != v else 0.0
                        )
                        new_u_end = u_end[v] + travel_time + wait_mid
                        back_time = float(self.dist_matrix[cust_idx, v]) / speed
                        return_time = service_time + back_time
                        new_deadline = min(t_dead[v], service_time + wait_limit)

                        if (
                            return_time <= new_deadline + 1e-9
                            and new_u_end + back_time <= endurance + 1e-9
                        ):
                            rank = (return_time, travel_dist, service_time)
                            if rank < best_rank:
                                best_rank = rank
                                best_vehicle = v
                                step_op = travel_time + wait_mid
                                best_update = (
                                    cust_idx,
                                    service_time,
                                    c_length[v] + travel_dist,
                                    u_cap[v] + demand,
                                    new_u_end,
                                    new_deadline,
                                    False,
                                    total_op_time[v] + step_op,
                                )

                # 2. Thử đóng chuyến hiện tại rồi mở chuyến mới
                if c_node[v] != v:
                    back_dist = float(self.dist_matrix[c_node[v], v])
                    c_length_reset = c_length[v] + back_dist
                    back_time_reset = back_dist / speed
                    c_time_reset = c_time[v] + back_time_reset

                    if demand <= capacity + 1e-9:
                        travel_dist_new = float(self.dist_matrix[v, cust_idx])
                        travel_time_new = travel_dist_new / speed
                        arrival_new = c_time_reset + travel_time_new
                        service_time_new = max(arrival_new, float(earliest))

                        if service_time_new <= float(latest) + 1e-9:
                            new_u_end_new = travel_time_new
                            back_time_new = float(self.dist_matrix[cust_idx, v]) / speed
                            return_time_new = service_time_new + back_time_new
                            new_deadline_new = min(
                                self.max_time, service_time_new + wait_limit
                            )

                            if (
                                return_time_new <= new_deadline_new + 1e-9
                                and new_u_end_new + back_time_new <= endurance + 1e-9
                            ):
                                rank_new = (
                                    return_time_new,
                                    travel_dist_new,
                                    service_time_new,
                                )
                                if rank_new < best_rank:
                                    best_rank = rank_new
                                    best_vehicle = v
                                    step_op_new = travel_time_new  # Từ depot xuất phát nên ko tính thời gian chờ
                                    best_update = (
                                        cust_idx,
                                        service_time_new,
                                        c_length_reset + travel_dist_new,
                                        demand,
                                        new_u_end_new,
                                        new_deadline_new,
                                        True,
                                        total_op_time[v] + back_time_reset + step_op_new,
                                    )

            # Cập nhật trạng thái sau khi đã tìm được best choice
            if best_vehicle != -1 and best_update is not None:
                v = best_vehicle
                (
                    n_c_node,
                    n_c_time,
                    n_c_len,
                    n_u_cap,
                    n_u_end,
                    n_t_dead,
                    is_reset,
                    n_total_op,
                ) = best_update

                if is_reset:
                    # Ghi nhận chặng về kho
                    c_time[v] += (
                        float(self.dist_matrix[c_node[v], v]) / self.agents_speed[v]
                    )
                    routes[v].append(v)

                c_node[v] = n_c_node
                c_time[v] = n_c_time
                c_length[v] = n_c_len
                u_cap[v] = n_u_cap
                u_end[v] = n_u_end
                t_dead[v] = n_t_dead
                total_op_time[v] = n_total_op
                routes[v].append(n_c_node)
            else:
                unserved.append(int(customer_id))

        # Đóng toàn bộ các route còn dang dở
        for v in range(self.num_agents):
            if c_node[v] != v:
                back_dist = float(self.dist_matrix[c_node[v], v])
                c_length[v] += back_dist
                back_time = back_dist / self.agents_speed[v]
                c_time[v] += back_time
                total_op_time[v] += back_time
                c_node[v] = v
                u_cap[v] = 0.0
                u_end[v] = 0.0
                t_dead[v] = self.max_time
                routes[v].append(v)

        total_distance = float(np.sum(c_length))
        makespan = float(np.max(c_time))

        # Tải dùng distance, Drone dùng thời gian hoạt động để tính giá
        usage_metric = np.where(self.is_truck, c_length, total_op_time)
        travel_part = float(np.sum(usage_metric * self.travel_price))

        used_vehicle = (c_length > 1e-8).astype(np.float32)
        rent_part = float(np.sum(used_vehicle * self.rent_price))
        total_cost = travel_part + rent_part

        score = self.objective_cost_from_parts(
            self.env.target, len(unserved), total_cost, makespan
        )

        solution = DecodedSolution(
            routes=routes,
            unserved_customers=unserved,
            score=score,
            makespan=makespan,
            total_distance=total_distance,
            total_cost=total_cost,
        )
        return solution

    def evaluate(self, chromosome: np.ndarray) -> float:
        return self.decode(chromosome).score

    def objective_cost_from_parts(
        self, target: str, unserved_count: int, total_cost: float, makespan: float
    ) -> float:
        if target == "mincost":
            return self.lambda_unserved * unserved_count + total_cost
        if target == "makespan":
            return self.lambda_makespan_unserved * unserved_count + makespan
        raise NotImplementedError(f"Unsupported target: {target}")

    def objective_cost(self, solution: DecodedSolution) -> float:
        return self.objective_cost_from_parts(
            self.env.target,
            len(solution.unserved_customers),
            solution.total_cost,
            solution.makespan,
        )

    def chromosome_seeds(self) -> list[np.ndarray]:
        customer_ids = np.arange(self.num_customers, dtype=np.int64)
        earliest = np.argsort(self.time_windows[:, 0])
        latest = np.argsort(self.time_windows[:, 1])
        freshness = np.argsort(self.waiting_time)
        demand_desc = np.argsort(-self.customer_demands)
        depot_dist = np.argsort(
            [
                float(self.dist_matrix[0, int(env_idx)])
                for env_idx in self.customer_env_indices
            ]
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

    def order_crossover(
        self, parent_a: np.ndarray, parent_b: np.ndarray, rng: random.Random
    ) -> np.ndarray:
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
        child = chromosome.copy()
        size = len(child)
        i, j = rng.sample(range(size), 2)
        gene = int(child[i])
        child = np.delete(child, i)
        child = np.insert(child, j, gene)
        return child.astype(np.int64, copy=False)

    def reverse_segment(self, chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
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
        current = chromosome.copy()
        current_score = self.evaluate(current)
        operators = (self.mutate, self.relocate_mutation, self.reverse_segment)

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
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        run_start_time = time.time()

        seeds = self.chromosome_seeds()
        population: list[np.ndarray] = [seed_perm.copy() for seed_perm in seeds]
        base_perm = np.arange(self.num_customers, dtype=np.int64)
        while len(population) < population_size:
            population.append(np_rng.permutation(base_perm))

        best_solution: DecodedSolution | None = None

        for generation in range(generations):
            decoded = [self.decode(chromosome) for chromosome in population]
            fitness = [item.score for item in decoded]
            order = sorted(range(len(population)), key=lambda idx: fitness[idx])
            population = [population[idx] for idx in order]
            decoded = [decoded[idx] for idx in order]
            fitness = [fitness[idx] for idx in order]

            if local_search_elites > 0:
                elite_ls_count = min(local_search_elites, len(population))
                for idx in range(elite_ls_count):
                    improved = self.local_search(
                        population[idx], rng, max_iters=local_search_iters
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

            if best_solution is None or decoded[0].score < best_solution.score:
                best_solution = decoded[0]

            if verbose and (
                generation == 0
                or (generation + 1) % max(1, progress_every) == 0
                or generation == generations - 1
            ):
                prefix = f"[{progress_label}] " if progress_label else "[GA] "
                print(
                    f"{prefix}{format_progress(generation + 1, generations, run_start_time, width=20)} gen={generation + 1:03d} best_score={decoded[0].score:.4f} unserved={len(decoded[0].unserved_customers)} makespan={decoded[0].makespan:.4f} distance={decoded[0].total_distance:.4f}"
                )

            next_population: list[np.ndarray] = [
                population[idx].copy() for idx in range(min(elite_size, len(population)))
            ]
            survivor_count = max(elite_size + 2, int(round(population_size * cull_ratio)))
            survivor_count = min(survivor_count, len(population))
            breeding_pool = population[:survivor_count]
            breeding_fitness = fitness[:survivor_count]

            immigrant_count = int(round(population_size * immigrant_ratio))
            immigrant_count = min(
                max(0, immigrant_count), max(0, population_size - len(next_population))
            )

            while len(next_population) < population_size:
                if (
                    immigrant_count > 0
                    and len(next_population) >= population_size - immigrant_count
                ):
                    next_population.append(np_rng.permutation(base_perm))
                    continue

                parent_a = self.tournament_select(
                    breeding_pool, breeding_fitness, tournament_size, rng
                )
                parent_b = self.tournament_select(
                    breeding_pool, breeding_fitness, tournament_size, rng
                )
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
        route_lists: list[list[int]] = []
        for v_idx, route in enumerate(solution.routes):
            if route:
                route_lists.append(list(route))
            else:
                route_lists.append([v_idx])

        horizon = max(len(route) for route in route_lists)
        action_rows = []
        for route in route_lists:
            last = route[-1]
            padded = route + [last] * (horizon - len(route))
            action_rows.append(padded)

        actions = torch.tensor(action_rows, dtype=torch.int64).unsqueeze(0)
        return actions

    def evaluate_with_env(self, solution: DecodedSolution) -> dict[str, float]:
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
        print("\nBest solution summary")
        print(f"  score          : {solution.score:.4f}")
        print(f"  unserved       : {len(solution.unserved_customers)}")
        print(f"  makespan       : {solution.makespan:.4f}")
        print(f"  total_distance : {solution.total_distance:.4f}")
        print(f"  total_cost     : {solution.total_cost:.4f}")
        if solution.unserved_customers:
            print(f"  unserved_ids   : {solution.unserved_customers}")

        for v_id, route in enumerate(solution.routes):
            display_route = []
            for node in route:
                if node < self.num_agents:
                    display_route.append(f"D{node}")
                else:
                    display_route.append(f"C{node - self.num_agents}")
            print(
                f"  vehicle {v_id}: {' -> '.join(display_route) if display_route else f'D{v_id}'}"
            )


# --- Helper Methods ---


def load_instance_from_td(
    td_loaded: torch.Tensor, batch_idx: int, target: str
) -> tuple[PVRPWDPVEnv, torch.Tensor, torch.Tensor]:
    if batch_idx >= td_loaded.batch_size[0]:
        raise IndexError(
            f"batch_idx={batch_idx} out of range for batch size {td_loaded.batch_size[0]}"
        )

    td_raw = td_loaded[batch_idx].unsqueeze(0)
    # Strip virtual padding (num_real_nodes, num_real_agents) so max_time and other
    # derived fields are computed only over real customers/agents. Without this,
    # padded agents with speed=0 make speed_min=0 and produce max_time=NaN, which
    # causes every insertion attempt in the GA decode to fail (unserved=100%).
    if "num_real_nodes" in td_raw.keys() and "num_real_agents" in td_raw.keys():
        td_raw = resample_batch_padding(td_raw)
    env = PVRPWDPVEnv(use_epoch_data=False, fallback_to_generator=False, target=target)
    td_reset = env.reset(td_raw.clone())
    return env, td_raw, td_reset


def load_instance(
    npz_path: str, batch_idx: int, target: str
) -> tuple[PVRPWDPVEnv, torch.Tensor, torch.Tensor]:
    td_loaded = load_npz_to_tensordict(npz_path)
    return load_instance_from_td(td_loaded, batch_idx, target)


def get_optional_batch_float(
    td_loaded: torch.Tensor, batch_idx: int, keys: tuple[str, ...]
) -> float:
    for key in keys:
        if key in td_loaded.keys():
            value = td_loaded[key][batch_idx]
            return float(value.reshape(-1)[0].item())
    return math.nan


def get_optional_batch_int(
    td_loaded: torch.Tensor, batch_idx: int, keys: tuple[str, ...]
) -> int | None:
    for key in keys:
        if key in td_loaded.keys():
            value = td_loaded[key][batch_idx]
            return int(value.reshape(-1)[0].item())
    return None


def select_batch_indices(
    td_loaded: torch.Tensor,
    batch_idx: int | None,
    num_batches: int | None,
    offset: int | None,
) -> list[int]:
    total_batches = int(td_loaded.batch_size[0])

    if batch_idx is not None:
        if batch_idx < 0 or batch_idx >= total_batches:
            raise IndexError(
                f"batch_idx={batch_idx} out of range for batch size {total_batches}"
            )
        if offset is not None:
            batch_offset = get_optional_batch_int(
                td_loaded, batch_idx, ("offsets", "offset")
            )
            if batch_offset is None:
                raise KeyError(
                    "--offset was passed, but the NPZ file has no 'offsets' or 'offset' key"
                )
            if batch_offset != offset:
                raise ValueError(
                    f"batch_idx={batch_idx} has offset={batch_offset}, not requested offset={offset}"
                )
        return [batch_idx]

    if offset is not None:
        if "offsets" in td_loaded.keys():
            offsets = td_loaded["offsets"]
        elif "offset" in td_loaded.keys():
            offsets = td_loaded["offset"]
        else:
            raise KeyError(
                "--offset was passed, but the NPZ file has no 'offsets' or 'offset' key"
            )
        flat_offsets = offsets.detach().cpu().reshape(-1)
        batch_indices = [
            idx for idx, value in enumerate(flat_offsets.tolist()) if int(value) == offset
        ]
    else:
        batch_indices = list(range(total_batches))

    if num_batches is not None:
        if num_batches < 0:
            raise ValueError(f"num_batches must be >= 0, got {num_batches}")
        batch_indices = batch_indices[:num_batches]

    return batch_indices


def summarize_results(results: list[InstanceResult], target: str) -> dict[str, float]:
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
        "feasible_rate": float(
            np.mean(
                [1.0 if item.done and item.unserved == 0 else 0.0 for item in results]
            )
        ),
    }


def save_results_csv(output_path: Path, results: list[InstanceResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "batch_idx",
                "offset",
                "cost",
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


def format_progress(current: int, total: int, start_time: float, width: int = 28) -> str:
    ratio = 1.0 if total <= 0 else max(0.0, min(1.0, current / total))
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(0.0, time.time() - start_time)
    eta = (elapsed / max(current, 1)) * max(total - current, 0)
    return f"[{bar}] {current}/{total} ({ratio * 100:5.1f}%) elapsed={elapsed:7.1f}s eta={eta:7.1f}s"


def worker_run_single_instance(
    npz_path: str,
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
    # Mỗi worker tự load riêng data của nó để không xung đột bộ nhớ chia sẻ
    env, td_raw, td_reset = load_instance(npz_path, batch_idx, target)
    reference_cost = get_optional_batch_float(td_raw, 0, ("costs", "cost"))
    offset = get_optional_batch_int(td_raw, 0, ("offsets", "offset"))
    solver = BasicPVRPWDPSolver(env=env, td_raw=td_raw, td_reset=td_reset)

    # Tắt hiển thị theo generation khi chạy song song để khỏi rác console
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
        progress_label=f"{target}|b={batch_idx:03d}",
        progress_every=generation_progress_every,
    )
    env_eval = solver.evaluate_with_env(best)

    if show_routes:
        print(f"\n[target={target}] batch={batch_idx}")
        solver.print_solution(best)

    return InstanceResult(
        target=target,
        batch_idx=batch_idx,
        offset=offset,
        cost=reference_cost,
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
    offset: int | None,
    show_routes: bool,
    show_generation_progress: bool,
    generation_progress_every: int,
    max_workers: int,
) -> list[InstanceResult]:
    td_loaded = load_npz_to_tensordict(npz_path)
    total_batches = int(td_loaded.batch_size[0])
    batch_indices = select_batch_indices(td_loaded, batch_idx, num_batches, offset)

    results: list[InstanceResult] = []
    start_time = time.time()

    offset_note = f", offset={offset}" if offset is not None else ""
    print(
        f"\n=== Running target={target} on {len(batch_indices)}/{total_batches} instances{offset_note} (Workers: {max_workers}) ==="
    )

    # Thực thi song song
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx in batch_indices:
            instance_seed = seed + idx + (0 if target == "mincost" else 10_000)

            future = executor.submit(
                worker_run_single_instance,
                npz_path,
                idx,
                target,
                population,
                generations,
                mutation_rate,
                elite_size,
                tournament_size,
                cull_ratio,
                immigrant_ratio,
                local_search_rate,
                local_search_elites,
                local_search_iters,
                instance_seed,
                show_routes and len(batch_indices) == 1,
                show_generation_progress,
                generation_progress_every,
            )
            futures[future] = idx

        for order_idx, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                cost_text = f"{result.cost:.4f}" if math.isfinite(result.cost) else "nan"
                print(
                    f"[{target}] {format_progress(order_idx, len(batch_indices), start_time)} batch={idx:03d} offset={result.offset} cost={cost_text} objective_cost={result.objective_cost:.4f} score={result.score:.4f} unserved={result.unserved} makespan={result.makespan:.4f} done={result.done}"
                )
            except Exception as exc:
                print(f"Batch {idx} generated an exception: {exc}")

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
    offset_suffix = f"_offset_{offset}" if offset is not None else ""
    output_file = (
        output_root / f"{Path(npz_path).stem}_{target}{offset_suffix}_results.csv"
    )
    save_results_csv(output_file, results)
    print(f"  saved_csv    : {output_file}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimized GA baseline for PVRPWDP (Vectorized + Multi-core)."
    )
    parser.add_argument(
        "--npz", required=True, help="Path to NPZ file containing PVRPWDP instances."
    )
    parser.add_argument(
        "--batch-idx", type=int, default=None, help="Run exactly one batch index."
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Run only the first N selected batches.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Only run batches whose NPZ offsets/offset value matches this number.",
    )
    parser.add_argument(
        "--target", choices=["makespan", "mincost", "both"], default="both"
    )
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=400)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--elite-size", type=int, default=4)
    parser.add_argument("--tournament-size", type=int, default=5)
    parser.add_argument("--cull-ratio", type=float, default=0.5)
    parser.add_argument("--immigrant-ratio", type=float, default=0.05)
    parser.add_argument("--local-search-rate", type=float, default=0.2)
    parser.add_argument("--local-search-elites", type=int, default=2)
    parser.add_argument("--local-search-iters", type=int, default=20)
    parser.add_argument("--show-generation-progress", action="store_true")
    parser.add_argument("--generation-progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--show-routes", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Number of CPU workers used to run batches in parallel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = ["mincost", "makespan"] if args.target == "both" else [args.target]
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
            offset=args.offset,
            show_routes=args.show_routes,
            show_generation_progress=args.show_generation_progress,
            generation_progress_every=args.generation_progress_every,
            max_workers=args.max_workers,
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

# uv run python ga_pvrpwdp.py --npz D:\k0d3\DATN\parco\test_data\test.npz --offset 1 --num-batches 10 --target makespan --population 100 --generations 400 --show-generation-progress --generation-progress-every 100 
