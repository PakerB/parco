"""HGS + split decoder baseline for PVRPWDP.

This script adapts the paper's HGS + SPA idea to the local PVRPWDP physics:

* Chromosomes are giant tours over customer ids.
* A dynamic-programming split decoder partitions each giant tour into
  contiguous vehicle blocks.
* Each vehicle block is split into one or more trips with PVRPWDP feasibility:
  capacity, time windows, freshness deadlines, and endurance.

The original paper solves MT-TD-VRP with time-dependent travel times and
identical vehicles. PVRPWDP in this repo uses static per-agent speeds and
heterogeneous trucks/drones, so the split stage below keeps the paper's
giant-tour/DP structure while replacing TD breakpoints with the environment's
deterministic step physics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import random
import time

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rl4co.data.utils import load_npz_to_tensordict

from ga_pvrpwdp import (
    InstanceResult,
    format_progress,
    load_instance,
    load_instance_from_td,
    save_results_csv,
    summarize_results,
    get_optional_batch_float,
    get_optional_batch_int,
    select_batch_indices,
)
from parco.envs.pvrpwdp import PVRPWDPVEnv

EPS = 1e-9


@dataclass(frozen=True)
class TripSimulation:
    actions: tuple[int, ...]
    finish_time: float
    distance: float
    operating_time: float
    travel_cost: float


@dataclass(frozen=True)
class BlockLabel:
    finish_time: float
    travel_cost: float
    distance: float
    operating_time: float
    routes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BlockSolution:
    finish_time: float
    travel_cost: float
    total_cost: float
    distance: float
    operating_time: float
    routes: tuple[tuple[int, ...], ...]


@dataclass
class DecodedSolution:
    routes: list[list[int]]
    unserved_customers: list[int]
    score: float
    makespan: float
    total_distance: float
    total_cost: float


@dataclass
class Individual:
    chromosome: np.ndarray
    solution: DecodedSolution


class HGSSplitPVRPWDPSolver:
    """Hybrid genetic search with a DP split decoder for PVRPWDP."""

    def __init__(
        self,
        env: PVRPWDPVEnv,
        td_raw: torch.Tensor,
        td_reset: torch.Tensor,
        max_labels: int = 24,
    ):
        self.env = env
        self.td_raw = td_raw
        self.td_reset = td_reset
        self.max_labels = max(1, int(max_labels))

        self.num_agents = int(td_reset["current_node"].shape[-1])
        self.num_customers = int(td_reset["locs"].shape[-2] - self.num_agents)
        self.max_time = float(td_reset["max_time"][0].item())

        self.locs = td_reset["locs"][0].cpu().numpy()
        demand_tensor = td_reset["demand"][0]
        if demand_tensor.ndim == 2:
            demand_tensor = demand_tensor.squeeze(0)
        self.customer_demands = demand_tensor[self.num_agents :].cpu().numpy()
        self.time_windows = td_reset["time_window"][0, self.num_agents :, :].cpu().numpy()
        self.waiting_time = td_reset["waiting_time"][0, self.num_agents :].cpu().numpy()
        self.agents_speed = np.maximum(
            td_reset["agents_speed"][0].cpu().numpy(),
            EPS,
        )
        self.agents_capacity = td_reset["agents_capacity"][0].cpu().numpy()
        self.agents_endurance = td_reset["agents_endurance"][0].cpu().numpy()
        self.customer_env_indices = np.arange(
            self.num_agents,
            self.num_agents + self.num_customers,
            dtype=np.int64,
        )

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

        bbox_diag = float(np.linalg.norm(self.locs.max(axis=0) - self.locs.min(axis=0)))
        truck_price = float(np.where(self.is_truck, self.travel_price, 0.0).max())
        drone_price = float(np.where(~self.is_truck, self.travel_price, 0.0).max())
        travel_max = 2.0 * bbox_diag * self.num_customers * truck_price
        time_max = self.max_time * max(1, self.num_customers) * drone_price
        self.lambda_unserved = travel_max + time_max + float(self.rent_price.sum()) + 1.0

        latest_max = float(self.time_windows[:, 1].max()) if self.num_customers else 0.0
        min_speed = float(np.min(self.agents_speed))
        self.lambda_makespan_unserved = (
            (2.0 * bbox_diag * self.num_customers / max(min_speed, EPS))
            + latest_max
            + 1.0
        )

        self._solution_cache: dict[tuple[int, ...], DecodedSolution] = {}

    @staticmethod
    def _truck_mask_from_capacity(capacity: np.ndarray) -> np.ndarray:
        thresh = 0.5 * (capacity.max() + capacity.min())
        return capacity > (thresh + 1e-6)

    def _vehicle_metric_cost(
        self,
        vehicle_id: int,
        distance: float,
        operating_time: float,
    ) -> float:
        metric = distance if self.is_truck[vehicle_id] else operating_time
        return float(metric * self.travel_price[vehicle_id])

    def _simulate_trip(
        self,
        vehicle_id: int,
        customer_ids: tuple[int, ...],
        available_time: float,
    ) -> TripSimulation | None:
        if not customer_ids:
            return None

        depot_idx = vehicle_id
        speed = float(self.agents_speed[vehicle_id])
        capacity = float(self.agents_capacity[vehicle_id])
        endurance = float(self.agents_endurance[vehicle_id])

        current_node = depot_idx
        current_time = float(available_time)
        used_capacity = 0.0
        used_endurance = 0.0
        trip_deadline = self.max_time
        distance = 0.0
        operating_time = 0.0
        actions: list[int] = []

        for customer_id in customer_ids:
            customer_id = int(customer_id)
            customer_idx = int(self.customer_env_indices[customer_id])
            demand = float(self.customer_demands[customer_id])
            if used_capacity + demand > capacity + EPS:
                return None

            step_dist = float(self.dist_matrix[current_node, customer_idx])
            travel_time = step_dist / speed
            arrival = current_time + travel_time
            earliest, latest = self.time_windows[customer_id]
            service_time = max(arrival, float(earliest))
            if service_time > float(latest) + EPS:
                return None

            mid_trip_wait = 0.0
            if current_node != depot_idx:
                mid_trip_wait = max(float(earliest) - arrival, 0.0)

            new_used_endurance = used_endurance + travel_time + mid_trip_wait
            back_time = float(self.dist_matrix[customer_idx, depot_idx]) / speed
            return_time = service_time + back_time
            new_deadline = min(
                trip_deadline, service_time + float(self.waiting_time[customer_id])
            )

            if return_time > new_deadline + EPS:
                return None
            if new_used_endurance + back_time > endurance + EPS:
                return None

            used_capacity += demand
            used_endurance = new_used_endurance
            trip_deadline = new_deadline
            current_time = service_time
            current_node = customer_idx
            distance += step_dist
            operating_time += travel_time + mid_trip_wait
            actions.append(customer_idx)

        back_dist = float(self.dist_matrix[current_node, depot_idx])
        back_time = back_dist / speed
        finish_time = current_time + back_time
        distance += back_dist
        operating_time += back_time
        actions.append(depot_idx)

        travel_cost = self._vehicle_metric_cost(vehicle_id, distance, operating_time)
        return TripSimulation(
            actions=tuple(actions),
            finish_time=finish_time,
            distance=distance,
            operating_time=operating_time,
            travel_cost=travel_cost,
        )

    def _prune_labels(self, labels: list[BlockLabel]) -> list[BlockLabel]:
        labels = sorted(labels, key=lambda item: (item.travel_cost, item.finish_time))
        kept: list[BlockLabel] = []
        for label in labels:
            dominated = False
            for other in kept:
                if (
                    other.travel_cost <= label.travel_cost + EPS
                    and other.finish_time <= label.finish_time + EPS
                ):
                    dominated = True
                    break
            if not dominated:
                kept.append(label)

        if len(kept) <= self.max_labels:
            return kept

        if self.env.target == "makespan":
            kept = sorted(kept, key=lambda item: (item.finish_time, item.travel_cost))
        else:
            kept = sorted(kept, key=lambda item: (item.travel_cost, item.finish_time))
        return kept[: self.max_labels]

    def _best_vehicle_block(
        self,
        tour: tuple[int, ...],
        vehicle_id: int,
        start: int,
        end: int,
        cache: dict[tuple[int, int, int], BlockSolution | None],
    ) -> BlockSolution | None:
        if start == end:
            return BlockSolution(
                finish_time=0.0,
                travel_cost=0.0,
                total_cost=0.0,
                distance=0.0,
                operating_time=0.0,
                routes=(),
            )

        key = (vehicle_id, start, end)
        if key in cache:
            return cache[key]

        labels_by_pos: dict[int, list[BlockLabel]] = {
            start: [
                BlockLabel(
                    finish_time=0.0,
                    travel_cost=0.0,
                    distance=0.0,
                    operating_time=0.0,
                    routes=(),
                )
            ]
        }

        for pos in range(start, end):
            labels = labels_by_pos.get(pos)
            if not labels:
                continue
            for label in labels:
                for nxt in range(pos + 1, end + 1):
                    trip = self._simulate_trip(
                        vehicle_id,
                        tour[pos:nxt],
                        label.finish_time,
                    )
                    if trip is None:
                        break

                    new_label = BlockLabel(
                        finish_time=trip.finish_time,
                        travel_cost=label.travel_cost + trip.travel_cost,
                        distance=label.distance + trip.distance,
                        operating_time=label.operating_time + trip.operating_time,
                        routes=label.routes + (trip.actions,),
                    )
                    labels_by_pos.setdefault(nxt, []).append(new_label)
                    labels_by_pos[nxt] = self._prune_labels(labels_by_pos[nxt])

        final_labels = labels_by_pos.get(end, [])
        if not final_labels:
            cache[key] = None
            return None

        if self.env.target == "makespan":
            best = min(
                final_labels, key=lambda item: (item.finish_time, item.travel_cost)
            )
        else:
            best = min(
                final_labels, key=lambda item: (item.travel_cost, item.finish_time)
            )

        total_cost = best.travel_cost + float(self.rent_price[vehicle_id])
        solution = BlockSolution(
            finish_time=best.finish_time,
            travel_cost=best.travel_cost,
            total_cost=total_cost,
            distance=best.distance,
            operating_time=best.operating_time,
            routes=best.routes,
        )
        cache[key] = solution
        return solution

    def decode(self, chromosome: np.ndarray) -> DecodedSolution:
        key = tuple(int(x) for x in chromosome.tolist())
        cached = self._solution_cache.get(key)
        if cached is not None:
            return cached

        solution = self._decode_split(key)
        if len(self._solution_cache) > 50_000:
            self._solution_cache.clear()
        self._solution_cache[key] = solution
        return solution

    def _decode_split(self, tour: tuple[int, ...]) -> DecodedSolution:
        n = len(tour)
        m = self.num_agents
        inf = float("inf")
        block_cache: dict[tuple[int, int, int], BlockSolution | None] = {}

        dp = [[inf] * (n + 1) for _ in range(m + 1)]
        parents: list[list[tuple[int, BlockSolution | None] | None]] = [
            [None] * (n + 1) for _ in range(m + 1)
        ]
        dp[0][0] = 0.0

        for vehicle_id in range(m):
            next_row = vehicle_id + 1
            for served in range(n + 1):
                prev_value = dp[vehicle_id][served]
                if not math.isfinite(prev_value):
                    continue

                if prev_value + EPS < dp[next_row][served]:
                    dp[next_row][served] = prev_value
                    parents[next_row][served] = (served, None)

                for end in range(served + 1, n + 1):
                    block = self._best_vehicle_block(
                        tour,
                        vehicle_id,
                        served,
                        end,
                        block_cache,
                    )
                    if block is None:
                        break

                    if self.env.target == "makespan":
                        candidate = max(prev_value, block.finish_time)
                    else:
                        candidate = prev_value + block.total_cost

                    if candidate + EPS < dp[next_row][end]:
                        dp[next_row][end] = candidate
                        parents[next_row][end] = (served, block)

        penalty = (
            self.lambda_makespan_unserved
            if self.env.target == "makespan"
            else self.lambda_unserved
        )
        best_served = min(
            range(n + 1),
            key=lambda i: dp[m][i] + penalty * (n - i),
        )
        best_value = dp[m][best_served]
        unserved = [int(customer_id) for customer_id in tour[best_served:]]

        routes: list[list[int]] = [[] for _ in range(m)]

        cursor = best_served
        for next_row in range(m, 0, -1):
            parent = parents[next_row][cursor]
            if parent is None:
                cursor = 0
                continue

            prev_cursor, block = parent
            vehicle_id = next_row - 1
            if block is not None:
                flat_actions: list[int] = []
                for trip_actions in block.routes:
                    flat_actions.extend(int(node) for node in trip_actions)
                routes[vehicle_id] = flat_actions
            cursor = prev_cursor

        if not math.isfinite(best_value):
            unserved = [int(customer_id) for customer_id in tour]

        return self._repair_and_score(routes, unserved, penalty)

    def _initial_repair_state(
        self,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        current_node = np.arange(self.num_agents, dtype=np.int64)
        current_time = np.zeros(self.num_agents, dtype=np.float64)
        current_length = np.zeros(self.num_agents, dtype=np.float64)
        operating_time = np.zeros(self.num_agents, dtype=np.float64)
        used_capacity = np.zeros(self.num_agents, dtype=np.float64)
        used_endurance = np.zeros(self.num_agents, dtype=np.float64)
        trip_deadline = np.full(self.num_agents, self.max_time, dtype=np.float64)
        return (
            current_node,
            current_time,
            current_length,
            operating_time,
            used_capacity,
            used_endurance,
            trip_deadline,
        )

    def _apply_route_action(
        self,
        vehicle_id: int,
        action: int,
        current_node: np.ndarray,
        current_time: np.ndarray,
        current_length: np.ndarray,
        operating_time: np.ndarray,
        used_capacity: np.ndarray,
        used_endurance: np.ndarray,
        trip_deadline: np.ndarray,
    ) -> None:
        depot_idx = vehicle_id
        speed = float(self.agents_speed[vehicle_id])
        action = int(action)

        if action < self.num_agents:
            back_dist = float(self.dist_matrix[current_node[vehicle_id], depot_idx])
            back_time = back_dist / speed
            current_length[vehicle_id] += back_dist
            current_time[vehicle_id] += back_time
            if current_node[vehicle_id] != depot_idx:
                operating_time[vehicle_id] += back_time
            current_node[vehicle_id] = depot_idx
            used_capacity[vehicle_id] = 0.0
            used_endurance[vehicle_id] = 0.0
            trip_deadline[vehicle_id] = self.max_time
            return

        customer_id = action - self.num_agents
        step_dist = float(self.dist_matrix[current_node[vehicle_id], action])
        travel_time = step_dist / speed
        arrival = current_time[vehicle_id] + travel_time
        earliest = float(self.time_windows[customer_id, 0])
        service_time = max(arrival, earliest)
        mid_trip_wait = 0.0
        if current_node[vehicle_id] != depot_idx:
            mid_trip_wait = max(earliest - arrival, 0.0)

        current_length[vehicle_id] += step_dist
        operating_time[vehicle_id] += travel_time + mid_trip_wait
        current_time[vehicle_id] = service_time
        used_capacity[vehicle_id] += float(self.customer_demands[customer_id])
        used_endurance[vehicle_id] += travel_time + mid_trip_wait

        new_deadline = service_time + float(self.waiting_time[customer_id])
        if current_node[vehicle_id] == depot_idx:
            trip_deadline[vehicle_id] = new_deadline
        else:
            trip_deadline[vehicle_id] = min(trip_deadline[vehicle_id], new_deadline)
        current_node[vehicle_id] = action

    def _customer_transition(
        self,
        vehicle_id: int,
        start_node: int,
        start_time: float,
        used_capacity: float,
        used_endurance: float,
        trip_deadline: float,
        customer_id: int,
    ) -> tuple[float, float, float, float, float, float, float, int] | None:
        depot_idx = vehicle_id
        customer_idx = int(self.customer_env_indices[customer_id])
        speed = float(self.agents_speed[vehicle_id])
        demand = float(self.customer_demands[customer_id])
        capacity = float(self.agents_capacity[vehicle_id])
        endurance = float(self.agents_endurance[vehicle_id])

        if used_capacity + demand > capacity + EPS:
            return None

        step_dist = float(self.dist_matrix[start_node, customer_idx])
        travel_time = step_dist / speed
        arrival = start_time + travel_time
        earliest, latest = self.time_windows[customer_id]
        service_time = max(arrival, float(earliest))
        if service_time > float(latest) + EPS:
            return None

        mid_trip_wait = 0.0
        if start_node != depot_idx:
            mid_trip_wait = max(float(earliest) - arrival, 0.0)

        new_used_endurance = used_endurance + travel_time + mid_trip_wait
        back_time = float(self.dist_matrix[customer_idx, depot_idx]) / speed
        return_time = service_time + back_time
        new_deadline = min(
            trip_deadline, service_time + float(self.waiting_time[customer_id])
        )

        if return_time > new_deadline + EPS:
            return None
        if new_used_endurance + back_time > endurance + EPS:
            return None

        step_op = travel_time + mid_trip_wait
        return (
            service_time,
            step_dist,
            step_op,
            used_capacity + demand,
            new_used_endurance,
            new_deadline,
            return_time,
            customer_idx,
        )

    def _repair_and_score(
        self,
        routes: list[list[int]],
        candidate_unserved: list[int],
        penalty: float,
    ) -> DecodedSolution:
        (
            current_node,
            current_time,
            current_length,
            operating_time,
            used_capacity,
            used_endurance,
            trip_deadline,
        ) = self._initial_repair_state()

        routes = [list(route) for route in routes]
        for vehicle_id, route in enumerate(routes):
            for action in route:
                self._apply_route_action(
                    vehicle_id,
                    action,
                    current_node,
                    current_time,
                    current_length,
                    operating_time,
                    used_capacity,
                    used_endurance,
                    trip_deadline,
                )

        remaining_unserved: list[int] = []
        for customer_id in candidate_unserved:
            best_rank = (math.inf, math.inf, math.inf)
            best_update = None

            for vehicle_id in range(self.num_agents):
                append = self._customer_transition(
                    vehicle_id,
                    int(current_node[vehicle_id]),
                    float(current_time[vehicle_id]),
                    float(used_capacity[vehicle_id]),
                    float(used_endurance[vehicle_id]),
                    float(trip_deadline[vehicle_id]),
                    int(customer_id),
                )
                if append is not None:
                    (
                        service_time,
                        step_dist,
                        step_op,
                        next_capacity,
                        next_endurance,
                        next_deadline,
                        return_time,
                        customer_idx,
                    ) = append
                    added_cost = self._vehicle_metric_cost(vehicle_id, step_dist, step_op)
                    rank = (return_time, added_cost, service_time)
                    if rank < best_rank:
                        best_rank = rank
                        best_update = (
                            vehicle_id,
                            False,
                            customer_idx,
                            service_time,
                            step_dist,
                            step_op,
                            next_capacity,
                            next_endurance,
                            next_deadline,
                            0.0,
                            0.0,
                        )

                if current_node[vehicle_id] == vehicle_id:
                    continue

                close_dist = float(self.dist_matrix[current_node[vehicle_id], vehicle_id])
                close_time = close_dist / float(self.agents_speed[vehicle_id])
                reset_time = float(current_time[vehicle_id]) + close_time
                restart = self._customer_transition(
                    vehicle_id,
                    vehicle_id,
                    reset_time,
                    0.0,
                    0.0,
                    self.max_time,
                    int(customer_id),
                )
                if restart is not None:
                    (
                        service_time,
                        step_dist,
                        step_op,
                        next_capacity,
                        next_endurance,
                        next_deadline,
                        return_time,
                        customer_idx,
                    ) = restart
                    added_metric = close_dist + step_dist
                    added_op = close_time + step_op
                    added_cost = self._vehicle_metric_cost(
                        vehicle_id,
                        added_metric,
                        added_op,
                    )
                    rank = (return_time, added_cost, service_time)
                    if rank < best_rank:
                        best_rank = rank
                        best_update = (
                            vehicle_id,
                            True,
                            customer_idx,
                            service_time,
                            step_dist,
                            step_op,
                            next_capacity,
                            next_endurance,
                            next_deadline,
                            close_dist,
                            close_time,
                        )

            if best_update is None:
                remaining_unserved.append(int(customer_id))
                continue

            (
                vehicle_id,
                closes_trip,
                customer_idx,
                service_time,
                step_dist,
                step_op,
                next_capacity,
                next_endurance,
                next_deadline,
                close_dist,
                close_time,
            ) = best_update

            if closes_trip:
                current_length[vehicle_id] += close_dist
                current_time[vehicle_id] += close_time
                operating_time[vehicle_id] += close_time
                current_node[vehicle_id] = vehicle_id
                used_capacity[vehicle_id] = 0.0
                used_endurance[vehicle_id] = 0.0
                trip_deadline[vehicle_id] = self.max_time
                routes[vehicle_id].append(vehicle_id)

            current_length[vehicle_id] += step_dist
            operating_time[vehicle_id] += step_op
            current_time[vehicle_id] = service_time
            current_node[vehicle_id] = customer_idx
            used_capacity[vehicle_id] = next_capacity
            used_endurance[vehicle_id] = next_endurance
            trip_deadline[vehicle_id] = next_deadline
            routes[vehicle_id].append(customer_idx)

        for vehicle_id in range(self.num_agents):
            if current_node[vehicle_id] == vehicle_id:
                continue
            back_dist = float(self.dist_matrix[current_node[vehicle_id], vehicle_id])
            back_time = back_dist / float(self.agents_speed[vehicle_id])
            current_length[vehicle_id] += back_dist
            current_time[vehicle_id] += back_time
            operating_time[vehicle_id] += back_time
            current_node[vehicle_id] = vehicle_id
            used_capacity[vehicle_id] = 0.0
            used_endurance[vehicle_id] = 0.0
            trip_deadline[vehicle_id] = self.max_time
            routes[vehicle_id].append(vehicle_id)

        usage_metric = np.where(self.is_truck, current_length, operating_time)
        travel_part = float(np.sum(usage_metric * self.travel_price))
        used_vehicle = (current_length > EPS).astype(np.float64)
        rent_part = float(np.sum(used_vehicle * self.rent_price))
        total_cost = travel_part + rent_part
        makespan = float(np.max(current_time))

        if self.env.target == "makespan":
            base_score = makespan
        else:
            base_score = total_cost
        score = base_score + penalty * len(remaining_unserved)

        return DecodedSolution(
            routes=routes,
            unserved_customers=remaining_unserved,
            score=float(score),
            makespan=float(makespan),
            total_distance=float(np.sum(current_length)),
            total_cost=float(total_cost),
        )

    def evaluate(self, chromosome: np.ndarray) -> float:
        return self.decode(chromosome).score

    def chromosome_seeds(self, rng: random.Random) -> list[np.ndarray]:
        customer_ids = np.arange(self.num_customers, dtype=np.int64)
        if self.num_customers == 0:
            return [customer_ids]

        depot = self.locs[0]
        customer_locs = self.locs[self.customer_env_indices]
        angles = np.arctan2(
            customer_locs[:, 1] - depot[1], customer_locs[:, 0] - depot[0]
        )
        depot_dist = np.array(
            [float(self.dist_matrix[0, int(idx)]) for idx in self.customer_env_indices]
        )

        seeds = [
            customer_ids,
            np.argsort(self.time_windows[:, 0]),
            np.argsort(self.time_windows[:, 1]),
            np.argsort(self.waiting_time),
            np.argsort(-self.customer_demands),
            np.argsort(depot_dist),
            np.argsort(angles),
        ]

        optics_seed = self._optional_optics_seed(rng)
        if optics_seed is not None:
            seeds.append(optics_seed)

        unique: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        for seed in seeds:
            seed = np.asarray(seed, dtype=np.int64)
            seed_key = tuple(int(x) for x in seed.tolist())
            if seed_key not in seen:
                seen.add(seed_key)
                unique.append(seed.copy())
        return unique

    def _optional_optics_seed(self, rng: random.Random) -> np.ndarray | None:
        try:
            from sklearn.cluster import OPTICS
        except Exception:
            return None

        if self.num_customers < 4:
            return None

        customer_locs = self.locs[self.customer_env_indices]
        min_samples = max(2, min(10, self.num_customers // 10))
        try:
            labels = OPTICS(min_samples=min_samples).fit_predict(customer_locs)
        except Exception:
            return None

        clusters: dict[int, list[int]] = {}
        for customer_id, label in enumerate(labels.tolist()):
            clusters.setdefault(int(label), []).append(customer_id)

        cluster_items = list(clusters.items())
        rng.shuffle(cluster_items)
        ordered: list[int] = []
        for _, members in cluster_items:
            rng.shuffle(members)
            ordered.extend(members)
        return np.asarray(ordered, dtype=np.int64)

    def order_crossover(
        self,
        parent_a: np.ndarray,
        parent_b: np.ndarray,
        rng: random.Random,
    ) -> np.ndarray:
        size = len(parent_a)
        if size < 2:
            return parent_a.copy()

        left, right = sorted(rng.sample(range(size), 2))
        child = np.full(size, -1, dtype=np.int64)
        child[left : right + 1] = parent_a[left : right + 1]
        used = set(int(x) for x in child[left : right + 1].tolist())

        insert_pos = (right + 1) % size
        for gene in parent_b.tolist():
            if int(gene) in used:
                continue
            child[insert_pos] = int(gene)
            insert_pos = (insert_pos + 1) % size
        return child

    def mutate(self, chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        operators = (
            self._relocate_one,
            self._relocate_pair,
            self._relocate_pair_reversed,
            self._swap_one,
            self._swap_pair_single,
            self._swap_pair_pair,
            self._reverse_segment,
            self._exchange_tails,
        )
        return operators[rng.randrange(len(operators))](chromosome, rng)

    @staticmethod
    def _relocate_one(chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        if len(chromosome) < 2:
            return chromosome.copy()
        child = chromosome.copy()
        i, j = rng.sample(range(len(child)), 2)
        gene = int(child[i])
        child = np.delete(child, i)
        child = np.insert(child, j, gene)
        return child.astype(np.int64, copy=False)

    @staticmethod
    def _relocate_pair(chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        if len(chromosome) < 3:
            return HGSSplitPVRPWDPSolver._relocate_one(chromosome, rng)
        i = rng.randrange(len(chromosome) - 1)
        pair = chromosome[i : i + 2].copy()
        remainder = np.delete(chromosome, [i, i + 1])
        j = rng.randrange(len(remainder) + 1)
        return np.insert(remainder, j, pair).astype(np.int64, copy=False)

    @staticmethod
    def _relocate_pair_reversed(chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        if len(chromosome) < 3:
            return HGSSplitPVRPWDPSolver._relocate_one(chromosome, rng)
        i = rng.randrange(len(chromosome) - 1)
        pair = chromosome[i : i + 2][::-1].copy()
        remainder = np.delete(chromosome, [i, i + 1])
        j = rng.randrange(len(remainder) + 1)
        return np.insert(remainder, j, pair).astype(np.int64, copy=False)

    @staticmethod
    def _swap_one(chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        child = chromosome.copy()
        if len(child) < 2:
            return child
        i, j = rng.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
        return child

    @staticmethod
    def _swap_pair_single(chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        if len(chromosome) < 3:
            return HGSSplitPVRPWDPSolver._swap_one(chromosome, rng)
        i = rng.randrange(len(chromosome) - 1)
        candidates = [idx for idx in range(len(chromosome)) if idx not in (i, i + 1)]
        j = rng.choice(candidates)
        values = [int(x) for x in chromosome.tolist()]
        pair = values[i : i + 2]
        single = values[j]
        if j < i:
            child = values[:j] + pair + values[j + 1 : i] + [single] + values[i + 2 :]
        else:
            child = values[:i] + [single] + values[i + 2 : j] + pair + values[j + 1 :]
        return np.asarray(child, dtype=np.int64)

    @staticmethod
    def _swap_pair_pair(chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        if len(chromosome) < 4:
            return HGSSplitPVRPWDPSolver._swap_one(chromosome, rng)
        i, j = sorted(rng.sample(range(len(chromosome) - 1), 2))
        if abs(i - j) <= 1:
            return HGSSplitPVRPWDPSolver._swap_one(chromosome, rng)
        child = chromosome.copy()
        pair_i = child[i : i + 2].copy()
        pair_j = child[j : j + 2].copy()
        child[i : i + 2] = pair_j
        child[j : j + 2] = pair_i
        return child

    @staticmethod
    def _reverse_segment(chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        child = chromosome.copy()
        if len(child) < 2:
            return child
        i, j = sorted(rng.sample(range(len(child)), 2))
        child[i : j + 1] = child[i : j + 1][::-1]
        return child

    @staticmethod
    def _exchange_tails(chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        child = chromosome.copy()
        if len(child) < 4:
            return HGSSplitPVRPWDPSolver._reverse_segment(chromosome, rng)
        i, j = sorted(rng.sample(range(1, len(child) - 1), 2))
        return np.concatenate([child[:i], child[j:], child[i:j]]).astype(np.int64)

    def local_search(
        self,
        chromosome: np.ndarray,
        rng: random.Random,
        max_iters: int = 12,
        neighborhood_trials: int = 16,
    ) -> np.ndarray:
        current = chromosome.copy()
        current_score = self.evaluate(current)

        for _ in range(max_iters):
            best = current
            best_score = current_score
            for _ in range(neighborhood_trials):
                candidate = self.mutate(current, rng)
                candidate_score = self.evaluate(candidate)
                if candidate_score + EPS < best_score:
                    best = candidate
                    best_score = candidate_score
            if best_score + EPS >= current_score:
                break
            current = best
            current_score = best_score
        return current

    def _make_individual(self, chromosome: np.ndarray) -> Individual:
        return Individual(chromosome=chromosome.copy(), solution=self.decode(chromosome))

    def _route_arcs(self, solution: DecodedSolution) -> set[tuple[int, int]]:
        arcs: set[tuple[int, int]] = set()
        for vehicle_id, route in enumerate(solution.routes):
            previous = vehicle_id
            for node in route:
                node = int(node)
                arcs.add((previous, node))
                previous = node
        return arcs

    def _biased_fitness(
        self,
        population: list[Individual],
        elite_fraction: float,
        close_fraction: float,
    ) -> list[float]:
        size = len(population)
        if size <= 1:
            return [0.0] * size

        cost_order = sorted(range(size), key=lambda idx: population[idx].solution.score)
        cost_rank = [0.0] * size
        for rank, idx in enumerate(cost_order, start=1):
            cost_rank[idx] = float(rank)

        arc_sets = [self._route_arcs(ind.solution) for ind in population]
        nclose = max(1, int(round(size * close_fraction)))
        diversity = [0.0] * size
        for i in range(size):
            distances: list[float] = []
            for j in range(size):
                if i == j:
                    continue
                denom = max(len(arc_sets[i]), len(arc_sets[j]), 1)
                common = len(arc_sets[i] & arc_sets[j])
                distances.append(1.0 - (common / denom))
            distances.sort()
            diversity[i] = float(np.mean(distances[:nclose])) if distances else 0.0

        diversity_order = sorted(
            range(size), key=lambda idx: diversity[idx], reverse=True
        )
        diversity_rank = [0.0] * size
        for rank, idx in enumerate(diversity_order, start=1):
            diversity_rank[idx] = float(rank)

        elite_count = max(1, int(round(size * elite_fraction)))
        diversity_weight = 1.0 - (elite_count / max(size, 1))
        return [
            cost_rank[idx] + diversity_weight * diversity_rank[idx] for idx in range(size)
        ]

    def _tournament_select(
        self,
        population: list[Individual],
        biased: list[float],
        rng: random.Random,
    ) -> Individual:
        if len(population) == 1:
            return population[0]
        idx_a, idx_b = rng.sample(range(len(population)), 2)
        return population[idx_a] if biased[idx_a] < biased[idx_b] else population[idx_b]

    def _select_survivors(
        self,
        population: list[Individual],
        mu: int,
        elite_fraction: float,
        close_fraction: float,
    ) -> list[Individual]:
        biased = self._biased_fitness(population, elite_fraction, close_fraction)
        order = sorted(range(len(population)), key=lambda idx: biased[idx])
        return [population[idx] for idx in order[:mu]]

    def _initial_population(
        self,
        mu: int,
        rng: random.Random,
        np_rng: np.random.Generator,
    ) -> list[Individual]:
        base_perm = np.arange(self.num_customers, dtype=np.int64)
        population: list[Individual] = []
        seen: set[tuple[int, ...]] = set()

        for seed_perm in self.chromosome_seeds(rng):
            key = tuple(int(x) for x in seed_perm.tolist())
            if key not in seen:
                seen.add(key)
                population.append(self._make_individual(seed_perm))

        while len(population) < mu:
            chromosome = np_rng.permutation(base_perm)
            key = tuple(int(x) for x in chromosome.tolist())
            if key in seen and self.num_customers > 1:
                continue
            seen.add(key)
            population.append(self._make_individual(chromosome))

        return population[:mu]

    def _diversify(
        self,
        population: list[Individual],
        mu: int,
        rng: random.Random,
        np_rng: np.random.Generator,
    ) -> list[Individual]:
        keep_count = max(1, mu // 2)
        population = sorted(population, key=lambda ind: ind.solution.score)[:keep_count]
        seen = {tuple(int(x) for x in ind.chromosome.tolist()) for ind in population}
        base_perm = np.arange(self.num_customers, dtype=np.int64)

        while len(population) < mu:
            if rng.random() < 0.35:
                seed_pool = self.chromosome_seeds(rng)
                chromosome = seed_pool[rng.randrange(len(seed_pool))].copy()
                chromosome = self.mutate(chromosome, rng)
            else:
                chromosome = np_rng.permutation(base_perm)
            key = tuple(int(x) for x in chromosome.tolist())
            if key in seen and self.num_customers > 1:
                continue
            seen.add(key)
            population.append(self._make_individual(chromosome))
        return population

    def run(
        self,
        mu: int = 14,
        lambda_size: int = 16,
        max_iters: int | None = None,
        mutation_rate: float = 0.2,
        local_search_rate: float = 0.35,
        local_search_iters: int = 12,
        elite_fraction: float = 0.6,
        close_fraction: float = 0.6,
        diversify_after: int = 500,
        seed: int = 42,
        verbose: bool = True,
        progress_label: str | None = None,
        progress_every: int = 100,
    ) -> DecodedSolution:
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        run_start_time = time.time()

        if max_iters is None:
            max_iters = max(50 * self.num_customers, 2000)

        population = self._initial_population(mu, rng, np_rng)
        best_solution = min(population, key=lambda ind: ind.solution.score).solution
        no_improve = 0

        for iteration in range(max_iters):
            biased = self._biased_fitness(population, elite_fraction, close_fraction)
            parent_a = self._tournament_select(population, biased, rng)
            parent_b = self._tournament_select(population, biased, rng)

            child_a = self.order_crossover(parent_a.chromosome, parent_b.chromosome, rng)
            child_b = self.order_crossover(parent_b.chromosome, parent_a.chromosome, rng)
            if rng.random() < mutation_rate:
                child_a = self.mutate(child_a, rng)
            if rng.random() < mutation_rate:
                child_b = self.mutate(child_b, rng)

            child = (
                child_a if self.evaluate(child_a) <= self.evaluate(child_b) else child_b
            )
            if rng.random() < local_search_rate:
                child = self.local_search(
                    child,
                    rng,
                    max_iters=local_search_iters,
                    neighborhood_trials=16,
                )

            population.append(self._make_individual(child))
            if len(population) >= mu + lambda_size:
                population = self._select_survivors(
                    population,
                    mu,
                    elite_fraction,
                    close_fraction,
                )

            current_best = min(population, key=lambda ind: ind.solution.score).solution
            if current_best.score + EPS < best_solution.score:
                best_solution = current_best
                no_improve = 0
            else:
                no_improve += 1

            if diversify_after > 0 and no_improve >= diversify_after:
                population = self._diversify(population, mu, rng, np_rng)
                no_improve = 0

            if verbose and (
                iteration == 0
                or (iteration + 1) % max(1, progress_every) == 0
                or iteration == max_iters - 1
            ):
                prefix = f"[{progress_label}] " if progress_label else "[HGS-Split] "
                print(
                    f"{prefix}{format_progress(iteration + 1, max_iters, run_start_time, width=20)} "
                    f"iter={iteration + 1:04d} "
                    f"best_score={best_solution.score:.4f} "
                    f"unserved={len(best_solution.unserved_customers)} "
                    f"makespan={best_solution.makespan:.4f} "
                    f"distance={best_solution.total_distance:.4f}"
                )

        return best_solution

    def solution_to_actions(self, solution: DecodedSolution) -> torch.Tensor:
        route_lists: list[list[int]] = []
        for vehicle_id, route in enumerate(solution.routes):
            route_lists.append(list(route) if route else [vehicle_id])

        horizon = max(1, max(len(route) for route in route_lists))
        action_rows = []
        for route in route_lists:
            last = route[-1]
            padded = route + [last] * (horizon - len(route))
            action_rows.append(padded)

        return torch.tensor(action_rows, dtype=torch.int64).unsqueeze(0)

    def evaluate_with_env(self, solution: DecodedSolution) -> dict[str, float]:
        actions = self.solution_to_actions(solution)
        td = self.td_reset.clone()
        for step in range(actions.shape[-1]):
            td.set("action", actions[:, :, step])
            td = self.env.step(td)["next"]

        raw_env_reward = float(self.env.get_reward(td.clone(), actions).item())
        return {
            "objective_reward": -solution.score,
            "objective_cost": solution.score,
            "raw_env_reward": raw_env_reward,
            "done": float(td["done"].float().item()),
            "visited_customers": float(td["visited"][0, self.num_agents :].sum().item()),
        }

    def print_solution(self, solution: DecodedSolution) -> None:
        print("\nBest solution summary (HGS + split)")
        print(f"  score          : {solution.score:.4f}")
        print(f"  unserved       : {len(solution.unserved_customers)}")
        print(f"  makespan       : {solution.makespan:.4f}")
        print(f"  total_distance : {solution.total_distance:.4f}")
        print(f"  total_cost     : {solution.total_cost:.4f}")
        if solution.unserved_customers:
            print(f"  unserved_ids   : {solution.unserved_customers}")

        for vehicle_id, route in enumerate(solution.routes):
            display_route = []
            for node in route:
                if node < self.num_agents:
                    display_route.append(f"D{node}")
                else:
                    display_route.append(f"C{node - self.num_agents}")
            print(
                f"  vehicle {vehicle_id}: "
                f"{' -> '.join(display_route) if display_route else f'D{vehicle_id}'}"
            )


def run_single_instance(
    td_loaded: torch.Tensor,
    batch_idx: int,
    target: str,
    mu: int,
    lambda_size: int,
    max_iters: int | None,
    mutation_rate: float,
    local_search_rate: float,
    local_search_iters: int,
    elite_fraction: float,
    close_fraction: float,
    diversify_after: int,
    max_labels: int,
    seed: int,
    show_routes: bool,
    show_generation_progress: bool,
    generation_progress_every: int,
) -> InstanceResult:
    env, td_raw, td_reset = load_instance_from_td(td_loaded, batch_idx, target)
    reference_cost = get_optional_batch_float(td_raw, 0, ("costs", "cost"))
    offset = get_optional_batch_int(td_raw, 0, ("offsets", "offset"))
    
    solver = HGSSplitPVRPWDPSolver(
        env=env,
        td_raw=td_raw,
        td_reset=td_reset,
        max_labels=max_labels,
    )
    best = solver.run(
        mu=mu,
        lambda_size=lambda_size,
        max_iters=max_iters,
        mutation_rate=mutation_rate,
        local_search_rate=local_search_rate,
        local_search_iters=local_search_iters,
        elite_fraction=elite_fraction,
        close_fraction=close_fraction,
        diversify_after=diversify_after,
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


def worker_run_single_instance(
    npz_path: str,
    batch_idx: int,
    target: str,
    mu: int,
    lambda_size: int,
    max_iters: int | None,
    mutation_rate: float,
    local_search_rate: float,
    local_search_iters: int,
    elite_fraction: float,
    close_fraction: float,
    diversify_after: int,
    max_labels: int,
    seed: int,
    show_routes: bool,
    show_generation_progress: bool,
    generation_progress_every: int,
) -> InstanceResult:
    env, td_raw, td_reset = load_instance(npz_path, batch_idx, target)
    reference_cost = get_optional_batch_float(td_raw, 0, ("costs", "cost"))
    offset = get_optional_batch_int(td_raw, 0, ("offsets", "offset"))
    
    solver = HGSSplitPVRPWDPSolver(
        env=env,
        td_raw=td_raw,
        td_reset=td_reset,
        max_labels=max_labels,
    )
    best = solver.run(
        mu=mu,
        lambda_size=lambda_size,
        max_iters=max_iters,
        mutation_rate=mutation_rate,
        local_search_rate=local_search_rate,
        local_search_iters=local_search_iters,
        elite_fraction=elite_fraction,
        close_fraction=close_fraction,
        diversify_after=diversify_after,
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
    mu: int,
    lambda_size: int,
    max_iters: int | None,
    mutation_rate: float,
    local_search_rate: float,
    local_search_iters: int,
    elite_fraction: float,
    close_fraction: float,
    diversify_after: int,
    max_labels: int,
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
        f"\n=== [HGS-Split] Running target={target} on "
        f"{len(batch_indices)}/{total_batches} instances{offset_note} (Workers: {max_workers}) ==="
    )

    if max_workers <= 1 or len(batch_indices) == 1:
        for order_idx, idx in enumerate(batch_indices, start=1):
            instance_seed = seed + idx + (0 if target == "mincost" else 10_000)
            result = run_single_instance(
                td_loaded=td_loaded,
                batch_idx=idx,
                target=target,
                mu=mu,
                lambda_size=lambda_size,
                max_iters=max_iters,
                mutation_rate=mutation_rate,
                local_search_rate=local_search_rate,
                local_search_iters=local_search_iters,
                elite_fraction=elite_fraction,
                close_fraction=close_fraction,
                diversify_after=diversify_after,
                max_labels=max_labels,
                seed=instance_seed,
                show_routes=show_routes and len(batch_indices) == 1,
                show_generation_progress=show_generation_progress,
                generation_progress_every=generation_progress_every,
            )
            results.append(result)
            cost_text = f"{result.cost:.4f}" if math.isfinite(result.cost) else "nan"
            print(
                f"[{target}] {format_progress(order_idx, len(batch_indices), start_time)} "
                f"batch={idx:03d} offset={result.offset} cost={cost_text} "
                f"objective_cost={result.objective_cost:.4f} "
                f"score={result.score:.4f} "
                f"unserved={result.unserved} "
                f"makespan={result.makespan:.4f} "
                f"done={result.done}"
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx in batch_indices:
                instance_seed = seed + idx + (0 if target == "mincost" else 10_000)
                future = executor.submit(
                    worker_run_single_instance,
                    npz_path,
                    idx,
                    target,
                    mu,
                    lambda_size,
                    max_iters,
                    mutation_rate,
                    local_search_rate,
                    local_search_iters,
                    elite_fraction,
                    close_fraction,
                    diversify_after,
                    max_labels,
                    instance_seed,
                    show_routes and len(batch_indices) == 1,
                    show_generation_progress,
                    generation_progress_every,
                )
                futures[future] = idx

            for order_idx, future in enumerate(
                concurrent.futures.as_completed(futures),
                start=1,
            ):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    cost_text = f"{result.cost:.4f}" if math.isfinite(result.cost) else "nan"
                    print(
                        f"[{target}] {format_progress(order_idx, len(batch_indices), start_time)} "
                        f"batch={idx:03d} offset={result.offset} cost={cost_text} "
                        f"objective_cost={result.objective_cost:.4f} "
                        f"score={result.score:.4f} "
                        f"unserved={result.unserved} "
                        f"makespan={result.makespan:.4f} "
                        f"done={result.done}"
                    )
                except Exception as exc:
                    print(f"Batch {idx} generated an exception: {exc}")

    summary = summarize_results(results, target)
    print(f"\nSummary for target={target} (HGS-Split)")
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
        output_root = Path(npz_path).resolve().parent / "hgs_split_results"
    
    output_root.mkdir(parents=True, exist_ok=True)
    offset_suffix = f"_offset_{offset}" if offset is not None else ""
    output_file = output_root / f"{Path(npz_path).stem}_{target}{offset_suffix}_hgs_split.csv"
    save_results_csv(output_file, results)
    print(f"  saved_csv    : {output_file}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HGS + DP split baseline for PVRPWDP.",
    )
    parser.add_argument("--npz", required=True, help="Path to NPZ file.")
    parser.add_argument("--batch-idx", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Only run batches whose NPZ offsets/offset value matches this number.",
    )
    parser.add_argument(
        "--target", choices=["makespan", "mincost", "both"], default="both"
    )
    parser.add_argument("--mu", type=int, default=14, help="Base population size.")
    parser.add_argument(
        "--lambda-size", type=int, default=16, help="Survivor trigger size."
    )
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--local-search-rate", type=float, default=0.35)
    parser.add_argument("--local-search-iters", type=int, default=12)
    parser.add_argument("--elite-fraction", type=float, default=0.6)
    parser.add_argument("--close-fraction", type=float, default=0.6)
    parser.add_argument("--diversify-after", type=int, default=500)
    parser.add_argument("--max-labels", type=int, default=24)
    parser.add_argument("--show-generation-progress", action="store_true")
    parser.add_argument("--generation-progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--show-routes", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = ["mincost", "makespan"] if args.target == "both" else [args.target]
    all_results: dict[str, list[InstanceResult]] = {}

    for target in targets:
        all_results[target] = run_target_over_dataset(
            npz_path=args.npz,
            target=target,
            mu=args.mu,
            lambda_size=args.lambda_size,
            max_iters=args.max_iters,
            mutation_rate=args.mutation_rate,
            local_search_rate=args.local_search_rate,
            local_search_iters=args.local_search_iters,
            elite_fraction=args.elite_fraction,
            close_fraction=args.close_fraction,
            diversify_after=args.diversify_after,
            max_labels=args.max_labels,
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
        print("\n=== Combined overview (HGS-Split) ===")
        print(f"  mincost_avg_obj_cost  : {mincost_summary['avg_objective_cost']:.4f}")
        print(f"  mincost_feasible_rate : {mincost_summary['feasible_rate']:.4f}")
        print(f"  makespan_avg_obj_cost : {makespan_summary['avg_objective_cost']:.4f}")
        print(f"  makespan_avg_makespan : {makespan_summary['avg_makespan']:.4f}")
        print(f"  makespan_feasible_rate: {makespan_summary['feasible_rate']:.4f}")


if __name__ == "__main__":
    main()
    
# uv run python ga_pvrpwdp_hgs_split.py --npz data/test_data/test.npz  --target both --mu 14 --lambda-size 16 --max-iters 500 --show-routes --show-generation-progress