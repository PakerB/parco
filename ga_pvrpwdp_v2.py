"""GA v2 for PVRPWDP: Separator + Round-Robin Vehicle Assignment encoding.

Chromosome = list of ints where:
  - customer ids are 0..N-1
  - SEP = -1 means "close current trip, switch to next vehicle (round-robin)"

Example with 3 vehicles:
  [5, 2, -1, 7, 1, -1, 8, -1, 4, 6, -1, 3]
  => vehicle 0 trip 1: [5, 2]
     vehicle 1 trip 1: [7, 1]
     vehicle 2 trip 1: [8]
     vehicle 0 trip 2: [4, 6]
     vehicle 1 trip 2: [3]
"""
from __future__ import annotations

import argparse
import concurrent.futures
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
# NOTE: v1's DecodedSolution uses `routes`, v2 uses `states` + `VehicleState`.
# v1 also does NOT define VehicleState. Therefore v2 defines its own dataclasses
# below and only imports the runner/utility helpers that are truly shared.
from ga_pvrpwdp import (
    InstanceResult,
    format_progress,
    load_instance,
    load_instance_from_td,
    save_results_csv,
    summarize_results,
)

SEP = -1


@dataclass
class VehicleState:
    """Mutable per-vehicle state used by the separator decoder."""
    vehicle_id: int
    depot_idx: int
    speed: float
    capacity: float
    endurance: float
    max_time: float
    current_node: int
    trip_deadline: float
    current_length: float = 0.0
    current_time: float = 0.0
    used_capacity: float = 0.0
    used_endurance: float = 0.0
    route_actions: list[int] = field(default_factory=list)

    def clone(self) -> "VehicleState":
        return VehicleState(
            vehicle_id=self.vehicle_id,
            depot_idx=self.depot_idx,
            speed=self.speed,
            capacity=self.capacity,
            endurance=self.endurance,
            max_time=self.max_time,
            current_node=self.current_node,
            trip_deadline=self.trip_deadline,
            current_length=self.current_length,
            current_time=self.current_time,
            used_capacity=self.used_capacity,
            used_endurance=self.used_endurance,
            route_actions=list(self.route_actions),
        )


@dataclass
class DecodedSolution:
    """v2 decoded solution: keeps full per-vehicle state instead of flat routes."""
    states: list[VehicleState]
    unserved_customers: list[int]
    score: float
    makespan: float
    total_distance: float
    total_cost: float


class SeparatorPVRPWDPSolver:
    """GA v2: separator + round-robin chromosome encoding."""

    def __init__(self, env: PVRPWDPVEnv, td_raw: torch.Tensor, td_reset: torch.Tensor):
        self.env = env
        self.td_raw = td_raw
        self.td_reset = td_reset

        self.num_agents = int(td_reset["current_node"].shape[-1])
        self.num_customers = int(td_reset["locs"].shape[-2] - self.num_agents)
        self.max_time = float(td_reset["max_time"][0].item())

        self.locs = td_reset["locs"][0].cpu().numpy()
        # demand is [B, m+N] (2D), not [B, m, m+N]. Index with [0, num_agents:].
        self.customer_demands = td_reset["demand"][0, self.num_agents:].cpu().numpy()
        self.time_windows = td_reset["time_window"][0, self.num_agents:, :].cpu().numpy()
        self.waiting_time = td_reset["waiting_time"][0, self.num_agents:].cpu().numpy()
        # Guard against any residual zero-speed agents to avoid divide-by-zero / NaN.
        self.agents_speed = np.maximum(
            td_reset["agents_speed"][0].cpu().numpy(), 1e-9
        )
        self.agents_capacity = td_reset["agents_capacity"][0].cpu().numpy()
        self.agents_endurance = td_reset["agents_endurance"][0].cpu().numpy()
        self.customer_env_indices = np.arange(self.num_agents, self.num_agents + self.num_customers)

        is_truck = self._truck_mask_from_capacity(self.agents_capacity)
        self.is_truck = is_truck
        # Use getattr with sensible defaults in case the env doesn't expose these
        # attributes (matches v1's defensive style).
        self.travel_price = np.where(
            is_truck,
            float(getattr(env, "travel_price_truck", 0.35 / 1000)),
            float(getattr(env, "travel_price_drone", 1.5 / 3600)),
        )
        self.rent_price = np.where(
            is_truck,
            float(getattr(env, "rent_price_truck", 40.0)),
            float(getattr(env, "rent_price_drone", 10.0)),
        )

        bbox_diag = float(np.linalg.norm(self.locs.max(axis=0) - self.locs.min(axis=0)))
        travel_max = 2.0 * bbox_diag * self.num_customers * float(self.travel_price.max())
        self.lambda_unserved = travel_max + float(self.rent_price.sum()) + 1.0
        latest_max = float(self.time_windows[:, 1].max())
        min_speed = float(np.min(self.agents_speed))
        self.lambda_makespan_unserved = (
            2.0 * bbox_diag * self.num_customers / max(min_speed, 1e-9)
        ) + latest_max + 1.0

        self._score_cache: dict[tuple[int, ...], tuple[float, DecodedSolution]] = {}

    @staticmethod
    def _truck_mask_from_capacity(capacity: np.ndarray) -> np.ndarray:
        thresh = 0.5 * (capacity.max() + capacity.min())
        return capacity > (thresh + 1e-6)

    def _distance(self, node_a: int, node_b: int) -> float:
        return float(np.linalg.norm(self.locs[node_a] - self.locs[node_b]))

    def _initial_states(self) -> list[VehicleState]:
        states: list[VehicleState] = []
        for agent_id in range(self.num_agents):
            states.append(VehicleState(
                vehicle_id=agent_id,
                depot_idx=agent_id,
                speed=float(self.agents_speed[agent_id]),
                capacity=float(self.agents_capacity[agent_id]),
                endurance=float(self.agents_endurance[agent_id]),
                max_time=self.max_time,
                current_node=agent_id,
                trip_deadline=self.max_time,
            ))
        return states

    def _close_trip(self, state: VehicleState) -> VehicleState:
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

    def _try_append_customer(
        self, state: VehicleState, customer_id: int
    ) -> VehicleState | None:
        customer_idx = int(self.customer_env_indices[customer_id])

        demand = float(self.customer_demands[customer_id])
        if state.used_capacity + demand > state.capacity + 1e-9:
            return None

        travel_dist = self._distance(state.current_node, customer_idx)
        travel_time = travel_dist / state.speed
        arrival = state.current_time + travel_time
        earliest, latest = self.time_windows[customer_id]
        service_time = max(arrival, float(earliest))
        if service_time > float(latest) + 1e-9:
            return None

        waiting_mid_trip = 0.0
        if state.current_node != state.depot_idx:
            waiting_mid_trip = max(float(earliest) - arrival, 0.0)

        used_endurance = state.used_endurance + travel_time + waiting_mid_trip
        back_time = self._distance(customer_idx, state.depot_idx) / state.speed
        return_time = service_time + back_time
        new_deadline = service_time + float(self.waiting_time[customer_id])
        effective_deadline = min(state.trip_deadline, new_deadline)

        if return_time > effective_deadline + 1e-9:
            return None
        if used_endurance + back_time > state.endurance + 1e-9:
            return None

        new_state = state.clone()
        new_state.current_length += travel_dist
        new_state.current_time = service_time
        new_state.current_node = customer_idx
        new_state.used_capacity += demand
        new_state.used_endurance = used_endurance
        new_state.trip_deadline = effective_deadline
        new_state.route_actions.append(customer_idx)
        return new_state

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def decode(self, chromosome: np.ndarray) -> DecodedSolution:
        key = tuple(int(x) for x in chromosome.tolist())
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached[1]

        states = self._initial_states()
        current_vehicle = 0
        unserved: list[int] = []

        for gene in chromosome.tolist():
            gene = int(gene)
            if gene == SEP:
                states[current_vehicle] = self._close_trip(states[current_vehicle])
                current_vehicle = (current_vehicle + 1) % self.num_agents
                continue

            customer_id = gene
            new_state = self._try_append_customer(states[current_vehicle], customer_id)
            if new_state is not None:
                states[current_vehicle] = new_state
            else:
                fallback_placed = False
                for offset in range(1, self.num_agents):
                    alt_vehicle = (current_vehicle + offset) % self.num_agents
                    alt_state = self._try_append_customer(states[alt_vehicle], customer_id)
                    if alt_state is not None:
                        states[alt_vehicle] = alt_state
                        fallback_placed = True
                        break
                if not fallback_placed:
                    unserved.append(customer_id)

        closed_states = [self._close_trip(s) for s in states]
        total_distance = float(sum(s.current_length for s in closed_states))
        makespan = float(max(s.current_time for s in closed_states))

        travel_part = float(sum(
            s.current_length * self.travel_price[i] for i, s in enumerate(closed_states)
        ))
        used_vehicle = np.array(
            [s.current_length > 1e-8 for s in closed_states], dtype=np.float32
        )
        rent_part = float(np.sum(used_vehicle * self.rent_price))
        total_cost = travel_part + rent_part

        score = self._objective_cost(len(unserved), total_cost, makespan)
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
        return self.decode(chromosome).score

    def _objective_cost(
        self, unserved_count: int, total_cost: float, makespan: float
    ) -> float:
        if self.env.target == "mincost":
            return self.lambda_unserved * unserved_count + total_cost
        if self.env.target == "makespan":
            return self.lambda_makespan_unserved * unserved_count + makespan
        raise NotImplementedError(f"Unsupported target: {self.env.target}")

    def objective_cost(self, solution: DecodedSolution) -> float:
        return self._objective_cost(
            len(solution.unserved_customers),
            solution.total_cost,
            solution.makespan,
        )

    # ------------------------------------------------------------------
    # Chromosome generation
    # ------------------------------------------------------------------
    def _insert_separators(
        self, customer_perm: np.ndarray, rng: random.Random
    ) -> np.ndarray:
        perm_list = customer_perm.tolist()
        n = len(perm_list)
        num_seps = max(1, self.num_agents - 1)
        num_seps = min(num_seps, max(1, n // 2))
        positions = sorted(rng.sample(range(1, n + 1), num_seps))
        result: list[int] = []
        prev = 0
        for pos in positions:
            result.extend(perm_list[prev:pos])
            result.append(SEP)
            prev = pos
        result.extend(perm_list[prev:])
        return np.array(result, dtype=np.int64)

    def chromosome_seeds(self, rng: random.Random) -> list[np.ndarray]:
        customer_ids = np.arange(self.num_customers, dtype=np.int64)
        perms = [
            customer_ids.copy(),
            np.argsort(self.time_windows[:, 0]).astype(np.int64),
            np.argsort(self.time_windows[:, 1]).astype(np.int64),
            np.argsort(self.waiting_time).astype(np.int64),
            np.argsort(-self.customer_demands).astype(np.int64),
        ]
        seeds: list[np.ndarray] = []
        for perm in perms:
            seeds.append(self._insert_separators(perm, rng))
        return seeds

    def random_chromosome(self, np_rng: np.random.Generator, rng: random.Random) -> np.ndarray:
        perm = np_rng.permutation(self.num_customers).astype(np.int64)
        return self._insert_separators(perm, rng)

    # ------------------------------------------------------------------
    # Crossover: preserve customer order from parents, re-insert SEPs
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_customers(chromosome: np.ndarray) -> np.ndarray:
        return chromosome[chromosome != SEP]

    def crossover(
        self, parent_a: np.ndarray, parent_b: np.ndarray, rng: random.Random
    ) -> np.ndarray:
        cust_a = self._extract_customers(parent_a)
        cust_b = self._extract_customers(parent_b)
        size = len(cust_a)
        left, right = sorted(rng.sample(range(size), 2))
        child_custs = np.full(size, -999, dtype=np.int64)
        child_custs[left: right + 1] = cust_a[left: right + 1]
        used = set(int(x) for x in child_custs[left: right + 1].tolist())
        insert_pos = (right + 1) % size
        for gene in cust_b.tolist():
            if gene in used:
                continue
            child_custs[insert_pos] = gene
            insert_pos = (insert_pos + 1) % size

        sep_positions_a = [i for i, g in enumerate(parent_a.tolist()) if g == SEP]
        sep_positions_b = [i for i, g in enumerate(parent_b.tolist()) if g == SEP]
        sep_positions = sep_positions_a if rng.random() < 0.5 else sep_positions_b
        child_list = child_custs.tolist()
        offset = 0
        for pos in sep_positions:
            insert_at = min(pos + offset, len(child_list))
            child_list.insert(insert_at, SEP)
            offset += 1
        return np.array(child_list, dtype=np.int64)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def mutate_swap_customers(self, chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        cust_indices = [i for i, g in enumerate(chromosome) if g != SEP]
        if len(cust_indices) < 2:
            return chromosome.copy()
        child = chromosome.copy()
        i, j = rng.sample(cust_indices, 2)
        child[i], child[j] = child[j], child[i]
        return child

    def mutate_move_separator(self, chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        child = chromosome.copy()
        sep_indices = [i for i, g in enumerate(child) if g == SEP]
        if not sep_indices:
            return child
        idx = rng.choice(sep_indices)
        child_list = child.tolist()
        child_list.pop(idx)
        if len(child_list) < 2:
            return chromosome.copy()
        new_pos = rng.randint(1, len(child_list) - 1)
        child_list.insert(new_pos, SEP)
        return np.array(child_list, dtype=np.int64)

    def mutate_add_remove_separator(self, chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        child_list = list(chromosome.tolist())
        sep_count = child_list.count(SEP)
        cust_count = len(child_list) - sep_count

        if sep_count > 1 and rng.random() < 0.5:
            sep_indices = [i for i, g in enumerate(child_list) if g == SEP]
            child_list.pop(rng.choice(sep_indices))
        elif cust_count > 1:
            cust_indices = [i for i, g in enumerate(child_list) if g != SEP]
            pos = rng.choice(cust_indices[1:])
            child_list.insert(pos, SEP)

        return np.array(child_list, dtype=np.int64)

    def mutate_reverse_segment(self, chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        cust_indices = [i for i, g in enumerate(chromosome) if g != SEP]
        if len(cust_indices) < 2:
            return chromosome.copy()
        child = chromosome.copy()
        ci, cj = sorted(rng.sample(range(len(cust_indices)), 2))
        i, j = cust_indices[ci], cust_indices[cj]
        segment = list(child[i: j + 1])
        custs_in_seg = [x for x in segment if x != SEP]
        custs_in_seg.reverse()
        c_iter = iter(custs_in_seg)
        for pos in range(i, j + 1):
            if child[pos] != SEP:
                child[pos] = next(c_iter)
        return child

    def mutate_relocate_customer(self, chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        cust_indices = [i for i, g in enumerate(chromosome) if g != SEP]
        if len(cust_indices) < 2:
            return chromosome.copy()
        child_list = list(chromosome.tolist())
        src = rng.choice(cust_indices)
        gene = child_list.pop(src)
        new_cust_indices = [i for i, g in enumerate(child_list) if g != SEP]
        if not new_cust_indices:
            child_list.append(gene)
        else:
            dst = rng.choice(new_cust_indices)
            child_list.insert(dst, gene)
        return np.array(child_list, dtype=np.int64)

    def mutate(self, chromosome: np.ndarray, rng: random.Random) -> np.ndarray:
        op = rng.random()
        if op < 0.25:
            return self.mutate_swap_customers(chromosome, rng)
        elif op < 0.45:
            return self.mutate_move_separator(chromosome, rng)
        elif op < 0.60:
            return self.mutate_add_remove_separator(chromosome, rng)
        elif op < 0.80:
            return self.mutate_reverse_segment(chromosome, rng)
        else:
            return self.mutate_relocate_customer(chromosome, rng)

    # ------------------------------------------------------------------
    # Local search
    # ------------------------------------------------------------------
    def local_search(
        self,
        chromosome: np.ndarray,
        rng: random.Random,
        max_iters: int = 20,
        neighborhood_trials: int = 12,
    ) -> np.ndarray:
        current = chromosome.copy()
        current_score = self.evaluate(current)

        for _ in range(max_iters):
            improved = False
            for _ in range(neighborhood_trials):
                candidate = self.mutate(current, rng)
                candidate_score = self.evaluate(candidate)
                if candidate_score + 1e-9 < current_score:
                    current = candidate
                    current_score = candidate_score
                    improved = True
                    break
            if not improved:
                break
        return current

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    @staticmethod
    def tournament_select(
        population: list[np.ndarray],
        fitness: list[float],
        tournament_size: int,
        rng: random.Random,
    ) -> np.ndarray:
        tournament_size = min(tournament_size, len(population))
        idxs = rng.sample(range(len(population)), tournament_size)
        best_idx = min(idxs, key=lambda idx: fitness[idx])
        return population[best_idx]

    # ------------------------------------------------------------------
    # Main GA loop
    # ------------------------------------------------------------------
    def run(
        self,
        population_size: int = 80,
        generations: int = 400,
        mutation_rate: float = 0.3,
        elite_size: int = 4,
        tournament_size: int = 5,
        cull_ratio: float = 0.5,
        immigrant_ratio: float = 0.05,
        local_search_rate: float = 0.15,
        local_search_elites: int = 2,
        local_search_iters: int = 15,
        seed: int = 42,
        verbose: bool = True,
        progress_label: str | None = None,
        progress_every: int = 10,
    ) -> DecodedSolution:
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        run_start_time = time.time()

        seeds = self.chromosome_seeds(rng)
        population: list[np.ndarray] = [s.copy() for s in seeds]
        while len(population) < population_size:
            population.append(self.random_chromosome(np_rng, rng))

        best_solution: DecodedSolution | None = None

        for generation in range(generations):
            decoded = [self.decode(c) for c in population]
            fitness = [d.score for d in decoded]
            order = sorted(range(len(population)), key=lambda idx: fitness[idx])
            population = [population[idx] for idx in order]
            decoded = [decoded[idx] for idx in order]
            fitness = [fitness[idx] for idx in order]

            if local_search_elites > 0:
                elite_ls_count = min(local_search_elites, len(population))
                for idx in range(elite_ls_count):
                    improved = self.local_search(population[idx], rng, max_iters=local_search_iters)
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
                prefix = f"[{progress_label}] " if progress_label else "[GAv2] "
                print(
                    f"{prefix}{format_progress(generation + 1, generations, run_start_time, width=20)} "
                    f"gen={generation + 1:03d} "
                    f"best_score={decoded[0].score:.4f} "
                    f"unserved={len(decoded[0].unserved_customers)} "
                    f"makespan={decoded[0].makespan:.4f} "
                    f"distance={decoded[0].total_distance:.4f}"
                )

            next_population: list[np.ndarray] = [
                population[idx].copy() for idx in range(min(elite_size, len(population)))
            ]
            survivor_count = max(elite_size + 2, int(round(population_size * cull_ratio)))
            survivor_count = min(survivor_count, len(population))
            breeding_pool = population[:survivor_count]
            breeding_fitness = fitness[:survivor_count]

            immigrant_count = int(round(population_size * immigrant_ratio))
            immigrant_count = min(max(0, immigrant_count), max(0, population_size - len(next_population)))

            while len(next_population) < population_size:
                if immigrant_count > 0 and len(next_population) >= population_size - immigrant_count:
                    next_population.append(self.random_chromosome(np_rng, rng))
                    continue

                parent_a = self.tournament_select(breeding_pool, breeding_fitness, tournament_size, rng)
                parent_b = self.tournament_select(breeding_pool, breeding_fitness, tournament_size, rng)
                child = self.crossover(parent_a, parent_b, rng)
                if rng.random() < mutation_rate:
                    child = self.mutate(child, rng)
                if rng.random() < local_search_rate:
                    child = self.local_search(child, rng, max_iters=max(1, local_search_iters // 2), neighborhood_trials=8)
                next_population.append(child)

            population = next_population

        assert best_solution is not None
        return best_solution

    # ------------------------------------------------------------------
    # Env evaluation & display (reuse v1 logic)
    # ------------------------------------------------------------------
    def solution_to_actions(self, solution: DecodedSolution) -> torch.Tensor:
        route_lists: list[list[int]] = []
        for state in solution.states:
            if state.route_actions:
                route_lists.append(list(state.route_actions))
            else:
                route_lists.append([state.depot_idx])
        horizon = max(len(r) for r in route_lists)
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
        objective_cost = self.objective_cost(solution)
        raw_env_reward = float(self.env.get_reward(td.clone(), actions).item())
        return {
            "objective_reward": -objective_cost,
            "objective_cost": objective_cost,
            "raw_env_reward": raw_env_reward,
            "done": float(td["done"].float().item()),
            "visited_customers": float(td["visited"][0, self.num_agents:].sum().item()),
        }

    def print_solution(self, solution: DecodedSolution) -> None:
        print("\nBest solution summary (v2 separator)")
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


# ==================================================================
# Runner functions (same structure as v1)
# ==================================================================

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
    env, td_raw, td_reset = load_instance_from_td(td_loaded, batch_idx, target)
    solver = SeparatorPVRPWDPSolver(env=env, td_raw=td_raw, td_reset=td_reset)
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
    """Worker cho ProcessPoolExecutor: mỗi tiến trình tự load data riêng."""
    env, td_raw, td_reset = load_instance(npz_path, batch_idx, target)
    solver = SeparatorPVRPWDPSolver(env=env, td_raw=td_raw, td_reset=td_reset)
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
    max_workers: int = 1,
) -> list[InstanceResult]:
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

    print(
        f"\n=== [v2-separator] Running target={target} on "
        f"{len(batch_indices)}/{total_batches} instances (Workers: {max_workers}) ==="
    )

    if max_workers <= 1 or len(batch_indices) == 1:
        # Chạy tuần tự trong cùng tiến trình: tái sử dụng td_loaded đã nạp sẵn.
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
    else:
        # Chạy song song: mỗi worker tự load npz của nó, tránh pickle tensordict.
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
                    print(
                        f"[{target}] {format_progress(order_idx, len(batch_indices), start_time)} "
                        f"batch={idx:03d} "
                        f"objective_cost={result.objective_cost:.4f} "
                        f"score={result.score:.4f} "
                        f"unserved={result.unserved} "
                        f"makespan={result.makespan:.4f} "
                        f"done={result.done}"
                    )
                except Exception as exc:
                    print(f"Batch {idx} generated an exception: {exc}")

    summary = summarize_results(results, target)
    print(f"\nSummary for target={target} (v2-separator)")
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
        output_root = Path(npz_path).resolve().parent / "ga_v2_results"
    output_root.mkdir(parents=True, exist_ok=True)
    output_file = output_root / f"{Path(npz_path).stem}_{target}_v2_results.csv"
    save_results_csv(output_file, results)
    print(f"  saved_csv    : {output_file}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GA v2 (separator encoding) for PVRPWDP.")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--batch-idx", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--target", choices=["makespan", "mincost", "both"], default="both")
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=400)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--elite-size", type=int, default=4)
    parser.add_argument("--tournament-size", type=int, default=5)
    parser.add_argument("--cull-ratio", type=float, default=0.5)
    parser.add_argument("--immigrant-ratio", type=float, default=0.05)
    parser.add_argument("--local-search-rate", type=float, default=0.15)
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
        help="Số nhân CPU sử dụng để chạy các batch song song (>=2 để bật ProcessPoolExecutor).",
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
            show_routes=args.show_routes,
            show_generation_progress=args.show_generation_progress,
            generation_progress_every=args.generation_progress_every,
            max_workers=args.max_workers,
        )

    if len(targets) == 2:
        mincost_summary = summarize_results(all_results["mincost"], "mincost")
        makespan_summary = summarize_results(all_results["makespan"], "makespan")
        print("\n=== Combined overview (v2-separator) ===")
        print(f"  mincost_avg_obj_cost  : {mincost_summary['avg_objective_cost']:.4f}")
        print(f"  mincost_feasible_rate : {mincost_summary['feasible_rate']:.4f}")
        print(f"  makespan_avg_obj_cost : {makespan_summary['avg_objective_cost']:.4f}")
        print(f"  makespan_avg_makespan : {makespan_summary['avg_makespan']:.4f}")
        print(f"  makespan_feasible_rate: {makespan_summary['feasible_rate']:.4f}")


if __name__ == "__main__":
    main()
