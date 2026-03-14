from typing import Optional

import torch

from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from parco.envs.epoch_data_env_base import EpochDataEnvBase
from parco.envs.pvrpwdp.generator import PVRPWDPGenerator


log = get_pylogger(__name__)

class PVRPWDPVEnv(EpochDataEnvBase):
    """Perishable Vehicle Routing Problem with Drones and Pickup (PVRPWDP) environment.
    
    PVRPWDP extends HCVRP with time windows and perishability constraints for pickup operations.
    In PVRPWDP, heterogeneous vehicles (trucks and drones) with different capacities, speeds, and 
    operational constraints are used to pick up perishable goods from customers and deliver them 
    back to the depot before the goods spoil.
    
    Key differences from HCVRP:
        - **Pickup operation**: Vehicles pick up goods FROM customers and bring them TO the depot
        - **Time windows**: Each customer has a time window [earliest, latest] for service
          - If vehicle arrives early (before earliest), it must wait until earliest time
          - Waiting time is included in endurance consumption for drones
        - **Perishability constraint**: Each picked-up item has a waiting_time (freshness duration)
          - Once picked up, the item must be delivered to depot within its waiting_time
          - If delivery exceeds waiting_time, the item spoils (becomes unusable)
        - **Trip deadline**: Accumulated deadline tracking to ensure freshness during multi-pickup trips
        - **Endurance constraint**: Drones have limited flight time (battery endurance)
          - Endurance = travel time + waiting time (if arrived early)
          - Endurance resets to 0 when vehicle returns to depot
        - **Instant reset at depot**: When vehicle returns to depot:
          - Capacity and endurance are instantly reset to full (reset time = 0)
          - Vehicle can immediately start next trip without waiting
    
    Observations:
        - Location of the depot
        - Locations, demands, time windows, and waiting times (freshness) of each customer
        - Current location, remaining capacity, and used endurance of each vehicle
        - Current time and trip deadline for freshness tracking
        - Type and speed of each vehicle (truck/drone)
    
    Constraints:
        - Each tour starts and ends at the depot for each vehicle
        - Each customer must be visited exactly once by one vehicle
        - Vehicles must not exceed their remaining capacity when picking up from customers
        - Service must occur within customer's time window [earliest, latest]
        - Picked-up items must be delivered to depot before spoiling (within waiting_time)
        - Drones must return to depot before battery runs out (endurance constraint)
        - Each vehicle can only return to its designated depot
    
    Finish Condition:
        - All vehicles have picked up from all customers and returned to the depot
        - All items delivered before spoiling
    
    Reward:
        - Lexicographic objective with two priorities:
          1. Feasibility: Maximize customers visited (hard constraint)
          2. Efficiency: Minimize makespan (completion time of slowest vehicle)
    
    Args:
        generator: An instance of PVRPWDPGenerator for generating problem instances
        generator_params: Parameters configuring the generator (num_loc, time windows, freshness, etc.)
        check_solution: Enable solution validation for debugging (slows down training)

    Input TensorDict Shape (from Generator):
        N: num customer
        M: num agent

        Node:
            depot:          [B, 2] 
            locs:           [B, N, 2]
            time_window:    [B, N, 2]
            demand:         [B, N]
            waiting_time:   [B, N]
        Veh:
            speed:          [B, M]
            capacity:       [B, M]
            endurance:      [B, M]
    
    Output Reset TensorDict Shape (after _reset):
        N: num customer
        M: num agent
        N+M: depot + customer 

        Node:
            locs:               [B, M+N, 2]   # M depot copies + N customer locs
            demand:             [B, M, M+N]   # Repeated for each agent
            time_window:        [B, M+N, 2]   # [earliest, latest] for depots and customers
            waiting_time:       [B, M+N]      # Freshness time for depots and customers
        
        Agent State:
            current_length:     [B, M]        # Total distance traveled by each agent
            current_time:       [B, M]        # Current time of each agent
            current_node:       [B, M]        # Current node index (0..M-1 = depots, M..M+N-1 = customers)
            trip_deadline:      [B, M]        # Deadline to return to depot (freshness constraint)
            depot_node:         [B, M]        # Home depot index for each agent
            used_capacity:      [B, M]        # Accumulated pickup from customers
            used_endurance:     [B, M]        # Accumulated battery/endurance used
            agents_capacity:    [B, M]        # Max capacity of each agent
            agents_speed:       [B, M]        # Speed of each agent
            agents_endurance:   [B, M]        # Max endurance/battery of each agent
            
        Tracking:
            i:                  [B, 1]        # Step counter
            visited:            [B, M+N]      # Boolean mask of visited nodes
            action_mask:        [B, M, M+N]   # Valid actions for each agent
            done:               [B]           # Episode completion flag
    """
    
    name = "pvrpwdp"
    
    def __init__(
        self,
        generator: PVRPWDPGenerator | None = None,
        generator_params: dict = {},
        check_solution: bool = False,
        epoch_data_dir: str | None = None,
        epoch_file_pattern: str = "epoch_{epoch}.npz",
        use_epoch_data: bool = True,
        fallback_to_generator: bool = True,
        **kwargs,
    ):
        if generator is None:
            generator = PVRPWDPGenerator(**generator_params)
        self.generator = generator
        
        super().__init__(
            epoch_data_dir=epoch_data_dir,
            epoch_file_pattern=epoch_file_pattern,
            use_epoch_data=use_epoch_data,
            fallback_to_generator=fallback_to_generator,
            **kwargs
        )

        self._make_spec(self.generator)

    def _reset(
        self, 
        td: Optional[TensorDict] = None, 
        batch_size: Optional[list] = None
    ) -> TensorDict:
        device = td.device

        # Record parameters
        # num_agents = self.generator.num_agents
        # num_loc_all = self.generator.num_loc + num_agents
        num_agents = td["agents_speed"].size(-1)
        num_loc_all = td["locs"].size(-2) + num_agents

        # Repeat the depot for each agent (i.e. each agent has its own depot, at the same place)
        depots = td["depot"]
        if depots.shape[-2] == 1 or depots.ndim == 2:
            depots = depots.unsqueeze(-2) if depots.ndim == 2 else depots
            depots = depots.repeat(1, num_agents, 1)

        # Padding depot demand as 0 to the demand
        demand_depot = torch.zeros(
            (*batch_size, num_agents), dtype=torch.float32, device=device
        )
        demand = torch.cat((demand_depot, td["demand"]), -1)

        # Repeat the demand for each agent, for convinent action mask calculation
        # Note that this will take more memory
        demand = demand.unsqueeze(-2).repeat(1, num_agents, 1)

        speeds = td["agents_speed"]
        speed_min = speeds.min(dim=-1, keepdim=True)[0]  # [B, 1]
        
        # Calculate max_time for depot time window: max(latest_i + travel_time_to_depot_with_min_speed)
        # Get customer locations and depot location
        customer_locs = td["locs"]  # [B, N, 2]
        depot_loc = td["depot"]  # [B, 2] or [B, 1, 2]
        if depot_loc.ndim == 2:
            depot_loc = depot_loc.unsqueeze(-2)  # [B, 1, 2]
        
        # Calculate distance from each customer to depot [B, N]
        dist_to_depot = torch.norm(customer_locs - depot_loc, p=2, dim=-1)  # [B, N]
        
        # Calculate travel time with minimum speed [B, N]
        travel_time_to_depot = dist_to_depot / speed_min  # [B, N]
        
        # Get latest time windows for customers [B, N]
        customer_latest = td["time_windows"][..., 1]  # [B, N]
        
        # Calculate max_time = max(latest_i + travel_time_i) [B, 1]
        max_time = (customer_latest + travel_time_to_depot).max(dim=-1, keepdim=True)[0]  # [B, 1]
        
        # Padding depot time_window as [0, max_time] for each depot
        # Shape: [B, m, 2] for depots, concat with [B, N, 2] for customers -> [B, m+N, 2]
        tw_depot = torch.stack([
            torch.zeros((*batch_size, num_agents), dtype=torch.float32, device=device),  # earliest = 0
            max_time.expand(-1, num_agents)  # latest = max_time, expand [B, 1] -> [B, m]
        ], dim=-1)  # [B, m, 2]
        time_window = torch.cat((tw_depot, td["time_windows"]), -2)  # [B, m+N, 2]

        # Padding depot waiting_time as max_time for each depot
        # Use max_time (consistent with truck endurance and time_scaler)
        # Shape: [B, m] for depots, concat with [B, N] for customers -> [B, m+N]
        waiting_time_depot = max_time.expand(-1, num_agents)  # [B, m]
        waiting_time = torch.cat((waiting_time_depot, td["waiting_time"]), -1)  # [B, m+N]

        # Init current node
        depot_node = torch.arange(num_agents, dtype=torch.int64, device=device)[
            None, ...
        ].repeat(*batch_size, 1)
        current_node = depot_node.clone()

        # Init visited
        visited = torch.zeros((*batch_size, num_loc_all), dtype=torch.bool, device=device)

        # Init action mask
        action_mask = torch.ones(
            (*batch_size, num_agents, num_loc_all), dtype=torch.bool, device=device
        )

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": torch.cat((depots, td["locs"]), -2),
                "demand": demand,
                "time_window": time_window,  # [B, m+N, 2]
                "waiting_time": waiting_time,  # [B, m+N]
                "current_length": torch.zeros(
                    (*batch_size, num_agents), dtype=torch.float32, device=device
                ),
                "current_time": torch.zeros(
                    (*batch_size, num_agents), dtype=torch.float32, device=device
                ),
                "trip_deadline": max_time.expand(-1, num_agents),  # [B, m] - use max_time instead of 1e6
                "max_time": max_time.squeeze(-1),  # [B] - store for resetting trip_deadline in _step
                "current_node": current_node,
                "previous_action": current_node.clone(),  # [B, m] - track previous action for stuck detection
                "depot_node": depot_node,
                "used_capacity": torch.zeros((*batch_size, num_agents), device=device),
                "used_endurance": torch.zeros((*batch_size, num_agents), device=device),
                "agents_capacity": td["agents_capacity"],
                "agents_speed": td["agents_speed"],
                "agents_endurance": td["agents_endurance"],
                "i": torch.zeros((*batch_size, 1), dtype=torch.int64, device=device),
                "visited": visited,
                "action_mask": action_mask,
                "done": torch.zeros((*batch_size,), dtype=torch.bool, device=device),
            },
            batch_size=batch_size,
            device=device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _step(self, td: TensorDict) -> TensorDict:
        """
        Keys:
            - action [batch_size, num_agents]: action taken by each agent
        """
        num_agents = td["current_node"].size(-1)

        # Update the current length
        current_loc = gather_by_index(td["locs"], td["action"])
        previous_loc = gather_by_index(td["locs"], td["current_node"])
        
        # If the agent is staying at the same node, do not update anything
        stay_flag = td["action"] == td["current_node"]
        
        # Calculate travel distance for this step only (previous -> current)
        step_distance = get_distance(previous_loc, current_loc)
        
        # Calculate travel time based on step distance (0 if staying)
        travel_time = (step_distance / td["agents_speed"]) * (~stay_flag).float()
        
        # Now update total current_length
        current_length = td["current_length"] + step_distance

        # Get waiting time of the selected node
        selected_waiting_time = gather_by_index(td["waiting_time"], td["action"])

        # Update the used capacity
        # Increase used capacity if not visiting the depot, otherwise set to 0
        selected_demand = gather_by_index(td["demand"], td["action"], dim=-1)
        selected_demand = selected_demand * (~stay_flag).float()
        used_capacity = (td["used_capacity"] + selected_demand) * (
            td["action"] >= num_agents
        ).float()
        
        # Update current_time: arrival time considering travel + respect time window earliest time
        arrival_time = td["current_time"] + travel_time
        selected_earliest = gather_by_index(td["time_window"][..., 0], td["action"])
        current_time = torch.maximum(arrival_time, selected_earliest)
        
        # Calculate actual endurance used:
        # - If departing from depot (current_node < num_agents): only count travel_time
        #   (can choose departure time to arrive exactly at earliest, no waiting needed)
        # - If during trip (current_node >= num_agents): count travel_time + waiting_time
        #   (must wait if arrived early, cannot control departure time mid-trip)
        is_departing_from_depot = td["current_node"] < num_agents
        waiting_time_at_node = torch.clamp(selected_earliest - arrival_time, min=0.0) * (~stay_flag).float()
        
        # Only add waiting time if NOT departing from depot
        total_endurance_used = torch.where(
            is_departing_from_depot,
            travel_time,  # From depot: only travel time
            travel_time + waiting_time_at_node  # During trip: travel + waiting
        )
        
        # Update used_endurance: only accumulate if not visiting depot
        used_endurance = (td["used_endurance"] + total_endurance_used) * (
            td["action"] >= num_agents
        ).float()
        
        # Update trip_deadline based on current_node
        # If current_node is depot: trip_deadline = current_time + waiting_time
        # If current_node is not depot: trip_deadline = min(trip_deadline, current_time + waiting_time)
        # If staying at same node: keep old trip_deadline (no update)
        is_at_depot = td["current_node"] < num_agents  # True if at depot
        
        new_deadline = current_time + selected_waiting_time
        trip_deadline_updated = torch.where(
            is_at_depot,
            new_deadline,  # At depot: set to new deadline
            torch.min(td["trip_deadline"], new_deadline)  # Not at depot: take minimum
        )
        
        # Keep old deadline if staying, use updated deadline if moving
        trip_deadline = torch.where(
            stay_flag,
            td["trip_deadline"],  # Stay: keep old deadline
            trip_deadline_updated  # Move: use updated deadline
        )
        
        # Reset trip_deadline to max_time when going back to depot
        # Use max_time from td instead of hardcoded 1e6
        max_time_expanded = td["max_time"].unsqueeze(-1).expand_as(trip_deadline)  # [B, m]
        trip_deadline = torch.where(
            td["action"] < num_agents,  # If going to depot
            max_time_expanded,  # Reset to max_time
            trip_deadline  # Otherwise keep current deadline
        )
        

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, td["action"], 1)

        # update the done and reward
        # Done condition 1: all customers visited
        all_visited = visited[..., num_agents:].sum(-1) == (visited.size(-1) - num_agents)
        done = all_visited
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_length": current_length,
                "current_time": current_time,
                "trip_deadline": trip_deadline,
                "current_node": td["action"],
                "previous_action": td["action"].clone(),  # Store action for next step's stuck detection
                "used_capacity": used_capacity,
                "used_endurance": used_endurance,
                "i": td["i"] + 1,
                "visited": visited,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))

        # === Impossibility Detection ===
        # Check if all agents are at depot with no valid customer action
        # This should not happen if data is feasible, but we detect it for safety
        impossible = self._check_impossibility(td, num_agents)
        if impossible.any():
            # log.warning("="*80)
            # log.warning("IMPOSSIBILITY DETECTED - Setting done=True")
            # log.warning("="*80)
            # batch_size = td.batch_size[0] if isinstance(td.batch_size, (list, tuple)) else 1
            # for b in range(batch_size):
            #     if impossible[b]:
            #         visited_count = td["visited"][b, num_agents:].sum().item()
            #         total_customers = td["visited"].shape[-1] - num_agents
            #         log.warning(
            #             f"  Batch {b}: All agents at depot, no customer reachable. "
            #             f"Visited {visited_count}/{total_customers} customers"
            #         )
            
            # Set done for impossible batches
            if td["done"].dim() > 1:
                td["done"] = td["done"] | impossible.unsqueeze(-1)
            else:
                td["done"] = td["done"] | impossible

        # Done condition 2: no agent can reach any unvisited customer
        # (all agents only have depot as valid action)
        # Must check AFTER action_mask is updated
        action_mask = td["action_mask"]
        has_customer_action = action_mask[..., num_agents:].any(dim=-1)  # [B, m] per agent
        any_agent_can_reach_customer = has_customer_action.any(dim=-1)   # [B] any agent
        no_progress_possible = ~any_agent_can_reach_customer & ~td["done"].squeeze(-1)
        if no_progress_possible.any():
            td["done"] = td["done"] | no_progress_possible.unsqueeze(-1)
        return td
    
    def _check_impossibility(self, td: TensorDict, num_agents: int) -> torch.Tensor:
        """
        Detect if problem is impossible to continue.
        
        Condition: All agents at depot + no valid customer action for any agent + not done yet
        
        Returns: [B] boolean tensor - True if impossible
        """
        # Check if all agents are at depot
        agent_at_depot = td["current_node"] < num_agents  # [B, m]
        all_at_depot = agent_at_depot.all(dim=-1)  # [B]
        
        # Check if any agent has valid customer action
        action_mask = td["action_mask"]  # [B, m, m+N]
        customer_actions = action_mask[..., num_agents:]  # [B, m, N]
        has_customer_action = customer_actions.any(dim=-1)  # [B, m]
        any_agent_can_reach = has_customer_action.any(dim=-1)  # [B]
        
        # Impossible = all at depot + no agent can reach customer + not done
        done = td["done"].squeeze(-1) if td["done"].dim() > 1 else td["done"]  # [B]
        impossible = all_at_depot & ~any_agent_can_reach & ~done
        
        return impossible
    
    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for PVRPWDP.
        
        Cost = alpha * (makespan / max_time) + beta * (unvisited / total_customers)
        
        Where:
            - makespan = max(current_time of all agents) = thời gian về depot muộn nhất
            - max_time = thời gian tối đa của bài toán (dùng để chuẩn hóa)
            - unvisited = số khách chưa thăm
            - total_customers = tổng số khách
            - alpha = 10 (trọng số makespan)
            - beta = 1 (trọng số unvisited)
        
        Reward = -cost (RL4CO convention: maximize reward = minimize cost)
        """
        alpha = 0.0
        beta = 20.0
        
        num_agents = td["current_node"].size(-1)
        num_customers = td["visited"].shape[-1] - num_agents  # total locations - depot slots
        
        # Makespan: thời gian về depot muộn nhất trong các agent
        makespan = td["current_time"].max(dim=-1)[0]  # [B]
        
        # Chuẩn hóa makespan bởi max_time
        max_time = td["max_time"]  # [B]
        makespan_normalized = makespan / max_time  # [B], trong khoảng [0, 1+]
        
        # Số khách không thăm được (chỉ đếm customer, bỏ depot slots)
        unvisited_count = (~td["visited"][..., num_agents:]).sum(dim=-1).float()  # [B]
        unvisited_ratio = unvisited_count / num_customers  # [B], trong khoảng [0, 1]
        
        # Cost = alpha * (makespan / max_time) + beta * (unvisited / total)
        cost = alpha * makespan_normalized + beta * unvisited_ratio
        
        # Reward = -cost
        return -cost

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        batch_size = td.batch_size
        num_agents = td["current_node"].size(-1)

        # Init action mask for each agent with all not visited nodes
        action_mask = torch.repeat_interleave(
            ~td["visited"][..., None, :], dim=-2, repeats=num_agents
        )

        # Can not visit the node if the demand is more than the remaining capacity
        remain_capacity = td["agents_capacity"] - td["used_capacity"]
        within_capacity_flag = td["demand"] <= remain_capacity[..., None]  # TODO: check
        action_mask &= within_capacity_flag

        # === PVRPWDP-specific constraints: Time window and deadline ===
        # For each agent and each potential customer node:
        # 1. Calculate arrival time at customer
        # 2. Check if arrival time <= latest time window
        # 3. Calculate time to return to depot after visiting customer
        # 4. Check if return time <= trip_deadline
        
        # Get current locations of all agents [B, m, 2]
        current_locs = gather_by_index(td["locs"], td["current_node"])  # [B, m, 2]
        
        # Get all node locations [B, m+N, 2]
        all_locs = td["locs"]  # [B, m+N, 2]
        
        # Calculate distance from each agent to each node [B, m, m+N]
        # Expand dimensions: current_locs [B, m, 1, 2], all_locs [B, 1, m+N, 2]
        dist_to_nodes = torch.cdist(
            current_locs, 
            all_locs, 
            p=2
        )  # [B, m, m+N]
        
        # Calculate travel time to each node [B, m, m+N]
        # agents_speed shape: [B, m] -> [B, m, 1]
        travel_time_to_nodes = dist_to_nodes / td["agents_speed"].unsqueeze(-1)
        
        # Calculate arrival time at each node [B, m, m+N]
        # current_time shape: [B, m] -> [B, m, 1]
        arrival_time = td["current_time"].unsqueeze(-1) + travel_time_to_nodes
        
        # Check if we're departing from depot (for waiting time calculation)
        is_at_depot = td["current_node"] < num_agents  # [B, m]
        
        # Calculate waiting time at each node if arrived early
        # time_window[..., 0] is earliest time [B, m+N]
        earliest_times = td["time_window"][..., 0].unsqueeze(-2)  # [B, 1, m+N]
        waiting_time = torch.clamp(earliest_times - arrival_time, min=0.0)  # [B, m, m+N]
        
        # Waiting time only applies if NOT at depot (can optimize departure from depot)
        # is_at_depot [B, m] -> [B, m, 1]
        waiting_time = waiting_time * (~is_at_depot.unsqueeze(-1)).float()
        
        # Service time = arrival time + waiting time (respect earliest time window)
        service_time = torch.maximum(arrival_time, earliest_times)  # [B, m, m+N]
        
        # Time window constraint: service time must <= latest time
        # time_window[..., 1] is latest time [B, m+N]
        latest_times = td["time_window"][..., 1].unsqueeze(-2)  # [B, 1, m+N]
        within_time_window = service_time <= latest_times  # [B, m, m+N]
        
        # Apply time window constraint (only for customer nodes, not depots)
        customer_mask = torch.ones_like(action_mask)
        customer_mask[..., :num_agents] = True  # Always allow depot in this check
        within_time_window = within_time_window | ~customer_mask
        action_mask &= within_time_window
        
        # Deadline constraint: ensure can return to depot before trip_deadline
        # Calculate distance from each potential node back to each agent's depot
        # Depot locations: td["locs"][..., :num_agents, :] [B, m, 2]
        depot_locs = td["locs"][..., :num_agents, :]  # [B, m, 2]
        
        # For each agent i, calculate distance from each node to depot i
        # all_locs [B, m+N, 2], depot_locs [B, m, 2]
        # We need [B, m, m+N] where [b, i, j] = distance from node j to depot i
        dist_to_depot = torch.cdist(
            depot_locs,  # [B, m, 2]
            all_locs,    # [B, m+N, 2]
            p=2
        )  # [B, m, m+N]
        
        # Calculate travel time from each node back to depot [B, m, m+N]
        travel_time_to_depot = dist_to_depot / td["agents_speed"].unsqueeze(-1)
        
        # Calculate total time when returning to depot [B, m, m+N]
        # = service_time + travel_time_to_depot
        return_time = service_time + travel_time_to_depot
        
        # Check if return time <= trip_deadline [B, m, m+N]
        # trip_deadline shape: [B, m] -> [B, m, 1]
        within_deadline = return_time <= td["trip_deadline"].unsqueeze(-1)
        
        # Apply deadline constraint (only for customer nodes, not depots)
        within_deadline = within_deadline | ~customer_mask
        action_mask &= within_deadline
        
        # === Endurance constraint: ensure enough battery/endurance for trip ===
        # Calculate endurance needed: travel to node + waiting (if early & not from depot) + return to depot
        # For agents at depot: waiting_time is 0 (can optimize departure)
        # For agents mid-trip: waiting_time is counted (must wait if early)
        endurance_to_node = travel_time_to_nodes.clone()  # [B, m, m+N]
        
        # Add waiting time only if NOT at depot (same logic as in _step)
        endurance_to_node = torch.where(
            is_at_depot.unsqueeze(-1),  # [B, m, 1]
            travel_time_to_nodes,  # From depot: only travel time
            travel_time_to_nodes + waiting_time  # Mid-trip: travel + waiting
        )
        
        # Total endurance needed = endurance to node + travel back to depot
        total_endurance_needed = endurance_to_node + travel_time_to_depot  # [B, m, m+N]
        
        # Check if total endurance (already used + needed) <= agent's max endurance
        # agents_freshness is the max endurance/battery capacity [B, m]
        # used_endurance is how much already used [B, m]
        total_endurance = td["used_endurance"].unsqueeze(-1) + total_endurance_needed  # [B, m, m+N]
        within_endurance = total_endurance <= td["agents_endurance"].unsqueeze(-1)  # [B, m, m+N]
        
        # Apply endurance constraint (only for customer nodes, not depots)
        # Depot visits always allowed (reset endurance to 0)
        within_endurance = within_endurance | ~customer_mask
        action_mask &= within_endurance
        
        # === Stuck state detection: If action will repeat, force progression ===
        # Check if action will be the same as current_node (all agents stuck)
        # For agents ngoài depot, mask out current_node to force them toward depot
        if "previous_action" in td.keys():
            action_will_repeat = td["current_node"] == td["previous_action"]  # [B, m]
            
            if action_will_repeat.any():
                # Agents NOT at depot + action will repeat → force them to move
                # Block their current_node from action_mask so next step must choose depot
                is_at_depot = td["current_node"] < num_agents  # [B, m]
                should_block_current = action_will_repeat & ~is_at_depot  # [B, m]
                
                if should_block_current.any():
                    # Vectorized: Create mask to block current_node
                    # For affected agents, set their current_node action to False
                    block_mask = torch.zeros_like(action_mask, dtype=torch.bool)  # [B, m, m+N]
                    
                    # Scatter ones at positions where should block current_node
                    for b in range(batch_size[0]):
                        for m_idx in range(num_agents):
                            if should_block_current[b, m_idx]:
                                current_idx = td["current_node"][b, m_idx].item()
                                block_mask[b, m_idx, current_idx] = True
                    
                    # Apply block: set these positions to False
                    action_mask = action_mask & ~block_mask
        
        # === Original depot isolation logic ===
        # The depot is not available if **all** the agents are at the depot and the task is not finished
        all_back_flag = torch.sum(td["current_node"] >= num_agents, dim=-1) == 0
        has_finished_early = all_back_flag & ~td["done"]

        depot_mask = ~has_finished_early[..., None]  # 1 means we can visit

        # If no available nodes outside (all visited), make the depot always available
        all_visited_flag = (
            torch.sum(~td["visited"][..., num_agents:], dim=-1, keepdim=True) == 0
        )
        depot_mask |= all_visited_flag

        # Update the depot mask in the action mask
        eye_matrix = torch.eye(num_agents, device=td.device)
        eye_matrix = eye_matrix[None, ...].repeat(*batch_size, 1, 1).bool()
        eye_matrix &= depot_mask[..., None]
        action_mask[..., :num_agents] = eye_matrix

        # === Safety net: ensure each agent always has at least one valid action ===
        # If an agent has NO valid action (all customers and depot masked), allow its depot.
        # This prevents action_mask from being entirely False, which would cause crashes.
        # Note: Data is guaranteed to be feasible, so this should rarely trigger.
        # If it does, impossibility detection in _step() will catch it and set done=True.
        no_valid_action = ~action_mask.any(dim=-1)  # [B, m] - True if agent has no valid action
        if no_valid_action.any():
            # For each agent with no valid action, allow only its depot
            depot_fix = torch.zeros_like(action_mask)  # [B, m, m+N]
            depot_eye = torch.eye(num_agents, device=td.device).bool()
            depot_eye = depot_eye[None, ...].repeat(*batch_size, 1, 1)  # [B, m, m]
            # Pad to full action space: [B, m, m] -> [B, m, m+N]
            depot_fix[..., :num_agents] = depot_eye
            # Replace mask for affected agents (allow only their depot)
            action_mask = torch.where(
                no_valid_action.unsqueeze(-1),  # [B, m, 1] - agents with no valid action
                depot_fix,                        # [B, m, m+N] - allow only their depot
                action_mask                       # [B, m, m+N] - keep existing mask
            )

        return action_mask

    def _make_spec(self, generator: PVRPWDPGenerator):
        """Tạo observation và action specs."""
        from torchrl.data import Bounded, Composite, Unbounded

        self.observation_spec = Composite(
            locs=Bounded(
                low=0.0,
                high=1.0,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            demand=Bounded(
                low=0.0,
                high=1.0,
                shape=(generator.num_loc + 1,),
                dtype=torch.float32,
            ),
            time_windows=Bounded(
                low=0.0,
                high=1.0,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            current_node=Unbounded(shape=(1,), dtype=torch.int64),
            visited=Unbounded(shape=(generator.num_loc + 1,), dtype=torch.bool),
            action_mask=Unbounded(shape=(generator.num_loc + 1,), dtype=torch.bool),
        )

        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc + 1,
        )

        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        pass

    @staticmethod
    def render(td: TensorDict, actions=None, ax=None):
        """Render PVRPWDPV2 environment visualization."""
        from .render import render
        return render(td, actions, ax)
