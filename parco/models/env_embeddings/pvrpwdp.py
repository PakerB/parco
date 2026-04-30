import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index

from parco.models.nn.positional_encoder import PositionalEncoder

from .communication import BaseMultiAgentContextEmbedding


class PVRPWDPDynamicEmbedding(nn.Module):
    """PVRPWDP Dynamic Embedding for decoder K/V updates.
    
    Projects 5 per-node features into dynamic key, value, and logit_key updates.
    These are ADDED to the static K/V from the encoder at each decode step.
    
    Features (per node, aggregated from all agents):
        1. min_slack / time_scaler: worst-case urgency
        2. mean_slack / time_scaler: average difficulty
        3. num_reachable / num_agents: reachability ratio
        4. min_future_slack / time_scaler: forward risk
        5. is_visited: binary flag
    
    Args:
        embed_dim: Dimension of embeddings
        linear_bias: Whether to use bias in linear layers
    """

    def __init__(self, embed_dim: int = 128, linear_bias: bool = False, **kwargs):
        super(PVRPWDPDynamicEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.dyn_kv = nn.Linear(5, 3 * embed_dim, bias=linear_bias)

    def forward(self, td):
        num_agents = td["current_node"].size(-1)
        N_total = td["locs"].size(-2)  # M+N
        N = N_total - num_agents
        
        # Check if heuristic features are cached
        if "slack_matrix" not in td.keys():
            # Fallback: return zeros (backward compatible)
            B = td.batch_size[0] if isinstance(td.batch_size, (list, tuple)) else td.batch_size
            z = torch.zeros(B, N_total, self.embed_dim, device=td.device)
            return z, z, z
        
        # Get cached scalers
        time_scaler = td["time_scaler"].unsqueeze(-1)  # [B, 1, 1]
        
        # slack_matrix: [B, M, M+N] → aggregate to [B, M+N]
        slack_matrix = td["slack_matrix"]  # [B, M, M+N]
        action_mask = td["action_mask"]  # [B, M, M+N]
        
        # Min slack per node (across agents), only over agents that can reach
        slack_for_min = slack_matrix.masked_fill(~action_mask, float('inf'))  # [B, M, M+N]
        min_slack = slack_for_min.min(dim=-2)[0]  # [B, M+N]
        # Where no agent can reach, set to 0
        no_agent_reach = ~action_mask.any(dim=-2)  # [B, M+N]
        min_slack = min_slack.masked_fill(no_agent_reach, 0.0)
        
        # Mean slack per node (across agents that can reach)
        slack_sum = (slack_matrix * action_mask.float()).sum(dim=-2)  # [B, M+N]
        reach_count = action_mask.float().sum(dim=-2).clamp(min=1e-6)  # [B, M+N]
        mean_slack = slack_sum / reach_count  # [B, M+N]
        
        # num_reachable ratio (only for customers, pad depot with 1.0)
        num_reachable_ratio = torch.ones(min_slack.shape, device=td.device)  # [B, M+N]
        if "num_reachable" in td.keys():
            num_reachable_ratio[..., num_agents:] = td["num_reachable"] / num_agents
        
        # min_future_slack (only for customers, pad depot with 0)
        min_fs = torch.zeros(min_slack.shape, device=td.device)  # [B, M+N]
        if "min_future_slack" in td.keys():
            # Aggregate across agents: take min over agents
            mfs = td["min_future_slack"]  # [B, M, N]
            customer_mask = action_mask[..., num_agents:]  # [B, M, N]
            mfs_for_min = mfs.masked_fill(~customer_mask, float('inf'))
            min_fs_per_node = mfs_for_min.min(dim=-2)[0]  # [B, N]
            no_agent_reach_cust = ~customer_mask.any(dim=-2)  # [B, N]
            min_fs_per_node = min_fs_per_node.masked_fill(no_agent_reach_cust, 0.0)
            min_fs[..., num_agents:] = min_fs_per_node
        
        # is_visited
        is_visited = td["visited"].float()  # [B, M+N]
        
        # Stack features [B, M+N, 5]
        features = torch.stack([
            min_slack / time_scaler.squeeze(-1),
            mean_slack / time_scaler.squeeze(-1),
            num_reachable_ratio,
            min_fs / time_scaler.squeeze(-1),
            is_visited,
        ], dim=-1)  # [B, M+N, 5]
        
        # Project to 3 * embed_dim
        dyn_proj = self.dyn_kv(features)  # [B, M+N, 3*D]
        dyn_k, dyn_v, dyn_l = dyn_proj.chunk(3, dim=-1)  # 3 × [B, M+N, D]
        
        return dyn_k, dyn_v, dyn_l


class PVRPWDPInitEmbedding(nn.Module):
    """PVRPWDP Initial Embedding for encoder.
    
    Embeds depot, agents, and clients with their features.
    Note that in PVRPWDP capacities are not the same for all agents and
    they need to be rescaled by max capacity.
    
    Scalers are computed dynamically from the data:
    - demand_scaler = max(demand of all clients, capacity of all agents)
    - speed_scaler = max(speed of all agents)
    - endurance_scaler = max(endurance of all agents) if normalize_endurance_by_max=True
                        else time_scaler
    - time_scaler = max(latest time window of all clients)
    
    Args:
        embed_dim: Dimension of embeddings
        linear_bias: Whether to use bias in linear layers
        use_polar_feats: Whether to add polar coordinates (distance and angle to depot) for clients
        normalize_endurance_by_max: If True, normalize endurance by max(endurance of all agents).
                                   If False, normalize by time_scaler (for time consistency).
                                   Default: True
    """

    def __init__(
        self,
        embed_dim: int = 128,
        linear_bias: bool = False,
        use_polar_feats: bool = True,
        normalize_endurance_by_max: bool = False,
    ):
        super(PVRPWDPInitEmbedding, self).__init__()
        # depot feats: [x0, y0]
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.pos_embedding_proj = nn.Linear(embed_dim, embed_dim, linear_bias)
        self.alpha = nn.Parameter(torch.Tensor([1]))
        # agent feats: [x0, y0, capacity, speed, endurance]
        self.init_embed_agents = nn.Linear(5, embed_dim, linear_bias)
        # combine depot and agent embeddings
        self.init_embed_depot_agents = nn.Linear(2 * embed_dim, embed_dim, linear_bias)
        # client feats: [x, y, demand, earliest, latest, waiting_time] + optional [dist_to_depot, angle_to_depot]
        client_feats_dim = 8 if use_polar_feats else 6
        self.init_embed_clients = nn.Linear(client_feats_dim, embed_dim, linear_bias)

        self.use_polar_feats = use_polar_feats
        self.normalize_endurance_by_max = normalize_endurance_by_max

    def forward(self, td):
        num_agents = td["action_mask"].shape[-2]  # [B, m, m+N]
        depot_locs = td["locs"][..., :num_agents, :]
        agents_locs = td["locs"][..., :num_agents, :]
        clients_locs = td["locs"][..., num_agents:, :]

        # Compute dynamic scalers from data
        demands = td["demand"][..., num_agents:]  # [B, N] (skip depot zeros)
        capacities = td["agents_capacity"]  # [B, m]
        speeds = td["agents_speed"]  # [B, m]
        endurances = td["agents_endurance"]  # [B, m]
        time_windows = td["time_window"][..., num_agents:, :]  # [B, N, 2]
        
        # demand_scaler = max(all demands, all capacities)
        demand_scaler = torch.maximum(
            demands.max(dim=-1, keepdim=True)[0],  # [B, 1]
            capacities.max(dim=-1, keepdim=True)[0]  # [B, 1]
        ).unsqueeze(-1)  # [B, 1, 1]
        
        # speed_scaler = max(all speeds)
        speed_scaler = speeds.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B, 1, 1]
        
        # time_scaler = max(all latest time windows)
        time_scaler = time_windows[..., 1].max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B, 1, 1]
        
        # endurance_scaler: either max(all endurances) or time_scaler
        if self.normalize_endurance_by_max:
            endurance_scaler = endurances.max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B, 1, 1]
        else:
            endurance_scaler = time_scaler  # Use time_scaler for endurance normalization
        
        # Clamp scalers to avoid division by zero or extreme values
        demand_scaler = torch.clamp(demand_scaler, min=1e-6)
        speed_scaler = torch.clamp(speed_scaler, min=1e-6)
        time_scaler = torch.clamp(time_scaler, min=1e-6)
        endurance_scaler = torch.clamp(endurance_scaler, min=1e-6)
        
        # Compute distance scaler from actual coordinates (for normalizing spatial features)
        # Use max distance from depot to any location
        all_locs = td["locs"]  # [B, M+N, 2]
        depot_loc = depot_locs[..., 0:1, :]  # [B, 1, 2] - first agent's depot
        distances = torch.norm(all_locs - depot_loc, p=2, dim=-1, keepdim=True)  # [B, M+N, 1]
        distance_scaler = distances.max(dim=-2, keepdim=True)[0]  # [B, 1, 1]
        # Avoid division by zero
        distance_scaler = torch.where(
            distance_scaler > 1e-6,
            distance_scaler,
            torch.ones_like(distance_scaler)
        )

        # Depots embedding with positional encoding (normalize by distance scaler)
        depot_locs_normalized = depot_locs / distance_scaler  # [B, 1, 1] broadcasts to [B, M, 2]
        depots_embedding = self.init_embed_depot(depot_locs_normalized)
        pos_embedding = self.pos_encoder(depots_embedding, add=False)
        pos_embedding = self.alpha * self.pos_embedding_proj(pos_embedding)
        depot_embedding = depots_embedding + pos_embedding

        # Agents embedding (normalize coordinates by distance scaler)
        agents_locs_normalized = agents_locs / distance_scaler  # [B, 1, 1] broadcasts to [B, M, 2]
        agents_feats = torch.cat(
            [
                agents_locs_normalized,
                td["agents_capacity"][..., None] / demand_scaler,
                td["agents_speed"][..., None] / speed_scaler,
                td["agents_endurance"][..., None] / endurance_scaler,
            ],
            dim=-1,
        )
        agents_embedding = self.init_embed_agents(agents_feats)

        # Combine depot and agents embeddings
        depot_agents_feats = torch.cat([depot_embedding, agents_embedding], dim=-1)
        depot_agents_embedding = self.init_embed_depot_agents(depot_agents_feats)

        # Clients embedding
        earliest = time_windows[..., 0]  # [B, N]
        latest = time_windows[..., 1]    # [B, N]
        waiting_times = td["waiting_time"][..., num_agents:]  # [B, N] - only customer waiting times (slice on last dim)
        
        # Normalize client locations by distance scaler (NOT min-max)
        clients_locs_normalized = clients_locs / distance_scaler  # [B, 1, 1] broadcasts to [B, N, 2]
        
        clients_feats = torch.cat(
            [
                clients_locs_normalized,               # [B, N, 2] - normalized by distance!
                demands[..., None] / demand_scaler,    # [B, N, 1]
                earliest[..., None] / time_scaler,     # [B, N, 1]
                latest[..., None] / time_scaler,       # [B, N, 1]
                waiting_times[..., None] / time_scaler, # [B, N, 1] - freshness normalized
            ], 
            dim=-1
        )  # [B, N, 6]

        if self.use_polar_feats:
            # Convert to polar coordinates using ACTUAL (unnormalized) coordinates
            # This preserves the physical meaning of distance and angle
            depot_actual = depot_locs[..., 0:1, :]  # [B, 1, 2]
            client_locs_centered = clients_locs - depot_actual  # [B, N, 2]
            
            # Compute distance and normalize by distance_scaler
            dist_to_depot = torch.norm(client_locs_centered, p=2, dim=-1, keepdim=True)  # [B, N, 1]
            dist_to_depot_normalized = dist_to_depot / distance_scaler  # [B, N, 1]
            
            # Angle is scale-invariant, so no normalization needed
            angle_to_depot = torch.atan2(
                client_locs_centered[..., 1:], client_locs_centered[..., :1]
            )  # [B, N, 1]
            
            clients_feats = torch.cat(
                [clients_feats, dist_to_depot_normalized, angle_to_depot], dim=-1
            )

        clients_embedding = self.init_embed_clients(clients_feats)

        # === Cache scalers into td for downstream use (ContextEmbedding, DynamicEmbedding) ===
        # These are computed once here (before the decode loop) and reused every step.
        # Stored as [B, 1] so embedding modules can unsqueeze as needed.
        td.update({
            "demand_scaler": demand_scaler.squeeze(-1),   # [B, 1, 1] → [B, 1]
            "time_scaler": time_scaler.squeeze(-1),       # [B, 1, 1] → [B, 1]
            "speed_scaler": speed_scaler.squeeze(-1),     # [B, 1, 1] → [B, 1]
            "distance_scaler": distance_scaler.squeeze(-1), # [B, 1, 1] → [B, 1]
        })

        return torch.cat(
            [depot_agents_embedding, clients_embedding], -2
        )  # [B, m+N, hdim]


class PVRPWDPContextEmbedding(BaseMultiAgentContextEmbedding):

    """PVRPWDP Context Embedding with claim_embed and cached scalers.
    
    Agent features (8 total):
        1-3: current_time, remaining_capacity, remaining_endurance
        4-6: time_to_depot, time_to_deadline, effective_time_limit
        7: min_slack_of_agent (urgency)
        8: num_reachable_ratio (coverage)
    
    Claim embed: weighted attention over customer nodes using 6 heuristic signals.
    """

    def __init__(
        self,
        embed_dim,
        agent_feat_dim=3,  # current_time, remaining_capacity, remaining_endurance
        global_feat_dim=1,
        normalize_endurance_by_max=False,
        use_time_to_depot=True,
        use_claim_embed=True,
        **kwargs,
    ):
        if use_time_to_depot:
            agent_feat_dim += 3  # time_to_depot + time_to_deadline + effective_time_limit
        agent_feat_dim += 2  # min_slack_of_agent + num_reachable_ratio
        super(PVRPWDPContextEmbedding, self).__init__(
            embed_dim, agent_feat_dim, global_feat_dim, **kwargs
        )
        self.normalize_endurance_by_max = normalize_endurance_by_max
        self.use_time_to_depot = use_time_to_depot
        self.use_claim_embed = use_claim_embed
        
        if use_claim_embed:
            self.claim_score_proj = nn.Linear(6, 1, bias=True)
            # Override project_context to accept 5D instead of 4D
            self.project_context = nn.Linear(embed_dim * 5, embed_dim, bias=False)

    def _agent_state_embedding(self, embeddings, td, num_agents, num_cities):
        # Use cached scalers from td (computed once in env._reset)
        if "demand_scaler" in td.keys():
            demand_scaler = td["demand_scaler"].unsqueeze(-1)  # [B, 1, 1]
            time_scaler = td["time_scaler"].unsqueeze(-1)  # [B, 1, 1]
        else:
            # Fallback: compute scalers (backward compatible)
            demands = td["demand"][..., num_agents:]  # [B, N]
            capacities = td["agents_capacity"]  # [B, m]
            time_windows = td["time_window"][..., num_agents:, :]  # [B, N, 2]
            demand_scaler = torch.maximum(
                demands.max(dim=-1, keepdim=True)[0],
                capacities.max(dim=-1, keepdim=True)[0]
            ).unsqueeze(-1)
            time_scaler = time_windows[..., 1].max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        
        # Clamp scalers
        demand_scaler = demand_scaler.clamp(min=1e-6)
        time_scaler = time_scaler.clamp(min=1e-6)
        
        # Use current_time directly from td (already computed in env)
        current_time_normalized = td["current_time"] / time_scaler.squeeze(-1)  # [B, M]
        
        # Compute remaining endurance normalized by time_scaler (same reference as all time features)
        remaining_endurance_norm = (td["agents_endurance"] - td["used_endurance"]) / time_scaler.squeeze(-1)  # [B, M]
        
        context_feats = torch.stack(
            [
                current_time_normalized,  # current time (normalized by time_scaler)
                (td["agents_capacity"] - td["used_capacity"])
                / demand_scaler.squeeze(-1),  # remaining capacity (normalized by demand_scaler)
                remaining_endurance_norm,  # remaining endurance (normalized by time_scaler)
            ],
            dim=-1,
        )
        if self.use_time_to_depot:
            depot = td["locs"][..., 0:1, :]
            cur_loc = gather_by_index(td["locs"], td["current_node"])
            dist_to_depot = torch.norm(cur_loc - depot, p=2, dim=-1, keepdim=True)
            
            agent_speed_safe = td["agents_speed"][..., None].clamp(min=1e-6)
            time_to_depot = dist_to_depot / agent_speed_safe
            
            time_scaler_safe = time_scaler.clamp(min=1e-6)
            time_to_depot_normalized = time_to_depot / time_scaler_safe
            
            time_to_deadline = (td["trip_deadline"] - td["current_time"])[..., None]
            time_to_deadline_normalized = time_to_deadline / time_scaler_safe
            
            remaining_endurance_actual = td["agents_endurance"] - td["used_endurance"]
            time_to_deadline_actual = td["trip_deadline"] - td["current_time"]
            effective_time_limit_actual = torch.minimum(
                remaining_endurance_actual, time_to_deadline_actual
            )
            effective_time_limit = effective_time_limit_actual / time_scaler_safe.squeeze(-1)
            
            context_feats = torch.cat([
                context_feats, 
                time_to_depot_normalized,
                time_to_deadline_normalized,
                effective_time_limit[..., None]
            ], dim=-1)
        
        # --- New features: min_slack_of_agent and num_reachable_ratio ---
        if "slack_matrix" in td.keys():
            slack_k = td["slack_matrix"][..., num_agents:]  # [B, M, N] customer slack
            customer_mask = td["action_mask"][..., num_agents:]  # [B, M, N]
            # Min slack for each agent (over reachable customers)
            slack_for_min = slack_k.masked_fill(~customer_mask, float('inf'))
            min_slack_agent = slack_for_min.min(dim=-1)[0]  # [B, M]
            # If agent has no reachable customer, set to 0
            no_reach = ~customer_mask.any(dim=-1)  # [B, M]
            min_slack_agent = min_slack_agent.masked_fill(no_reach, 0.0)
            min_slack_agent_norm = min_slack_agent / time_scaler.squeeze(-1)
            
            # Num reachable ratio per agent
            num_reach_agent = customer_mask.float().sum(dim=-1)  # [B, M]
            num_reach_ratio = num_reach_agent / num_cities
        else:
            min_slack_agent_norm = torch.zeros_like(td["current_time"])
            num_reach_ratio = torch.ones_like(td["current_time"])
        
        context_feats = torch.cat([
            context_feats,
            min_slack_agent_norm[..., None],  # [B, M, 1]
            num_reach_ratio[..., None],  # [B, M, 1]
        ], dim=-1)
        
        return self.proj_agent_feats(context_feats)

    def _claim_embedding(self, embeddings, td, num_agents, num_cities):
        """Compute claim_embed: weighted attention over customer embeddings.
        
        6 scoring signals per (agent k, customer node i):
            1. -slack[k,i] / time_scaler  (urgency)
            2. 1 / (num_reachable[i] + 1)  (exclusivity)
            3. travel_time_to_depot[k,i] / time_scaler  (depot distance)
            4. -min_future_slack[k,i] / time_scaler  (forward risk)
            5. demand[i] / remaining_capacity[k]  (capacity fit)
            6. -reachability_loss[k,i]  (reachability risk)
        """
        if "slack_matrix" not in td.keys():
            # Fallback: return zeros
            B = td.batch_size[0] if isinstance(td.batch_size, (list, tuple)) else td.batch_size
            return torch.zeros(B, num_agents, self.embed_dim, device=td.device)
        
        time_scaler = td["time_scaler"].unsqueeze(-1)  # [B, 1, 1]
        time_scaler_safe = time_scaler.clamp(min=1e-6)
        
        # 1. Urgency: -slack[k,i]
        slack_customers = td["slack_matrix"][..., num_agents:]  # [B, M, N]
        f1 = -slack_customers / time_scaler_safe  # [B, M, N]
        
        # 2. Exclusivity: 1 / (num_reachable[i] + 1)
        f2 = 1.0 / (td["num_reachable"].unsqueeze(-2) + 1.0)  # [B, 1, N] → broadcast [B, M, N]
        f2 = f2.expand_as(f1)
        
        # 3. Depot distance (travel time)
        if "dist_to_depot_static" in td.keys():
            dist_depot = td["dist_to_depot_static"][..., num_agents:]  # [B, M, N]
        else:
            depot_locs = td["locs"][..., :num_agents, :]
            customer_locs = td["locs"][..., num_agents:, :]
            dist_depot = torch.cdist(depot_locs, customer_locs)  # [B, M, N]
        travel_depot = dist_depot / td["agents_speed"].unsqueeze(-1).clamp(min=1e-6)
        f3 = travel_depot / time_scaler_safe  # [B, M, N]
        
        # 4. Forward risk: -min_future_slack[k,i]
        f4 = -td["min_future_slack"] / time_scaler_safe  # [B, M, N]
        
        # 5. Capacity fit: demand[i] / remaining_capacity[k]
        customer_demand = td["demand"][..., num_agents:]  # [B, N]
        remain_cap = (td["agents_capacity"] - td["used_capacity"]).clamp(min=1e-6)  # [B, M]
        f5 = customer_demand.unsqueeze(-2) / remain_cap.unsqueeze(-1)  # [B, M, N]
        
        # 6. Reachability risk: -reachability_loss[k,i]
        f6 = -td["reachability_loss"]  # [B, M, N]
        
        # Stack and project: [B, M, N, 6] → [B, M, N, 1]
        features = torch.stack([f1, f2, f3, f4, f5, f6], dim=-1)  # [B, M, N, 6]
        scores = self.claim_score_proj(features).squeeze(-1)  # [B, M, N]
        
        # Mask unreachable nodes before softmax
        customer_mask = td["action_mask"][..., num_agents:]  # [B, M, N]
        scores = scores.masked_fill(~customer_mask, float('-inf'))
        
        # Handle case where agent has no reachable customer
        all_masked = ~customer_mask.any(dim=-1, keepdim=True)  # [B, M, 1]
        scores = scores.masked_fill(all_masked, 0.0)  # prevent NaN in softmax
        
        weights = torch.softmax(scores, dim=-1)  # [B, M, N]
        # Zero out weights for agents with no reachable customer
        weights = weights.masked_fill(all_masked, 0.0)
        
        # Weighted sum of customer embeddings
        node_emb = embeddings[:, num_agents:, :]  # [B, N, D]
        # Use einsum for clarity: claim[b,k,d] = sum_n weights[b,k,n] * node_emb[b,n,d]
        claim_embed = torch.einsum('bmn,bnd->bmd', weights, node_emb)  # [B, M, D]
        
        return claim_embed

    def forward(self, embeddings, td):
        # Override forward to inject claim_embed
        num_agents = td["action_mask"].shape[-2]
        num_cities = td["locs"].shape[-2] - num_agents
        cur_node_embedding = gather_by_index(
            embeddings, td["current_node"]
        )  # [B, M, hdim]
        depot_embedding = gather_by_index(embeddings, td["depot_node"])  # [B, M, hdim]
        agent_state_embed = self._agent_state_embedding(
            embeddings, td, num_agents=num_agents, num_cities=num_cities
        )  # [B, M, hdim]
        global_embed = self._global_state_embedding(
            embeddings, td, num_agents=num_agents, num_cities=num_cities
        )  # [B, M, hdim]
        
        if self.use_claim_embed:
            claim_embed = self._claim_embedding(
                embeddings, td, num_agents=num_agents, num_cities=num_cities
            )  # [B, M, hdim]
            context_embed = torch.cat(
                [cur_node_embedding, depot_embedding, agent_state_embed, global_embed, claim_embed], dim=-1
            )
        else:
            context_embed = torch.cat(
                [cur_node_embedding, depot_embedding, agent_state_embed, global_embed], dim=-1
            )
        
        context_embed = self.project_context(context_embed)
        h_comm = self.communication_layers(context_embed)
        if self.norm is not None:
            h_comm = self.norm(h_comm)
        return h_comm

    def _global_state_embedding(self, embeddings, td, num_agents, num_cities):
        global_feats = torch.cat(
            [
                td["visited"][..., num_agents:].sum(-1)[..., None]
                / num_cities,  # number of visited cities / total
            ],
            dim=-1,
        )
        return self.proj_global_feats(global_feats)[..., None, :].repeat(1, num_agents, 1)
