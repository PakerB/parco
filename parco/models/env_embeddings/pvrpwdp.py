import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index

from parco.models.nn.positional_encoder import PositionalEncoder

from .communication import BaseMultiAgentContextEmbedding

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
        demands = td["demand"][..., 0, num_agents:]  # [B, N]
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

        return torch.cat(
            [depot_agents_embedding, clients_embedding], -2
        )  # [B, m+N, hdim]


class PVRPWDPContextEmbedding(BaseMultiAgentContextEmbedding):

    """TODO"""

    def __init__(
        self,
        embed_dim,
        agent_feat_dim=3,  # current_time, remaining_capacity, remaining_endurance
        global_feat_dim=1,
        normalize_endurance_by_max=False,
        use_time_to_depot=True,
        **kwargs,
    ):
        if use_time_to_depot:
            agent_feat_dim += 3  # time_to_depot + time_to_deadline + effective_time_limit
        super(PVRPWDPContextEmbedding, self).__init__(
            embed_dim, agent_feat_dim, global_feat_dim, **kwargs
        )
        self.normalize_endurance_by_max = normalize_endurance_by_max
        self.use_time_to_depot = use_time_to_depot

    def _agent_state_embedding(self, embeddings, td, num_agents, num_cities):
        # Compute dynamic scalers from data (same as in PVRPWDPInitEmbedding)
        demands = td["demand"][..., 0, num_agents:]  # [B, N]
        capacities = td["agents_capacity"]  # [B, m]
        speeds = td["agents_speed"]  # [B, m]
        endurances = td["agents_endurance"]  # [B, m]
        time_windows = td["time_window"][..., num_agents:, :]  # [B, N, 2] - customers only, exclude depot!
        
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
        
        # Use current_time directly from td (already computed in env)
        current_time_normalized = td["current_time"] / time_scaler.squeeze(-1)  # [B, M]
        
        # Compute remaining endurance normalized by time_scaler (same reference as all time features)
        remaining_endurance_norm = (td["agents_endurance"] - td["used_endurance"]) / time_scaler.squeeze(-1)  # [B, M]
        
        context_feats = torch.stack(
            [
                current_time_normalized,  # current time (normalized by time_scaler)
                (td["agents_capacity"] - td["used_capacity"])
                / demand_scaler.squeeze(-1),  # remaining capacity (normalized by demand_scaler)
                remaining_endurance_norm,  # remaining endurance (normalized by time_scaler - SAME as other time features)
            ],
            dim=-1,
        )
        if self.use_time_to_depot:
            # NOTE: We should use actual (unnormalized) coordinates for distance calculation
            # because agents_speed is in actual units (m/s), not normalized units
            # The time computation needs to match the env's physics
            depot = td["locs"][..., 0:1, :]
            cur_loc = gather_by_index(td["locs"], td["current_node"])
            dist_to_depot = torch.norm(cur_loc - depot, p=2, dim=-1, keepdim=True)
            
            # Avoid division by zero for speed (add small epsilon)
            agent_speed_safe = torch.where(
                td["agents_speed"][..., None] > 1e-6,
                td["agents_speed"][..., None],
                torch.ones_like(td["agents_speed"][..., None]) * 1e-6
            )
            time_to_depot = dist_to_depot / agent_speed_safe
            
            # Avoid division by zero for time_scaler (should not happen but be safe)
            time_scaler_safe = torch.where(
                time_scaler > 1e-6,
                time_scaler,
                torch.ones_like(time_scaler)
            )
            time_to_depot_normalized = time_to_depot / time_scaler_safe
            
            # Time to deadline = deadline - current_time (already in actual time units)
            time_to_deadline = (td["trip_deadline"] - td["current_time"])[..., None]
            time_to_deadline_normalized = time_to_deadline / time_scaler_safe
            
            # Compute effective time limit = min(remaining_endurance, time_to_deadline)
            # IMPORTANT: Compute min BEFORE normalization using actual time values (same units)
            remaining_endurance_actual = td["agents_endurance"] - td["used_endurance"]  # [B, M] actual time
            time_to_deadline_actual = (td["trip_deadline"] - td["current_time"])  # [B, M] actual time
            
            # Min of actual values (both in actual time units)
            effective_time_limit_actual = torch.minimum(
                remaining_endurance_actual,
                time_to_deadline_actual
            )  # [B, M]
            
            # Then normalize by time_scaler (SAME as all other time features)
            effective_time_limit = effective_time_limit_actual / time_scaler_safe.squeeze(-1)  # [B, M] normalized by time_scaler
            
            context_feats = torch.cat([
                context_feats, 
                time_to_depot_normalized,      # Travel time normalized by time_scaler
                time_to_deadline_normalized,   # Deadline normalized by time_scaler
                effective_time_limit[..., None]  # Effective limit normalized by time_scaler [B, M, 1]
            ], dim=-1)
        return self.proj_agent_feats(context_feats)

    def _global_state_embedding(self, embeddings, td, num_agents, num_cities):
        global_feats = torch.cat(
            [
                td["visited"][..., num_agents:].sum(-1)[..., None]
                / num_cities,  # number of visited cities / total
            ],
            dim=-1,
        )
        return self.proj_global_feats(global_feats)[..., None, :].repeat(1, num_agents, 1)
