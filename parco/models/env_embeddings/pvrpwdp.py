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
        capacities = td["capacity"]  # [B, m]
        speeds = td["speed"]  # [B, m]
        endurances = td["endurance"]  # [B, m]
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

        # Depots embedding with positional encoding
        depots_embedding = self.init_embed_depot(depot_locs)
        pos_embedding = self.pos_encoder(depots_embedding, add=False)
        pos_embedding = self.alpha * self.pos_embedding_proj(pos_embedding)
        depot_embedding = depots_embedding + pos_embedding

        # Agents embedding
        agents_feats = torch.cat(
            [
                agents_locs,
                td["capacity"][..., None] / demand_scaler,
                td["speed"][..., None] / speed_scaler,
                td["endurance"][..., None] / endurance_scaler,
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
        waiting_times = td["waiting_time"][num_agents:]  # [B, N] - only customer waiting times
        
        clients_feats = torch.cat(
            [
                clients_locs,                          # [B, N, 2]
                demands[..., None] / demand_scaler,    # [B, N, 1]
                earliest[..., None] / time_scaler,     # [B, N, 1]
                latest[..., None] / time_scaler,       # [B, N, 1]
                waiting_times[..., None] / time_scaler, # [B, N, 1] - freshness normalized
            ], 
            dim=-1
        )  # [B, N, 6]

        if self.use_polar_feats:
            # Convert to polar coordinates
            depot = depot_locs[..., 0:1, :]
            client_locs_centered = clients_locs - depot  # centering
            dist_to_depot = torch.norm(client_locs_centered, p=2, dim=-1, keepdim=True)
            angle_to_depot = torch.atan2(
                client_locs_centered[..., 1:], client_locs_centered[..., :1]
            )
            clients_feats = torch.cat(
                [clients_feats, dist_to_depot, angle_to_depot], dim=-1
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
        agent_feat_dim=3,  # Changed from 2 to 3: current_time, remaining_capacity, remaining_endurance
        global_feat_dim=1,
        normalize_endurance_by_max=False,
        use_time_to_depot=True,
        **kwargs,
    ):
        if use_time_to_depot:
            agent_feat_dim += 2  # time_to_depot + time_to_deadline
        super(PVRPWDPContextEmbedding, self).__init__(
            embed_dim, agent_feat_dim, global_feat_dim, **kwargs
        )
        self.normalize_endurance_by_max = normalize_endurance_by_max
        self.use_time_to_depot = use_time_to_depot

    def _agent_state_embedding(self, embeddings, td, num_agents, num_cities):
        # Compute dynamic scalers from data (same as in PVRPWDPInitEmbedding)
        demands = td["demand"][..., 0, num_agents:]  # [B, N]
        capacities = td["capacity"]  # [B, m]
        speeds = td["speed"]  # [B, m]
        endurances = td["endurance"]  # [B, m]
        time_windows = td["time_window"][..., :, :]  # [B, N, 2]
        
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
        
        context_feats = torch.stack(
            [
                current_time_normalized,  # current time (normalized)
                (td["agents_capacity"] - td["used_capacity"])
                / demand_scaler.squeeze(-1),  # remaining capacity
                (td["agents_endurance"] - td["used_endurance"])
                / endurance_scaler.squeeze(-1),  # remaining endurance (normalized by endurance_scaler)
            ],
            dim=-1,
        )
        if self.use_time_to_depot:
            depot = td["locs"][..., 0:1, :]
            cur_loc = gather_by_index(td["locs"], td["current_node"])
            dist_to_depot = torch.norm(cur_loc - depot, p=2, dim=-1, keepdim=True)
            time_to_depot = dist_to_depot / td["agents_speed"][..., None]
            time_to_depot_normalized = time_to_depot / time_scaler
            
            # Time to deadline = deadline - current_time (already in actual time units)
            time_to_deadline = (td["trip_deadline"] - td["current_time"])[..., None]
            time_to_deadline_normalized = time_to_deadline / time_scaler
            
            context_feats = torch.cat([context_feats, time_to_depot_normalized, time_to_deadline_normalized], dim=-1)
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
