"""Attention Injection Modules for Pointer Attention logits.

Two modes:
    - MLPAttentionInjection: MLP that mixes logits + 3 heuristic features.
    - ProjectedAdditiveInjection: Linear(3→1) bias added to logits.

Note: Pointer attention returns logits as [B, M, N] (heads already reduced).
      These modules operate on 3D tensors, NOT per-head 4D tensors.
"""

import torch
import torch.nn as nn


class MLPAttentionInjection(nn.Module):
    """MLP mixing of logits + 3 heuristic features (MatNet-style).
    
    Input: logits [B, M, N] + 3 features [B, M, N]
    Output: mixed_logits [B, M, N]
    
    MLP: Linear(4→hidden) → relu → Linear(hidden→1)
    """
    
    def __init__(self, num_heads: int = 8, hidden_dim: int = 16):
        super(MLPAttentionInjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, logits, td, time_scaler):
        """
        Args:
            logits: [B, M, N_total] pointer attention logits (post head-reduction)
            td: TensorDict with cached features
            time_scaler: [B, 1] or [B, 1, 1]
        Returns:
            mixed_logits: [B, M, N_total] modified logits
        """
        num_agents = td["current_node"].size(-1)
        N_total = logits.size(-1)
        
        if "slack_matrix" not in td.keys():
            return logits
        
        ts = time_scaler.view(-1, 1, 1) if time_scaler.dim() == 2 else time_scaler  # [B, 1, 1]
        ts = ts.clamp(min=1e-6)
        
        # 3 heuristic features [B, M, N_total]
        slack = td["slack_matrix"] / ts  # [B, M, M+N]
        
        # future_slack and reachability_loss are [B, M, N] — pad depot columns with 0
        B, M = slack.shape[0], slack.shape[1]
        future_slack_full = torch.zeros(B, M, N_total, device=logits.device)
        reachability_full = torch.zeros(B, M, N_total, device=logits.device)
        if "min_future_slack" in td.keys():
            future_slack_full[..., num_agents:] = td["min_future_slack"] / ts
        if "reachability_loss" in td.keys():
            reachability_full[..., num_agents:] = td["reachability_loss"]
        
        # Stack: [B, M, N_total, 4]
        combined = torch.stack([logits, slack, future_slack_full, reachability_full], dim=-1)
        
        # MLP: shared weights across positions
        hidden = torch.relu(self.fc1(combined))  # [B, M, N_total, hidden]
        mixed_logits = self.fc2(hidden).squeeze(-1)  # [B, M, N_total]
        
        return mixed_logits


class ProjectedAdditiveInjection(nn.Module):
    """Linear additive bias from 3 heuristic features.
    
    Input: logits [B, M, N] + 3 features [B, M, N]
    Output: logits + bias [B, M, N]
    
    bias = Linear(3→1)(features).
    """
    
    def __init__(self):
        super(ProjectedAdditiveInjection, self).__init__()
        self.proj = nn.Linear(3, 1, bias=True)
    
    def forward(self, logits, td, time_scaler):
        """
        Args:
            logits: [B, M, N_total] pointer attention logits (post head-reduction)
            td: TensorDict with cached features
            time_scaler: [B, 1] or [B, 1, 1]
        Returns:
            score: [B, M, N_total] = logits + bias
        """
        num_agents = td["current_node"].size(-1)
        N_total = logits.size(-1)
        
        if "slack_matrix" not in td.keys():
            return logits
        
        ts = time_scaler.view(-1, 1, 1) if time_scaler.dim() == 2 else time_scaler  # [B, 1, 1]
        ts = ts.clamp(min=1e-6)
        
        # 3 features [B, M, N_total]
        slack = td["slack_matrix"] / ts
        
        B, M = slack.shape[0], slack.shape[1]
        future_slack_full = torch.zeros(B, M, N_total, device=logits.device)
        reachability_full = torch.zeros(B, M, N_total, device=logits.device)
        if "min_future_slack" in td.keys():
            future_slack_full[..., num_agents:] = td["min_future_slack"] / ts
        if "reachability_loss" in td.keys():
            reachability_full[..., num_agents:] = td["reachability_loss"]
        
        # Stack: [B, M, N_total, 3]
        features = torch.stack([slack, future_slack_full, reachability_full], dim=-1)
        
        # Project: [B, M, N_total, 3] → [B, M, N_total]
        bias = self.proj(features).squeeze(-1)
        
        # Add bias to logits
        return logits + bias
