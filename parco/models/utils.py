import torch

from rl4co.utils.ops import gather_by_index


def replace_key_td(td, key, replacement):
    # TODO: check if best way in TensorDict?
    td.pop(key)
    td[key] = replacement
    return td


def resample_batch_padding(td):
    """
    Resample batch in PADDING MODE: remove virtual nodes/agents based on metadata.
    
    This function is used when data is pre-padded to fixed size (e.g., [B, 100, 2] for locs)
    with metadata tracking the real size. It removes virtual padding per instance.
    
    Metadata fields expected:
        - num_real_nodes: [B] - number of real customers (others are virtual, at [0,0])
        - num_real_agents: [B] - number of real agents (others are virtual, speed=0)
    
    Args:
        td: TensorDict with padded data and metadata
        
    Returns:
        td: TensorDict with virtual nodes/agents removed per instance (format cũ - compact)
    
    Note:
        After resample, batch has OLD format (compact):
        - locs: [B, max_real_nodes, 2] (no padding)
        - demand: [B, max_real_nodes] (no padding)
        - agents_speed: [B, max_real_agents] (no padding)
        All instances in batch are trimmed to max real size in batch.
    """
    batch_size = td.batch_size[0] if isinstance(td.batch_size, torch.Size) else td.batch_size
    device = td.device
    
    # Check if we have metadata
    if "num_real_nodes" not in td.keys() or "num_real_agents" not in td.keys():
        print("⚠️  Padding mode metadata not found (num_real_nodes, num_real_agents). Returning batch as-is.")
        return td
    
    # Extract metadata [B] - number of real nodes/agents per instance
    num_real_nodes = td["num_real_nodes"]  # [B]
    num_real_agents = td["num_real_agents"]  # [B]
    
    # Get max real values in batch for slicing
    # Each instance may have different num_real_nodes, so we trim ALL to batch max
    # This ensures uniform batch shape [B, max_real_nodes, ...] = OLD FORMAT
    max_real_nodes = num_real_nodes.max().item()
    max_real_agents = num_real_agents.max().item()
    
    # ============ LOCATION DATA (slice by max_real_nodes) ============
    if "locs" in td.keys():
        td = replace_key_td(td, "locs", td["locs"][..., :max_real_nodes, :])
    
    # ============ CUSTOMER-SPECIFIC ATTRIBUTES (slice by max_real_nodes) ============
    if "demand" in td.keys():
        td = replace_key_td(td, "demand", td["demand"][..., :max_real_nodes])
    
    if "time_windows" in td.keys():
        td = replace_key_td(td, "time_windows", td["time_windows"][..., :max_real_nodes, :])
    
    if "waiting_time" in td.keys():
        td = replace_key_td(td, "waiting_time", td["waiting_time"][..., :max_real_nodes])
    
    # ============ AGENT-SPECIFIC ATTRIBUTES (slice by max_real_agents) ============
    # PVRPWDP
    if "agents_capacity" in td.keys():
        td = replace_key_td(td, "agents_capacity", td["agents_capacity"][..., :max_real_agents])
    
    if "agents_speed" in td.keys():
        td = replace_key_td(td, "agents_speed", td["agents_speed"][..., :max_real_agents])
    
    if "agents_endurance" in td.keys():
        td = replace_key_td(td, "agents_endurance", td["agents_endurance"][..., :max_real_agents])
    
    # HCVRP / OMDCPDP
    if "capacity" in td.keys():
        td = replace_key_td(td, "capacity", td["capacity"][..., :max_real_agents])
    
    if "speed" in td.keys():
        td = replace_key_td(td, "speed", td["speed"][..., :max_real_agents])
    
    # OMDCPDP specific
    if "pickup_et" in td.keys():
        td = replace_key_td(td, "pickup_et", td["pickup_et"][..., : max_real_nodes // 2])
    
    if "delivery_et" in td.keys():
        td = replace_key_td(td, "delivery_et", td["delivery_et"][..., : max_real_nodes // 2])
    
    # Note: num_agents is not needed for env._reset() which calculates it from agents_speed.size(-1)
    # But we keep the num_real_nodes and num_real_agents metadata for debugging/inspection
    
    # TensorDict batch_size is now: [B, max_real_nodes, ...]
    # This matches the OLD format (compact, no padding)
    
    return td


def resample_batch(td, num_agents, num_locs):
    # Remove depots until num_agents
    td.set_("num_agents", torch.full((*td.batch_size,), num_agents, device=td.device))
    if "depots" in td.keys():
        # note that if we have "depot" instead, this will automatically
        # be repeated inside the environment
        td = replace_key_td(td, "depots", td["depots"][..., :num_agents, :])

    if "pickup_et" in td.keys():
        # Ensure num_locs is even for omdcpdp
        num_locs = num_locs - 1 if num_locs % 2 == 0 else num_locs
        # also, set the "num_agents" key to the new number of agents
        td.set_("num_agents", torch.full((*td.batch_size,), num_agents, device=td.device))

    # ============ LOCATION DATA (slice by num_locs) ============
    td = replace_key_td(td, "locs", td["locs"][..., :num_locs, :])

    # For early time windows (OMDCPDP)
    if "pickup_et" in td.keys():
        td = replace_key_td(td, "pickup_et", td["pickup_et"][..., : num_locs // 2])
    if "delivery_et" in td.keys():
        td = replace_key_td(td, "delivery_et", td["delivery_et"][..., : num_locs // 2])

    # ============ AGENT-SPECIFIC ATTRIBUTES (slice by num_agents) ============
    # HCVRP / OMDCPDP
    if "capacity" in td.keys():
        td = replace_key_td(td, "capacity", td["capacity"][..., :num_agents])

    if "speed" in td.keys():
        td = replace_key_td(td, "speed", td["speed"][..., :num_agents])

    # PVRPWDP
    if "agents_capacity" in td.keys():
        td = replace_key_td(td, "agents_capacity", td["agents_capacity"][..., :num_agents])
    
    if "agents_speed" in td.keys():
        td = replace_key_td(td, "agents_speed", td["agents_speed"][..., :num_agents])
    
    if "agents_endurance" in td.keys():
        td = replace_key_td(td, "agents_endurance", td["agents_endurance"][..., :num_agents])

    # ============ CUSTOMER-SPECIFIC ATTRIBUTES (slice by num_locs) ============
    if "demand" in td.keys():
        td = replace_key_td(td, "demand", td["demand"][..., :num_locs])

    # Time windows
    if "time_windows" in td.keys():
        td = replace_key_td(td, "time_windows", td["time_windows"][..., :num_locs, :])

    # Waiting time (PVRPWDP) - only customers (depots added in env._reset)
    if "waiting_time" in td.keys():
        td = replace_key_td(td, "waiting_time", td["waiting_time"][..., :num_locs])
    
    # Pickup/Delivery time windows (OMDCPDP)
    if "pickup_et" in td.keys():
        td = replace_key_td(td, "pickup_et", td["pickup_et"][..., :num_locs // 2])
    
    if "delivery_et" in td.keys():
        td = replace_key_td(td, "delivery_et", td["delivery_et"][..., :num_locs // 2])

    return td


def get_log_likelihood(log_p, actions=None, mask=None, return_sum: bool = False):
    """Get log likelihood of selected actions

    Args:
        log_p: [batch, n_agents, (decode_len), n_nodes]
        actions: [batch, n_agents, (decode_len)]
        mask: [batch, n_agents, (decode_len)]
    """

    # NOTE: we do not use this since it is more inefficient, we do it in the decoder
    if actions is not None:
        if log_p.dim() > 3:
            log_p = gather_by_index(log_p, actions, dim=-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        log_p[mask] = 0

    assert (
        log_p > -1000
    ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    # TODO: check the return sum argument.
    # TODO: Also, should we sum over agents too?
    if return_sum:
        return log_p.sum(-1)  # [batch, num_agents]
    else:
        return log_p  # [batch, num_agents, (decode_len)]
