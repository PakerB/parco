import torch
from tensordict import TensorDict

def replace_key_td(td, key, replacement):
    td.pop(key)
    td[key] = replacement
    return td

def resample_batch_padding(td):
    batch_size = td.batch_size[0] if isinstance(td.batch_size, torch.Size) else td.batch_size
    
    num_real_nodes = td["num_real_nodes"]  # [B]
    max_real_nodes = num_real_nodes.max().item()
    
    if "locs" in td.keys():
        td = replace_key_td(td, "locs", td["locs"][..., :max_real_nodes, :])
        
    return td

# Create dummy td
B = 2
max_pad = 10
td = TensorDict({
    "num_real_nodes": torch.tensor([5, 8]),
    "num_real_agents": torch.tensor([2, 3]),
    "locs": torch.zeros((B, max_pad, 2))
}, batch_size=[B])

print("Before:", td["locs"].shape)
td_resampled = resample_batch_padding(td)
print("After:", td_resampled["locs"].shape)
print("Instance 0 still has padded nodes?", td_resampled["locs"].shape[1] > td["num_real_nodes"][0].item())
