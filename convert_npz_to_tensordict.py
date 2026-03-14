"""
Convert legacy npz files to TensorDict-compatible format for PVRPWDP environment.

This script reads npz files with raw numpy arrays and converts them to the format
expected by the PVRPWDP environment, matching the generator output structure.
"""

import os
import numpy as np
import torch
from tensordict import TensorDict
from rl4co.data.utils import save_tensordict_to_npz

def convert_npz_to_pvrpwdp_format(input_file, output_file):
    """Convert a legacy npz file to PVRPWDP TensorDict format.
    
    Args:
        input_file: Path to input .npz file with raw arrays
        output_file: Path to output .npz file in TensorDict format
    """
    print(f"Loading {input_file}...")
    data = np.load(input_file, allow_pickle=True)
    
    # Extract data
    depot = data['depot']  # [B, 2]
    locs = data['locs']    # [B, N, 2]
    demand = data['demand']  # [B, N]
    time_windows = data['time_windows']  # [B, N, 2]
    freshness = data['freshness']  # [B, N] (waiting_time)
    
    # Vehicle data
    num_trucks = int(data['num_trucks'])
    num_drones = int(data['num_drones'])
    truck_capacity = float(data['truck_capacity'])
    drone_capacity = float(data['drone_capacity'])
    truck_speed = float(data['truck_speed'])
    drone_speed = float(data['drone_speed'])
    endurance = float(data['endurance'])
    
    batch_size = depot.shape[0]
    num_loc = locs.shape[1]
    num_agents = num_trucks + num_drones
    
    print(f"  Batch size: {batch_size}")
    print(f"  Num locations: {num_loc}")
    print(f"  Num agents: {num_agents} ({num_trucks} trucks + {num_drones} drones)")
    
    # Create TensorDict in PVRPWDP generator format
    td = TensorDict({
        # Location data
        'depot': torch.from_numpy(depot).float(),  # [B, 2]
        'locs': torch.from_numpy(locs).float(),    # [B, N, 2]
        
        # Customer attributes
        'demand': torch.from_numpy(demand).float(),  # [B, N]
        'time_windows': torch.from_numpy(time_windows).float(),  # [B, N, 2]
        'waiting_time': torch.from_numpy(freshness).float(),  # [B, N]
        
        # Agent attributes
        'agents_capacity': torch.tensor([truck_capacity] * num_trucks + [drone_capacity] * num_drones).float().unsqueeze(0).repeat(batch_size, 1),  # [B, M]
        'agents_speed': torch.tensor([truck_speed] * num_trucks + [drone_speed] * num_drones).float().unsqueeze(0).repeat(batch_size, 1),  # [B, M]
        'agents_endurance': torch.tensor([float('inf')] * num_trucks + [endurance] * num_drones).float().unsqueeze(0).repeat(batch_size, 1),  # [B, M]
        
    }, batch_size=[batch_size])
    
    # Save in TensorDict format
    print(f"Saving to {output_file}...")
    save_tensordict_to_npz(td, output_file)
    print(f"✅ Converted successfully!\n")
    
    return td


def convert_directory(input_dir, output_dir):
    """Convert all npz files in a directory.
    
    Args:
        input_dir: Directory containing input .npz files
        output_dir: Directory to save converted files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    npz_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npz')])
    
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} files to convert")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}\n")
    
    for filename in npz_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            convert_npz_to_pvrpwdp_format(input_path, output_path)
        except Exception as e:
            print(f"❌ Error converting {filename}: {e}\n")
    
    print(f"\n✅ Conversion complete! Converted {len(npz_files)} files.")


if __name__ == "__main__":
    import sys
    
    # Default paths
    train_input = "train_data_npz"
    train_output = "data/train_data_npz"
    
    val_input = "val_data/val.npz"
    val_output = "data/val_data/val.npz"
    
    test_input = "test_data/test.npz"
    test_output = "data/test_data/test.npz"
    
    print("=" * 80)
    print("PVRPWDP NPZ Format Converter")
    print("=" * 80)
    print()
    
    # Convert training data
    if os.path.exists(train_input):
        print("Converting training data...")
        convert_directory(train_input, train_output)
        print()
    
    # Convert validation data
    if os.path.exists(val_input):
        print("Converting validation data...")
        os.makedirs(os.path.dirname(val_output), exist_ok=True)
        convert_npz_to_pvrpwdp_format(val_input, val_output)
        print()
    
    # Convert test data
    if os.path.exists(test_input):
        print("Converting test data...")
        os.makedirs(os.path.dirname(test_output), exist_ok=True)
        convert_npz_to_pvrpwdp_format(test_input, test_output)
        print()
    
    print("=" * 80)
    print("✅ All conversions complete!")
    print("=" * 80)
