"""
Convert TWdata TXT files to NPZ with epoch-safe distribution.

Requirements:
- 500K txt files in TWdata/*/
- Format: a.b.c.d.e.txt
  - a.b.c = instance code (same coords, different TW)
  - d.e = variant
- Constraint: Same instance code NOT in same epoch
- Output: 20 NPZ files × 10K instances = 200K total

Strategy:
1. Scan all txt files
2. Group by instance code (a.b.c)
3. Round-robin distribute variants to epochs
4. Convert to NPZ

Output NPZ Format (NEW V5 - Vehicle Arrays - PVRPWDP Compatible):
- depot: [B, 2] - depot coordinates
- locs: [B, N, 2] - customer coordinates (no depot)
- time_windows: [B, N, 2] - customer time windows (with 's' for PVRPWDP compatibility)
- demand: [B, N] - customer demands
- waiting_time: [B, N] - waiting time limit for each customer (freshness)
- agents_speed: [B, M] - vehicle speeds [truck_0, truck_1, ..., drone_0, drone_1, ...]
- agents_capacity: [B, M] - vehicle capacities [truck_0, truck_1, ..., drone_0, drone_1, ...]
- agents_endurance: [B, M] - vehicle endurance
  * Trucks: max_time (calculated as max(latest_i + travel_time_to_depot) - consistent with time_scaler)
  * Drones: fixed endurance from file (700s)

Where:
- B = batch size
- N = number of customers (20)
- M = total vehicles (num_trucks + num_drones)
- Vehicles ordered: all trucks first, then all drones
"""

import numpy as np
import glob
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import random

def parse_filename(filepath):
    """
    Extract instance code and variant from filename.
    
    Example: 20.35.1.1.1.txt → code='20.35.1', variant='1.1'
    """
    stem = Path(filepath).stem  # Remove .txt
    parts = stem.split('.')
    
    if len(parts) >= 5:
        code = '.'.join(parts[:3])  # a.b.c
        variant = '.'.join(parts[3:])  # d.e
    else:
        # Fallback
        code = stem
        variant = '0'
    
    return code, variant

def parse_pvrpwdp_txt(filepath):
    """
    Parse PVRPWDP instance from txt file.
    
    Expected format:
    Line 1: trucks_count N
    Line 2: drones_count M
    Line 3: customers N
    Line 4: depot x y
    Line 5: Truck_speed(m/s) ...
    Line 6: Endurance_drone_speed[m/s] ...
    Line 7: Endurance_fixed_time[s] ...
    Line 8: Waiting_time_limit(s) ...
    Line 9: truck capacity(kg) ...
    Line 10: drone capacity(kg) ...
    Line 11: Header
    Line 12-N+11: customers (x y dronable demand tw_start tw_end)
    
    Returns:
        dict with NEW V5 format (PVRPWDP compatible):
        - depot: [2] - depot coordinates
        - locs: [N, 2] - customer coordinates (no depot)
        - demand: [N] - customer demands (no depot)
        - time_windows: [N, 2] - customer time windows (no depot) - with 's'
        - waiting_time: [N] - waiting time limit (freshness) for each customer
        - agents_speed: [M] - vehicle speeds [trucks..., drones...]
        - agents_capacity: [M] - vehicle capacities [trucks..., drones...]
        - agents_endurance: [M] - vehicle endurance
          * Trucks: max_time (calculated as max(latest_i + travel_time_to_depot) - consistent with time_scaler)
          * Drones: fixed endurance from file (700s)
        
        Where M = num_trucks + num_drones
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header (new format)
        num_trucks = int(lines[0].strip().split()[1])
        num_drones = int(lines[1].strip().split()[1])
        N = int(lines[2].strip().split()[1])
        
        # Parse speeds and times
        truck_speed = float(lines[4].strip().split()[1])  # m/s
        drone_speed = float(lines[5].strip().split()[1])  # m/s
        drone_endurance = float(lines[6].strip().split()[1])  # seconds
        freshness = float(lines[7].strip().split()[1])  # seconds (waiting time limit)
        
        # Parse capacities
        truck_capacity_str = lines[8].strip().split()[2]  # "truck capacity(kg) 200"
        drone_capacity_str = lines[9].strip().split()[2]  # "drone capacity(kg) 2,27"
        
        truck_capacity = float(truck_capacity_str)
        # Handle comma as decimal separator for drone capacity
        drone_capacity = float(drone_capacity_str.replace(',', '.'))
        
        # Parse depot (line 3: "depot x y")
        depot_parts = lines[3].strip().split()
        depot_x, depot_y = float(depot_parts[1]), float(depot_parts[2])
        depot = [depot_x, depot_y]
        
        # Parse customers only (depot stored separately)
        locs = []  # Customers only - NO depot
        demand = []  # Customers only - NO depot
        time_windows = []  # Customers only - NO depot (with 's' for PVRPWDP)
        
        # Parse customers (lines 11 onwards, skip header at line 10)
        for i in range(11, 11 + N):  # N customers starting at line 11 (0-indexed)
            parts = lines[i].strip().split()
            x, y = float(parts[0]), float(parts[1])
            d = float(parts[3])
            tw_start, tw_end = float(parts[4]), float(parts[5])
            
            locs.append([x, y])
            demand.append(d)
            time_windows.append([tw_start, tw_end])
        
        # Build vehicle arrays (M = num_trucks + num_drones)
        # Order: [truck_0, truck_1, ..., drone_0, drone_1, ...]
        M = num_trucks + num_drones
        
        # Speed array: all trucks same speed, all drones same speed
        agents_speed = [truck_speed] * num_trucks + [drone_speed] * num_drones
        
        # Capacity array: all trucks same capacity, all drones same capacity
        agents_capacity = [truck_capacity] * num_trucks + [drone_capacity] * num_drones
        
        # Endurance array calculation:
        # - Drones: fixed endurance from file (700s)
        # - Trucks: NO endurance limit (can operate indefinitely)
        #   Set to max_time (calculated same way as env.py for consistency)
        
        # Calculate max_time = max(latest_i + travel_time_i_to_depot)
        # This matches the time_scaler calculation in env.py
        min_speed = min(truck_speed, drone_speed)
        max_time = 0.0
        for loc, tw in zip(locs, time_windows):
            x, y = loc[0], loc[1]
            distance_to_depot = np.sqrt((x - depot_x)**2 + (y - depot_y)**2)
            travel_time = distance_to_depot / min_speed
            latest = tw[1]  # latest time window
            time_with_return = latest + travel_time
            max_time = max(max_time, time_with_return)
        
        # Trucks have no endurance constraint - use max_time
        truck_endurance = max_time  # No battery limit (consistent with time_scaler)
        
        # Build endurance array
        agents_endurance = [truck_endurance] * num_trucks + [drone_endurance] * num_drones
        
        # Waiting time array: same freshness for all customers
        waiting_time = [freshness] * N
        
        return {
            'depot': np.array(depot, dtype=np.float32),  # [2]
            'locs': np.array(locs, dtype=np.float32),  # [N, 2]
            'demand': np.array(demand, dtype=np.float32),  # [N]
            'time_windows': np.array(time_windows, dtype=np.float32),  # [N, 2] - with 's'
            'waiting_time': np.array(waiting_time, dtype=np.float32),  # [N]
            'agents_speed': np.array(agents_speed, dtype=np.float32),  # [M]
            'agents_capacity': np.array(agents_capacity, dtype=np.float32),  # [M]
            'agents_endurance': np.array(agents_endurance, dtype=np.float32),  # [M]
            'num_agents': np.array(M, dtype=np.int64),  # total vehicles (trucks + drones) for this instance
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def collect_and_group_files(data_dir):
    """
    Collect all txt files and group by instance code.
    
    Returns:
        dict: {instance_code: [filepath1, filepath2, ...]}
    """
    print("Collecting files...")
    
    # Find all txt files
    pattern = f"{data_dir}/**/*.txt"
    all_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(all_files)} txt files")
    
    # Group by instance code
    groups = defaultdict(list)
    
    for filepath in tqdm(all_files, desc="Grouping"):
        code, variant = parse_filename(filepath)
        groups[code].append(filepath)
    
    print(f"Found {len(groups)} unique instance codes")
    
    if len(groups) == 0:
        print("❌ ERROR: No valid files found! Check data_dir path.")
        return groups
    
    print(f"Average variants per code: {len(all_files) / len(groups):.1f}")
    
    return groups

def distribute_to_epochs(groups, num_epochs=20, instances_per_epoch=10000):
    """
    Distribute file groups to epochs ensuring no code collision.
    
    Strategy:
    - Round-robin assign variants of same code to different epochs
    - Shuffle within epochs for randomness
    
    Returns:
        list of lists: [epoch0_files, epoch1_files, ...]
    """
    print(f"\nDistributing to {num_epochs} epochs...")
    
    epochs = [[] for _ in range(num_epochs)]
    
    # Convert to list for shuffling
    codes = list(groups.keys())
    random.shuffle(codes)
    
    # Round-robin assign variants
    for code in tqdm(codes, desc="Distributing"):
        variants = groups[code]
        random.shuffle(variants)
        
        # Assign each variant to different epoch
        for i, filepath in enumerate(variants):
            epoch_idx = i % num_epochs
            epochs[epoch_idx].append(filepath)
    
    # Trim to desired size and shuffle
    for i in range(num_epochs):
        random.shuffle(epochs[i])
        epochs[i] = epochs[i][:instances_per_epoch]
    
    # Print stats
    for i, epoch_files in enumerate(epochs):
        print(f"Epoch {i}: {len(epoch_files)} instances")
    
    return epochs

def convert_epoch_to_npz(epoch_files, output_path):
    """
    Convert list of txt files to single NPZ with NEW V5 format.
    """
    print(f"\nConverting {len(epoch_files)} files to {output_path}...")
    
    instances = []
    
    for filepath in tqdm(epoch_files, desc="Parsing"):
        inst = parse_pvrpwdp_txt(filepath)
        if inst is not None:
            instances.append(inst)
    
    if len(instances) == 0:
        print("No valid instances!")
        return
    
    # Stack arrays - NEW V5 format (PVRPWDP compatible)
    data = {
        'depot': np.stack([inst['depot'] for inst in instances]),  # [B, 2]
        'locs': np.stack([inst['locs'] for inst in instances]),  # [B, N, 2]
        'demand': np.stack([inst['demand'] for inst in instances]),  # [B, N]
        'time_windows': np.stack([inst['time_windows'] for inst in instances]),  # [B, N, 2] - with 's'
        'waiting_time': np.stack([inst['waiting_time'] for inst in instances]),  # [B, N]
        'agents_speed': np.stack([inst['agents_speed'] for inst in instances]),  # [B, M]
        'agents_capacity': np.stack([inst['agents_capacity'] for inst in instances]),  # [B, M]
        'agents_endurance': np.stack([inst['agents_endurance'] for inst in instances]),  # [B, M]
    'num_agents': np.stack([inst['num_agents'] for inst in instances]),  # [B]
    }
    
    # Save
    np.savez_compressed(output_path, **data)
    
    # Print stats
    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Saved {len(instances)} instances ({file_size_mb:.1f} MB)")
    print(f"\n📦 NPZ Structure (V5 PVRPWDP Compatible Format):")
    print(f"  depot:            {data['depot'].shape} - depot coordinates [B, 2]")
    print(f"  locs:             {data['locs'].shape} - customer coordinates [B, N, 2]")
    print(f"  demand:           {data['demand'].shape} - customer demands [B, N]")
    print(f"  time_windows:     {data['time_windows'].shape} - customer time windows [B, N, 2]")
    print(f"  waiting_time:     {data['waiting_time'].shape} - waiting time limits [B, N]")
    print(f"  agents_speed:     {data['agents_speed'].shape} - vehicle speeds [B, M]")
    print(f"  agents_capacity:  {data['agents_capacity'].shape} - vehicle capacities [B, M]")
    print(f"  agents_endurance: {data['agents_endurance'].shape} - vehicle endurance [B, M]")
    print(f"  num_agents:       {data['num_agents'].shape} - number of vehicles per instance [B]")
    print(f"  ✅ Format: V5 PVRPWDP compatible (M = trucks + drones)")
    
    # Show sample for first instance
    if len(instances) > 0:
        M = data['agents_speed'].shape[1]
        # Count trucks by checking if endurance > 1000 (drones are typically 700s)
        # More robust: check if endurance is close to drone_endurance (700)
        drone_endurance_threshold = 800  # anything below this is likely a drone
        num_drones = int(np.sum(data['agents_endurance'][0] < drone_endurance_threshold))
        num_trucks = M - num_drones
        print(f"\n  Sample instance 0:")
        print(f"    Fleet: {num_trucks} trucks + {num_drones} drones = {M} vehicles")
        print(f"    Speed: {data['agents_speed'][0]}")
        print(f"    Capacity: {data['agents_capacity'][0]}")
        print(f"    Endurance: {data['agents_endurance'][0]}")

def main():
    """
    Main conversion pipeline.
    """
    # Config
    data_dir = 'TWdata'  # Fixed: correct relative path from train_data/
    output_dir = 'train_data_npz'  # Output folder
    num_epochs = 20  # 20 epochs for full dataset
    instances_per_epoch = 2000  # 10K instances per epoch
    
    # Create output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Collect and group files
    groups = collect_and_group_files(data_dir)
    
    # Step 2: Distribute to epochs
    epochs = distribute_to_epochs(groups, num_epochs, instances_per_epoch)
    
    # Step 3: Convert each epoch to NPZ
    for i, epoch_files in enumerate(epochs):
        output_path = f"{output_dir}/pvrpwdp_epoch_{i:02d}.npz"
        convert_epoch_to_npz(epoch_files, output_path)
    
    print("\n✅ Conversion complete!")
    print(f"Output: {output_dir}/pvrpwdp_epoch_*.npz")
    print(f"Total: {num_epochs} files × {instances_per_epoch} instances")


if __name__ == '__main__':
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    main()
