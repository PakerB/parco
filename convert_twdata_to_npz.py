"""
Convert JSON files to NPZ with random distribution.

Features:
- Stream-load JSON files (one at a time to save RAM)
- Extract 50 versions per JSON (10 solutions × 5 offsets)
- Configurable cutoff: MAX_CUSTOMERS, MAX_TRUCKS, MAX_DRONES
- Random vehicle permutation per version (for training diversity)
- Hash-based distribution to NPZ files (random + deterministic)
- Configurable NPZ size: 100K, 200K, 500K instances/file

Output NPZ Format (V5 - PVRPWDP Compatible):
- depot: [B, 2] - depot coordinates
- locs: [B, N, 2] - customer coordinates (no depot, max N = MAX_CUSTOMERS)
- time_windows: [B, N, 2] - customer time windows
- demand: [B, N] - customer demands
- waiting_time: [B, N] - freshness limits
- agents_speed: [B, M] - vehicle speeds (M = MAX_TRUCKS + MAX_DRONES, random order)
- agents_capacity: [B, M] - vehicle capacities (random order)
- agents_endurance: [B, M] - vehicle endurance (random order)

Where:
- B = INSTANCES_PER_NPZ (100K, 200K, or 500K)
- N = MAX_CUSTOMERS
- M = MAX_TRUCKS + MAX_DRONES
- Vehicles are RANDOMLY PERMUTED per version for training diversity
"""

import numpy as np
import json
import hashlib
from pathlib import Path
from tqdm import tqdm
import random
from datetime import datetime

def get_seed_from_month():
    """
    Get random seed from seconds since month start.
    
    Example: 26/3/2026 14:30:45 → ~2,160,000+ seconds since 1/3/2026
    """
    now = datetime.now()
    month_start = datetime(now.year, now.month, 1)
    seconds_since_month_start = int((now - month_start).total_seconds())
    return seconds_since_month_start


def parse_json_instance(filepath):
    """
    Parse JSON file and extract basic info.
    
    Returns:
        dict with 'metadata' and 'customers' and 'solutions'
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def extract_pvrpwdp_version(json_data, solution_idx, offset_idx, 
                            max_customers, max_trucks, max_drones,
                            random_permutation=True):
    """
    Extract one version from JSON with random vehicle permutation.
    
    Args:
        json_data: parsed JSON data
        solution_idx: 1-10
        offset_idx: 1-5
        max_customers: cutoff for customers
        max_trucks: cutoff for trucks
        max_drones: cutoff for drones
        random_permutation: if True, shuffle vehicles randomly
    
    Returns:
        dict with PVRPWDP instance data (one version)
    """
    try:
        metadata = json_data['metadata']
        customers = json_data['customers']
        solutions = json_data['solutions']
        
        # Get this solution
        sol = solutions[solution_idx - 1]  # 0-indexed
        offset = sol['offsets'][str(offset_idx)]  # offsets are keyed as strings "1", "2", etc
        
        # Parse metadata
        num_trucks = metadata['trucks']
        num_drones = metadata['drones']
        depot = metadata['depot']  # [x, y]
        truck_speed = metadata['truck_speed']
        drone_speed = metadata['drone_speed']
        
        # ✅ Cutoff trucks and drones (take min of actual and max)
        num_trucks = min(num_trucks, max_trucks)
        num_drones = min(num_drones, max_drones)
        M = num_trucks + num_drones
        
        # Parse customers (cutoff to max_customers)
        N = min(len(customers), max_customers)
        
        locs = []
        demand = []
        time_windows = []
        
        for i in range(N):
            customer = customers[i]
            locs.append([customer['x'], customer['y']])
            demand.append(customer['demand'])
            
            # Time window from offset
            cid = str(customer['id'])
            if cid in offset['tw']:
                tw_values = offset['tw'][cid]
                if len(tw_values) == 2:
                    time_windows.append([tw_values[0], tw_values[1]])
                elif len(tw_values) == 1:
                    # Only one value → use as deadline
                    time_windows.append([0, tw_values[0]])
                else:
                    # Empty or invalid → use default
                    time_windows.append([0, 2475])
            else:
                time_windows.append([0, 2475])  # default
        
        # ✅ Build vehicle arrays with RANDOM PERMUTATION and PADDING
        # Create vehicle list: trucks + drones, padding to max_trucks + max_drones
        trucks_list = ['truck'] * num_trucks + ['truck'] * (max_trucks - num_trucks)  # Pad trucks
        drones_list = ['drone'] * num_drones + ['drone'] * (max_drones - num_drones)  # Pad drones
        vehicles = trucks_list + drones_list  # Combine
        
        if random_permutation:
            random.shuffle(vehicles)  # ✅ Random permutation per version
        
        # Build agents arrays based on permuted order
        agents_speed = []
        agents_capacity = []
        agents_endurance = []
        
        # Standard values (from TWdata format)
        truck_capacity = 200.0  # example
        drone_capacity = 2.27   # example
        drone_endurance = 700.0  # seconds
        
        # Calculate truck endurance (max_time for time_scaler consistency)
        min_speed = min(truck_speed, drone_speed)
        max_time = 0.0
        for loc, tw in zip(locs, time_windows):
            x, y = loc[0], loc[1]
            distance_to_depot = np.sqrt((x - depot[0])**2 + (y - depot[1])**2)
            travel_time = distance_to_depot / min_speed
            latest = tw[1]
            time_with_return = latest + travel_time
            max_time = max(max_time, time_with_return)
        
        truck_endurance = max_time
        
        # Fill agents arrays in permuted vehicle order (including padded vehicles)
        for vehicle in vehicles:
            if vehicle == 'truck':
                agents_speed.append(truck_speed)
                agents_capacity.append(truck_capacity)
                agents_endurance.append(truck_endurance)
            else:  # drone
                agents_speed.append(drone_speed)
                agents_capacity.append(drone_capacity)
                agents_endurance.append(drone_endurance)
        
        # ✅ Verify all agents arrays have exactly max_trucks + max_drones elements
        expected_M = max_trucks + max_drones
        assert len(agents_speed) == expected_M, f"agents_speed size {len(agents_speed)} != {expected_M}"
        assert len(agents_capacity) == expected_M, f"agents_capacity size {len(agents_capacity)} != {expected_M}"
        assert len(agents_endurance) == expected_M, f"agents_endurance size {len(agents_endurance)} != {expected_M}"
        
        # Freshness (waiting_time)
        waiting_time = [2475.0] * N  # default freshness
        
        return {
            'depot': np.array(depot, dtype=np.float32),  # [2]
            'locs': np.array(locs, dtype=np.float32),  # [N, 2]
            'demand': np.array(demand, dtype=np.float32),  # [N]
            'time_windows': np.array(time_windows, dtype=np.float32),  # [N, 2]
            'waiting_time': np.array(waiting_time, dtype=np.float32),  # [N]
            'agents_speed': np.array(agents_speed, dtype=np.float32),  # [M]
            'agents_capacity': np.array(agents_capacity, dtype=np.float32),  # [M]
            'agents_endurance': np.array(agents_endurance, dtype=np.float32),  # [M]
        }
    
    except Exception as e:
        print(f"Error extracting version (sol={solution_idx}, offset={offset_idx}): {e}")
        import traceback
        traceback.print_exc()
        return None

def collect_json_files(json_folder):
    """
    Scan and collect all JSON files from folder.
    
    Returns:
        sorted list of JSON file paths
    """
    print(f"📂 Scanning folder: {json_folder}")
    
    json_folder = Path(json_folder)
    json_files = sorted(json_folder.glob("*.json"))
    
    print(f"✅ Found {len(json_files)} JSON files")
    
    if len(json_files) == 0:
        print("❌ ERROR: No JSON files found! Check folder path.")
        return []
    
    return json_files


def get_npz_index_for_version(instance_idx, version_idx, num_npzs):
    """
    Hash-based distribution: random + deterministic.
    
    Args:
        instance_idx: 0 to num_instances-1
        version_idx: 0 to 49 (50 versions per instance)
        num_npzs: number of NPZ files
    
    Return:
        NPZ file index (0 to num_npzs-1)
    """
    key = f"{instance_idx}_{version_idx}"
    hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return hash_val % num_npzs

def convert_json_to_npz(json_files, output_dir, instances_per_npz,
                       max_customers, max_trucks, max_drones,
                       output_prefix="pvrpwdp_batch",
                       random_permutation=True, seed=None):
    """
    Convert JSON files to NPZ with streaming and sequential distribution.
    
    Args:
        json_files: list of JSON file paths
        output_dir: directory to save NPZ files
        instances_per_npz: size of each NPZ file (100K, 200K, 500K)
        max_customers: cutoff for customers per instance
        max_trucks: cutoff for trucks per instance
        max_drones: cutoff for drones per instance
        output_prefix: prefix for output file names (default: "pvrpwdp_batch")
        random_permutation: if True, shuffle vehicles per version
        seed: random seed (if None, use month-based seed)
    """
    
    if seed is None:
        seed = get_seed_from_month()
    
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"\n🌱 Random seed: {seed}")
    print(f"📦 Config: instances_per_npz={instances_per_npz}")
    print(f"📦 Config: MAX_CUSTOMERS={max_customers}, MAX_TRUCKS={max_trucks}, MAX_DRONES={max_drones}")
    print(f"📦 Config: output_prefix={output_prefix}")
    print(f"📦 Config: random_permutation={random_permutation}")
    
    # Create output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate number of NPZ files needed
    total_instances = len(json_files) * 50  # 50 versions per JSON
    num_npzs = total_instances // instances_per_npz
    
    print(f"\n📊 Statistics:")
    print(f"   JSON files: {len(json_files)}")
    print(f"   Total instances: {total_instances}")
    print(f"   Instances per NPZ: {instances_per_npz}")
    print(f"   NPZ files to create: {num_npzs}")
    
    # Initialize buffers for each NPZ file
    npz_buffers = {i: [] for i in range(num_npzs)}
    
    # Stream process each JSON file
    print(f"\n🔄 Converting JSON files...")
    
    instance_counter = 0
    global_instance_idx = 0  # ✅ Global counter for sequential distribution
    
    for json_idx, json_file in enumerate(tqdm(json_files, desc="Processing JSON files")):
        # ✅ Stream-load one JSON file at a time
        json_data = parse_json_instance(json_file)
        
        if json_data is None:
            continue
        
        # Extract 50 versions (10 solutions × 5 offsets)
        for sol_idx in range(1, 11):      # 1-10
            for offset_idx in range(1, 6):  # 1-5
                
                # ✅ Extract version with random permutation
                instance = extract_pvrpwdp_version(
                    json_data, sol_idx, offset_idx,
                    max_customers, max_trucks, max_drones,
                    random_permutation=random_permutation
                )
                
                if instance is None:
                    continue
                
                # ✅ SEQUENTIAL distribution: round-robin to fill each NPZ exactly
                npz_idx = global_instance_idx // instances_per_npz
                
                # Stop if we have enough NPZ files
                if npz_idx >= num_npzs:
                    break
                
                npz_buffers[npz_idx].append(instance)
                global_instance_idx += 1
                instance_counter += 1
            
            if global_instance_idx // instances_per_npz >= num_npzs:
                break
        
        if global_instance_idx // instances_per_npz >= num_npzs:
            break
        
        # ✅ Unload JSON to save RAM
        del json_data
    
    # ✅ Save all NPZ files
    print(f"\n💾 Saving NPZ files...")
    
    for npz_idx in tqdm(range(num_npzs), desc="Saving NPZ files"):
        if len(npz_buffers[npz_idx]) == 0:
            continue
        
        # ✅ Verify size
        actual_size = len(npz_buffers[npz_idx])
        if actual_size != instances_per_npz:
            print(f"   ⚠️  File {npz_idx}: {actual_size} instances (expected {instances_per_npz})")
        
        # Stack instances
        instances = npz_buffers[npz_idx]
        
        data = {
            'depot': np.stack([inst['depot'] for inst in instances]),  # [B, 2]
            'locs': np.stack([inst['locs'] for inst in instances]),    # [B, N, 2]
            'demand': np.stack([inst['demand'] for inst in instances]),  # [B, N]
            'time_windows': np.stack([inst['time_windows'] for inst in instances]),  # [B, N, 2]
            'waiting_time': np.stack([inst['waiting_time'] for inst in instances]),  # [B, N]
            'agents_speed': np.stack([inst['agents_speed'] for inst in instances]),  # [B, M]
            'agents_capacity': np.stack([inst['agents_capacity'] for inst in instances]),  # [B, M]
            'agents_endurance': np.stack([inst['agents_endurance'] for inst in instances]),  # [B, M]
        }
        
        # Save NPZ with custom prefix
        output_path = f"{output_dir}/{output_prefix}_{npz_idx:02d}.npz"
        np.savez_compressed(output_path, **data)  # type: ignore
        
        file_size_mb = Path(output_path).stat().st_size / 1024 / 1024
        print(f"   ✅ {output_path}: {len(instances):,} instances ({file_size_mb:.1f} MB)")
        
        # Print sample
        if npz_idx == 0:
            M = data['agents_speed'].shape[1]
            N = data['locs'].shape[1]
            print(f"\n📋 Sample instance structure:")
            print(f"   depot: {data['depot'][0]}")
            print(f"   locs: {data['locs'][0, :3, :]} (first 3 customers)")
            print(f"   demand: {data['demand'][0, :3]}")
            print(f"   time_windows: {data['time_windows'][0, :3, :]}")
            print(f"   agents_speed: {data['agents_speed'][0]}")
            print(f"   agents_capacity: {data['agents_capacity'][0]}")
            print(f"   agents_endurance: {data['agents_endurance'][0]}")
            print(f"   Shape summary: N={N} customers, M={M} vehicles")
    
    print(f"\n✅ Conversion complete!")
    print(f"📂 Output: {output_dir}/{output_prefix}_*.npz")



def main():
    """
    Main conversion pipeline: JSON → NPZ
    """
    # ✅ CONFIGURATION (User can modify these)
    JSON_FOLDER = "instances_json"  # Folder containing JSON files (change path here!)
    OUTPUT_DIR = "train_data_npz"   # Output folder for NPZ files
    OUTPUT_PREFIX = "pvrpwdp_batch" # ✅ Prefix for output file names
    
    INSTANCES_PER_NPZ = 100_000      # NPZ file size (options: 100_000, 200_000, 500_000)
    MAX_CUSTOMERS = 100              # Cutoff for customers per instance
    MAX_TRUCKS = 10                  # Cutoff for trucks
    MAX_DRONES = 10                  # Cutoff for drones
    RANDOM_PERMUTATION = True        # Random shuffle vehicles per version
    
    # Get seed from month
    seed = get_seed_from_month()
    
    # Step 1: Collect JSON files
    json_files = collect_json_files(JSON_FOLDER)
    
    if len(json_files) == 0:
        print("❌ No JSON files found. Exiting.")
        return
    
    # Step 2: Convert to NPZ
    convert_json_to_npz(
        json_files,
        OUTPUT_DIR,
        INSTANCES_PER_NPZ,
        MAX_CUSTOMERS,
        MAX_TRUCKS,
        MAX_DRONES,
        output_prefix=OUTPUT_PREFIX,  # ✅ Pass custom prefix
        random_permutation=RANDOM_PERMUTATION,
        seed=seed
    )


if __name__ == '__main__':
    main()
