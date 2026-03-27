"""
Script to read and inspect NPZ files from npz_test folder.
"""

import numpy as np
from pathlib import Path

def inspect_npz_file(npz_path):
    """
    Load and inspect NPZ file structure and contents.
    """
    print(f"\n{'='*80}")
    print(f"📦 Inspecting: {npz_path.name}")
    print(f"{'='*80}")
    
    # Load NPZ file
    data = np.load(npz_path)
    
    print(f"\n📋 Keys in NPZ file:")
    for key in sorted(data.keys()):
        print(f"   - {key}")
    
    print(f"\n📊 Data shapes:")
    for key in sorted(data.keys()):
        arr = data[key]
        print(f"   {key:20s}: {str(arr.shape):20s} dtype: {arr.dtype}")
    
    print(f"\n📈 Statistics:")
    batch_size = data['depot'].shape[0]
    num_customers = data['locs'].shape[1]
    num_vehicles = data['agents_speed'].shape[1]
    
    print(f"   Batch size: {batch_size} instances")
    print(f"   Customers per instance: {num_customers}")
    print(f"   Vehicles per instance: {num_vehicles}")
    
    print(f"\n🔍 Sample instance (batch[0]):")
    
    print(f"\n   Depot (coordinates):")
    print(f"      {data['depot'][0]}")
    
    print(f"\n   First 5 customer locations:")
    for i in range(min(5, num_customers)):
        print(f"      Customer {i}: {data['locs'][0, i]}")
    
    print(f"\n   First 5 demands:")
    print(f"      {data['demand'][0, :5]}")
    
    print(f"\n   First 5 time windows:")
    for i in range(min(5, num_customers)):
        print(f"      Customer {i}: {data['time_windows'][0, i]}")
    
    print(f"\n   Vehicle speeds:")
    print(f"      {data['agents_speed'][0]}")
    
    print(f"\n   Vehicle capacities:")
    print(f"      {data['agents_capacity'][0]}")
    
    print(f"\n   Vehicle endurance:")
    print(f"      {data['agents_endurance'][0]}")
    
    print(f"\n   Waiting times (first 5):")
    print(f"      {data['waiting_time'][0, :5]}")
    
    # Check for NaN values
    print(f"\n⚠️  NaN/Inf Check:")
    has_issues = False
    for key in sorted(data.keys()):
        arr = data[key]
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"   ❌ {key}: {nan_count} NaN, {inf_count} Inf")
            has_issues = True
    
    if not has_issues:
        print(f"   ✅ No NaN/Inf values found!")
    
    # Close the npz file
    data.close()


def main():
    """
    Main: scan folder and inspect all NPZ files.
    """
    npz_folder = Path("npz_test")
    
    print(f"\n🔍 Scanning folder: {npz_folder.absolute()}")
    
    # Find all NPZ files
    npz_files = sorted(npz_folder.glob("*.npz"))
    
    print(f"✅ Found {len(npz_files)} NPZ file(s)")
    
    if len(npz_files) == 0:
        print("❌ No NPZ files found!")
        return
    
    # Inspect each NPZ file
    for npz_file in npz_files:
        inspect_npz_file(npz_file)
    
    print(f"\n{'='*80}")
    print(f"✅ Inspection complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
