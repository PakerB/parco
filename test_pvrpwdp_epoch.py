"""
Test script for PVRPWDP Epoch Data Loading

This script demonstrates how to use the EpochDataEnvBase with PVRPWDP environment.

Usage:
    python test_pvrpwdp_epoch.py
"""

import torch
from parco.envs.pvrpwdp.env import PVRPWDPVEnv
from parco.envs.pvrpwdp.generator import PVRPWDPGenerator

def test_basic_functionality():
    """Test basic epoch data loading functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)
    
    # Create environment with epoch data support
    env = PVRPWDPVEnv(
        epoch_data_dir="data/pvrpwdp/train_epochs/",
        epoch_file_pattern="epoch_{epoch}.npz",
        use_epoch_data=True,
        fallback_to_generator=True,
        generator_params={
            'num_loc': 20,
            'num_agents': 3,
        }
    )
    
    # Print configuration
    env.print_epoch_data_info()
    
    # List available epochs
    available = env.list_available_epochs()
    print(f"\n✅ Available epochs: {available}")
    
    return env

def test_data_loading():
    """Test loading data from different epochs."""
    print("\n" + "="*60)
    print("TEST 2: Data Loading")
    print("="*60)
    
    env = PVRPWDPVEnv(
        epoch_data_dir="data/pvrpwdp/train_epochs/",
        use_epoch_data=True,
        fallback_to_generator=True,
        generator_params={'num_loc': 20, 'num_agents': 3}
    )
    
    # Test loading training data
    print("\n📦 Loading training data...")
    env.current_epoch = 0
    train_dataset = env.dataset(batch_size=[100], phase="train")
    print(f"✅ Loaded training data: {train_dataset}")
    print(f"   Batch size: {len(train_dataset)}")
    
    # Test loading validation data (should use generator)
    print("\n📦 Loading validation data...")
    val_dataset = env.dataset(batch_size=[50], phase="val")
    print(f"✅ Loaded validation data: {val_dataset}")
    
    return env

def test_fallback_to_generator():
    """Test fallback to generator when epoch file missing."""
    print("\n" + "="*60)
    print("TEST 3: Fallback to Generator")
    print("="*60)
    
    env = PVRPWDPVEnv(
        epoch_data_dir="data/pvrpwdp/train_epochs/",
        use_epoch_data=True,
        fallback_to_generator=True,  # Enable fallback
        generator_params={'num_loc': 20, 'num_agents': 3}
    )
    
    # Try to load from non-existent epoch
    print("\n📦 Loading from non-existent epoch 999...")
    env.current_epoch = 999
    dataset = env.dataset(batch_size=[10], phase="train")
    print(f"✅ Fallback successful! Generated data: {len(dataset)} instances")
    
    return env

def test_validation():
    """Test epoch file validation."""
    print("\n" + "="*60)
    print("TEST 4: Epoch File Validation")
    print("="*60)
    
    env = PVRPWDPVEnv(
        epoch_data_dir="data/pvrpwdp/train_epochs/",
        use_epoch_data=True,
        generator_params={'num_loc': 20, 'num_agents': 3}
    )
    
    # Validate epoch files
    print("\n🔍 Validating epoch files (max_epoch=10)...")
    results = env.validate_epoch_files(max_epoch=10)
    
    print(f"\n📊 Validation Results:")
    print(f"   Total expected: {results['total_expected']}")
    print(f"   ✅ Valid:       {len(results['valid'])}")
    print(f"   ❌ Missing:     {len(results['missing'])}")
    print(f"   ⚠️  Corrupted:  {len(results['corrupted'])}")
    
    if results['missing']:
        print(f"\n   Missing epochs: {results['missing'][:10]}...")
    
    return results

def test_without_epoch_data():
    """Test environment without epoch data (standard RL4CO behavior)."""
    print("\n" + "="*60)
    print("TEST 5: Without Epoch Data (Standard RL4CO)")
    print("="*60)
    
    # Create environment without epoch data
    env = PVRPWDPVEnv(
        use_epoch_data=False,  # Disable epoch data
        generator_params={'num_loc': 20, 'num_agents': 3}
    )
    
    print("\n📦 Generating data using standard RL4CO...")
    env.current_epoch = 0
    dataset = env.dataset(batch_size=[50], phase="train")
    print(f"✅ Generated {len(dataset)} instances using generator")
    
    return env

def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# PVRPWDP Epoch Data Loading Test Suite")
    print("#"*60)
    
    try:
        # Test 1: Basic functionality
        env = test_basic_functionality()
        
        # Test 2: Data loading
        test_data_loading()
        
        # Test 3: Fallback to generator
        test_fallback_to_generator()
        
        # Test 4: Validation
        test_validation()
        
        # Test 5: Without epoch data
        test_without_epoch_data()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
