"""
Generate Epoch Training Data for PVRPWDP

This script generates training data files for each epoch, allowing for:
1. Curriculum learning with increasing difficulty
2. Pre-generated data to save computation during training
3. Reproducible training data

Usage:
    # Generate 100 epochs with default settings
    python generate_pvrpwdp_epochs.py
    
    # Generate with custom settings
    python generate_pvrpwdp_epochs.py --num_epochs 50 --batch_size 500 --num_loc 30
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from parco.envs.pvrpwdp.generator import PVRPWDPGenerator
from rl4co.data.utils import save_tensordict_to_npz


def generate_epoch_data(
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 1000,
    num_loc: int = 20,
    num_agents: int = 3,
    **generator_kwargs
):
    """Generate training data for multiple epochs.
    
    Args:
        output_dir: Directory to save epoch files
        num_epochs: Number of epochs to generate
        batch_size: Number of instances per epoch
        num_loc: Number of customer locations
        num_agents: Number of vehicles
        **generator_kwargs: Additional arguments for PVRPWDPGenerator
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("PVRPWDP Epoch Data Generation")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size:       {batch_size}")
    print(f"Num locations:    {num_loc}")
    print(f"Num agents:       {num_agents}")
    print("="*60 + "\n")
    
    # Create generator
    generator = PVRPWDPGenerator(
        num_loc=num_loc,
        num_agents=num_agents,
        **generator_kwargs
    )
    
    # Generate data for each epoch
    for epoch in tqdm(range(num_epochs), desc="Generating epochs"):
        # Generate data
        td = generator(
            batch_size=[batch_size],
            current_epoch=epoch,
            max_epochs=num_epochs
        )
        
        # Save to file
        filename = output_path / f"epoch_{epoch}.npz"
        save_tensordict_to_npz(td, filename)
        
        if epoch % 10 == 0:
            tqdm.write(f"✅ Saved epoch {epoch} to {filename}")
    
    print("\n" + "="*60)
    print("✅ Generation Complete!")
    print("="*60)
    print(f"Generated {num_epochs} epoch files")
    print(f"Total instances: {num_epochs * batch_size}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")


def generate_with_curriculum(
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 1000,
    start_num_loc: int = 10,
    end_num_loc: int = 50,
    num_agents: int = 3,
    **generator_kwargs
):
    """Generate training data with curriculum learning (increasing difficulty).
    
    Args:
        output_dir: Directory to save epoch files
        num_epochs: Number of epochs to generate
        batch_size: Number of instances per epoch
        start_num_loc: Starting number of locations (easy)
        end_num_loc: Ending number of locations (hard)
        num_agents: Number of vehicles
        **generator_kwargs: Additional arguments for PVRPWDPGenerator
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("PVRPWDP Curriculum Learning Data Generation")
    print("="*60)
    print(f"Output directory:  {output_dir}")
    print(f"Number of epochs:  {num_epochs}")
    print(f"Batch size:        {batch_size}")
    print(f"Locations range:   {start_num_loc} → {end_num_loc} (curriculum)")
    print(f"Num agents:        {num_agents}")
    print("="*60 + "\n")
    
    # Generate data for each epoch with increasing difficulty
    for epoch in tqdm(range(num_epochs), desc="Generating curriculum"):
        # Calculate number of locations for this epoch (linear curriculum)
        progress = epoch / max(num_epochs - 1, 1)
        current_num_loc = int(start_num_loc + (end_num_loc - start_num_loc) * progress)
        
        # Create generator for this epoch
        generator = PVRPWDPGenerator(
            num_loc=current_num_loc,
            num_agents=num_agents,
            **generator_kwargs
        )
        
        # Generate data
        td = generator(
            batch_size=[batch_size],
            current_epoch=epoch,
            max_epochs=num_epochs
        )
        
        # Save to file
        filename = output_path / f"epoch_{epoch}.npz"
        save_tensordict_to_npz(td, filename)
        
        if epoch % 10 == 0:
            tqdm.write(f"✅ Epoch {epoch}: {current_num_loc} locations → {filename}")
    
    print("\n" + "="*60)
    print("✅ Curriculum Generation Complete!")
    print("="*60)
    print(f"Generated {num_epochs} epoch files with curriculum learning")
    print(f"Difficulty range: {start_num_loc} → {end_num_loc} locations")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate PVRPWDP epoch training data")
    
    # Basic settings
    parser.add_argument("--output_dir", type=str, default="data/pvrpwdp/train_epochs/",
                        help="Output directory for epoch files")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of epochs to generate")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Number of instances per epoch")
    
    # Problem settings
    parser.add_argument("--num_loc", type=int, default=20,
                        help="Number of customer locations")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="Number of vehicles")
    
    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning (increasing difficulty)")
    parser.add_argument("--start_num_loc", type=int, default=10,
                        help="Starting number of locations for curriculum")
    parser.add_argument("--end_num_loc", type=int, default=50,
                        help="Ending number of locations for curriculum")
    
    # Generator settings
    parser.add_argument("--min_demand", type=int, default=1,
                        help="Minimum customer demand")
    parser.add_argument("--max_demand", type=int, default=10,
                        help="Maximum customer demand")
    parser.add_argument("--capacity", type=float, default=40.0,
                        help="Vehicle capacity")
    parser.add_argument("--endurance", type=float, default=10.0,
                        help="Drone endurance (battery)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Truck speed")
    parser.add_argument("--drone_speed", type=float, default=2.0,
                        help="Drone speed")
    parser.add_argument("--tw_expansion", type=float, default=3.0,
                        help="Time window expansion factor")
    parser.add_argument("--freshness_factor", type=float, default=2.0,
                        help="Freshness (waiting_time) factor")
    
    args = parser.parse_args()
    
    # Prepare generator kwargs
    generator_kwargs = {
        'min_demand': args.min_demand,
        'max_demand': args.max_demand,
        'capacity': args.capacity,
        'endurance': args.endurance,
        'speed': args.speed,
        'drone_speed': args.drone_speed,
        'tw_expansion': args.tw_expansion,
        'freshness_factor': args.freshness_factor,
    }
    
    # Generate data
    if args.curriculum:
        generate_with_curriculum(
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            start_num_loc=args.start_num_loc,
            end_num_loc=args.end_num_loc,
            num_agents=args.num_agents,
            **generator_kwargs
        )
    else:
        generate_epoch_data(
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            num_loc=args.num_loc,
            num_agents=args.num_agents,
            **generator_kwargs
        )


if __name__ == "__main__":
    main()
