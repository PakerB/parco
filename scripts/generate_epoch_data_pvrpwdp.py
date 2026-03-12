"""
Script to generate epoch-based training data for PVRPWDP environment.

This script generates pre-computed training data for each epoch with curriculum learning,
allowing faster training and reproducible experiments.

Usage:
    python scripts/generate_epoch_data_pvrpwdp.py \\
        --output_dir data/pvrpwdp/train_epochs/ \\
        --num_epochs 100 \\
        --batch_size 10000 \\
        --num_loc 20 \\
        --num_agents 3

Curriculum Learning Strategies:
    - Time window tightness increases over epochs
    - Optional: number of locations increases
    - Optional: vehicle capacity decreases
    - Optional: waiting time (freshness) decreases
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from parco.envs.pvrpwdp.generator import PVRPWDPGenerator
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def generate_epoch_data(
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 10000,
    num_loc: int = 20,
    num_agents: int = 3,
    curriculum_strategy: str = "tw_tightness",
    **generator_kwargs,
):
    """Generate training data for each epoch with curriculum learning.
    
    Args:
        output_dir: Output directory for epoch files
        num_epochs: Number of epochs to generate
        batch_size: Batch size for each epoch
        num_loc: Number of customer locations
        num_agents: Number of agents/vehicles
        curriculum_strategy: One of:
            - 'tw_tightness': Tighten time windows over epochs
            - 'num_loc': Increase number of locations
            - 'capacity': Decrease vehicle capacity
            - 'waiting_time': Decrease freshness duration
            - 'combined': Combine multiple strategies
            - 'none': No curriculum, same difficulty
        **generator_kwargs: Additional arguments for PVRPWDPGenerator
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Generating {num_epochs} epochs of data")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Number of locations: {num_loc}")
    log.info(f"Number of agents: {num_agents}")
    log.info(f"Curriculum strategy: {curriculum_strategy}")
    
    for epoch in tqdm(range(num_epochs), desc="Generating epochs"):
        # Calculate progress (0.0 to 1.0)
        progress = epoch / max(num_epochs - 1, 1)
        
        # Apply curriculum learning strategy
        gen_params = {
            'num_loc': num_loc,
            'num_agents': num_agents,
            **generator_kwargs
        }
        
        if curriculum_strategy == 'tw_tightness':
            # Tighten time windows over epochs
            # Early epochs: wide windows [30, 50]
            # Late epochs: tight windows [20, 40]
            gen_params['tw_width_min'] = 30 - progress * 10
            gen_params['tw_width_max'] = 50 - progress * 10
            
        elif curriculum_strategy == 'num_loc':
            # Increase number of locations over epochs
            # Early epochs: 10 locations
            # Late epochs: 50 locations
            gen_params['num_loc'] = 10 + int(progress * 40)
            
        elif curriculum_strategy == 'capacity':
            # Decrease vehicle capacity over epochs
            # Makes problem harder (need more trips)
            base_capacity = generator_kwargs.get('capacity_scale', 1.0)
            gen_params['capacity_scale'] = base_capacity * (1.0 - progress * 0.3)
            
        elif curriculum_strategy == 'waiting_time':
            # Decrease waiting time (freshness) over epochs
            # Makes perishability constraint tighter
            base_waiting = generator_kwargs.get('waiting_time_scale', 1.0)
            gen_params['waiting_time_scale'] = base_waiting * (1.0 - progress * 0.5)
            
        elif curriculum_strategy == 'combined':
            # Combine multiple strategies for maximum curriculum effect
            gen_params['tw_width_min'] = 30 - progress * 10
            gen_params['tw_width_max'] = 50 - progress * 10
            gen_params['num_loc'] = max(10, num_loc - int(progress * (num_loc - 10)))
            base_capacity = generator_kwargs.get('capacity_scale', 1.0)
            gen_params['capacity_scale'] = base_capacity * (1.0 - progress * 0.2)
            
        elif curriculum_strategy == 'none':
            # No curriculum learning, same difficulty for all epochs
            pass
        else:
            log.warning(f"Unknown curriculum strategy '{curriculum_strategy}', using 'none'")
        
        # Create generator with curriculum parameters
        generator = PVRPWDPGenerator(**gen_params)
        
        # Generate data
        td = generator(batch_size=[batch_size])
        
        # Convert TensorDict to numpy dict for saving
        data_dict = {}
        for key, value in td.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.cpu().numpy()
        
        # Save to compressed npz file
        output_file = output_path / f"epoch_{epoch}.npz"
        np.savez_compressed(output_file, **data_dict)
        
        # Log progress (every 10 epochs or last epoch)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            log.info(f"Epoch {epoch}: saved to {output_file.name}")
            if curriculum_strategy != 'none':
                log.info(f"  Curriculum params: {gen_params}")
    
    log.info(f"✅ Successfully generated {num_epochs} epoch files in {output_dir}")
    
    # Print summary
    total_size_mb = sum(f.stat().st_size for f in output_path.glob("*.npz")) / 1024 / 1024
    log.info(f"Total data size: {total_size_mb:.2f} MB")
    log.info(f"Average file size: {total_size_mb / num_epochs:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate epoch-based training data for PVRPWDP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/pvrpwdp/train_epochs/",
        help="Output directory for epoch files"
    )
    
    # Data generation parameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for each epoch"
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=20,
        help="Number of customer locations"
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=3,
        help="Number of agents/vehicles"
    )
    
    # Curriculum learning
    parser.add_argument(
        "--curriculum",
        type=str,
        default="tw_tightness",
        choices=['tw_tightness', 'num_loc', 'capacity', 'waiting_time', 'combined', 'none'],
        help="Curriculum learning strategy"
    )
    
    # Generator parameters
    parser.add_argument(
        "--tw_width_min",
        type=float,
        default=None,
        help="Minimum time window width (overrides curriculum)"
    )
    parser.add_argument(
        "--tw_width_max",
        type=float,
        default=None,
        help="Maximum time window width (overrides curriculum)"
    )
    parser.add_argument(
        "--capacity_scale",
        type=float,
        default=1.0,
        help="Scale factor for vehicle capacity"
    )
    parser.add_argument(
        "--waiting_time_scale",
        type=float,
        default=1.0,
        help="Scale factor for waiting time (freshness)"
    )
    
    args = parser.parse_args()
    
    # Prepare generator kwargs
    generator_kwargs = {}
    if args.tw_width_min is not None:
        generator_kwargs['tw_width_min'] = args.tw_width_min
    if args.tw_width_max is not None:
        generator_kwargs['tw_width_max'] = args.tw_width_max
    generator_kwargs['capacity_scale'] = args.capacity_scale
    generator_kwargs['waiting_time_scale'] = args.waiting_time_scale
    
    # Generate data
    generate_epoch_data(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_loc=args.num_loc,
        num_agents=args.num_agents,
        curriculum_strategy=args.curriculum,
        **generator_kwargs,
    )


if __name__ == "__main__":
    main()
