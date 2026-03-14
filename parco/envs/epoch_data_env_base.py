"""
Epoch-based Data Loading Environment Base

This module provides a base class for RL4CO environments that load training data
from pre-generated epoch files instead of generating data on-the-fly.

Key Features:
    - Load data from a folder containing multiple .npz files (one per epoch)
    - Automatically select the correct file based on current epoch
    - Fallback to generator if epoch file not found
    - Validation and test datasets still use standard RL4CO loading

Usage Example:
    ```python
    from parco.envs.epoch_data_env_base import EpochDataEnvBase
    from parco.envs.pvrpwdp.generator import PVRPWDPGenerator
    
    class MyEnv(EpochDataEnvBase):
        def __init__(self, **kwargs):
            super().__init__(
                epoch_data_dir="data/train_epochs/",
                epoch_file_pattern="epoch_{epoch}.npz",
                generator=PVRPWDPGenerator(),
                **kwargs
            )
    
    # During training
    env.current_epoch = 10  # Set by trainer
    dataset = env.dataset(batch_size=[1000], phase="train")
    # Will load from "data/train_epochs/epoch_10.npz"
    ```

File Naming Convention:
    - Default pattern: "epoch_{epoch}.npz"
    - Custom pattern: Use {epoch} as placeholder
    - Examples: 
        * "epoch_0.npz", "epoch_1.npz", ...
        * "train_data_epoch_{epoch}.npz"
        * "pvrpwdp_e{epoch}.npz"
"""

import os
from pathlib import Path
from typing import Optional

import torch
from tensordict.tensordict import TensorDict

from rl4co.data.dataset import TensorDictDataset
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class EpochDataEnvBase(RL4COEnvBase):
    """Base class for RL4CO environments that load training data from epoch files.
    
    This class extends RL4COEnvBase to support loading training data from a folder
    containing pre-generated .npz files, one for each epoch. This is useful when:
    1. You want to use curriculum learning with increasing difficulty per epoch
    2. You want to pre-generate all training data to save computation during training
    3. You want to use externally generated data (e.g., from OR-Tools, expert solutions)
    
    The environment will automatically load the correct file based on `current_epoch`.
    If the epoch file is not found, it falls back to the generator.
    
    Args:
        epoch_data_dir: Directory containing epoch data files (e.g., "data/train_epochs/")
        epoch_file_pattern: File naming pattern with {epoch} placeholder (default: "epoch_{epoch}.npz")
        use_epoch_data: Whether to use epoch data for training (default: True)
        fallback_to_generator: Whether to use generator if epoch file not found (default: True)
        **kwargs: Additional arguments passed to RL4COEnvBase
    
    Attributes:
        current_epoch: Current training epoch (set by trainer)
        max_epochs: Maximum number of epochs (set by trainer)
    
    Note:
        - Validation and test datasets are still loaded using standard RL4CO methods
        - Only training data uses epoch-based loading
    """
    
    def __init__(
        self,
        *,
        epoch_data_dir: str = None,
        epoch_file_pattern: str = "epoch_{epoch}.npz",
        use_epoch_data: bool = True,
        fallback_to_generator: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.epoch_data_dir = epoch_data_dir
        self.epoch_file_pattern = epoch_file_pattern
        self.use_epoch_data = use_epoch_data
        self.fallback_to_generator = fallback_to_generator
        
        # These will be set by the trainer
        self.current_epoch = 0
        self.max_epochs = None
        
        # Validate epoch_data_dir if use_epoch_data is True
        if self.use_epoch_data:
            if self.epoch_data_dir is None:
                log.warning(
                    "use_epoch_data=True but epoch_data_dir is None. "
                    "Epoch data loading will be disabled. "
                    "Set epoch_data_dir to enable epoch-based data loading."
                )
                self.use_epoch_data = False
            else:
                epoch_data_path = Path(self.epoch_data_dir)
                if not epoch_data_path.exists():
                    log.warning(
                        f"Epoch data directory '{self.epoch_data_dir}' does not exist. "
                        f"Epoch data loading will be disabled unless the directory is created."
                    )
                elif not epoch_data_path.is_dir():
                    log.error(
                        f"Epoch data path '{self.epoch_data_dir}' is not a directory. "
                        f"Epoch data loading will be disabled."
                    )
                    self.use_epoch_data = False
    
    def get_epoch_file_path(self, epoch: int) -> str:
        """Get the file path for a specific epoch.
        
        Args:
            epoch: Epoch number
            
        Returns:
            Full path to the epoch data file
            
        Example:
            >>> env.get_epoch_file_path(10)
            'data/train_epochs/epoch_10.npz'
        """
        if self.epoch_data_dir is None:
            return None
        
        filename = self.epoch_file_pattern.format(epoch=epoch)
        return os.path.join(self.epoch_data_dir, filename)
    
    def _load_epoch_data(self, epoch: int, batch_size: list) -> Optional[TensorDict]:
        """Load data for a specific epoch from file.
        
        Args:
            epoch: Epoch number
            batch_size: Desired batch size (may be used for validation)
            
        Returns:
            TensorDict with loaded data, or None if file not found
        """
        file_path = self.get_epoch_file_path(epoch)
        
        if file_path is None:
            return None
        
        if not os.path.exists(file_path):
            if self.fallback_to_generator:
                log.warning(
                    f"Epoch file '{file_path}' not found for epoch {epoch}. "
                    f"Falling back to generator."
                )
            else:
                log.error(
                    f"Epoch file '{file_path}' not found for epoch {epoch} "
                    f"and fallback_to_generator=False. Cannot load data."
                )
            return None
        
        try:
            log.info(f"Loading epoch {epoch} data from '{file_path}'")
            td = load_npz_to_tensordict(file_path)
            
            # Validate batch size if specified
            if batch_size:
                # Handle both int and list batch_size
                expected_batch = batch_size[0] if isinstance(batch_size, (list, tuple)) else batch_size
                actual_batch = td.batch_size[0]
                
                if actual_batch != expected_batch:
                    log.warning(
                        f"Loaded data batch size {actual_batch} does not match "
                        f"requested batch size {expected_batch}. Using loaded size."
                    )
            
            return td
            
        except Exception as e:
            import traceback
            log.error(
                f"Error loading epoch file '{file_path}': {str(e)}\n"
                f"Full traceback:\n{traceback.format_exc()}"
            )
            if not self.fallback_to_generator:
                # Re-raise exception if fallback disabled for better debugging
                raise
            return None
    
    def dataset(self, batch_size=[], phase="train", filename=None):
        """Return a dataset of observations.
        
        For training phase:
            - If use_epoch_data=True: Load from epoch file based on current_epoch
            - If epoch file not found: Use generator (if fallback enabled)
            
        For validation/test phases:
            - Use standard RL4CO loading (from val_file/test_file or generator)
        
        Args:
            batch_size: Batch size for the dataset
            phase: One of "train", "val", "test"
            filename: Override filename (standard RL4CO behavior)
            
        Returns:
            TensorDictDataset with the loaded/generated data
        """
        # For non-training phases, use standard RL4CO loading
        if phase != "train":
            return super().dataset(batch_size, phase, filename)
        
        # For training phase, check if we should use epoch data
        if not self.use_epoch_data or self.epoch_data_dir is None:
            # Use standard RL4CO loading (generator or train_file)
            return super().dataset(batch_size, phase, filename)
        
        # Override with filename if provided
        if filename is not None:
            log.info(f"Overriding epoch data with filename: {filename}")
            return super().dataset(batch_size, phase, filename)
        
        # Try to load epoch data
        td = self._load_epoch_data(self.current_epoch, batch_size)
        
        # If epoch data not found and fallback enabled, use generator
        if td is None:
            if self.fallback_to_generator:
                log.info(f"Generating data for epoch {self.current_epoch} using generator")
                td = self.generator(
                    batch_size,
                    current_epoch=self.current_epoch,
                    max_epochs=self.max_epochs
                )
            else:
                raise FileNotFoundError(
                    f"Epoch file not found for epoch {self.current_epoch} "
                    f"and fallback_to_generator=False"
                )
        
        return self.dataset_cls(td)
    
    def list_available_epochs(self) -> list:
        """List all available epoch files in the epoch_data_dir.
        
        Returns:
            List of epoch numbers that have corresponding data files
            
        Example:
            >>> env.list_available_epochs()
            [0, 1, 2, 5, 10, 15, 20]
        """
        if self.epoch_data_dir is None or not os.path.exists(self.epoch_data_dir):
            return []
        
        epoch_pattern = self.epoch_file_pattern.replace("{epoch}", "*")
        from glob import glob
        
        files = glob(os.path.join(self.epoch_data_dir, epoch_pattern))
        
        # Extract epoch numbers from filenames
        epochs = []
        for file_path in files:
            filename = os.path.basename(file_path)
            try:
                # Try to extract epoch number from filename
                # This is a simple heuristic, may need adjustment for complex patterns
                import re
                match = re.search(r'(\d+)', filename)
                if match:
                    epoch = int(match.group(1))
                    epochs.append(epoch)
            except:
                continue
        
        return sorted(epochs)
    
    def validate_epoch_files(self, max_epoch: Optional[int] = None) -> dict:
        """Validate that all epoch files exist and are loadable.
        
        Args:
            max_epoch: Maximum epoch to check (default: self.max_epochs)
            
        Returns:
            Dictionary with validation results:
                - 'missing': List of missing epoch numbers
                - 'corrupted': List of corrupted epoch numbers (files exist but can't load)
                - 'valid': List of valid epoch numbers
                - 'total_expected': Total number of expected epochs
                
        Example:
            >>> results = env.validate_epoch_files(max_epoch=100)
            >>> print(f"Missing: {len(results['missing'])}/{results['total_expected']}")
        """
        if max_epoch is None:
            max_epoch = self.max_epochs if self.max_epochs is not None else 100
        
        results = {
            'missing': [],
            'corrupted': [],
            'valid': [],
            'total_expected': max_epoch
        }
        
        for epoch in range(max_epoch):
            file_path = self.get_epoch_file_path(epoch)
            
            if not os.path.exists(file_path):
                results['missing'].append(epoch)
            else:
                try:
                    td = load_npz_to_tensordict(file_path)
                    results['valid'].append(epoch)
                except Exception as e:
                    log.warning(f"Corrupted file for epoch {epoch}: {str(e)}")
                    results['corrupted'].append(epoch)
        
        return results
    
    def print_epoch_data_info(self):
        """Print information about epoch data configuration and availability."""
        print("\n" + "="*60)
        print("EPOCH DATA CONFIGURATION")
        print("="*60)
        print(f"Epoch Data Directory: {self.epoch_data_dir}")
        print(f"File Pattern:         {self.epoch_file_pattern}")
        print(f"Use Epoch Data:       {self.use_epoch_data}")
        print(f"Fallback to Generator: {self.fallback_to_generator}")
        print(f"Current Epoch:        {self.current_epoch}")
        print(f"Max Epochs:           {self.max_epochs}")
        
        if self.use_epoch_data and self.epoch_data_dir:
            available_epochs = self.list_available_epochs()
            print(f"\nAvailable Epochs:     {len(available_epochs)}")
            if available_epochs:
                print(f"Epoch Range:          {min(available_epochs)} - {max(available_epochs)}")
                
                # Show sample epochs
                if len(available_epochs) <= 10:
                    print(f"Epochs:               {available_epochs}")
                else:
                    print(f"Sample Epochs:        {available_epochs[:5]} ... {available_epochs[-5:]}")
                
                # Check current epoch
                if self.current_epoch in available_epochs:
                    print(f"✅ Current epoch {self.current_epoch} file exists")
                else:
                    print(f"⚠️  Current epoch {self.current_epoch} file NOT FOUND")
        
        print("="*60 + "\n")


# Example usage and testing
if __name__ == "__main__":
    # This is just for demonstration
    print(__doc__)
