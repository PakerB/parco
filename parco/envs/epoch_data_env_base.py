"""
Epoch-based Data Loading Environment Base (Multi-Part Lazy Loading)

This module provides a base class for RL4CO environments that load training data
from pre-generated epoch files partitioned into multiple parts.

Key Features:
    - Lazy loading: Only loads one part of an epoch into RAM at a time.
    - Automatic Globbing: Finds all parts belonging to the current epoch.
    - Full debug suite included.

File Naming Convention:
    - Default pattern: "epoch_{epoch:02d}_{part:02d}.npz"
    - Examples:
        * "epoch_00_00.npz", "epoch_00_01.npz" (Epoch 0 parts)
        * "epoch_01_00.npz", "epoch_01_01.npz" (Epoch 1 parts)
"""

import os
import glob
import re
from pathlib import Path
from typing import Optional

import torch
from tensordict.tensordict import TensorDict

from rl4co.data.dataset import TensorDictDataset
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MultiPartLazyDataset(torch.utils.data.Dataset):
    """
    Dataset giữ các part của 1 Epoch duy nhất.
    Nạp/xả từng part lên RAM để không bị tràn bộ nhớ.
    """

    def __init__(self, epoch, part_files, samples_per_file):
        self.epoch = epoch
        self.part_files = sorted(part_files)
        self.samples_per_file = samples_per_file

        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # Shard per part so all DDP ranks load the same part file at the same time.
        self.local_length = self.samples_per_file // self.world_size
        self.total_samples = len(self.part_files) * self.local_length

        self.current_part_idx = -1
        self.current_td = None
        # Cache các item đã "tháo" thành dict cho part đang nạp để
        # tương thích định dạng với rl4co.data.dataset.TensorDictDataset
        self._current_items = None

    def __len__(self):
        return self.total_samples

    def _ensure_part_loaded(self, part_idx: int):
        if self.current_part_idx == part_idx:
            return
        file_path = self.part_files[part_idx]
        file_name = os.path.basename(file_path)
        log.info(
            f"🔄 [Epoch {self.epoch:02d}] - Giải phóng RAM & Nạp Part {part_idx:02d}: {file_name}"
        )
        td = load_npz_to_tensordict(file_path)
        self.current_td = td
        # Disassemble TensorDict -> list[dict[str, Tensor]] giống TensorDictDataset
        # để collate_fn hoạt động đúng với rl4co._dataloader_single.
        n = td.batch_size[0]
        self._current_items = [
            {key: value[i] for key, value in td.items()} for i in range(n)
        ]
        self.current_part_idx = part_idx

    def __getitem__(self, idx):
        part_idx_seq = idx // self.local_length
        local_idx_in_part = idx % self.local_length
        global_idx = (
            part_idx_seq * self.samples_per_file
            + local_idx_in_part
            + self.rank * self.local_length
        )
        part_idx = global_idx // self.samples_per_file
        item_idx = global_idx % self.samples_per_file
        self._ensure_part_loaded(part_idx)
        return self._current_items[item_idx]

    @staticmethod
    def collate_fn(batch):
        """Collate list of dicts thành một TensorDict batched.

        Sao chép hành vi của `rl4co.data.dataset.TensorDictDataset.collate_fn`
        để tương thích với `rl4co.models.rl.common.base._dataloader_single`.
        """
        # Trường hợp hiếm: nếu item là TensorDict (khi __getitem__ bị override),
        # dùng torch.stack trực tiếp.
        if isinstance(batch[0], TensorDict):
            return torch.stack(batch, dim=0)
        return TensorDict(
            {key: torch.stack([b[key] for b in batch]) for key in batch[0].keys()},
            batch_size=torch.Size([len(batch)]),
        )


class EpochDataEnvBase(RL4COEnvBase):

    def __init__(
        self,
        *,
        epoch_data_dir: str = None,
        epoch_file_pattern: str = "epoch_{epoch:02d}_{part:02d}.npz",
        use_epoch_data: bool = True,
        fallback_to_generator: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.epoch_data_dir = epoch_data_dir
        self.epoch_file_pattern = epoch_file_pattern
        self.use_epoch_data = use_epoch_data
        self.fallback_to_generator = fallback_to_generator

        self.current_epoch = 0
        self.max_epochs = None

        if self.use_epoch_data:
            if self.epoch_data_dir is None:
                log.warning("use_epoch_data=True but epoch_data_dir is None. Disabled.")
                self.use_epoch_data = False
            else:
                epoch_data_path = Path(self.epoch_data_dir)
                if not epoch_data_path.exists() or not epoch_data_path.is_dir():
                    log.warning(
                        f"Epoch data directory '{self.epoch_data_dir}' is invalid/missing."
                    )
                    self.use_epoch_data = False

    def dataset(self, batch_size=[], phase="train", filename=None):
        if phase != "train":
            return super().dataset(batch_size, phase, filename)

        # 1. Tìm tất cả các file part của epoch hiện tại
        pattern_with_epoch = self.epoch_file_pattern.replace(
            "{epoch:02d}", f"{self.current_epoch:02d}"
        )
        search_pattern = pattern_with_epoch.replace("{part:02d}", "*")

        search_path = os.path.join(self.epoch_data_dir, search_pattern)
        part_files = sorted(glob.glob(search_path))

        if not part_files or not self.use_epoch_data:
            error_msg = f"❌ LỖI CHÍ MẠNG: Không tìm thấy bất kỳ file nào cho Epoch {self.current_epoch} với định dạng '{search_pattern}'"
            log.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Detect samples per file from the first part
        first_td = load_npz_to_tensordict(part_files[0])
        SAMPLES_PER_FILE = len(first_td)
        del first_td

        log.info(
            f"Epoch {self.current_epoch:02d}: {len(part_files)} parts x {SAMPLES_PER_FILE} samples = {len(part_files)*SAMPLES_PER_FILE} total."
        )
        return MultiPartLazyDataset(self.current_epoch, part_files, SAMPLES_PER_FILE)

    def list_available_epochs(self) -> list:
        """Đã được nâng cấp để đọc format epoch_ab_cd.npz"""
        if self.epoch_data_dir is None or not os.path.exists(self.epoch_data_dir):
            return []

        # Tìm tất cả file có chữ epoch_
        search_pattern = self.epoch_file_pattern.replace("{epoch:02d}", "*").replace(
            "{part:02d}", "*"
        )
        files = sorted(glob.glob(os.path.join(self.epoch_data_dir, search_pattern)))

        epochs = set()
        for file_path in files:
            filename = os.path.basename(file_path)
            # Dùng regex để bắt đúng cái số ab trong epoch_ab_cd
            match = re.search(r"epoch_(\d+)_", filename)
            if match:
                epochs.add(int(match.group(1)))

        return sorted(list(epochs))

    def validate_epoch_files(self, max_epoch: Optional[int] = None) -> dict:
        """Kiểm tra xem mỗi epoch có ít nhất 1 part hợp lệ không"""
        if max_epoch is None:
            max_epoch = self.max_epochs if self.max_epochs is not None else 100

        results = {
            "missing": [],
            "corrupted": [],
            "valid": [],
            "total_expected": max_epoch,
        }

        for epoch in range(max_epoch):
            pattern_with_epoch = self.epoch_file_pattern.replace(
                "{epoch:02d}", f"{epoch:02d}"
            )
            search_pattern = pattern_with_epoch.replace("{part:02d}", "*")
            part_files = sorted(
                glob.glob(os.path.join(self.epoch_data_dir, search_pattern))
            )

            if not part_files:
                results["missing"].append(epoch)
            else:
                try:
                    # Test load part đầu tiên để xem file có bị lỗi (corrupted) không
                    td = load_npz_to_tensordict(part_files[0])
                    results["valid"].append(epoch)
                except Exception as e:
                    log.warning(f"Corrupted file in epoch {epoch}: {str(e)}")
                    results["corrupted"].append(epoch)

        return results

    def print_epoch_data_info(self):
        """Giữ nguyên hàm print debug gốc của bạn"""
        print("\n" + "=" * 60)
        print("EPOCH DATA CONFIGURATION")
        print("=" * 60)
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
                print(
                    f"Epoch Range:          {min(available_epochs)} - {max(available_epochs)}"
                )

                if len(available_epochs) <= 10:
                    print(f"Epochs:               {available_epochs}")
                else:
                    print(
                        f"Sample Epochs:        {available_epochs[:5]} ... {available_epochs[-5:]}"
                    )

                if self.current_epoch in available_epochs:
                    print(f"✅ Current epoch {self.current_epoch} files exist")
                else:
                    print(f"⚠️  Current epoch {self.current_epoch} files NOT FOUND")

        print("=" * 60 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print(__doc__)
