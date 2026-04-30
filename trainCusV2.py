import os
import torch
from rl4co.utils.trainer import RL4COTrainer
from parco.envs.pvrpwdp import PVRPWDPVEnv
from parco.models import PARCORLModule, PARCOPolicy
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from datetime import timedelta
import lightning.pytorch as pl


# ✅ Callback để update env.current_epoch mỗi epoch
class UpdateEpochCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        """Update env.current_epoch khi epoch bắt đầu"""
        current_epoch = trainer.current_epoch
        pl_module.env.current_epoch = current_epoch
        if trainer.is_global_zero:
            print(f"✅ Updated env.current_epoch = {current_epoch}")


class CustomPARCORLModule(PARCORLModule):
    def __init__(
        self,
        env,
        policy,
        batch_size=64,
        **kwargs,
    ):
        super().__init__(env, policy, batch_size=batch_size, **kwargs)
        self._batch_size = batch_size

    def train_dataloader(self):
        current_epoch = (
            self.trainer.current_epoch if self.trainer else self.env.current_epoch
        )
        self.env.current_epoch = current_epoch
        print(f"Nap du lieu cho Epoch {current_epoch:02d}...")

        # Giai phong dataset cu
        import gc

        if hasattr(self, "train_dataset") and self.train_dataset is not None:
            del self.train_dataset
            self.train_dataset = None
        gc.collect()

        # Tao dataset moi cho epoch hien tai
        self.train_dataset = self.env.dataset(self._batch_size)

        dl_old = super().train_dataloader()
        return DataLoader(
            dl_old.dataset,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=dl_old.collate_fn,
        )

    def _get_distributed_sampler(self, dataset):
        """Create DistributedSampler for val/test when DDP is active."""
        if torch.distributed.is_initialized():
            from torch.utils.data.distributed import DistributedSampler

            return DistributedSampler(dataset, shuffle=False)
        return None

    def val_dataloader(self):
        dl_old = super().val_dataloader()
        sampler = self._get_distributed_sampler(dl_old.dataset)
        return DataLoader(
            dl_old.dataset,
            batch_size=self._batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=dl_old.collate_fn,
        )

    def test_dataloader(self):
        dl_old = super().test_dataloader()
        sampler = self._get_distributed_sampler(dl_old.dataset)
        return DataLoader(
            dl_old.dataset,
            batch_size=self._batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=dl_old.collate_fn,
        )

    def configure_optimizers(self):
        # BỔ SUNG SCHEDULER ĐỂ TỰ ĐỘNG GIẢM LEARNING RATE KHI REWARD KHÔNG TĂNG
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/reward"},
        }


if __name__ == "__main__":
    print("🚀 Khởi động quá trình huấn luyện bằng Python Script...")

    # 6. Determinism: Use fixed seed across all processes
    pl.seed_everything(42, workers=True)

    # 1. Setup đường dẫn (Tự động lấy thư mục hiện tại làm gốc)
    project_root = os.getcwd()
    epoch_data_dir = os.path.join(project_root, "data", "train_data_npz")
    val_file = os.path.join(project_root, "data", "val_data", "val.npz")
    test_file = os.path.join(project_root, "data", "test_data", "test.npz")

    # ========== CHECKPOINT & EPOCH CONFIG ==========
    checkpoint_dir = os.path.join(project_root, "data", "checkpoint")
    checkpoint_name = "parco_checkpoint.ckpt"  # Tên checkpoint để chèn vào
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # Kiểm tra checkpoint có tồn tại không
    if os.path.exists(checkpoint_path):
        checkpoint_path = checkpoint_path  # Sử dụng checkpoint nếu tồn tại
        start_epoch = 5  # Mặc định epoch = 0
        print(f"✓ Checkpoint tìm thấy: {checkpoint_path}")
        print(f"✓ Epoch bắt đầu: {start_epoch}\n")
    else:
        checkpoint_path = None  # Không có checkpoint, train từ đầu
        start_epoch = 0
        print("✓ Không tìm thấy checkpoint, train từ đầu")
        print(f"✓ Epoch bắt đầu: {start_epoch}\n")

    # 2. Khởi tạo Environment
    env = PVRPWDPVEnv(
        epoch_data_dir=epoch_data_dir,
        epoch_file_pattern="epoch_{epoch:02d}_{part:02d}.npz",
        use_epoch_data=True,
        fallback_to_generator=False,
        val_file=val_file,
        test_file=test_file,
    )

    # Set current_epoch để load file npz tương ứng
    env.current_epoch = start_epoch

    env.print_epoch_data_info()

    # 3. Khoi tao Policy (voi feature moi)
    policy = PARCOPolicy(
        env_name=env.name,
        embed_dim=128,
        agent_handler="highprob",
        normalization="rms",
        context_embedding_kwargs={
            "normalization": "rms",
            "norm_after": False,
            "normalize_endurance_by_max": False,
            "use_time_to_depot": True,
            "use_claim_embed": True,
        },
        decoder_kwargs={
            "attention_injection_mode": "projected",  # "none" | "projected" | "mlp"
        },
        norm_after=False,
    )

    # 4. Khoi tao Model
    model = CustomPARCORLModule(
        env,
        policy=policy,
        batch_size=64,
        num_augment=10,
        use_padding_mode=True,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0},
    )

    # Không lưu env và policy objects vào checkpoint
    model.save_hyperparameters(ignore=["env", "policy"])

    # 5. Cấu hình Trainer (KÍCH HOẠT DDP - BỎ ACCUMULATE ĐỂ MAX TỐC ĐỘ)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_root, "my_checkpoints"),
        save_top_k=-1,
        every_n_epochs=1,
        filename="parco-{epoch:02d}",
    )
    csv_logger = CSVLogger(os.path.join(project_root, "csv_logs"), name="parco_training")

    # Thêm callback để update env.current_epoch mỗi khi epoch thay đổi
    epoch_callback = UpdateEpochCallback()

    trainer = RL4COTrainer(
        max_epochs=21,
        accelerator="gpu",
        devices=2,
        strategy=DDPStrategy(
            find_unused_parameters=True, timeout=timedelta(seconds=5400)
        ),
        precision="bf16-mixed",
        # accumulate_grad_batches=4,
        reload_dataloaders_every_n_epochs=1,
        use_distributed_sampler=False,
        logger=csv_logger,
        callbacks=[checkpoint_callback, epoch_callback],
        gradient_clip_val=1.0,
        # Không truyền tham số accumulate_grad_batches vào đây nữa
    )

    # 6. Chạy Train
    print("Training...")

    if checkpoint_path:
        trainer.fit(model, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model)

    # 7. Lưu weights cuối cùng
    # (Bắt buộc phải check is_global_zero để 2 GPU không tranh nhau ghi đè 1 file)
    if trainer.is_global_zero:
        save_path = os.path.join(project_root, "parco_policy_final.pth")
        torch.save(model.policy.state_dict(), save_path)
        print(f"✅ Đã lưu file model thành công tại: {save_path}")
        print("\n✅ Quá trình huấn luyện hoàn tất!")
