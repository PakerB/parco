import os
import torch
from rl4co.utils.trainer import RL4COTrainer
from parco.envs.pvrpwdp import PVRPWDPVEnv
from parco.models import PARCORLModule, PARCOPolicy
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import CSVLogger

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class CustomPARCORLModule(PARCORLModule):
    def train_dataloader(self):
        dl_old = super().train_dataloader()
        custom_sampler = DistributedSampler(dl_old.dataset, shuffle=False)
        return DataLoader(
            dl_old.dataset,
            batch_size=self.train_batch_size, # Sử dụng thuộc tính đã được thiết lập trong setup()
            sampler=custom_sampler, 
            num_workers=4,
            collate_fn=dl_old.collate_fn
        )

    def val_dataloader(self):
        dl_old = super().val_dataloader()
        custom_sampler = DistributedSampler(dl_old.dataset, shuffle=False)
        return DataLoader(
            dl_old.dataset,
            batch_size=self.val_batch_size, # Sử dụng thuộc tính đã được thiết lập trong setup()
            sampler=custom_sampler,
            num_workers=4,
            collate_fn=dl_old.collate_fn
        )

    def test_dataloader(self):
        dl_old = super().test_dataloader()
        custom_sampler = DistributedSampler(dl_old.dataset, shuffle=False)
        return DataLoader(
            dl_old.dataset,
            batch_size=self.test_batch_size, # Sử dụng thuộc tính đã được thiết lập trong setup()
            sampler=custom_sampler,
            num_workers=4,
            collate_fn=dl_old.collate_fn
        )


class EpochDataCallback(Callback):
    """Callback để update env.current_epoch mỗi khi trainer epoch thay đổi"""
    def __init__(self, env):
        super().__init__()
        self.env = env
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Gọi tại đầu mỗi epoch để update env.current_epoch"""
        # trainer.current_epoch bắt đầu từ 0
        self.env.current_epoch = trainer.current_epoch
        print(f"📊 Epoch {trainer.current_epoch}: Loading pvrpwdp_epoch_{trainer.current_epoch:02d}.npz")

if __name__ == '__main__':
    print("🚀 Khởi động quá trình huấn luyện bằng Python Script...")
    
    # 1. Setup đường dẫn (Tự động lấy thư mục hiện tại làm gốc)
    project_root = os.getcwd()
    epoch_data_dir = os.path.join(project_root, "data", "train_data_npz")
    val_file = os.path.join("val_data", "val.npz")
    test_file = os.path.join("test_data", "test.npz")
    
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
        epoch_file_pattern="epoch_{epoch:02d}.npz",
        use_epoch_data=True,
        fallback_to_generator=False,  
        val_file=val_file,
        test_file=test_file,
    )
    
    # Set current_epoch để load file npz tương ứng
    env.current_epoch = start_epoch
    
    env.print_epoch_data_info()
    
    # 3. Khởi tạo Policy
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
        },
        norm_after=False,
    )

    # 4. Khởi tạo Model 
    # TỐI ƯU TỐC ĐỘ: Mỗi GPU cõng 128 mẫu. Tổng 2 GPU = 256 mẫu/bước.
    # Sử dụng class CustomPARCORLModule
    model = CustomPARCORLModule(
        env, 
        policy=policy,
        train_data_size=400384,     
        val_data_size=20736,      
        
        # Thiết lập kích thước batch riêng biệt
        batch_size=256,            # Sẽ được dùng làm train_batch_size
        val_batch_size=256,        # Kích thước batch cho validation
        test_batch_size=256,       # Kích thước batch cho test
        
        num_augment=16,            
        use_padding_mode=True,  
        optimizer_kwargs={'lr': 1e-4, 'weight_decay': 0},
    )

    # Không lưu env và policy objects vào checkpoint
    model.save_hyperparameters(ignore=['env', 'policy'])

    # 5. Cấu hình Trainer (KÍCH HOẠT DDP - BỎ ACCUMULATE ĐỂ MAX TỐC ĐỘ)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(project_root, "my_checkpoints"),
        save_top_k=-1,
        every_n_epochs=1,
        filename="parco-{epoch:02d}"
    )
    csv_logger = CSVLogger(os.path.join(project_root, "csv_logs"), name="parco_training")
    
    # Thêm callback để update env.current_epoch mỗi khi epoch thay đổi
    epoch_data_callback = EpochDataCallback(env)

    trainer = RL4COTrainer(
        max_epochs=5,          # 
        accelerator="gpu",
        devices=2,              #  Dùng cả 2 card T4
        strategy="ddp_find_unused_parameters_true",
        reload_dataloaders_every_n_epochs=1,
        use_distributed_sampler=False,     
        accumulate_grad_batches=2,       
        logger=csv_logger,               
        callbacks=[checkpoint_callback, epoch_data_callback],  # Thêm callback
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
        save_path = os.path.join(project_root, 'parco_policy_final.pth')
        torch.save(model.policy.state_dict(), save_path)
        print(f"✅ Đã lưu file model thành công tại: {save_path}")
        print("\n✅ Quá trình huấn luyện hoàn tất!")