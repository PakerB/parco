# EPOCH-BASED DATA LOADING FOR PARCO

## 📋 TỔNG QUAN

Module `EpochDataEnvBase` cho phép load dữ liệu training từ các file `.npz` được sinh trước theo từng epoch, thay vì dùng generator trong quá trình training.

**Lợi ích:**
- ✅ **Curriculum Learning**: Mỗi epoch có độ khó tăng dần
- ✅ **Pre-generated Data**: Tiết kiệm thời gian training (không phải generate data mỗi lần)
- ✅ **External Data**: Có thể dùng data từ OR-Tools, expert solutions, v.v.
- ✅ **Reproducibility**: Data giống nhau cho mỗi epoch, dễ debug và so sánh
- ✅ **Backward Compatible**: Vẫn hỗ trợ fallback về generator nếu file không tồn tại

---

## 1. CÀI ĐẶT

### 1.1. Import

```python
from parco.envs.epoch_data_env_base import EpochDataEnvBase
```

### 1.2. Tạo Environment kế thừa

```python
from parco.envs.epoch_data_env_base import EpochDataEnvBase
from parco.envs.pvrpwdp.generator import PVRPWDPGenerator

class PVRPWDPEpochEnv(EpochDataEnvBase):
    """PVRPWDP Environment with epoch-based data loading."""
    
    name = "pvrpwdp_epoch"
    
    def __init__(
        self,
        generator: PVRPWDPGenerator = None,
        generator_params: dict = {},
        epoch_data_dir: str = "data/pvrpwdp/train_epochs/",
        epoch_file_pattern: str = "epoch_{epoch}.npz",
        **kwargs,
    ):
        if generator is None:
            generator = PVRPWDPGenerator(**generator_params)
        
        super().__init__(
            generator=generator,
            epoch_data_dir=epoch_data_dir,
            epoch_file_pattern=epoch_file_pattern,
            use_epoch_data=True,
            fallback_to_generator=True,
            **kwargs
        )
        
        self._make_spec(self.generator)
    
    # Copy all methods from PVRPWDPVEnv (_reset, _step, _get_reward, etc.)
    # ... (giữ nguyên implementation của PVRPWDPVEnv)
```

---

## 2. CHUẨN BỊ DỮ LIỆU

### 2.1. Cấu trúc thư mục

```
data/
└── pvrpwdp/
    ├── train_epochs/
    │   ├── epoch_0.npz
    │   ├── epoch_1.npz
    │   ├── epoch_2.npz
    │   └── ...
    ├── val/
    │   └── validation.npz
    └── test/
        └── test.npz
```

### 2.2. Sinh dữ liệu cho các epoch

```python
import torch
import numpy as np
from parco.envs.pvrpwdp import PVRPWDPVEnv, PVRPWDPGenerator
from pathlib import Path

def generate_epoch_data(
    output_dir: str = "data/pvrpwdp/train_epochs/",
    num_epochs: int = 100,
    batch_size: int = 10000,
    num_loc: int = 20,
    num_agents: int = 3,
):
    """Generate training data for each epoch with curriculum learning."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"Generating data for epoch {epoch}/{num_epochs}...")
        
        # Curriculum learning: increase difficulty over epochs
        # Example: increase time window tightness
        tw_width_min = 30 - (epoch / num_epochs) * 10  # From 30 to 20
        tw_width_max = 50 - (epoch / num_epochs) * 10  # From 50 to 40
        
        generator = PVRPWDPGenerator(
            num_loc=num_loc,
            num_agents=num_agents,
            tw_width_min=tw_width_min,
            tw_width_max=tw_width_max,
        )
        
        # Generate data
        td = generator(batch_size=[batch_size])
        
        # Save to file
        output_file = output_path / f"epoch_{epoch}.npz"
        
        # Convert TensorDict to numpy dict
        data_dict = {}
        for key, value in td.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.cpu().numpy()
        
        np.savez_compressed(output_file, **data_dict)
        print(f"  Saved to {output_file}")
    
    print(f"\nGenerated {num_epochs} epoch files in {output_dir}")

# Usage
if __name__ == "__main__":
    generate_epoch_data(
        output_dir="data/pvrpwdp/train_epochs/",
        num_epochs=100,
        batch_size=10000,
        num_loc=20,
        num_agents=3,
    )
```

---

## 3. SỬ DỤNG TRONG TRAINING

### 3.1. Config file (YAML)

```yaml
# configs/env/pvrpwdp_epoch.yaml
defaults:
  - pvrpwdp  # Inherit from base pvrpwdp config

# Override class
_target_: parco.envs.pvrpwdp_epoch_env.PVRPWDPEpochEnv

# Epoch data settings
epoch_data_dir: "data/pvrpwdp/train_epochs/"
epoch_file_pattern: "epoch_{epoch}.npz"
use_epoch_data: true
fallback_to_generator: true

# Validation and test files (standard)
val_file: "data/pvrpwdp/val/validation.npz"
test_file: "data/pvrpwdp/test/test.npz"
```

### 3.2. Training script

```python
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    # Environment will automatically load epoch data
    env = hydra.utils.instantiate(cfg.env)
    
    # Model
    model = hydra.utils.instantiate(cfg.model, env=env)
    
    # Trainer
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        # ... other configs
    )
    
    # Train
    # Environment will automatically:
    # - Load data from epoch_{current_epoch}.npz for training
    # - Use validation.npz for validation
    # - Use test.npz for testing
    trainer.fit(model)

if __name__ == "__main__":
    main()
```

---

## 4. API REFERENCE

### 4.1. `EpochDataEnvBase.__init__`

```python
def __init__(
    self,
    *,
    epoch_data_dir: str = None,           # Thư mục chứa epoch files
    epoch_file_pattern: str = "epoch_{epoch}.npz",  # Pattern tên file
    use_epoch_data: bool = True,          # Bật/tắt epoch data loading
    fallback_to_generator: bool = True,   # Fallback về generator nếu không tìm thấy file
    **kwargs,
)
```

**Parameters:**
- `epoch_data_dir`: Đường dẫn đến thư mục chứa epoch files
- `epoch_file_pattern`: Pattern tên file với `{epoch}` placeholder
- `use_epoch_data`: Bật/tắt epoch data loading
- `fallback_to_generator`: Nếu `True`, dùng generator khi file không tồn tại

### 4.2. `get_epoch_file_path(epoch: int) -> str`

Lấy đường dẫn file cho epoch cụ thể.

```python
>>> env.get_epoch_file_path(10)
'data/pvrpwdp/train_epochs/epoch_10.npz'
```

### 4.3. `list_available_epochs() -> list`

Liệt kê tất cả epoch files có sẵn.

```python
>>> env.list_available_epochs()
[0, 1, 2, 5, 10, 15, 20, 25, 30]
```

### 4.4. `validate_epoch_files(max_epoch: int) -> dict`

Kiểm tra tính hợp lệ của epoch files.

```python
>>> results = env.validate_epoch_files(max_epoch=100)
>>> print(results)
{
    'missing': [3, 4, 6, 7, ...],
    'corrupted': [],
    'valid': [0, 1, 2, 5, 10, ...],
    'total_expected': 100
}
```

### 4.5. `print_epoch_data_info()`

In thông tin về epoch data configuration.

```python
>>> env.print_epoch_data_info()
============================================================
EPOCH DATA CONFIGURATION
============================================================
Epoch Data Directory: data/pvrpwdp/train_epochs/
File Pattern:         epoch_{epoch}.npz
Use Epoch Data:       True
Fallback to Generator: True
Current Epoch:        10
Max Epochs:           100

Available Epochs:     50
Epoch Range:          0 - 99
Sample Epochs:        [0, 1, 2, 3, 4] ... [95, 96, 97, 98, 99]
✅ Current epoch 10 file exists
============================================================
```

---

## 5. CURRICULUM LEARNING STRATEGIES

### 5.1. Độ khó tăng dần theo epoch

```python
def generate_curriculum_data(epoch: int, num_epochs: int):
    """Generate data with increasing difficulty."""
    
    # Strategy 1: Tighten time windows
    tw_width_min = 30 - (epoch / num_epochs) * 10  # 30 → 20
    tw_width_max = 50 - (epoch / num_epochs) * 10  # 50 → 40
    
    # Strategy 2: Increase number of locations
    num_loc = 10 + int((epoch / num_epochs) * 40)  # 10 → 50
    
    # Strategy 3: Decrease vehicle capacity
    capacity_scale = 1.0 - (epoch / num_epochs) * 0.3  # 1.0 → 0.7
    
    # Strategy 4: Increase perishability (shorter waiting_time)
    waiting_time_scale = 1.0 - (epoch / num_epochs) * 0.5  # 1.0 → 0.5
    
    generator = PVRPWDPGenerator(
        num_loc=num_loc,
        tw_width_min=tw_width_min,
        tw_width_max=tw_width_max,
        capacity_scale=capacity_scale,
        waiting_time_scale=waiting_time_scale,
    )
    
    return generator(batch_size=[10000])
```

### 5.2. Kết hợp nhiều chiến lược

```python
# Warmup (epoch 0-20): Easy instances
if epoch < 20:
    num_loc = 10
    tw_width = 50
    
# Main training (epoch 20-80): Gradually increase difficulty
elif epoch < 80:
    progress = (epoch - 20) / 60
    num_loc = 10 + int(progress * 40)
    tw_width = 50 - progress * 20
    
# Final phase (epoch 80-100): Hard instances
else:
    num_loc = 50
    tw_width = 30
```

---

## 6. BEST PRACTICES

### ✅ **DO:**
1. **Validate epoch files before training**
   ```python
   env = PVRPWDPEpochEnv(...)
   results = env.validate_epoch_files(max_epoch=100)
   if results['missing']:
       print(f"⚠️  Missing {len(results['missing'])} files!")
   ```

2. **Use meaningful file patterns**
   ```python
   # Good
   epoch_file_pattern="pvrpwdp_n20_e{epoch}.npz"
   
   # Bad
   epoch_file_pattern="data_{epoch}.npz"  # Too generic
   ```

3. **Enable fallback for robustness**
   ```python
   fallback_to_generator=True  # Training won't crash if file missing
   ```

4. **Check available epochs**
   ```python
   env.print_epoch_data_info()  # Before training
   ```

### ❌ **DON'T:**
1. **Don't hardcode paths**
   ```python
   # Bad
   epoch_data_dir="C:/Users/me/data/epochs/"
   
   # Good
   epoch_data_dir="data/train_epochs/"  # Relative path
   ```

2. **Don't skip validation**
   ```python
   # Bad: Start training without checking
   trainer.fit(model)
   
   # Good: Validate first
   results = env.validate_epoch_files()
   assert len(results['corrupted']) == 0
   trainer.fit(model)
   ```

3. **Don't use same data for all epochs**
   ```python
   # Bad: No curriculum learning
   for epoch in range(100):
       generate_same_data(epoch)
   
   # Good: Increase difficulty
   for epoch in range(100):
       generate_curriculum_data(epoch, num_epochs=100)
   ```

---

## 7. TROUBLESHOOTING

### ❓ **File not found error**
```
FileNotFoundError: Epoch file not found for epoch 10 and fallback_to_generator=False
```

**Solution:**
1. Check file exists: `ls data/train_epochs/epoch_10.npz`
2. Validate pattern: `env.get_epoch_file_path(10)`
3. Enable fallback: `fallback_to_generator=True`

### ❓ **Batch size mismatch warning**
```
WARNING: Loaded data batch size 5000 does not match requested batch size 10000
```

**Solution:**
- This is just a warning, training will continue
- To fix: Regenerate epoch files with correct batch size

### ❓ **Corrupted file error**
```
ERROR: Error loading epoch file 'epoch_5.npz': ...
```

**Solution:**
1. Validate files: `env.validate_epoch_files()`
2. Delete corrupted file
3. Regenerate: `generate_epoch_data(start_epoch=5, end_epoch=6)`

### ❓ **Memory issues**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size in epoch files
- Use compressed format: `np.savez_compressed()`
- Load data on CPU: `td.to("cpu")`

---

## 8. VÍ DỤ ĐẦY ĐỦ

### 8.1. Generate data

```python
# scripts/generate_epoch_data.py
import torch
import numpy as np
from pathlib import Path
from parco.envs.pvrpwdp import PVRPWDPGenerator

def main():
    output_dir = Path("data/pvrpwdp/train_epochs/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(100):
        # Curriculum: tighter time windows over time
        progress = epoch / 100
        tw_width_min = 30 - progress * 10
        tw_width_max = 50 - progress * 10
        
        generator = PVRPWDPGenerator(
            num_loc=20,
            num_agents=3,
            tw_width_min=tw_width_min,
            tw_width_max=tw_width_max,
        )
        
        td = generator(batch_size=[10000])
        
        # Save
        data_dict = {k: v.cpu().numpy() for k, v in td.items() if isinstance(v, torch.Tensor)}
        np.savez_compressed(output_dir / f"epoch_{epoch}.npz", **data_dict)
        
        print(f"✅ Epoch {epoch}: tw_width=[{tw_width_min:.1f}, {tw_width_max:.1f}]")

if __name__ == "__main__":
    main()
```

### 8.2. Use in training

```python
# train.py
from parco.envs.pvrpwdp_epoch_env import PVRPWDPEpochEnv

# Create environment
env = PVRPWDPEpochEnv(
    epoch_data_dir="data/pvrpwdp/train_epochs/",
    epoch_file_pattern="epoch_{epoch}.npz",
    use_epoch_data=True,
    fallback_to_generator=True,
)

# Validate before training
env.print_epoch_data_info()
results = env.validate_epoch_files(max_epoch=100)
print(f"Valid: {len(results['valid'])}, Missing: {len(results['missing'])}")

# Train
model = PARCOPolicy(env=env)
trainer = Trainer(max_epochs=100)
trainer.fit(model)
```

---

## 9. SO SÁNH VỚI RL4CO STANDARD

| Aspect | RL4CO Standard | EpochDataEnvBase |
|--------|----------------|------------------|
| **Data Source** | Generator on-the-fly | Pre-generated epoch files |
| **Curriculum** | ❌ Not built-in | ✅ Supported |
| **Reproducibility** | ⚠️  Depends on seed | ✅ Exact same data |
| **Training Speed** | ⚠️  Slower (generate each time) | ✅ Faster (pre-generated) |
| **Memory** | ✅ Lower | ⚠️  Higher (store all files) |
| **Flexibility** | ✅ Easy to change | ⚠️  Need to regenerate |
| **External Data** | ❌ Not supported | ✅ Supported |

---

## 10. KẾT LUẬN

`EpochDataEnvBase` là một extension hữu ích cho RL4CO khi bạn cần:
- ✅ **Curriculum learning** với độ khó tăng dần
- ✅ **Pre-generated data** để tăng tốc training
- ✅ **Reproducibility** với data cố định
- ✅ **External data** từ OR-Tools, expert solutions, v.v.

**Không cần sửa RL4CO library**, chỉ cần kế thừa `EpochDataEnvBase` trong project parco! 🚀
