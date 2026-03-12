# PVRPWDP Epoch Data Training

Hệ thống training với epoch data cho phép load dữ liệu từ file pre-generated thay vì generate on-the-fly.

## 🎯 Lợi ích

### 1. **Curriculum Learning**
- Tăng độ khó dần theo epoch (10 → 50 locations)
- Model học tốt hơn với incremental difficulty

### 2. **Reproducibility**
- Cùng training data cho mọi run
- Dễ compare results giữa các experiments

### 3. **Performance**
- Không tốn thời gian generate data mỗi epoch
- Training nhanh hơn ~10-20%

## 📦 Cấu trúc File

```
data/pvrpwdp/train_epochs/
├── epoch_0.npz       # Epoch 0: 10 locations (easy)
├── epoch_1.npz       # Epoch 1: 11 locations
├── epoch_2.npz
├── ...
├── epoch_49.npz
└── epoch_50.npz      # Epoch 50: 50 locations (hard)
```

Mỗi file chứa:
- **depot**: [B, 2] - Depot location
- **locs**: [B, N, 2] - Customer locations
- **demand**: [B, N] - Customer demands
- **time_window**: [B, N, 2] - Time windows [earliest, latest]
- **waiting_time**: [B, N] - Freshness constraints
- **speed**: [B, M] - Vehicle speeds
- **capacity**: [B, M] - Vehicle capacities
- **endurance**: [B, M] - Drone endurance

## 🚀 Sử dụng

### Bước 1: Generate Epoch Data

#### Option A: Fixed Difficulty (recommended để test)
```bash
# Generate 10 epochs, mỗi epoch 100 instances, 20 locations
python generate_pvrpwdp_epochs.py \
    --num_epochs 10 \
    --batch_size 100 \
    --num_loc 20 \
    --num_agents 3 \
    --output_dir data/pvrpwdp/train_epochs/
```

#### Option B: Curriculum Learning (recommended để train)
```bash
# Generate 100 epochs với curriculum: 10 → 50 locations
python generate_pvrpwdp_epochs.py \
    --curriculum \
    --num_epochs 100 \
    --batch_size 1000 \
    --start_num_loc 10 \
    --end_num_loc 50 \
    --num_agents 3 \
    --output_dir data/pvrpwdp/train_epochs/
```

#### Tùy chỉnh generator parameters:
```bash
python generate_pvrpwdp_epochs.py \
    --num_epochs 50 \
    --batch_size 500 \
    --num_loc 30 \
    --num_agents 5 \
    --min_demand 1 \
    --max_demand 10 \
    --capacity 40.0 \
    --endurance 10.0 \
    --speed 1.0 \
    --drone_speed 2.0 \
    --tw_expansion 3.0 \
    --freshness_factor 2.0
```

### Bước 2: Test Epoch Data Loading

```bash
python test_pvrpwdp_epoch.py
```

Expected output:
```
============================================================
EPOCH DATA CONFIGURATION
============================================================
Epoch Data Directory: data/pvrpwdp/train_epochs/
File Pattern:         epoch_{epoch}.npz
Use Epoch Data:       True
Fallback to Generator: True
Current Epoch:        0
Max Epochs:           None

Available Epochs:     10
Epoch Range:          0 - 9
✅ Current epoch 0 file exists
============================================================
```

### Bước 3: Train với Epoch Data

#### Option A: Sử dụng config file

Tạo config `configs/experiment/pvrpwdp.yaml`:
```yaml
# @package _global_

defaults:
  - override /env: pvrpwdp_epoch.yaml
  - override /model: parco.yaml
  - override /trainer: default.yaml

# Model config
model:
  num_augment: 8

# Trainer config
trainer:
  max_epochs: 100
  
# Environment config
env:
  epoch_data_dir: "data/pvrpwdp/train_epochs/"
  epoch_file_pattern: "epoch_{epoch}.npz"
  use_epoch_data: true
  fallback_to_generator: true
```

Train:
```bash
python train.py experiment=pvrpwdp
```

#### Option B: Sử dụng command line

```bash
python train.py \
    env=pvrpwdp_epoch \
    env.epoch_data_dir="data/pvrpwdp/train_epochs/" \
    env.use_epoch_data=true \
    model=parco \
    trainer.max_epochs=100
```

## 🔧 Cấu hình

### Environment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epoch_data_dir` | str | None | Directory chứa epoch files |
| `epoch_file_pattern` | str | "epoch_{epoch}.npz" | File naming pattern |
| `use_epoch_data` | bool | True | Enable epoch data loading |
| `fallback_to_generator` | bool | True | Use generator if file missing |

### Generator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_loc` | int | 20 | Số lượng customers |
| `num_agents` | int | 3 | Số lượng vehicles |
| `min_demand` | int | 1 | Min customer demand |
| `max_demand` | int | 10 | Max customer demand |
| `capacity` | float | 40.0 | Vehicle capacity |
| `endurance` | float | 10.0 | Drone battery endurance |
| `speed` | float | 1.0 | Truck speed |
| `drone_speed` | float | 2.0 | Drone speed (faster) |
| `tw_expansion` | float | 3.0 | Time window size factor |
| `freshness_factor` | float | 2.0 | Freshness constraint factor |

## 📊 Monitoring

### Check available epochs:
```python
from parco.envs.pvrpwdp.env import PVRPWDPVEnv

env = PVRPWDPVEnv(epoch_data_dir="data/pvrpwdp/train_epochs/")
epochs = env.list_available_epochs()
print(f"Available epochs: {epochs}")
```

### Validate epoch files:
```python
results = env.validate_epoch_files(max_epoch=100)
print(f"Valid: {len(results['valid'])}")
print(f"Missing: {len(results['missing'])}")
print(f"Corrupted: {len(results['corrupted'])}")
```

### Print epoch info:
```python
env.print_epoch_data_info()
```

## 🔄 Workflow

```
1. Generate epoch data
   ↓
2. Validate data files
   ↓
3. Test data loading
   ↓
4. Train model với epoch data
   ↓
5. Monitor training (epoch files automatically loaded)
```

## 💡 Best Practices

### 1. **Pre-generate đủ epochs**
```bash
# Generate thêm 10-20% epochs để tránh thiếu
python generate_pvrpwdp_epochs.py --num_epochs 120  # For 100 epoch training
```

### 2. **Validate trước khi train**
```bash
python test_pvrpwdp_epoch.py  # Check all files are valid
```

### 3. **Sử dụng fallback_to_generator=True**
- Nếu epoch file thiếu/corrupt → tự động generate
- Không dừng training giữa chừng

### 4. **Curriculum Learning**
```bash
# Start easy → gradually increase difficulty
python generate_pvrpwdp_epochs.py \
    --curriculum \
    --start_num_loc 10 \
    --end_num_loc 50
```

### 5. **Backup epoch data**
```bash
# Epoch data rất quý! Backup ngay
tar -czf train_epochs_backup.tar.gz data/pvrpwdp/train_epochs/
```

## 🐛 Troubleshooting

### Issue: "Epoch file not found"
**Solution:**
1. Check `epoch_data_dir` path đúng chưa
2. Check file naming pattern: `epoch_{epoch}.npz`
3. Enable `fallback_to_generator=True` để tự động generate

### Issue: "Corrupted epoch file"
**Solution:**
```bash
# Re-generate epoch bị corrupted
python generate_pvrpwdp_epochs.py --num_epochs 1 --batch_size 1000
# Rename to correct epoch number
mv data/pvrpwdp/train_epochs/epoch_0.npz data/pvrpwdp/train_epochs/epoch_X.npz
```

### Issue: Training slow với epoch data
**Check:**
- Disk I/O speed (SSD recommended)
- File size không quá lớn (max ~100MB/file)
- Batch size phù hợp với memory

## 📝 Example Configs

### Small Dataset (Testing)
```yaml
env:
  epoch_data_dir: "data/pvrpwdp/train_epochs_small/"
  generator_params:
    num_loc: 10
    num_agents: 2
```

### Large Dataset (Production)
```yaml
env:
  epoch_data_dir: "data/pvrpwdp/train_epochs_large/"
  generator_params:
    num_loc: 50
    num_agents: 5
```

### Curriculum Dataset
```yaml
env:
  epoch_data_dir: "data/pvrpwdp/train_epochs_curriculum/"
  # Each epoch has different num_loc (pre-generated)
```

## 🎓 Advanced Usage

### Custom epoch file pattern:
```python
env = PVRPWDPVEnv(
    epoch_data_dir="data/custom/",
    epoch_file_pattern="pvrpwdp_e{epoch}_data.npz"
)
# Will load: pvrpwdp_e0_data.npz, pvrpwdp_e1_data.npz, ...
```

### Load specific epoch:
```python
env.current_epoch = 10
dataset = env.dataset(batch_size=[100], phase="train")
# Will load: epoch_10.npz
```

### Disable epoch data (standard RL4CO):
```python
env = PVRPWDPVEnv(
    use_epoch_data=False  # Use generator like normal
)
```
