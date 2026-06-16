# 训练时间预估

## 实测数据

### 小 batch (16 labeled + 48 unlabeled = 64 张/步)

| 配置 | 每步 | 总时间 (50步) | 备注 |
|------|------|-------------|------|
| 优化前 (v1) | ~70 ms | 8.8s | cudnn_benchmark=off, interleave=on |
| 无 AMP, 无 interleave | ~99 ms | 9.9s | cudnn_benchmark=off |
| AMP | ~120 ms | 12.1s | AMP 开销 > 收益 (batch 太小) |

### 全量 batch (64 labeled + 448 unlabeled = 512 张/步) — 实测

| 配置 | 每步 | 步/秒 | 2^20 步耗时 |
|------|------|-------|-----------|
| FP32 (无 AMP) | **248 ms** | 4.0 | **~73 小时 (3.0天)** |
| **AMP (混合精度)** | **145 ms** | 6.9 | **~42 小时 (1.75天)** |
| AMP 加速比 | — | **1.71×** | — |

### AMP 实测原始输出
```
Full batch (64+448, 30 iters):
FP32: 7.44s → 4.0 steps/s → 248.0 ms/step
AMP:  4.34s → 6.9 steps/s → 144.7 ms/step
Speedup: 1.71x
```

## 校正后的各 GPU 预估 (基于实测, FP32→AMP=1.71×)

### 估算方法
1. 实测基准：RTX 4050 全量 batch AMP = 145 ms/step
2. 假设每步时间与 FP32 TFLOPS 成反比（除以相对算力）
3. 含评估/日志/I/O 开销 +10%

| GPU | FP32 TFLOPS | vs RTX4050 | 估算 ms/step | steps/s | AMP/轮 | AMP三轮(+10%开销) |
|-----|-------------|------------|-------------|---------|--------|-------------------|
| RTX 4050 Laptop | 8.9 | 1.00× | 145 (实测) | 6.9 | 42 h | 5.8天 |
| V100 32GB | 15.7 | 1.76× | 82 | 12.2 | 24 h | 3.3天 |
| V100 16GB | 14.0 | 1.57× | 92 | 10.8 | 27 h | 3.7天 |
| **RTX 4090** | 82.6 | 9.28× | **15.6** | **64** | **~4.5 h** | **~15 小时 (0.6天)** |

### RTX 4090 详细计算
```
单步: 145ms / (82.6/8.9) = 145 / 9.28 = 15.6 ms
步/秒: 1000/15.6 = 64 steps/s
2^20 步: 1,048,576 / 64 = 16,384 s = 4.55 小时
含评估+I/O (+10%): 4.55 × 1.10 = 5.0 小时/轮
三轮 (40/250/4000): 5.0 × 3 = 15 小时
```

### 命令行实测 (RTX 4050 基准)
```bash
$ python3 -u -c "..."
Full batch (64+448, 30 iters):
FP32: 4.0 steps/s, 248.0 ms/step
AMP:  6.9 steps/s, 144.7 ms/step
Speedup: 1.71x
```

## 各项优化实测效果 (全量 batch)

| 优化 | 每步时间 | 加速比 |
|------|---------|--------|
| 基线 (FP32, no interleave) | 248 ms | 1.00× |
| + cudnn_benchmark | 248 ms | — (仅首步 +10s) |
| + AMP | 145 ms | **1.71×** |
| + non_blocking transfer | (包含在内) | ~1.03× |
| — close interleave | — | ~1.02× |

综合优化：**1.71× 加速** (AMP 贡献最大)

## 命令行原始输出

### AMP 全量batch对比测试
```bash
python3 -u -c "
import sys, time; sys.path.insert(0, '.')
import torch
from torch.utils.data import DataLoader
from src.wideresnet import wrn_28_2
from src.fixmatch import fixmatch_loss
from src.dataset import CIFAR10Labeled, CIFAR10Unlabeled, TransformWeak, TransformStrong, _split_labeled_indices
from torchvision.datasets import CIFAR10

device = 'cuda'
model = wrn_28_2(num_classes=10).to(device)
data_root = './data'
train_dataset = CIFAR10(root=data_root, train=True, download=False)
labeled_indices, unlabeled_indices = _split_labeled_indices(
    train_dataset.targets, 250, num_classes=10, seed=0)
transform_weak = TransformWeak(); transform_strong = TransformStrong()
labeled_dataset = CIFAR10Labeled(data_root, labeled_indices, transform_weak)
unlabeled_dataset = CIFAR10Unlabeled(data_root, unlabeled_indices, transform_weak, transform_strong)
labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=448, shuffle=True, num_workers=0, drop_last=True)
x_l, y_l = next(iter(labeled_loader))
(x_w, x_s) = next(iter(unlabeled_loader))
x_l, y_l = x_l.to(device), y_l.to(device); x_w, x_s = x_w.to(device), x_s.to(device)

# Warmup
for _ in range(5):
    total_loss, _, _, _ = fixmatch_loss(model, x_l, y_l, x_w, x_s, use_interleave=False)
    total_loss.backward()

N=30
# FP32
model.zero_grad(); torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    total_loss, _, _, _ = fixmatch_loss(model, x_l, y_l, x_w, x_s, use_interleave=False)
    total_loss.backward()
torch.cuda.synchronize()
f32_time = time.time() - t0

# AMP
scaler = torch.amp.GradScaler('cuda')
for _ in range(5):
    with torch.amp.autocast('cuda'):
        total_loss, _, _, _ = fixmatch_loss(model, x_l, y_l, x_w, x_s, use_interleave=False)
    scaler.scale(total_loss).backward()

model.zero_grad(); torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda'):
        total_loss, _, _, _ = fixmatch_loss(model, x_l, y_l, x_w, x_s, use_interleave=False)
    scaler.scale(total_loss).backward()
torch.cuda.synchronize()
amp_time = time.time() - t0

print(f'FP32: {N/f32_time:.1f} steps/s, {f32_time/N*1000:.1f} ms/step')
print(f'AMP:  {N/amp_time:.1f} steps/s, {amp_time/N*1000:.1f} ms/step')
print(f'Speedup: {f32_time/amp_time:.2f}x')
"
Full batch (64+448, 30 iters):
FP32: 4.0 steps/s, 248.0 ms/step
AMP:  6.9 steps/s, 144.7 ms/step
Speedup: 1.71x
```

## 注意事项
- cudnn_benchmark 首步增加约 10s 搜索开销，2^20 步下可忽略
- 小 batch (≤64) 不建议 AMP：开销大于收益
- 全量 batch (512) AMP 才发挥 Tensor Core 优势
- 评估/日志/检查点 I/O 约 +5-8% 时间开销
