# 性能优化验证

## 优化项列表

| # | 优化 | 实现方式 | 状态 |
|---|------|---------|------|
| 1 | AMP 混合精度 | `torch.amp.autocast('cuda')` + `GradScaler` | 已实现 |
| 2 | 关闭 interleave | `--use_interleave` flag, 默认 False | 已实现 |
| 3 | non_blocking 传输 | `.to(device, non_blocking=True)` | 已实现 |
| 4 | cudnn_benchmark | `torch.backends.cudnn.benchmark = True` | 已实现 |
| 5 | pin_memory | DataLoader `pin_memory=True` | 已实现 |
| 6 | 数据预取 | DataLoader `prefetch_factor=2` | 未实现 |

## 实测结果

### 测试 1: 小 batch AMP (不推荐)
batch=16+48=64 张/步:
```
FP32:  9.9s/50步 → ~99 ms/step
AMP:   12.1s/50步 → ~120 ms/step
AMP 慢 21% — FP16 开销 > 收益 (batch 太小)
```

### 测试 2: 全量 batch AMP (推荐)
batch=64+448=512 张/步:
```
FP32: 248 ms/step → 4.0 steps/s → 73 小时/轮
AMP:  145 ms/step → 6.9 steps/s → 42 小时/轮
AMP 加速 1.71×
```

### 测试 3: cudnn_benchmark 效应
```
关闭: 首步 ~0.6s
开启: 首步 ~12.3s (+11.7s 卷积算法搜索)
后续步骤无差异
```
对 2^20 步训练：首次开销可忽略。

### 测试 4: interleave 效应
```
interleave=True:  无显著速度差异 (单 GPU)
interleave=False: 无显著速度差异 (单 GPU)
```
interleave 仅影响多 GPU BN 统计，单 GPU 关闭不影响速度或精度。

## 结论
- **AMP** 在全量 batch 下是最有效的优化（1.71×）
- **cudnn_benchmark** 长期训练启用（启动代价可忽略）
- **non_blocking** + **pin_memory** 微小幅优化
- 单 GPU 下关闭 **interleave** 无害
- 综合优化后 RTX 4050 全量训练：**~42 小时/轮**
