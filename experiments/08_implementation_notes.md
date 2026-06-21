# 代码实现记录

## 项目结构
```
hw3/
├── train.py                 # 训练入口脚本
├── config/
│   └── default.yaml         # 超参数配置
├── src/
│   ├── __init__.py          # 模块入口
│   ├── augmentations.py     # 弱增强 + 强增强 (RandAugment+Cutout)
│   ├── dataset.py           # 半监督数据划分 + DataLoader
│   ├── wideresnet.py        # WideResNet-28-2 骨干网络
│   └── fixmatch.py          # FixMatch 损失/EMA/interleave
├── report/
│   └── main.tex             # LaTeX 实验报告模板
├── experiments/             # 实验数据记录
├── checkpoints/             # 模型检查点
├── logs/                    # TensorBoard 日志
├── note.md                  # 实施计划
├── task.md                  # 实验要求
└── requirements.txt         # Python 依赖
```

## 参考来源
1. FixMatch 论文: Sohn et al., NeurIPS 2020. https://arxiv.org/abs/2001.07685
2. 官方 TF 实现: https://github.com/google-research/fixmatch (阅读算法流程，未直接复制代码)
3. USB 库: https://github.com/microsoft/Semi-supervised-learning
4. WideResNet: Zagoruyko & Komodakis, BMVC 2016. https://arxiv.org/abs/1605.07146
5. RandAugment: Cubuk et al., NeurIPS 2020. https://arxiv.org/abs/1909.13719
6. TorchSSL: https://github.com/TorchSSL/TorchSSL

## 自写核心模块
- 伪标签生成 (softmax → argmax → confidence mask): `src/fixmatch.py:fixmatch_loss()`
- 阈值判断 (max_prob >= τ): 同上函数
- 损失计算 (supervised CE + masked unsupervised CE): 同上函数
- EMA 更新与恢复: `src/fixmatch.py:EMA` 和 `EMAWithRestore`
- 半监督数据划分 (等量每类取样): `src/dataset.py:_split_labeled_indices()`
- 强增强 Cutout 实现: `src/augmentations.py:TransformStrong._cutout()`

## 使用的外部组件
- torchvision.transforms (标准库)
- torchvision.datasets.CIFAR10 (标准库)
- torch.optim.SGD (标准库)
- torch.optim.lr_scheduler.LambdaLR (标准库)

## 论文对齐修复记录

### 问题发现
训练结果始终低于论文预期：40 labels best 59.67%（论文 92.53%），250 labels best ~78%（论文 95.14%）。
且训练末期 accuracy 持续下跌，40 labels final 仅 29.20%，250 labels final 约 66%。
排查后发现 4 处与论文的偏差。

### 修复 1: LR 调度公式（Critical）
- **原代码**：`CosineAnnealingLR(eta_min=0)` → lr 最终衰减到 0
- **论文公式**：`η cos(7πk / 16K)`，终点 lr ≈ 0.00585
- **影响**：lr=0 导致训练末期模型进入确认偏误螺旋（错误伪标签 → 更错误 → 退化）
- **修复**：train.py, test_usb.py 改用 `LambdaLR` 实现论文公式

### 修复 2: 置信度阈值符号
- **原代码**：`max_probs >= confidence_threshold`
- **论文**：`1(max(q_b) > τ)` — 严格大于
- **修复**：`src/fixmatch.py:133` 改为 `>`

### 修复 3: labeled 数据纳入 unlabeled 池
- **原代码**：labeled 与 unlabeled 索引互斥（`class_indices[:K]` vs `class_indices[K:]`）
- **论文脚注 2**："include all labeled data as part of unlabeled data without their labels"
- **修复**：`src/dataset.py:_split_labeled_indices()` 中 unlabeled_indices 包含全部样本

### 修复 4: RandAugment 随机幅度 + Cutout 均值填充
- **原代码**：torchvision `RandAugment`（固定 magnitude）、Cutout 填 0（黑）
- **论文**："magnitude is randomly sampled from a pre-defined range"；Cutout "gray"
- **修复**：`src/augmentations.py` 实现 `RandAugmentPerSample`（每次变换随机 magnitude）、Cutout 填充 CIFAR-10 每通道均值

### 性能优化（训练速度）
- `train.py` 中 labeled_iter/unlabeled_iter 改为 `itertools.cycle()` 避免 DataLoader worker 反复创建销毁
- labeled_loader 使用 `num_workers=0`（数据量小，零进程开销）
