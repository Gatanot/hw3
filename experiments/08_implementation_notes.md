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
- torchvision.transforms.RandAugment (标准库)
- torchvision.datasets.CIFAR10 (标准库)
- torch.optim.SGD (标准库)
- torch.optim.lr_scheduler.CosineAnnealingLR (标准库)
