# FixMatch 半监督图像分类 — 项目实施计划

## 参考文献
- [1] FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence (Sohn et al., NeurIPS 2020)
- [2] MixMatch: A Holistic Approach to Semi-Supervised Learning (Berthelot et al., NeurIPS 2019)
- [3] USB: A Unified Semi-supervised Learning Benchmark (Wang et al., NeurIPS 2022)
- [4] 原始 TF 实现: https://github.com/google-research/fixmatch
- [5] USB 库: https://github.com/microsoft/Semi-supervised-learning

---

## 阶段 0：文献阅读与算法理解 (done)

**FixMatch 核心流程：**
1. **有监督分支**：标注图像 → 弱增强(flip+shift) → 交叉熵损失
2. **无监督分支**：
   - 同一无标注图像，分别施加弱增强和强增强
   - 弱增强 → argmax → 伪标签 (需 confidence > τ=0.95)
   - 强增强(RandAugment + Cutout) → 用伪标签计算交叉熵
3. **总损失**：L_sup + λ * L_unsup (λ=1)

**关键超参数**：
- Backbone: WideResNet-28-2
- 优化器: SGD + momentum (0.9) + nesterov
- LR: 0.03, cosine decay, warmup 0
- Batch: labeled=64, unlabeled=448 (μ=7)
- 总步数: 2^20 = 1,048,576
- EMA decay: 0.999
- Weight decay: 0.0005
- 置信度阈值: 0.95

**与 MixMatch 的区别**：
- MixMatch: 多次增强取平均 + 温度锐化 + MixUp 混合
- FixMatch: 仅一次弱增强→伪标签 + 一次强增强→一致性，更简洁

---

## 阶段 1：项目骨架搭建
- 创建目录结构
- `requirements.txt`
- LaTeX 实验报告模板 (`report/`)
- 配置文件 (`config/`)
- Git 初始化

## 阶段 2：数据加载模块（小规模 CPU 验证）
- CIFAR-10 半监督划分 (40/250/4000 标注)
- 弱增强: RandomHorizontalFlip + RandomCrop(32, padding=4)
- 强增强: RandAugment + Cutout
- 用少量数据 (如 40 labeled) 验证 DataLoader 正确性

## 阶段 3：WideResNet-28-2 骨干网络
- 实现 WideResNet (depth=28, widen_factor=2)
- CPU 小规模前向传播验证形状正确
- 参考: 论文附录 + 开源实现

## 阶段 4：FixMatch 核心模块
- **伪标签生成**：弱增强 → argmax → 伪标签（仅保留 confidence > τ）
- **阈值判断**：mask = max_prob >= τ
- **损失计算**：supervised CE + λ * unsupervised CE (masked)
- CPU 小规模数据上跑通一个 batch，验证 loss 下降

## 阶段 5：完整训练 Pipeline（CPU 小规模验证 → GPU 全量训练）
- 训练循环: 有监督 + 无监督 + EMA 更新
- LR schedule: cosine decay
- 日志记录 (tensorboard 或 wandb)
- 先在 CPU 上用 40 张标注、少量迭代验证流程
- 再 GPU 全量训练 (40/250/4000 三种设置)

## 阶段 6：评估与结果记录
- 在 CIFAR-10 test set 上评估准确率
- 记录 3 次运行的平均值 ± 标准差
- 与论文表格对比

## 阶段 7：USB 库对比实验
- 安装 semilearn (USB)
- 使用 USB 的 FixMatch 实现训练 CIFAR-10 (40/250/4000)
- 对比自实现 vs USB 效果

## 阶段 8：实验报告（LaTeX）
- 算法原理描述
- 实现细节（注明参考来源）
- 实验结果表格与对比分析
- FixMatch vs MixMatch 异同分析
- 结论

---

## 迭代策略
1. **CPU 小规模**：每个模块先用 CPU + 小数据 (单 batch) 验证正确性
2. **GPU 全量**：确认正确后扩展到完整训练
3. **编码规则**：核心模块自行编写，参考开源代码思路但不直接复制
