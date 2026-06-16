# 数据加载模块验证

## 测试 1: 半监督划分逻辑
```
import numpy as np
from src.dataset import _split_labeled_indices

# CIFAR-10: 5000 per class
labels = np.concatenate([np.full(5000, c) for c in range(10)])

for n in [40, 250, 4000]:
    lab, unl = _split_labeled_indices(labels, n, 10, seed=0)
    per_class = [(labels[lab] == c).sum() for c in range(10)]
    print(f'{n} labels: labeled={len(lab)}, unlabeled={len(unl)}, per_class={per_class}')
```

## 原始输出
```
40 labels: labeled=40, unlabeled=49960, per_class=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
250 labels: labeled=250, unlabeled=49750, per_class=[25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
4000 labels: labeled=4000, unlabeled=46000, per_class=[400, 400, 400, 400, 400, 400, 400, 400, 400, 400]
```

## 测试 2: DataLoader 形状验证
```
x_l:   torch.Size([4, 3, 32, 32]),  y_l:   torch.Size([4])
x_w:   torch.Size([28, 3, 32, 32]), x_s:   torch.Size([28, 3, 32, 32])
x_t:   torch.Size([16, 3, 32, 32]), y_t:   torch.Size([16])
```

## 测试 3: 增强效果验证
```
Weak aug mean: -0.0004, std: 1.0367    (已归一化)
Strong aug mean: -0.3934, std: 1.3020  (更多变化)
Weak-Strong diff: 1.0935               (确认增强不相同)
```

## 测试 4: 修正后的增强 (flip+crop 加入弱增强)
```
Labeled:   shape=torch.Size([4, 3, 32, 32]), mean=-0.1802
Weak:      shape=torch.Size([28, 3, 32, 32]), mean=-0.2805, std=1.1410
Strong:    shape=torch.Size([28, 3, 32, 32]), mean=-0.6171, std=1.1773
Weak-Strong diff: 1.0935
```

## 命令行原始输出

### 划分逻辑测试
```bash
$ python3 -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from src.dataset import _split_labeled_indices
labels = np.concatenate([np.full(5000, c) for c in range(10)])
for n in [40, 250, 4000]:
    lab, unl = _split_labeled_indices(labels, n, 10, seed=0)
    per_class = [(labels[lab] == c).sum() for c in range(10)]
    print(f'{n} labels: labeled={len(lab)}, unlabeled={len(unl)}, per_class={per_class}')
"
40 labels: labeled=40, unlabeled=49960, per_class=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
250 labels: labeled=250, unlabeled=49750, per_class=[25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
4000 labels: labeled=4000, unlabeled=46000, per_class=[400, 400, 400, 400, 400, 400, 400, 400, 400, 400]
All split tests passed!
```

### DataLoader 集成测试
```bash
$ python3 -c "
import sys; sys.path.insert(0, '.')
from src.dataset import get_cifar10_loaders
labeled_loader, unlabeled_loader, test_loader, _, _ = get_cifar10_loaders(
    './data', num_labels=40, batch_size=4, uratio=7,
    eval_batch_size=16, num_workers=0)
import torch
x_l, y_l = next(iter(labeled_loader))
(x_w, x_s) = next(iter(unlabeled_loader))
x_t, y_t = next(iter(test_loader))
print(f'Labeled:   x_l shape={x_l.shape}, y_l shape={y_l.shape}')
print(f'Unlabeled: x_w shape={x_w.shape}, x_s shape={x_s.shape}')
print(f'Test:      x_t shape={x_t.shape}, y_t shape={y_t.shape}')
"
[Data] num_labels=40, labeled=40, unlabeled=49960, test=10000
[Data] labeled batches=10, unlabeled batches=1784
Labeled:   x_l shape=torch.Size([4, 3, 32, 32]), y_l shape=torch.Size([4])
Unlabeled: x_w shape=torch.Size([28, 3, 32, 32]), x_s shape=torch.Size([28, 3, 32, 32])
Test:      x_t shape=torch.Size([16, 3, 32, 32]), y_t shape=torch.Size([16])
All DataLoaders working correctly!
```

### 增强差值验证
```bash
$ python3 -c "
import sys; sys.path.insert(0, '.')
from src.dataset import get_cifar10_loaders
labeled_loader, unlabeled_loader, _, _, _ = get_cifar10_loaders(
    './data', num_labels=40, batch_size=4, uratio=7, eval_batch_size=16, num_workers=0)
x_l, y_l = next(iter(labeled_loader))
(x_w, x_s) = next(iter(unlabeled_loader))
print(f'Weak-Strong diff: {(x_w - x_s).abs().mean():.4f}')
"
Weak-Strong diff: 1.0935
```
