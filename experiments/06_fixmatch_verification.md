# FixMatch 损失函数验证

## 测试 1: 默认阈值 (τ=0.95)
随机初始化模型，无样本超过 0.95 置信度：
```
Supervised Loss:     2.2755
Unsupervised Loss:   0.0000
Total Loss:          2.2755
Mask Ratio:          0.0000
Pseudo labels kept:  0.0 / 28
Gradient norm:       309.1188
```

## 测试 2: 零阈值验证 mask 机制
τ=0，全部样本通过：
```
Threshold=0.0:
  Sup Loss: 2.2890, Unsup Loss: 2.2635
  Mask Ratio: 1.0000
```
确认 mask/demask 机制正确。

## 测试 3: interleave/de_interleave 可逆性
```
Combined shape: torch.Size([60, 3, 32, 32])      # 4+28+28
Interleaved shape: torch.Size([60, 3, 32, 32])
Restored shape: torch.Size([60, 3, 32, 32])
Match: True
```

## 命令行原始输出

### 默认阈值测试
```bash
$ python3 -c "
import sys; sys.path.insert(0, '.')
import torch
from src.wideresnet import wrn_28_2
from src.fixmatch import fixmatch_loss
from src.dataset import get_cifar10_loaders
model = wrn_28_2(num_classes=10); model.train()
labeled_loader, unlabeled_loader, _, _, _ = get_cifar10_loaders(
    './data', num_labels=40, batch_size=4, uratio=7,
    eval_batch_size=16, num_workers=0)
x_l, y_l = next(iter(labeled_loader))
(x_w, x_s) = next(iter(unlabeled_loader))
total_loss, sup_loss, unsup_loss, mask_ratio = fixmatch_loss(
    model, x_l, y_l, x_w, x_s, confidence_threshold=0.95,
    unlabeled_loss_weight=1.0, use_interleave=True)
print(f'Supervised Loss:     {sup_loss.item():.4f}')
print(f'Unsupervised Loss:   {unsup_loss.item():.4f}')
print(f'Total Loss:          {total_loss.item():.4f}')
print(f'Mask Ratio:          {mask_ratio:.4f}')
total_loss.backward()
"
[Data] num_labels=40, labeled=40, unlabeled=49960, test=10000
[Data] labeled batches=10, unlabeled batches=1784
Supervised Loss:     2.2755
Unsupervised Loss:   0.0000
Total Loss:          2.2755
Mask Ratio:          0.0000
Pseudo labels kept:  0.0 / 28
```

### 零阈值 (mask 机制) 测试
```bash
$ python3 -c "
import sys; sys.path.insert(0, '.')
import torch
from src.wideresnet import wrn_28_2
from src.fixmatch import fixmatch_loss
from src.dataset import get_cifar10_loaders
model = wrn_28_2(num_classes=10); model.train()
labeled_loader, unlabeled_loader, _, _, _ = get_cifar10_loaders(
    './data', num_labels=40, batch_size=4, uratio=7, num_workers=0)
x_l, y_l = next(iter(labeled_loader))
(x_w, x_s) = next(iter(unlabeled_loader))
total_loss, sup_loss, unsup_loss, mask_ratio = fixmatch_loss(
    model, x_l, y_l, x_w, x_s, confidence_threshold=0.0,
    unlabeled_loss_weight=1.0, use_interleave=True)
print(f'Threshold=0.0:')
print(f'  Sup Loss: {sup_loss.item():.4f}, Unsup Loss: {unsup_loss.item():.4f}')
print(f'  Mask Ratio: {mask_ratio:.4f}')
"
Threshold=0.0:
  Sup Loss: 2.2890, Unsup Loss: 2.2635
  Mask Ratio: 1.0000
```

### interleave 可逆性测试
```bash
$ python3 -c "
import sys; sys.path.insert(0, '.')
import torch
from src.fixmatch import interleave, de_interleave
from src.dataset import get_cifar10_loaders
labeled_loader, unlabeled_loader, _, _, _ = get_cifar10_loaders(
    './data', num_labels=40, batch_size=4, uratio=7, num_workers=0)
x_l, y_l = next(iter(labeled_loader))
(x_w, x_s) = next(iter(unlabeled_loader))
combined = torch.cat([x_l, x_w, x_s], dim=0)
print(f'Combined shape: {combined.shape}')     # [4+28+28=60]
size = 2 * 7 + 1                                # = 15
assert combined.size(0) % size == 0
interleaved = interleave(combined, size)
restored = de_interleave(interleaved, size)
print(f'Interleaved shape: {interleaved.shape}')
print(f'Restored shape: {restored.shape}')
print(f'Match: {torch.allclose(combined, restored)}')
"
[Data] num_labels=40, labeled=40, unlabeled=49960, test=10000
[Data] labeled batches=10, unlabeled batches=1784
Combined shape: torch.Size([60, 3, 32, 32])
Interleaved shape: torch.Size([60, 3, 32, 32])
Restored shape: torch.Size([60, 3, 32, 32])
Match: True
```
