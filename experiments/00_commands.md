# 全部命令行记录

## 环境检查
```bash
nvidia-smi
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "from torchvision import transforms; print(hasattr(transforms, 'RandAugment'))"
python3 -c "from torchvision.datasets import CIFAR10; d=CIFAR10('./data', train=True, download=True); print(len(d))"
```

## 数据划分验证
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from src.dataset import _split_labeled_indices
labels = np.concatenate([np.full(5000, c) for c in range(10)])
for n in [40, 250, 4000]:
    lab, unl = _split_labeled_indices(labels, n, 10, seed=0)
    per_class = [(labels[lab] == c).sum() for c in range(10)]
    print(f'{n} labels: labeled={len(lab)}, unlabeled={len(unl)}, per_class={per_class}')
"
```

## DataLoader 验证
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
import torch
from src.dataset import get_cifar10_loaders
labeled_loader, unlabeled_loader, test_loader, _, _ = get_cifar10_loaders(
    './data', num_labels=40, batch_size=4, uratio=7, eval_batch_size=16, num_workers=0)
x_l, y_l = next(iter(labeled_loader))
(x_w, x_s) = next(iter(unlabeled_loader))
x_t, y_t = next(iter(test_loader))
print(f'Labeled: {x_l.shape}, {y_l.shape}')
print(f'Unlabeled: {x_w.shape}, {x_s.shape}')
print(f'Test: {x_t.shape}, {y_t.shape}')
"
```

## WRN-28-2 前向测试
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
import torch; import torch.nn as nn
from src.wideresnet import wrn_28_2
model = wrn_28_2(num_classes=10)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
x = torch.randn(2, 3, 32, 32)
out = model(x)
print(f'Input: {x.shape} -> Output: {out.shape}')
# Backward
model.train()
y = torch.randint(0, 10, (2,))
loss = nn.CrossEntropyLoss()(out, y)
loss.backward()
print(f'Loss: {loss.item():.4f}, grad: {sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None):.1f}')
"
```

## FixMatch 损失测试
```bash
# 默认阈值
python3 -c "
import sys; sys.path.insert(0, '.')
import torch
from src.wideresnet import wrn_28_2
from src.fixmatch import fixmatch_loss
from src.dataset import get_cifar10_loaders
model = wrn_28_2(num_classes=10); model.train()
labeled_loader, unlabeled_loader, _, _, _ = get_cifar10_loaders(
    './data', num_labels=40, batch_size=4, uratio=7, eval_batch_size=16, num_workers=0)
x_l, y_l = next(iter(labeled_loader))
(x_w, x_s) = next(iter(unlabeled_loader))
total_loss, sup_loss, unsup_loss, mask_ratio = fixmatch_loss(
    model, x_l, y_l, x_w, x_s, confidence_threshold=0.95, unlabeled_loss_weight=1.0)
print(f'Sup={sup_loss:.4f} Unsup={unsup_loss:.4f} Total={total_loss:.4f} Mask={mask_ratio:.4f}')
total_loss.backward()
print('OK')
"

# 零阈值验证
python3 -c "
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
    model, x_l, y_l, x_w, x_s, confidence_threshold=0.0, unlabeled_loss_weight=1.0)
print(f'Threshold=0: Sup={sup_loss:.4f} Unsup={unsup_loss:.4f} Mask={mask_ratio:.4f}')
"
```

## CPU 训练测试 (10步)
```bash
timeout 120 python3 -u train.py --num_labels 40 --device cpu --total_steps 10 \
    --log_interval 5 --eval_interval 100 --batch_size 4 --uratio 2 --num_workers 0
```

## GPU 训练测试 (50步)
```bash
timeout 120 python3 -u train.py --num_labels 40 --device cuda --total_steps 50 \
    --log_interval 10 --eval_interval 50 --batch_size 16 --uratio 3 --num_workers 2
```

## 安装依赖
```bash
pip install pyyaml tensorboard -q
```
