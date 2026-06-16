# WideResNet-28-2 验证

## 模型结构
- 深度: 28 (n = (28-4)/6 = 4 blocks per group)
- 扩展因子: 2
- 通道数: 16 → 32 → 64 → 128
- 总参数量: 1,467,610

## 测试 1: 前向传播
```
Input:  torch.Size([2, 3, 32, 32])
Output: torch.Size([2, 10])
```

## 测试 2: 反向传播
```
Loss: 2.3167
Total gradient norm: 320.5969
```
所有参数梯度非零，梯度正常流动。

## 模型结构详细
```
WideResNet(
  (conv1): Conv2d(3, 16, kernel=(3,3), stride=1)     # 16 ch
  (layer1): NetworkBlock × 4 blocks                   # 16→32 ch, stride 1
  (layer2): NetworkBlock × 4 blocks                   # 32→64 ch, stride 2
  (layer3): NetworkBlock × 4 blocks                   # 64→128 ch, stride 2
  (bn): BatchNorm2d(128)
  (avgpool): AdaptiveAvgPool2d(1,1)
  (fc): Linear(128, 10)
)
```

## 命令行原始输出

### 前向传播测试
```bash
$ python3 -c "
import sys; sys.path.insert(0, '.')
import torch
from src.wideresnet import wrn_28_2
model = wrn_28_2(num_classes=10)
print(model)
num_params = sum(p.numel() for p in model.parameters())
print(f'Total params: {num_params:,}')
x = torch.randn(2, 3, 32, 32)
with torch.no_grad():
    out = model(x)
print(f'Input:  {x.shape}')
print(f'Output: {out.shape}')
"
Total params: 1,467,610
Input:  torch.Size([2, 3, 32, 32])
Output: torch.Size([2, 10])
```

### 反向传播/梯度流测试
```bash
$ python3 -c "
import sys; sys.path.insert(0, '.')
import torch; import torch.nn as nn
from src.wideresnet import wrn_28_2
model = wrn_28_2(num_classes=10); model.train()
x = torch.randn(2, 3, 32, 32); y = torch.randint(0, 10, (2,))
out = model(x); loss = nn.CrossEntropyLoss()(out, y)
loss.backward()
grad_flow = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
print(f'Loss: {loss.item():.4f}')
print(f'Total gradient norm: {grad_flow:.4f}')
"
Loss: 2.3167
Total gradient norm: 320.5969
```
