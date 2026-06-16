# 实验环境

## 硬件
- CPU: (运行 `lscpu | grep "Model name"` 获取)
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB VRAM
- CUDA: 13.1, Driver: 591.74

## 软件
- Python: 3.13
- PyTorch: 2.11.0+cu128
- TorchVision: (含 RandAugment)

## GPU 算力参考
| GPU | FP32 TFLOPS | 显存 | 显存带宽 |
|-----|-------------|------|----------|
| RTX 4050 Laptop (本机) | 8.9 | 6 GB GDDR6 | 192 GB/s |
| V100 16GB PCIe | 14.0 | 16 GB HBM2 | 900 GB/s |
| V100 32GB SXM2 | 15.7 | 32 GB HBM2 | 900 GB/s |
| RTX 4090 | 82.6 | 24 GB GDDR6X | 1008 GB/s |
| A100 80GB | 19.5 | 80 GB HBM2e | 2039 GB/s |

说明:
- WRN-28-2 模型较小 (~1.5M 参数)，FP32 算力是主要瓶颈
- 显存带宽对训练影响较小（批数据量不大）
- RTX 4090 的 FP32 算力远超 V100，因为消费卡针对 FP32 优化

## 命令行原始输出

### PyTorch/CUDA 版本
```bash
$ python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
2.11.0+cu128
True
```

### torchvision RandAugment
```bash
$ python3 -c "from torchvision import transforms; print(hasattr(transforms, 'RandAugment'))"
True
```

### nvidia-smi
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.52.01              Driver Version: 591.74         CUDA Version: 13.1     |
|=========================================+========================+======================|
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 4050 ...    On  |   00000000:32:00.0 Off |                  N/A |
| N/A   63C    P4              9W /   61W |    1349MiB /   6141MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

### CUDA 基本功能测试
```bash
$ python3 -u -c "import torch; print('CUDA:', torch.cuda.is_available());
print('Device:', torch.cuda.get_device_name(0));
x = torch.randn(1000,1000,device='cuda'); y = torch.randn(1000,1000,device='cuda');
torch.cuda.synchronize(); z = x @ y; torch.cuda.synchronize(); print('Matrix OK')"
CUDA: True
Device: NVIDIA GeForce RTX 4050 Laptop GPU
Matrix OK
```
