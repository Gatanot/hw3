#!/usr/bin/env python3
"""
FixMatch 半监督 CIFAR-10 训练脚本。
用法: python train.py --num_labels 40 [--device cuda] [--total_steps 1048576]

优化:
  - AMP 混合精度 (--use_amp)
  - cudnn benchmark
  - non_blocking 数据传输
  - 单 GPU 下关闭 interleave (--use_interleave, 默认 False)
  - 数据预取 (pin_memory + non_blocking)
"""

import os
import sys
import argparse
import yaml
import time
from itertools import cycle

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.dataset import get_cifar10_loaders
from src.wideresnet import wrn_28_2
from src.fixmatch import fixmatch_loss, EMAWithRestore


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total * 100


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Using device: {device}")

    # cudnn 自动调优（首步会触发卷积算法搜索，增加约 10s 启动开销）
    if device.type == 'cuda' and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("[Train] cudnn.benchmark = True")

    set_seed(args.seed)

    # 数据加载
    labeled_loader, unlabeled_loader, test_loader, _, _ = get_cifar10_loaders(
        data_root=args.data_dir,
        num_labels=args.num_labels,
        batch_size=args.batch_size,
        uratio=args.uratio,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # 模型
    model = wrn_28_2(num_classes=10, dropout_rate=args.dropout_rate)
    model = model.to(device)

    # EMA
    ema = EMAWithRestore(model, decay=args.ema_decay)

    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay,
    )

    # 学习率调度: η cos(7πk / 16K) as per FixMatch paper
    total_steps = args.total_steps
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: math.cos(7 * math.pi * step / (16 * total_steps))
    )

    # AMP 混合精度
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None

    # 日志
    writer = SummaryWriter(log_dir=args.log_dir)

    # 训练循环
    model.train()
    best_acc = 0.0
    labeled_iter = cycle(labeled_loader)
    unlabeled_iter = cycle(unlabeled_loader)

    print(f"[Train] Starting training: {args.total_steps} steps "
          f"(AMP={args.use_amp}, interleave={args.use_interleave})")
    start_time = time.time()

    for step in range(args.total_steps):
        x_l, y_l = next(labeled_iter)
        x_w, x_s = next(unlabeled_iter)

        x_l, y_l = x_l.to(device, non_blocking=True), y_l.to(device, non_blocking=True)
        x_w, x_s = x_w.to(device, non_blocking=True), x_s.to(device, non_blocking=True)

        # FixMatch 损失 (AMP autocast)
        with torch.amp.autocast('cuda', enabled=args.use_amp):
            total_loss, sup_loss, unsup_loss, mask_ratio = fixmatch_loss(
                model, x_l, y_l, x_w, x_s,
                confidence_threshold=args.confidence_threshold,
                unlabeled_loss_weight=args.unlabeled_loss_weight,
                use_interleave=args.use_interleave,
            )

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        scheduler.step()

        # 更新 EMA
        ema.update()

        # 日志
        if (step + 1) % args.log_interval == 0 or step == 0:
            elapsed = time.time() - start_time
            lr_val = scheduler.get_last_lr()[0]
            steps_per_sec = (step + 1) / max(elapsed, 1e-6)
            print(f"[Step {step+1:7d}/{args.total_steps}] "
                  f"lr={lr_val:.6f} sup={sup_loss.item():.4f} "
                  f"unsup={unsup_loss.item():.4f} total={total_loss.item():.4f} "
                  f"mask={mask_ratio:.3f} {steps_per_sec:.1f}st/s time={elapsed:.1f}s")
            writer.add_scalar('train/sup_loss', sup_loss.item(), step)
            writer.add_scalar('train/unsup_loss', unsup_loss.item(), step)
            writer.add_scalar('train/total_loss', total_loss.item(), step)
            writer.add_scalar('train/mask_ratio', mask_ratio, step)
            writer.add_scalar('train/lr', lr_val, step)
            writer.add_scalar('train/steps_per_sec', steps_per_sec, step)

        # 评估
        if (step + 1) % args.eval_interval == 0 or step == 0:
            ema.apply_shadow()
            acc = evaluate(model, test_loader, device)
            ema.restore()
            model.train()

            print(f"[Eval  Step {step+1:7d}] Test Accuracy: {acc:.2f}%")
            writer.add_scalar('test/accuracy', acc, step)

            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'ema_shadow': ema.shadow,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(args.checkpoint_dir, 'best_model.pth'))

        # 保存检查点
        if (step + 1) % args.save_interval == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.checkpoint_dir, f'checkpoint_{step+1}.pth'))

    # 最终评估
    ema.apply_shadow()
    final_acc = evaluate(model, test_loader, device)
    ema.restore()

    total_time = time.time() - start_time
    print(f"\n[Final] Test Accuracy: {final_acc:.2f}% (best: {best_acc:.2f}%)")
    print(f"[Final] Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")

    writer.close()
    return best_acc, final_acc


def main():
    parser = argparse.ArgumentParser(description='FixMatch CIFAR-10 Training')

    # 配置
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='配置文件路径')

    # 数据集
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_labels', type=int, default=40,
                        help='标注数据量 (40, 250, 4000)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--uratio', type=int, default=7,
                        help='无标注/标注 batch 比例')
    parser.add_argument('--eval_batch_size', type=int, default=256)

    # 模型
    parser.add_argument('--dropout_rate', type=float, default=0.0)

    # FixMatch
    parser.add_argument('--confidence_threshold', type=float, default=0.95)
    parser.add_argument('--unlabeled_loss_weight', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # 优化器
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_true', default=True)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # 训练
    parser.add_argument('--total_steps', type=int, default=1048576,
                        help='总训练步数 (默认 2^20)')
    parser.add_argument('--log_interval', type=int, default=128)
    parser.add_argument('--eval_interval', type=int, default=1024)
    parser.add_argument('--save_interval', type=int, default=65536)
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='启用 AMP 混合精度 (FP16)')
    parser.add_argument('--use_interleave', action='store_true', default=False,
                        help='启用 interleave (多 GPU 需要，单 GPU 可关闭)')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=False,
                        help='启用 cudnn benchmark 自动调优 (首步 ~10s 开销，长训练建议启用)')

    # 系统
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')

    args = parser.parse_args()

    # 从 YAML 配置加载（CLI 参数会覆盖）
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    attr = f'{key}_{k}'
                    if hasattr(args, attr):
                        setattr(args, attr, v)

    pass  # total_steps 由 CLI 参数或默认值决定

    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"[Config] num_labels={args.num_labels}, batch_size={args.batch_size}, "
          f"uratio={args.uratio}, lr={args.lr}, total_steps={args.total_steps}")
    print(f"[Config] confidence_threshold={args.confidence_threshold}, "
          f"ema_decay={args.ema_decay}")

    best_acc, final_acc = train(args)
    print(f"\n[Result] num_labels={args.num_labels}, best_acc={best_acc:.2f}%, "
          f"final_acc={final_acc:.2f}%")


if __name__ == '__main__':
    main()
