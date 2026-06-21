#!/usr/bin/env python3
"""
小规模测试 USB (semilearn) FixMatch 实现
通过模拟 USB 训练循环，单独调用 train_step
"""
import sys, os, math, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.dataset import get_cifar10_loaders
from src.wideresnet import wrn_28_2
from src.fixmatch import EMAWithRestore


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WRNUSB(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = wrn_28_2(num_classes=num_classes)
    def forward(self, x):
        return {'logits': self.model(x), 'feat': None}


def run_usb_fixmatch(num_labels, total_steps, data_dir, seed=0):
    print(f"\n{'='*60}")
    print(f"USB FixMatch: {num_labels} labels, {total_steps} steps")
    print(f"{'='*60}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    labeled_loader, unlabeled_loader, test_loader, _, _ = get_cifar10_loaders(
        data_root=data_dir, num_labels=num_labels,
        batch_size=64, uratio=7, eval_batch_size=256,
        num_workers=2, seed=seed,
    )

    model = WRNUSB(num_classes=10).to(device)
    ema = EMAWithRestore(model, decay=0.999)

    optimizer = optim.SGD(
        model.parameters(), lr=0.03, momentum=0.9,
        nesterov=True, weight_decay=0.0005,
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: math.cos(7 * math.pi * step / (16 * total_steps))
    )

    ce_loss = nn.CrossEntropyLoss()
    cons_loss = nn.CrossEntropyLoss(reduction='none')

    model.train()
    best_acc = 0.0
    scaler = torch.cuda.amp.GradScaler()

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for step in range(total_steps):
        try:
            x_l, y_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            x_l, y_l = next(labeled_iter)

        try:
            x_w, x_s = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            x_w, x_s = next(unlabeled_iter)

        x_l, y_l = x_l.to(device), y_l.to(device)
        x_w, x_s = x_w.to(device), x_s.to(device)

        # USB FixMatch train_step logic
        with torch.cuda.amp.autocast(enabled=True):
            # Labeled forward
            outs_l = model(x_l)
            logits_l = outs_l['logits']

            # Unlabeled weak (no grad) + strong
            outs_ulb_s = model(x_s)
            logits_ulb_s = outs_ulb_s['logits']

            with torch.no_grad():
                outs_ulb_w = model(x_w)
                logits_ulb_w = outs_ulb_w['logits']

            # Supervised loss
            sup_loss = ce_loss(logits_l, y_l)

            # Pseudo-labeling with confidence threshold
            probs_ulb_w = torch.softmax(logits_ulb_w.detach(), dim=-1)
            max_probs, pseudo_labels = probs_ulb_w.max(dim=-1)
            mask = (max_probs >= 0.95).float()

            # Unsupervised loss (cross-entropy with pseudo-labels, masked)
            unsup_loss = cons_loss(logits_ulb_s, pseudo_labels)
            unsup_loss = (unsup_loss * mask).sum() / max(mask.sum(), 1)

            total_loss = sup_loss + 1.0 * unsup_loss

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update()

        mask_ratio = mask.mean().item()

        if (step + 1) % 128 == 0:
            print(f"[USB Step {step+1:5d}/{total_steps}] "
                  f"lr={scheduler.get_last_lr()[0]:.6f} sup={sup_loss.item():.4f} "
                  f"unsup={unsup_loss.item():.4f} total={total_loss.item():.4f} "
                  f"mask={mask_ratio:.3f}")

        if (step + 1) % 256 == 0 or step == 0:
            ema.apply_shadow()
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x_t, y_t in test_loader:
                    x_t, y_t = x_t.to(device), y_t.to(device)
                    out = model(x_t)['logits']
                    correct += (out.argmax(1) == y_t).sum().item()
                    total += y_t.size(0)
            acc = correct / total * 100
            ema.restore()
            model.train()
            print(f"[USB Eval  Step {step+1:5d}] Test Accuracy: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc

    ema.apply_shadow()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_t, y_t in test_loader:
            x_t, y_t = x_t.to(device), y_t.to(device)
            out = model(x_t)['logits']
            correct += (out.argmax(1) == y_t).sum().item()
            total += y_t.size(0)
    final_acc = correct / total * 100
    ema.restore()

    print(f"\n[USB Result] num_labels={num_labels}, best_acc={best_acc:.2f}%, final_acc={final_acc:.2f}%")
    return best_acc, final_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_labels', type=int, default=40)
    parser.add_argument('--total_steps', type=int, default=1024)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run_usb_fixmatch(args.num_labels, args.total_steps, args.data_dir, args.seed)


if __name__ == '__main__':
    main()
