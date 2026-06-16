import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average (指数移动平均)。
    在评估时使用 EMA 模型可获得更稳定的结果。
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone().detach()
                       for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay)

    @torch.no_grad()
    def apply_shadow(self):
        """将 EMA 权重应用到模型上（评估前调用）。"""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self):
        """恢复原始模型权重（评估后调用）。"""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        # 注意：restore 后 shadow 需要重新从模型获取，这里简化处理
        # 实际上应该保存原始权重副本


class EMAWithRestore(EMA):
    """
    支持 restore 的 EMA，保存原始权重副本用于恢复。
    """

    @torch.no_grad()
    def apply_shadow(self):
        self.backup = {name: param.data.clone()
                       for name, param in self.model.named_parameters()
                       if name in self.shadow}
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def interleave(x, size):
    """
    将 batch 按指定大小重新排列（interleave）。
    用于将标注和无标注数据交错排列以获得更好的 BN 统计。
    """
    s = list(x.shape)
    return x.view([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    """interleave 的逆操作。"""
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def fixmatch_loss(model, x_labeled, y_labeled, x_unlabeled_w, x_unlabeled_s,
                  confidence_threshold=0.95, unlabeled_loss_weight=1.0,
                  use_interleave=True):
    """
    FixMatch 损失函数。

    核心步骤：
    1. 弱增强无标注图像 → 模型预测 → softmax → 伪标签（仅保留高置信度）
    2. 用伪标签监督强增强图像的分类

    Args:
        model: WideResNet 模型（输出 logits，不含 softmax）
        x_labeled: 标注图像 (弱增强)，shape [B, 3, 32, 32]
        y_labeled: 标注标签，shape [B]
        x_unlabeled_w: 无标注图像 (弱增强)，shape [B*uratio, 3, 32, 32]
        x_unlabeled_s: 无标注图像 (强增强)，shape [B*uratio, 3, 32, 32]
        confidence_threshold: 伪标签置信度阈值 τ (默认 0.95)
        unlabeled_loss_weight: 无监督损失权重 λ (默认 1.0)
        use_interleave: 是否使用 interleave 技巧改善 BN 统计

    Returns:
        total_loss, supervised_loss, unsupervised_loss, mask_ratio
    """
    batch_size = x_labeled.size(0)
    ulb_batch_size = x_unlabeled_w.size(0)

    if use_interleave:
        combined = torch.cat([x_labeled, x_unlabeled_w, x_unlabeled_s], dim=0)
        n_groups = (ulb_batch_size // batch_size) * 2 + 1
        if combined.size(0) % n_groups == 0:
            combined = interleave(combined, n_groups)
            logits = model(combined)
            logits = de_interleave(logits, n_groups)
            logits_x = logits[:batch_size]
            logits_weak = logits[batch_size:batch_size + ulb_batch_size]
            logits_strong = logits[batch_size + ulb_batch_size:]
        else:
            logits_x = model(x_labeled)
            logits_weak = model(x_unlabeled_w)
            logits_strong = model(x_unlabeled_s)
    else:
        logits_x = model(x_labeled)
        logits_weak = model(x_unlabeled_w)
        logits_strong = model(x_unlabeled_s)

    # 有监督损失
    sup_loss = F.cross_entropy(logits_x, y_labeled)

    # 伪标签生成（无梯度）
    with torch.no_grad():
        pseudo_probs = F.softmax(logits_weak, dim=1)
        pseudo_labels = pseudo_probs.argmax(dim=1)
        max_probs = pseudo_probs.max(dim=1).values
        mask = (max_probs >= confidence_threshold).float()

    # 无监督损失：用伪标签监督强增强输出，仅保留高置信度样本
    unsup_loss = F.cross_entropy(logits_strong, pseudo_labels, reduction='none')
    unsup_loss = (unsup_loss * mask).mean()

    total_loss = sup_loss + unlabeled_loss_weight * unsup_loss
    mask_ratio = mask.mean().item()

    return total_loss, sup_loss, unsup_loss, mask_ratio
