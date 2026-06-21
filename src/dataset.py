import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10

from src.augmentations import TransformWeak, TransformStrong


def _split_labeled_indices(labels, num_labels_total, num_classes=10, seed=0):
    """
    从全部训练数据中均匀选取 num_labels_total 个标注样本。
    返回 labeled 和 unlabeled 的索引列表。
    Paper: unlabeled includes all labeled data without labels.
    """
    labels = np.array(labels)
    num_per_class = num_labels_total // num_classes
    rng = np.random.RandomState(seed)

    labeled_indices = []
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)
        labeled_indices.extend(class_indices[:num_per_class].tolist())

    unlabeled_indices = list(range(len(labels)))

    rng.shuffle(labeled_indices)
    rng.shuffle(unlabeled_indices)
    return labeled_indices, unlabeled_indices


class CIFAR10Labeled(Dataset):
    """
    标注数据 Dataset，返回弱增强图像和标签。
    """

    def __init__(self, data_root, labeled_indices, transform_weak=None):
        self.dataset = CIFAR10(root=data_root, train=True, download=False)
        self.indices = labeled_indices
        self.transform = transform_weak if transform_weak is not None else TransformWeak()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return self.transform(img), label


class CIFAR10Unlabeled(Dataset):
    """
    无标注数据 Dataset，返回 (弱增强图像, 强增强图像)。
    """

    def __init__(self, data_root, unlabeled_indices,
                 transform_weak=None, transform_strong=None):
        self.dataset = CIFAR10(root=data_root, train=True, download=False)
        self.indices = unlabeled_indices
        self.weak = transform_weak if transform_weak is not None else TransformWeak()
        self.strong = transform_strong if transform_strong is not None else TransformStrong()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.dataset[self.indices[idx]]
        return self.weak(img), self.strong(img)


class CIFAR10Test(Dataset):
    """
    CIFAR-10 测试集 Dataset。
    """

    def __init__(self, data_root, transform=None):
        self.dataset = CIFAR10(root=data_root, train=False, download=False)
        self.transform = transform if transform is not None else TransformWeak()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform(img), label


def get_cifar10_loaders(data_root, num_labels, batch_size=64, uratio=7,
                         eval_batch_size=256, num_workers=4, seed=0):
    """
    构建半监督 CIFAR-10 DataLoader。

    Returns:
        labeled_loader, unlabeled_loader, test_loader, labeled_indices, unlabeled_indices
    """
    train_dataset = CIFAR10(root=data_root, train=True, download=False)
    labeled_indices, unlabeled_indices = _split_labeled_indices(
        train_dataset.targets, num_labels, num_classes=10, seed=seed
    )

    transform_weak = TransformWeak()
    transform_strong = TransformStrong(num_ops=2, max_magnitude=10, cutout_size=16)

    labeled_dataset = CIFAR10Labeled(data_root, labeled_indices, transform_weak)
    unlabeled_dataset = CIFAR10Unlabeled(data_root, unlabeled_indices,
                                         transform_weak, transform_strong)
    test_dataset = CIFAR10Test(data_root, TransformWeak())

    labeled_loader = DataLoader(
        labeled_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=batch_size * uratio, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"[Data] num_labels={num_labels}, "
          f"labeled={len(labeled_indices)}, unlabeled={len(unlabeled_indices)}, "
          f"test={len(test_dataset)}")
    print(f"[Data] labeled batches={len(labeled_loader)}, "
          f"unlabeled batches={len(unlabeled_loader)}")
    return labeled_loader, unlabeled_loader, test_loader, labeled_indices, unlabeled_indices
