import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import RandAugment


class TransformWeak:
    """
    弱增强：RandomHorizontalFlip + RandomCrop(padding=4) + Normalize。
    CIFAR-10 标准化参数。
    """

    def __init__(self, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, x):
        return self.transform(x)


class TransformStrong:
    """
    强增强：RandAugment + Cutout + Normalize。
    Cutout 实现为随机遮挡 16x16 区域。
    """

    def __init__(self, num_ops=2, magnitude=10, cutout_size=16,
                 mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        self.randaugment = RandAugment(num_ops=num_ops, magnitude=magnitude)
        self.cutout_size = cutout_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, x):
        x = self.randaugment(x)
        x = self.to_tensor(x)
        if self.cutout_size > 0:
            x = self._cutout(x)
        x = self.normalize(x)
        return x

    def _cutout(self, img):
        """随机遮挡 cutout_size × cutout_size 区域。"""
        h, w = img.shape[1], img.shape[2]
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.cutout_size // 2, 0, h)
        y2 = np.clip(y + self.cutout_size // 2, 0, h)
        x1 = np.clip(x - self.cutout_size // 2, 0, w)
        x2 = np.clip(x + self.cutout_size // 2, 0, w)
        img[:, y1:y2, x1:x2] = 0.0
        return img
