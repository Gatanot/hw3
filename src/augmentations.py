import random
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_sharpness
from PIL import ImageOps, Image


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


class RandAugmentPerSample:
    """
    RandAugment with per-sample random magnitude, as FixMatch paper specifies:
    "the magnitude is randomly sampled from a pre-defined range at each training step"
    Magnitude sampled uniformly from [1, max_magnitude] for each transformation.
    """

    _OP_NAMES = [
        'AutoContrast', 'Equalize', 'Rotate', 'Solarize', 'Color',
        'Posterize', 'Contrast', 'Brightness', 'Sharpness', 'ShearX',
        'ShearY', 'TranslateX', 'TranslateY', 'Identity',
    ]

    def __init__(self, num_ops=2, max_magnitude=10):
        self.num_ops = num_ops
        self.max_magnitude = max_magnitude

    def _apply_op(self, img, op_name, magnitude):
        if op_name == 'Identity':
            return img
        if op_name == 'AutoContrast':
            return ImageOps.autocontrast(img)
        if op_name == 'Equalize':
            return ImageOps.equalize(img)
        if op_name == 'Rotate':
            return img.rotate(magnitude, fillcolor=128)
        if op_name == 'Solarize':
            return ImageOps.solarize(img, int(255 - magnitude * 2.55))
        if op_name == 'Color':
            return adjust_saturation(img, max(0, magnitude * 0.18))
        if op_name == 'Posterize':
            bits = int(8 - magnitude * 0.4)
            return ImageOps.posterize(img, max(1, bits))
        if op_name == 'Contrast':
            return adjust_contrast(img, max(0, magnitude * 0.18))
        if op_name == 'Brightness':
            return adjust_brightness(img, max(0, magnitude * 0.18))
        if op_name == 'Sharpness':
            return adjust_sharpness(img, max(0, magnitude * 0.18))
        if op_name == 'ShearX':
            return img.transform(img.size, Image.AFFINE,
                                 (1, magnitude * 0.03, 0, 0, 1, 0), fillcolor=128)
        if op_name == 'ShearY':
            return img.transform(img.size, Image.AFFINE,
                                 (1, 0, 0, magnitude * 0.03, 1, 0), fillcolor=128)
        if op_name == 'TranslateX':
            return img.transform(img.size, Image.AFFINE,
                                 (1, 0, magnitude * 0.33, 0, 1, 0), fillcolor=128)
        if op_name == 'TranslateY':
            return img.transform(img.size, Image.AFFINE,
                                 (1, 0, 0, 0, 1, magnitude * 0.33), fillcolor=128)
        return img

    def __call__(self, img):
        ops = random.sample(self._OP_NAMES, self.num_ops)
        for op_name in ops:
            magnitude = random.uniform(1, self.max_magnitude)
            img = self._apply_op(img, op_name, magnitude)
        return img


class TransformStrong:
    """
    强增强：RandAugment (random magnitude per op) + Cutout + Normalize。
    Cutout 填充为 CIFAR-10 每通道均值 (gray fill per paper)。
    """

    def __init__(self, num_ops=2, max_magnitude=10, cutout_size=16,
                 mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        self.randaugment = RandAugmentPerSample(num_ops=num_ops, max_magnitude=max_magnitude)
        self.cutout_size = cutout_size
        self.cutout_fill = [m for m in mean]
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
        """随机遮挡 cutout_size x cutout_size 区域，填充为数据集均值（灰色）。"""
        h, w = img.shape[1], img.shape[2]
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.cutout_size // 2, 0, h)
        y2 = np.maximum(y1 + 1, np.clip(y + self.cutout_size // 2, 0, h))
        x1 = np.clip(x - self.cutout_size // 2, 0, w)
        x2 = np.maximum(x1 + 1, np.clip(x + self.cutout_size // 2, 0, w))
        for c in range(3):
            img[c, y1:y2, x1:x2] = self.cutout_fill[c]
        return img
