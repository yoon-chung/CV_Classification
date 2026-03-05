"""
dataset.py — Dataset & MixUp/CutMix
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class DocDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.array(Image.open(self.image_dir / row['ID']).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']
        if self.is_test:
            return image
        return image, row['target']


# ============================================
# MixUp / CutMix
# ============================================

def mixup_data(x, y, alpha=0.4):
    """두 이미지를 비율로 섞기"""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    """이미지 일부를 다른 이미지로 교체"""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return mixed_x, y, y[index], lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp/CutMix 혼합 Loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
