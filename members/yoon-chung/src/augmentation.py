"""
augmentation.py — 온라인 증강 전략

최종 설정:
  - B3: v3 (왜곡 제외, 강한 블러/노이즈)
  - ConvNeXt: v2 (중간 강도)
  - TTA: 10개 (회전4 + HFlip + VFlip + 복합4)
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size, version='v2'):
    if version == 'v1':
        return _train_v1(img_size)
    elif version == 'v2':
        return _train_v2(img_size)
    elif version == 'v3':
        return _train_v3(img_size)
    else:
        raise ValueError(f'Unknown version: {version}')


def _train_v1(img_size):
    """v1: 기본 — 회전 + Flip만"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=180, border_mode=0, value=0, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=img_size // 8, max_width=img_size // 8,
                        fill_value=0, p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])


def _train_v2(img_size):
    """v2: 강화 — 블러/노이즈 추가 (ConvNeXt 최적)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=180, border_mode=0, value=0, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=img_size // 8, max_width=img_size // 8,
                        fill_value=0, p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])


def _train_v3(img_size):
    """v3: 최강 — 더 강한 블러/노이즈 + 색상 변형 (B3 최적)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=180, border_mode=0, value=0, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        A.GaussNoise(var_limit=(10.0, 80.0), p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), p=0.4),
        A.CoarseDropout(max_holes=10, max_height=img_size // 6, max_width=img_size // 6,
                        fill_value=0, p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])


def get_valid_transforms(img_size):
    """검증/추론용 (증강 없음)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])


def get_tta_transforms(img_size):
    """TTA 10개: 회전4 + HFlip + VFlip + 복합변환4"""
    norm = [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    return [
        A.Compose([A.Resize(img_size, img_size)] + norm),
        A.Compose([A.Resize(img_size, img_size), A.Rotate(limit=(90,90), p=1)] + norm),
        A.Compose([A.Resize(img_size, img_size), A.Rotate(limit=(180,180), p=1)] + norm),
        A.Compose([A.Resize(img_size, img_size), A.Rotate(limit=(270,270), p=1)] + norm),
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1)] + norm),
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1)] + norm),
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1),
                   A.Rotate(limit=(90,90), p=1)] + norm),
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1),
                   A.Rotate(limit=(90,90), p=1)] + norm),
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1),
                   A.Rotate(limit=(180,180), p=1)] + norm),
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1),
                   A.Rotate(limit=(180,180), p=1)] + norm),
    ]


def get_hard_valid_transforms(img_size):
    """Test 조건 반영 Val 변환 — LB 예측력 향상용"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=180, border_mode=0, value=0, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
