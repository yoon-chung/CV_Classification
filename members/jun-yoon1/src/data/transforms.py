import albumentations as A
from albumentations.pytorch import ToTensorV2


def _base_aug(image_size: int):
    return [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
        A.ShiftScaleRotate(
            shift_limit=0.02,
            scale_limit=0.08,
            rotate_limit=8,
            border_mode=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.2
        ),
        A.Normalize(),
        ToTensorV2(),
    ]


def _strong_v1(image_size: int):
    return [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.12,
            rotate_limit=15,
            border_mode=0,
            p=0.7,
        ),
        A.Perspective(scale=(0.03, 0.08), p=0.35),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.GaussNoise(var_limit=(10.0, 40.0)),
            ],
            p=0.35,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.25, contrast_limit=0.25
                ),
                A.CLAHE(clip_limit=2.0),
            ],
            p=0.3,
        ),
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.25),
        A.Normalize(),
        ToTensorV2(),
    ]


def _strong_v2(image_size: int):
    return [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.10,
            rotate_limit=12,
            border_mode=0,
            p=0.7,
        ),
        A.Perspective(scale=(0.02, 0.07), p=0.4),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.GaussNoise(var_limit=(10.0, 45.0)),
            ],
            p=0.45,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3
                ),
                A.CLAHE(clip_limit=2.0),
                A.Sharpen(alpha=(0.15, 0.3), lightness=(0.7, 1.2)),
            ],
            p=0.35,
        ),
        A.ImageCompression(quality_lower=35, quality_upper=90, p=0.35),
        A.CoarseDropout(
            max_holes=6,
            max_height=max(8, image_size // 18),
            max_width=max(8, image_size // 18),
            min_holes=1,
            fill_value=255,
            p=0.2,
        ),
        A.Normalize(),
        ToTensorV2(),
    ]


def build_train_transform(
    image_size: int, strong: bool = False, strong_profile: str = "v2"
) -> A.Compose:
    if not strong:
        aug = _base_aug(image_size)
    elif strong_profile == "v1":
        aug = _strong_v1(image_size)
    else:
        aug = _strong_v2(image_size)
    return A.Compose(aug)


def build_val_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
