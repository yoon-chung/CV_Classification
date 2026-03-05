"""
preprocess.py — 오프라인 증강 & 오버샘플링
용도: 학습 전 데이터 준비 (1회 실행)
참고: 최종 제출에서는 오프라인 증강 미사용 (온라인 증강만으로 LB 0.9526 달성)
"""
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import albumentations as A


def get_document_aug_transform():
    """문서/스캔 특화 증강 — Augraphy 대체 (OpenCV + Albumentations)"""
    return A.Compose([
        A.Rotate(limit=180, border_mode=0, value=0, p=0.7),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        A.GaussNoise(var_limit=(10.0, 80.0), p=0.4),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.RandomToneCurve(scale=0.1, p=0.2),
        A.ImageCompression(quality_lower=40, quality_upper=95, p=0.4),
        A.RandomResizedCrop(size=(512, 512), scale=(0.75, 1.0), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=40, max_width=40,
                        fill_value=0, p=0.2),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                       num_shadows_lower=1, num_shadows_upper=2,
                       shadow_dimension=5, p=0.2),
    ])


def oversample_with_doc_aug(df_train, src_dir, dst_dir, target_count=100,
                             extra_per_class=0):
    """
    문서 특화 오프라인 증강
    - 소수 클래스: target_count까지 채움
    - 전체 클래스: extra_per_class만큼 추가
    """
    dst_dir = Path(dst_dir)
    src_dir = Path(src_dir)

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)

    aug_transform = get_document_aug_transform()
    records = []

    for _, row in df_train.iterrows():
        shutil.copy2(src_dir / row['ID'], dst_dir / row['ID'])
        records.append({'ID': row['ID'], 'target': row['target']})

    for cls_id in sorted(df_train['target'].unique()):
        cls_files = df_train[df_train['target'] == cls_id]['ID'].tolist()
        n_original = len(cls_files)

        if n_original < target_count:
            n_needed = (target_count - n_original) + extra_per_class
        else:
            n_needed = extra_per_class

        if n_needed <= 0:
            continue

        for i in range(n_needed):
            src_fname = random.choice(cls_files)
            img = cv2.imread(str(src_dir / src_fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug_img = aug_transform(image=img)['image']

            aug_fname = f'aug_{cls_id:02d}_{i:04d}.jpg'
            cv2.imwrite(str(dst_dir / aug_fname),
                        cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            records.append({'ID': aug_fname, 'target': cls_id})

        print(f'  class {cls_id:2d}: {n_original}장 → {n_original + n_needed}장 (+{n_needed})')

    df_aug = pd.DataFrame(records)
    print(f'\n✅ 문서 증강 완료: {len(df_train)} → {len(df_aug)}장')
    print(f'   저장: {dst_dir}')
    return df_aug