"""
inference.py — TTA 추론 & 앙상블
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import DocDataset
from augmentation import get_valid_transforms, get_tta_transforms


@torch.no_grad()
def predict_tta(model, df_test, test_dir, cfg):
    """TTA 적용 추론"""
    model.eval()
    device = cfg['device']
    transforms = get_tta_transforms(cfg['img_size']) if cfg['use_tta'] \
                 else [get_valid_transforms(cfg['img_size'])]

    print(f'TTA: {len(transforms)}개 변환')
    all_probs = np.zeros((len(df_test), cfg['num_classes']))

    for t_idx, tfm in enumerate(transforms):
        print(f'  TTA {t_idx + 1}/{len(transforms)}...')
        ds = DocDataset(df_test, test_dir, tfm, is_test=True)
        loader = DataLoader(ds, batch_size=cfg['batch_size'] * 2,
                            shuffle=False, num_workers=cfg['num_workers'],
                            pin_memory=True)
        probs = []
        for images in tqdm(loader, leave=False):
            outputs = model(images.to(device))
            probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        all_probs += np.concatenate(probs, axis=0)

    all_probs /= len(transforms)
    return np.argmax(all_probs, axis=1), all_probs


def save_submission(predictions, df_test, save_path):
    """제출 CSV 저장"""
    df_sub = df_test.copy()
    df_sub['target'] = predictions
    df_sub.to_csv(save_path, index=False)
    print(f'제출 파일: {save_path}')
    return df_sub


def ensemble_from_files(prob_files, df_test, output_dir, save_name='submission_ensemble.csv'):
    """저장된 probs 파일들 soft voting 앙상블"""
    output_dir = Path(output_dir)
    prob_list = []
    for f in prob_files:
        fp = output_dir / f if not Path(f).is_absolute() else Path(f)
        if fp.exists():
            prob_list.append(np.load(fp))
            print(f'  {fp.name}')
        else:
            print(f'  {fp.name} (스킵)')

    if not prob_list:
        print('로드된 파일 없음')
        return None, None

    print(f'\n{len(prob_list)}개 모델 soft voting')
    avg_probs = np.mean(prob_list, axis=0)
    predictions = np.argmax(avg_probs, axis=1)

    save_path = output_dir / save_name
    save_submission(predictions, df_test, save_path)

    max_p = avg_probs.max(axis=1)
    print(f'Confidence: mean={max_p.mean():.3f} | >0.9: {(max_p > 0.9).sum()}장')
    return predictions, avg_probs
