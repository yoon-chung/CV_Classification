"""
train.py — 학습 & 검증 함수
"""
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

import timm

from dataset import mixup_data, cutmix_data, mix_criterion


def build_model(model_name, num_classes, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def get_class_weights(df_train, num_classes, device):
    """클래스 빈도 역수 기반 가중치"""
    class_counts = df_train['target'].value_counts().sort_index().values
    cw = 1.0 / class_counts
    cw = cw / cw.sum() * num_classes
    return torch.FloatTensor(cw).to(device)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, cfg):
    """1 epoch 학습 (MixUp/CutMix 포함)"""
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []
    device = cfg['device']

    for images, labels in tqdm(loader, desc='  Train', leave=False):
        images = images.to(device)
        labels = labels.to(device)

        r = np.random.rand()
        if r < cfg['mix_prob'] / 2 and cfg['use_mixup']:
            images, y_a, y_b, lam = mixup_data(images, labels, cfg['mixup_alpha'])
            outputs = model(images)
            loss = mix_criterion(criterion, outputs, y_a, y_b, lam)
        elif r < cfg['mix_prob'] and cfg['use_cutmix']:
            images, y_a, y_b, lam = cutmix_data(images, labels, cfg['cutmix_alpha'])
            outputs = model(images)
            loss = mix_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(labels_all, preds_all, average='macro')
    return epoch_loss, epoch_f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in tqdm(loader, desc='  Valid', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    return running_loss / len(loader.dataset), f1_score(labels_all, preds_all, average='macro')
