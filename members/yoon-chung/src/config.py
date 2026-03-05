"""
하이퍼파라미터 & 경로 설정
"""
from pathlib import Path

# ============================================
# 팀 작업용 경로
# ============================================
PROJECT_ROOT = Path('..')
DATA_DIR     = PROJECT_ROOT / 'data'
TRAIN_DIR    = DATA_DIR / 'train'
TEST_DIR     = DATA_DIR / 'test'
TRAIN_CSV    = DATA_DIR / 'train.csv'
META_CSV     = DATA_DIR / 'meta.csv'
SAMPLE_CSV   = DATA_DIR / 'sample_submission.csv'
OUTPUT_DIR   = PROJECT_ROOT / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# 하이퍼파라미터
# ============================================
CFG = {
    # 기본
    'seed': 42,
    'num_classes': 17,
    'device': 'cuda',

    # 모델
    'model_name': 'efficientnet_b3',
    'pretrained': True,
    'img_size': 384,

    # 학습
    'epochs': 30,
    'batch_size': 16,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'patience': 7,
    'num_workers': 2,

    # 클래스 불균형 대응
    'use_class_weight': True,

    # MixUp / CutMix
    'use_mixup': False,
    'mixup_alpha': 0.4,
    'use_cutmix': False,
    'cutmix_alpha': 1.0,
    'mix_prob': 0.5,

    # TTA
    'use_tta': True,

    # WandB
    'use_wandb': True,
    'wandb_project': 'doc-classification',
    'wandb_run_name': None,       # None이면 자동 생성
}
