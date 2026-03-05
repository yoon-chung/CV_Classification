# Document Type Classification

`jun-yoon1` 멤버의 실험 파이프라인 작업 영역입니다.
3개 모델 앙상블 기준의 학습/튜닝/제출 스크립트를 포함합니다.

## Directory
```text
.
├── configs/
│   ├── base/       # 공통 데이터/모델/학습 설정
│   ├── explore/    # 실험용 설정(간단 오버레이)
│   └── tune/       # 튜닝용 설정(간단 오버레이)
├── scripts/
│   ├── prepare_train_v1.py
│   ├── search_ensemble_weights.py
│   ├── infer_ensemble.py
│   ├── run_explore.sh
│   ├── run_tune.sh
│   ├── run_submit.sh
│   └── train.py
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── notebooks/
│   └── README.md
├── data/
├── outputs/
└── README.md
```

## Data Cleansing
`train_v1.csv` 기준:
- 라벨 수정 5건 반영
- 샘플 삭제 2건 반영
- 총 샘플 수 `1570 -> 1568`

실행:
```bash
python scripts/prepare_train_v1.py
```

EDA 요약 생성:
```bash
python scripts/eda_report.py
```

출력:
- `outputs/eda/class_distribution.csv`
- `outputs/eda/summary.csv`
- `outputs/eda/missing_images.csv` (있는 경우)

## Server Profile (Detected)
- GPU: RTX 3090 24GB VRAM
- CPU: AMD Threadripper PRO 3975WX (64 vCPU)
- RAM: 251GB

권장 운영:
- `image_size=384` 중심
- 모델별 batch size 개별 적용
- `num_workers=12`
- Explore 단계 `folds=3`, Tune 단계 `folds=5`

## EDA/Cleansing Recommendation
1. 라벨 품질 점검
- 클래스별 confusion 상위 오분류 샘플을 수집해 라벨 오염 재검토
- OCR 없이 가능한 수준에서 제목/레이아웃 유사 클래스(예: 진단서/소견서) 육안 검수

2. 데이터 분포 점검
- 클래스별 개수 및 불균형 비율 확인
- 해상도/종횡비/회전 각도 분포 확인
- 너무 어둡거나 과노이즈 샘플 비율 점검

3. 중복/근접중복 탐지
- pHash 기반 near-duplicate 점검
- 동일 문서 촬영 변형(밝기/각도만 다른 샘플) 비율 파악

4. 저성능 클래스 보강
- mixup/cutmix는 제외
- 저성능 클래스에만 클래스-조건부 증강 강도 상향:
  - `Affine/Perspective`, `MotionBlur/GaussianBlur`, `Brightness/Contrast`, `JPEGCompression`
- 고성능 클래스는 약한 증강 유지해 과증강 부작용 방지

현재 적용:
- 취약 클래스: `3, 7, 14`
- 강증강 프로필: `strong_profile: v2`

## Ensemble Direction
기본 전략:
- 3개 모델 독립 학습
- fold OOF 기반 가중치 soft-voting
- 클래스별 F1이 낮은 클래스에 대해 앙상블 가중치 조정

권장 3모델 (timm pretrained 가용):
- `convnext_small.in12k_ft_in1k_384`
- `tf_efficientnetv2_s.in1k`
- `swin_tiny_patch4_window7_224.ms_in1k`

## Run
탐색(빠른 검증):
```bash
bash members/jun-yoon1/scripts/run_explore.sh 2026-02-27 ensemble3 explore aug_low_f1_v1 3
```

튜닝(최종 검증):
```bash
bash members/jun-yoon1/scripts/run_tune.sh 2026-02-27 ensemble3 tune v1 5
bash members/jun-yoon1/scripts/run_tune.sh 2026-02-27 ensemble3 tune v2 5
```

OOF 기반 가중치 자동 탐색 + 제출 파일 생성:
```bash
bash members/jun-yoon1/scripts/run_submit.sh 2026-02-27 ensemble3 tune v1 3000
bash members/jun-yoon1/scripts/run_submit.sh 2026-02-27 ensemble3 tune v2 3000
```

생성 결과:
- `experiments/.../ensemble/weights.csv`
- `experiments/.../ensemble/best_score.csv`
- `experiments/.../submissions/submission_weighted.csv`

단일 스모크 테스트(1모델/1fold/1epoch):
```bash
python scripts/train.py \
  --date 2026-02-27 \
  --model ensemble3 \
  --method smoke \
  --direction sanity \
  --models tf_efficientnetv2_s.in1k \
  --folds 1 \
  --epochs 1
```

## Experiment Rule
- 저장 규칙: `experiments/{date}/{model}/{method}/{direction}/`
- 예시: `experiments/2026-02-27/convnext_small/ce_clsaug_v1/hard_cls_boost/`
