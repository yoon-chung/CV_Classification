# Run Layout (Team Repo)

본 문서는 `RUNS_DIR` 하위에 생성되는 1회 실행(run)의 디렉토리 레이아웃과 산출물(artifacts) 계약(contract) 규격을 정의합니다.

## Run Root Path

Hydra의 `hydra.run.dir` 설정을 따르며, Hive-style 파티셔닝을 사용합니다:

```
${RUNS_DIR}/${runner_profile}/date={YYYY-MM-DD}/run_id={run_id}/
```

- `RUNS_DIR`: 실행 산출물 루트 디렉토리 (예: `members/<github_id>/outputs`). `ROOT_DIR` 기준 상대 경로로 해석되며 기본값은 `outputs`입니다.
- `runner_profile`: 실행 프로필 (예: `local_proxy`, `local_confirm`, `solve`).
- `date`: UTC 기준 실행 시작 날짜. 형식은 `YYYY-MM-DD`.
- `run_id`: 1회 실행을 유일하게 식별하는 문자열 (예: `20260226T031500Z-acde1234`).

예시:

```
./outputs/local_proxy/date=2026-02-26/run_id=20260226T031500Z-acde1234/
```

큐 실행(동일 큐에서 여러 run을 묶어 추적)에서는 `sweep_id` 파티션을 사용합니다:

```
${RUNS_DIR}/${runner_profile}/date={YYYY-MM-DD}/sweep_id={sweep_id}/run_id={run_id}/
```

- `sweep_id`: 동일한 sweep/배치 실행 묶음을 식별하는 문자열 (예: `23-12-40`).

예시:

```
./outputs/local_proxy/date=2026-02-27/sweep_id=23-12-40/run_id=0_lr-1e-4__batch_size-16/
```

## Required Outputs

아래 파일/디렉토리는 run root 하위에 생성되어야 합니다.

### Metadata

- `meta.json`
  - 필수 필드(권장):
    - `run_id`, `runner_profile`, `sweep_id`
    - `argv`, `hydra` (mode/output_dir/job info)
    - `start_time_utc`, `end_time_utc`, `exit_status`
    - `git.sha`, `git.dirty`
    - `env_fingerprint`
    - `cfg_allowlist`, `cfg_fingerprint`

### Hydra

- `.hydra/`
  - Hydra가 덤프하는 설정 디렉토리.
  - 민감정보(예: API key, 토큰)는 Hydra config에 포함하지 않고 환경변수로만 주입합니다.

### Reports (metrics + arrays)

- `reports/val_metrics.json`
- `reports/classwise_f1.json`
- `reports/confusion_matrix.json`
- `reports/val_preds.npy`
- `reports/val_targets.npy`
- `reports/data_audit.json` (클래스 분포, 이미지 통계 등)
- `reports/perf.json` (wall time, images/sec, gpu mem)

### Checkpoints

- `checkpoints/last.pt`
- `checkpoints/best_macro_f1=<score>.pt`
- `checkpoints/best_loss=<score>.pt` (옵션)

### Submission

- `submissions/submission.csv` (infer 단계 수행 시)

## Storage Destination Policy

아래 표는 산출물이 어디에 저장되는지(로컬/S3/W&B)를 현재 구현 기준으로 정리한 내용입니다.

| 산출물 종류 | 기본 저장 위치 | 비고 |
| --- | --- | --- |
| Run 산출물 원본 (`reports/`, `checkpoints/`, `submissions/`, `meta.json`, `app.log`) | 로컬 (`outputs/.../run_id=.../`) | 항상 로컬에 생성됨 |
| S3 업로드 파일 | S3 + 로컬 원본 유지 | upload 단계에서 whitelist 규칙 통과 파일만 업로드 |
| W&B 메타데이터 | W&B (online) 또는 로컬 export | 현재 구현은 스칼라/요약 메타데이터만 기록 |

## Privacy Rules

아래 정보는 run 산출물(`meta.json`, `reports/*.json` 등)에 기록하지 않습니다.

- hostname
- CPU/GPU 모델명
- 절대 경로 (특히 `<HOME_DIR>/` 포함 경로)

대신 아래 항목만 허용합니다.

- `runner_id`: 실행자/러너를 구분하는 식별자(예: 환경변수로 주입된 짧은 문자열)
- `env_fingerprint`: 환경을 재현 가능하게 만드는 최소 정보의 해시/요약 (예: Python/torch 버전 + requirements lock 해시)

## Group-wise Split Rules

검증 split은 다음 규칙을 따릅니다.

1. 데이터프레임에 `doc_id` 또는 `document_id` 컬럼이 존재하면 이를 `group_key`로 사용합니다.
   - 같은 문서에서 파생된 이미지가 train/val에 섞이는 데이터 누수를 방지합니다.
2. `group_key`가 없으면 stratified split을 사용합니다.
   - 라벨 컬럼(예: `target`) 기준으로 `StratifiedKFold` 또는 `StratifiedShuffleSplit`을 적용합니다.

## Privacy Verification

산출물에 민감 정보가 포함되어 있는지 아래 명령어로 확인할 수 있습니다.

```bash
# 금지 문자열(hostname, <HOME_DIR>/, API 키 패턴 등) 검색
grep -rE "(hostname|<HOME_DIR>/|AI_TOKEN_)" ./members/<github_id>/outputs/local_proxy/date=2026-02-26/run_id=...
```
