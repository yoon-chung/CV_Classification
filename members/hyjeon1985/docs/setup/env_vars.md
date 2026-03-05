# Environment Variables Reference

이 프로젝트의 모든 설정은 환경 변수를 통해 제어됩니다. 민감한 정보는 절대 코드나 Git에 포함하지 않습니다.

`members/<github_id>/scripts/*.sh`는 `MEMBER_ID`를 자동으로 추론하며, 파이썬 엔트리(`python -m experiment`)가 개인 `.env`를 자동으로 로드합니다.
필요 시 `CV_PRJ_ROOT`로 프로젝트 루트를, `ROOT_DIR`로 멤버 워크스페이스 루트를 명시할 수 있습니다.

## 1) 환경 변수 목록 (연관값 묶음)

환경 변수는 아래 3개 묶음으로 관리합니다.

1. **실험 실행 관련 변수**
2. **프로젝트 폴더/경로 관리 변수**
3. **서비스/도구 연동 변수**

### A. 실험 실행 관련

| 이름 | 기본값 | 사용 레이어 | 비고 |
| --- | --- | --- | --- |
| `MAX_CONCURRENCY` | `1` | configs (`runner_profile/local_proxy.yaml`) | 로컬 실행 동시성 |
| `NUM_WORKERS` | `4` | configs (`runner_profile/*.yaml`) | dataloader worker |
| `WANDB_MODE` | `disabled` | configs (`scenario/*.yaml`) + runtime | `disabled/offline/online` |
| `CVDC_CACHE_NAMESPACE` | `cvdc` | configs (`base/default.yaml`, `scenario/*.yaml`) | 캐시 네임스페이스 |
| `CVDC_CACHE_VERSION` | `v1` | configs (`base/default.yaml`, `scenario/*.yaml`) | 캐시 버전 |
| `CVDC_CODE_SALT` | `null` | configs (`base/default.yaml`, `scenario/*.yaml`) | 코드 salt |
| `CVDC_DATA_SALT` | `null` | configs (`base/default.yaml`, `scenario/*.yaml`) | 데이터 salt |
| `CVDC_REQUIRE_CUDA` | `0` | scripts + runtime nodes (`nodes/{train,eval,infer}.py`) | `1`이면 CUDA 불가 시 즉시 실패(=CPU fallback 금지) |
| `RUNNER_ID` | `<github_id>` | runtime | 실행 주체 식별자 |
| `REUSE_ENABLED` | `true` | runtime | 이전 산출물 재사용 |
| `GC_ENABLED` | `true` | runtime | GC 활성화 |
| `GC_KEEP_RECENT` | `1` | runtime | 최근 N개 queue 보존 |

### B. 프로젝트 폴더/경로 관리

| 이름 | 기본값 | 사용 레이어 | 비고 |
| --- | --- | --- | --- |
| `ROOT_DIR` | `members/<github_id>` | runtime | 멤버 워크스페이스 루트 |
| `CONFIG_DIR` | `configs` | runtime | Hydra config 루트 (`ROOT_DIR` 기준 상대) |
| `DOCS_DIR` | `docs` | runtime | 문서 루트 (`ROOT_DIR` 기준 상대) |
| `DATA_DIR` | `data` | configs (`base/default.yaml`, `base/data.yaml`) | 입력 데이터 루트 (`ROOT_DIR` 기준 상대) |
| `CACHE_DIR` | `cache` | configs (`base/default.yaml`, `scenario/*.yaml`) | 캐시 루트 (`ROOT_DIR/cache`) |
| `RUNS_DIR` | `outputs` | configs (`base/default.yaml`, `base/data.yaml`) | 실행 산출물 루트 (`ROOT_DIR` 기준 상대) |
| `LOG_DIR` | `logs` | configs (`base/default.yaml`) + runtime | 로그 루트 (`RUNS_DIR` 하위) |

### C. 서비스/도구 연동

| 이름 | 기본값 | 사용 레이어 | 비고 |
| --- | --- | --- | --- |
| `S3_BUCKET` | (비어있음) | runtime (`src/store/backends/s3.py`) | 실제 업로드 백엔드가 읽는 버킷 키 |
| `S3_PREFIX` | `cvdc` | runtime (`src/store/backends/s3.py`) | 실제 업로드 백엔드 prefix |
| `S3_DEDUP_BY_HASH` | `1` | runtime (`src/store/backends/s3.py`) | CAS dedup |
| `S3_DRY_RUN` | `0` | runtime (`src/store/backends/s3.py`) | 실제 업로드 없이 로컬 staging만 수행 |
| `S3_UPLOAD_WHITELIST` | (비어있음) | runtime (`src/store/backends/s3.py`) | 업로드 허용 패턴(쉼표 구분) |
| `AWS_ACCESS_KEY_ID` | (비어있음) | boto3 | AWS 인증 |
| `AWS_SECRET_ACCESS_KEY` | (비어있음) | boto3 | AWS 인증 |
| `AWS_REGION` | `ap-northeast-2` | boto3 | AWS 리전 |
| `WANDB_API_KEY` | (비어있음) | wandb SDK | W&B 인증 |
| `WANDB_ENTITY` | (비어있음) | wandb SDK | W&B entity |
| `WANDB_PROJECT` | `cv-doc-class` | wandb SDK | W&B project |
| `SLACK_NOTIFY` | `0` | notify runtime | Slack 알림 on/off |
| `SLACK_WEBHOOK_URL` | (비어있음) | notify runtime | webhook 연동 |
| `SLACK_BOT_TOKEN` | (비어있음) | notify runtime | bot 연동 |
| `SLACK_CHANNEL_ID` | (비어있음) | notify runtime | 채널 지정 |
| `HF_TOKEN` | (비어있음) | HF SDK | Hugging Face 인증 |
| `HUGGINGFACE_HUB_TOKEN` | `${HF_TOKEN}` | HF SDK | 호환 토큰 |
| `HF_HOME` | `cache/huggingface` | HF SDK | HF 캐시 루트 (`ROOT_DIR` 기준 상대) |
| `HF_HUB_CACHE` | `cache/huggingface/hub` | HF SDK | Hub 캐시 (`ROOT_DIR` 기준 상대) |
| `TRANSFORMERS_CACHE` | `cache/huggingface/hub` | Transformers | 모델 캐시 (`ROOT_DIR` 기준 상대) |
| `HF_XET_HIGH_PERFORMANCE` | `1` | HF SDK | 전송 모드 |

- `WANDB_MODE=online`이면 `WANDB_API_KEY`, `WANDB_ENTITY`가 필요합니다.
- S3 업로드를 사용하면 `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET`이 필요합니다.
- Slack 알림은 `SLACK_WEBHOOK_URL` 또는 (`SLACK_BOT_TOKEN` + `SLACK_CHANNEL_ID`) 중 하나가 필요합니다.
- Hugging Face private/gated 모델을 사용하면 `HF_TOKEN`이 필요합니다.
- `--dummy-data` 없이 실행하면 `DATA_DIR` 하위 실데이터가 필요합니다.
- 팀 정책상 실데이터는 루트 `data/`가 아니라 `members/<github_id>/data/`에만 둡니다.

## 2) RUNNER_ID 가이드라인

`RUNNER_ID`는 실험 결과에서 누가 실행했는지 구분하기 위해 사용합니다.

- **개인정보 보호**: 실제 이름이나 이메일 대신 별명(예: `runner-01`, `gpu-server-a`)을 사용합니다.
- **용도**: W&B 실행 이름이나 S3 경로에 포함되어 실험 이력을 추적하는 데 도움을 줍니다.

## 3) 보안 및 비밀 정보 관리

- **커밋 금지**: `.env` 파일이나 비밀 정보가 포함된 스크립트는 절대 Git에 커밋하지 않습니다.
- **.gitignore**: `members/<github_id>/.env`가 Git에 포함되지 않도록 확인합니다.
- **공유 방식**: 팀원 간 비밀 정보 공유는 보안 메신저나 별도의 안전한 채널을 이용합니다.

## 4) .env 템플릿

`members/<github_id>/.env` 파일을 만들고 아래 내용을 복사하여 수정합니다.

```bash
# ===== A) 실험 실행 관련 =====

# Runner / execution
MAX_CONCURRENCY=1
NUM_WORKERS=4

# Experiment cache identity
CVDC_CACHE_NAMESPACE=cvdc
CVDC_CACHE_VERSION=v1
CVDC_CODE_SALT=
CVDC_DATA_SALT=

# Runtime behavior
CVDC_REQUIRE_CUDA=1
WANDB_MODE=disabled
RUNNER_ID=<github_id>
REUSE_ENABLED=true
GC_ENABLED=true
GC_KEEP_RECENT=1

# ===== B) 프로젝트 폴더/경로 관리 =====
ROOT_DIR=members/<github_id>
CONFIG_DIR=configs
DOCS_DIR=docs

DATA_DIR=data
CACHE_DIR=cache
RUNS_DIR=outputs
LOG_DIR=logs

# ===== C) 서비스/도구 연동 =====

# S3 upload
S3_BUCKET=
S3_PREFIX=cvdc
S3_DEDUP_BY_HASH=1
S3_DRY_RUN=0
S3_UPLOAD_WHITELIST=

# AWS credentials for boto3
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=ap-northeast-2

# W&B auth/project
WANDB_API_KEY=
WANDB_ENTITY=
WANDB_PROJECT=cv-doc-class

# Slack notification
SLACK_NOTIFY=0
SLACK_WEBHOOK_URL=
SLACK_BOT_TOKEN=
SLACK_CHANNEL_ID=

# Hugging Face
HF_TOKEN=
HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}
HF_HOME=cache/huggingface
HF_HUB_CACHE=cache/huggingface/hub
TRANSFORMERS_CACHE=cache/huggingface/hub
HF_XET_HIGH_PERFORMANCE=1
```

경로 해석 규칙:
- 상대 경로(`data`, `cache`, `outputs` 등)는 항상 `ROOT_DIR`(멤버 워크스페이스) 기준으로 절대 경로로 변환됩니다.
- 따라서 `.env`가 멤버 폴더에 있어도, 경로 값에 멤버 ID를 하드코딩할 필요가 없습니다.

추가 가이드:

- `docs/setup/wandb_cloud.md`
- `docs/setup/s3_artifacts.md`
- `docs/setup/huggingface_hub.md`
- `docs/setup/github_finegrained_pat.md`
