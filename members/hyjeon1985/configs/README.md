# Hydra Experiment Configurations

본 디렉토리는 `hyjeon1985` 멤버의 실험 설정을 관리하는 Hydra 설정 루트입니다. 모든 실험은 본 설정을 통해 정의되고 실행됩니다.

## Configs Root
Hydra 설정의 루트 경로는 `members/hyjeon1985/configs/`입니다. 모든 상대 경로는 이 위치를 기준으로 해석됩니다.

## Single Entrypoint
`experiment.yaml`은 본 설정 시스템의 단일 진입점(Entrypoint)입니다. 실행 시 별도의 `--config-name`을 지정하지 않으면 기본적으로 이 파일을 로드합니다.

## Composition Axes
설정은 책임 분리에 따라 다음과 같은 축(Axes)으로 구성됩니다.

- `scenario/`: 실행 환경 및 경로 설정 (데이터셋 경로, 로그 디렉토리, S3 설정 등)
- `runner_profile/`: 실행 정책 (런처 종류, 리소스 할당, 타임아웃 등)
- `kind/`: 실험 모드 선택 (`explore`, `tune`, `solve`)

### Upload Policy
- **S3**: `upload.s3.enabled` 설정을 통해 제어합니다.
- **W&B**: `wandb.mode`를 통해 메타데이터 업로드 여부를 결정하며, 아티팩트 업로드는 금지합니다.

## Experiment Kind
`experiment.kind` 설정을 통해 실험의 성격을 정의합니다.

- `explore`: 새로운 가설이나 방식을 빠르게 탐색하는 단계
- `tune`: 확정된 방식 위에서 하이퍼파라미터를 최적화하는 단계
- `solve`: 최종 제출을 위해 최적의 설정으로 학습 및 추론을 수행하는 단계

## Pipeline
실험 파이프라인은 `pipeline.step`과 `pipeline.stop_after`를 통해 제어합니다.

- **Standard Chain**: `prep` → `train` → `eval` → `infer` → `submission` → `upload`
- `pipeline.step=full`: 전체 파이프라인을 처음부터 끝까지 실행합니다.
- `pipeline.stop_after`: 실행을 중단할 마지막 단계를 지정합니다(해당 단계 포함).

## Override Rules
Hydra의 dot-path 문법을 사용하여 명령행에서 설정을 자유롭게 덮어쓸 수 있습니다.
예: `optimizer.lr=3e-4`

## Caching & Queue Strategy
효율적인 실험 실행을 위해 캐싱과 큐 전략을 지원합니다.

- **Caching**: `pipeline.cache.*` 및 `explore.stages[].cache.*` 키를 통해 제어합니다. 캐시 키는 데이터, 코드, 설정의 해시값을 조합하여 계산됩니다.
- **Queue**: `runner.queue.strategy`를 통해 실험 실행 순서를 관리합니다(예: `fifo`).

## Preflight (Fail-Fast)
실행 전 설정의 유효성을 검증하여 실패를 조기에 방지합니다.

- **Keys**: `preflight.mode`, `preflight.ruleset`
- **Priority**: CLI > kind > scenario > base
- **Defaults**:
  - `base`: `warn`/`confirm` (경고 후 사용자 확인)
  - `local`: `strict`/`local_fast` (엄격한 검증, 빠른 로컬 실행)
  - `solve`: `strict`/`solve` (최종 실행을 위한 엄격한 검증)

세부 규칙은 [Preflight README](../src/experiment/ops/preflight/README.md)를 참고합니다.

## WandB (Metadata-Only)
실험 로그 기록을 위해 W&B를 사용하되, 메타데이터 기록으로 제한합니다.

- **Default Mode**: `disabled`
- **Offline Workflow**: `WANDB_MODE=offline`으로 실행 후 `wandb sync`로 업로드합니다.
- **Keys**: `wandb.dir`, `wandb.log_interval`, `wandb.metrics` 등
- **Restriction**: 아티팩트(모델 가중치, 대용량 데이터 등) 업로드는 엄격히 금지합니다.

## Artifacts Level
`artifacts.level` 설정을 통해 저장할 산출물의 수준을 결정합니다.

- `minimal`: 필수 로그 및 최종 결과물만 저장
- `full`: 중간 체크포인트 및 상세 분석 데이터를 포함하여 저장

## How It Integrates
실험 패키지는 Hydra의 `initialize_config_dir`와 `compose` API를 사용하여 본 설정을 통합합니다.
- **Contract**: `upload.s3.enabled=false`인 경우 S3 클라이언트를 생성하지 않아야 합니다.

## 참고
과거 실험 설정 구성은 git 히스토리를 참고합니다.

## Examples
자주 사용되는 실행 예시는 다음과 같습니다. (저장소 루트 기준)

1. **로컬 환경에서 탐색 실험 실행**
   ```bash
   PYTHONPATH=src python -m experiment scenario=local runner_profile=local_proxy kind=explore
   ```

2. **W&B 온라인 모드로 실행**
   ```bash
   PYTHONPATH=src python -m experiment scenario=local wandb.mode=online
   ```

3. **특정 하이퍼파라미터 튜닝 실행**
   ```bash
   PYTHONPATH=src python -m experiment kind=tune tune=night1_lr
   ```

4. **최종 제출용 실행 (업로드 단계까지)**
   ```bash
   PYTHONPATH=src python -m experiment kind=solve solve=submission pipeline.stop_after=upload
   ```

5. **주요 파라미터 직접 덮어쓰기**
   ```bash
   PYTHONPATH=src python -m experiment optimizer.lr=3e-4 model.backbone=resnet50 preprocess.target_size=384
   ```

6. **탐색 시나리오 변경 (Focus)**
   ```bash
   PYTHONPATH=src python -m experiment explore=team_focus
   ```

7. **탐색 시나리오 변경 (Expand)**
   ```bash
   PYTHONPATH=src python -m experiment explore=team_expand
   ```
