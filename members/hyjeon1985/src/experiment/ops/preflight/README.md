# Preflight Package

Preflight는 실험 실행 전 운영상의 오류를 조기에 발견(fail-fast)하기 위한 검증 단계입니다. 설정 오류나 금지된 토큰 사용 등을 체크하여 리소스 낭비를 방지합니다.

## Purpose

실험 파이프라인이 본격적으로 시작되기 전에, 설정(Config)이나 스펙(Spec) 수준에서 발생할 수 있는 명백한 오류를 사전에 차단하는 것이 목적입니다. 이를 통해 잘못된 설정으로 인한 불필요한 GPU 자원 낭비나 잘못된 데이터 기록을 방지합니다.

## Config Keys

Preflight 동작은 다음 두 가지 키를 통해 제어됩니다.

- `preflight.mode`: `off | warn | strict`
  - `off`: 모든 체크를 건너뛰고 `passed=True`를 반환합니다.
  - `warn`: 이슈가 발견되어도 로그만 남기고 실행을 계속합니다. (기본값)
  - `strict`: `error` 등급의 이슈가 발견되면 `PreflightError`를 발생시켜 실행을 즉시 중단합니다.
- `preflight.ruleset`: `local_fast | confirm | solve`
  - 실행 환경이나 목적에 따라 적용할 규칙 세트를 선택합니다.

**우선순위 및 기본값:**
1. CLI 인자 (예: `preflight.mode=strict`)
2. 실험 종류(`kind`) 기반 기본값:
   - `explore`: `mode=warn`, `ruleset=local_fast`
   - `tune`: `mode=warn`, `ruleset=confirm`
   - `solve`: `mode=strict`, `ruleset=solve`
3. 베이스 설정

## Rulesets

각 규칙 세트는 다음과 같은 체크 항목을 포함합니다.

- `local_fast`: 로컬 환경에서의 빠른 검증을 위한 규칙 모음.
  - `spec.no_legacy_override_tokens`: 금지된 레거시 토큰(`stage`, `optimizer_lr` 등) 사용 여부 체크.
  - `spec.no_env_injection`: `WANDB_MODE` 등 환경 변수를 override로 주입하는지 체크.
  - `cfg.compose_success`: Hydra 설정 구성(Compose) 성공 여부.
  - `cfg.local_has_s3_disabled`: `scenario=local`일 때 S3 업로드가 비활성화되어 있는지 체크.
  - `cfg.wandb_artifacts_disabled`: W&B artifact 업로드가 비활성화되어 있는지 체크 (metadata-only 정책).
  - `cfg.wandb_mode_valid`: W&B 모드가 유효한지(`disabled`, `offline`, `online`) 체크.
- `confirm`: MVP 단계에서는 `local_fast`와 동일한 규칙을 적용합니다.
- `solve`: MVP 단계에서는 `confirm`과 동일한 규칙을 적용하며, `strict` 모드 사용을 권장합니다.

## Modes

- `off`: 모든 체크를 스킵하며, 항상 `report.passed=True`를 반환합니다.
- `warn`: 이슈 발견 시 경고 로그를 출력하되, 예외를 발생시키지 않고 실행을 허용합니다.
- `strict`: `error` 등급의 이슈가 하나라도 발견되면 `PreflightError`를 발생시켜 전체 프로세스를 중단합니다.

## Integration

Preflight는 `request_experiment()` 함수 내에서 다음과 같은 순서로 통합되어 실행됩니다.

1. `load_experiment_spec()`: YAML 실험 스펙 로드.
2. `planner.parse()` / `planner.plan()`: 실행 계획(Schedule) 수립.
3. **`run_preflight()`**: 수립된 계획과 스펙을 바탕으로 사전 검증 수행.
4. `executor.execute()`: 검증 통과 시(또는 `warn` 모드일 때) 실험 실행.

Preflight 단계는 부수 효과가 없는(side-effect free) 불변(immutable) 작업으로 설계되어 있어, 실행 계획이나 설정을 변경하지 않고 오직 검증 결과만 보고합니다.

## Examples

### 1. Local strict
로컬 탐색 실험(`kind=explore`)을 수행하면서도 엄격한 검증을 원하는 경우:
```bash
python -m src.experiment --spec-name experiment/explore_sample preflight.mode=strict
```
이 경우 `local_fast` 규칙 세트가 적용되며, 이슈 발견 시 실행이 중단됩니다.

### 2. Solve strict
최종 제출용 실험(`kind=solve`)을 실행할 때:
```bash
python -m src.experiment --spec-name experiment/solve_submission
```
`kind=solve`의 기본값에 의해 `strict` 모드와 `solve` 규칙 세트가 자동으로 적용되어 안전한 실행을 보장합니다.
