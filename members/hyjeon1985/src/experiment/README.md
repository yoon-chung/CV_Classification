# src.experiment

`src.experiment`는 실험 수행 자체보다, **실험 의뢰(Request) 추상화**를 중심으로 설계합니다.

핵심 원칙:
- 실험 패키지는 저장소 경로, S3, 모니터링 백엔드 세부를 직접 알지 않습니다.
- 실험 타입은 `explore | tune | solve` 3가지만 허용합니다.
- 저장/알림/모니터링은 `ExperimentClient`가 담당합니다.
- 스펙/계약 모델 검증은 `pydantic` 기반으로 처리합니다.

## 공개 개념

- `ExperimentKind`
  - `explore`, `tune`, `solve` strict enum
- `ExperimentClient` (Protocol)
  - `request_keyword() -> str`
  - 선택 구현: `accept_schedule(schedule) -> None`
  - 선택 구현: `accept_summary(summary) -> None`
- `ExperimentPlanner` (ABC)
  - spec 해석 + directive 생성 + schedule 생성
- `ExperimentExecutor` (ABC)
  - schedule 실행(또는 dry-run) 후 summary 반환

## 책임 분리

- Planner
  - YAML 스펙을 이해 가능한 구조로 해석합니다.
  - `request_keyword`를 받아 `experiment_id`/`run_id` 생성 규칙에 반영합니다.
  - 실행 순서(스케줄)만 생성합니다.
- Executor
  - 스케줄을 실행합니다(기본 구현은 passive입니다).
- Client
  - 스케줄/요약을 받아 저장/전달/통합 처리합니다.
  - 로컬 경로, 클라우드 경로, 모니터링 연동 정책을 소유합니다.

## 엔트리포인트 (MVP)

- CLI: `python -m experiment scenario=local runner_profile=local_proxy kind=explore`
- Config Axes: `scenario`, `runner_profile`, `kind` (Hydra composition)
- Pipeline Nodes: `prep`, `train`, `eval`, `infer`, `submission`, `upload`
- QA Tools: `preflight` (fail-fast validation)

## 스펙 스키마(요약)

```yaml
experiment:
  type: explore  # explore | tune | solve
  config_name: explore/default

plan:
  items:
    - run_id: e001
      priority: 10
      overrides:
        - pipeline.stop_after=eval
```

주의:
- `experiment.kind` 같은 별칭은 지원하지 않습니다.
- 실행 산출물 경로/저장소 위치는 스펙에 두지 않습니다.

## 모델링 원칙 (Pydantic)

- YAML 로딩 직후 pydantic 모델 검증을 수행합니다.
- unknown key는 금지(`extra=forbid`)하여 오타를 즉시 실패시킵니다.
- 스펙 파싱과 도메인 계약(`ExperimentSpec`, `ExperimentDirective*`, `WorkSchedule*`)은 pydantic 모델을 사용합니다.

## 참고

과거 설계/구현은 git 히스토리를 참고합니다.
