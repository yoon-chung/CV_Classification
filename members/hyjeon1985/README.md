# 개인 워크스페이스(`hyjeon1985`)

본 디렉토리는 팀 저장소 내에서 `hyjeon1985` 멤버가 실험을 수행하고 결과를 정리하기 위한 개인 작업공간입니다.
루트 [README.md](/home/jhoya/Workspaces/ai-22-cv-cv-team1/README.md)는 대회 결과 산출물 관점의 요약 문서이며, 본 문서는 개인 파이프라인 운영 관점의 문서를 연결·정리합니다.

## 디렉토리 구성

- `configs/`: 개인 실험 설정(Hydra) 루트입니다.
  - 자세한 내용: [configs/README.md](/home/jhoya/Workspaces/ai-22-cv-cv-team1/members/hyjeon1985/configs/README.md)
- `src/`: 실험 파이프라인 구현 코드입니다.
  - 설계 개요: [src/experiment/README.md](/home/jhoya/Workspaces/ai-22-cv-cv-team1/members/hyjeon1985/src/experiment/README.md)
- `scripts/`: 실행 편의 스크립트입니다.
- `docs/`: 운영/설정/산출물 규격 문서입니다.
  - 실행 산출물 규격: [docs/run_layout.md](/home/jhoya/Workspaces/ai-22-cv-cv-team1/members/hyjeon1985/docs/run_layout.md)
  - 외부 서비스 연동: `docs/setup/*.md`
- `outputs/`: 실행 산출물이 생성되는 기본 경로입니다(대용량이 될 수 있어 Git에 포함하지 않습니다).
- `archive/`: 제출/분석 재현을 위한 최소 산출물 아카이브입니다(필요 파일만 선별 보관합니다).
- `data/`: 대회 원본 데이터 로컬 보관 경로입니다(Git에 포함하지 않습니다).
- `cache/`: 모델/데이터 캐시 경로입니다(Git에 포함하지 않습니다).

## 보안 및 운영 원칙

- 민감정보(API 키, 토큰 등)는 코드/문서/Git에 포함하지 않으며, `.env` 또는 런타임 환경변수로만 주입합니다.
- 대회 원본 데이터는 저장소에 커밋하지 않습니다.
- 아카이브는 재현에 필요한 최소 산출물만 보관하는 것을 원칙으로 합니다.

## 빠른 링크

- 환경변수 레퍼런스: [docs/setup/env_vars.md](/home/jhoya/Workspaces/ai-22-cv-cv-team1/members/hyjeon1985/docs/setup/env_vars.md)
- W&B 설정(메타데이터 중심): [docs/setup/wandb_cloud.md](/home/jhoya/Workspaces/ai-22-cv-cv-team1/members/hyjeon1985/docs/setup/wandb_cloud.md)
- S3 아티팩트 저장: [docs/setup/s3_artifacts.md](/home/jhoya/Workspaces/ai-22-cv-cv-team1/members/hyjeon1985/docs/setup/s3_artifacts.md)
- Slack 알림: [docs/setup/slack_notifications.md](/home/jhoya/Workspaces/ai-22-cv-cv-team1/members/hyjeon1985/docs/setup/slack_notifications.md)

