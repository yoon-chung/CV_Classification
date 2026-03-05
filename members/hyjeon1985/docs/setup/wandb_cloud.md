# Weights & Biases (W&B) Setup

이 저장소는 실험 메타데이터 기록과 시각화를 위해 W&B를 사용합니다. 모든 실험 로그는 클라우드에 저장되어 팀원들과 실시간으로 공유할 수 있습니다.

## 1) W&B 프로젝트 생성

1. [W&B 홈페이지](https://wandb.ai/)에 로그인합니다.
2. 메인 페이지 오른쪽 상단의 **Create new project** 버튼을 클릭합니다.
3. **Project name** 필드에 프로젝트 이름을 입력합니다(예: `cv-doc-class`).
4. **Visibility**를 설정하고(팀 프로젝트 권장) **Create project**를 클릭합니다.

## 2) 주요 식별자 확인

환경 변수 설정에 필요한 정보는 다음과 같이 확인합니다.

- **WANDB_ENTITY**: W&B 사용자 이름 또는 팀 이름입니다. 프로젝트 URL(`wandb.ai/<entity>/<project>`)에서 확인할 수 있습니다.
- **WANDB_PROJECT**: 위에서 생성한 프로젝트 이름입니다.
- **API Key**: [User Settings](https://wandb.ai/settings) 페이지 하단의 **API keys** 섹션에서 복사할 수 있습니다.

## 3) 환경 변수 설정

`.env` 파일 또는 셸 환경에 아래 변수를 설정합니다.

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `WANDB_API_KEY` | (필수) | W&B 인증을 위한 API 키 (절대 커밋 금지) |
| `WANDB_ENTITY` | (필수) | W&B 팀 또는 사용자 계정명 |
| `WANDB_PROJECT` | `cv-doc-class` | 기록될 프로젝트 이름 |
| `WANDB_MODE` | `online` | `online`, `offline`, `disabled` 중 선택 |
| `LOG_DIR` | `logs` | `RUNS_DIR` 하위 로그 루트 디렉토리 |
| `RUNS_DIR` | `outputs` | W&B 로컬 로그는 기본적으로 `${RUNS_DIR}/wandb` 하위에 저장됩니다. |

## 4) 비용 및 용량 가드레일 (Free Tier)

W&B 무료 티어의 용량 제한을 준수하기 위해 아래 정책을 따릅니다.

- **파일 업로드 금지**: 모델 체크포인트(`.pth`)나 대용량 데이터셋은 W&B Artifacts에 업로드하지 않습니다. 대신 S3를 사용합니다.
- **이미지 로깅 최소화**: 학습 과정의 시각화 이미지는 에폭당 1~2개로 제한하거나, 특정 간격으로만 기록합니다.
- **로그 주기 조절**: 너무 잦은 로깅은 대시보드 응답성을 떨어뜨리므로 적절한 step 간격을 유지합니다.

## 5) 비활성화 및 오프라인 모드

- **WANDB_MODE=disabled**: W&B 로직이 완전히 비활성화됩니다. 인터넷 연결이 없거나 테스트 실행 시 사용합니다.
- **WANDB_MODE=offline**: 로컬에만 로그를 남기고 서버로 전송하지 않습니다. 나중에 `wandb sync` 명령어로 업로드할 수 있습니다.

## 6) 현재 저장 범위 (메타데이터 중심)

- 현재 구현에서 W&B 연동은 upload stage의 메타데이터 기록(스칼라/요약) 중심입니다.
- 모델 체크포인트/예측 배열/리포트 파일 원본은 W&B 아티팩트로 업로드하지 않습니다.
- `WANDB_MODE=offline`에서는 로컬 `outputs/wandb/` 하위에 run 로그가 저장됩니다.
- explore 오케스트레이터 자식 run은 부모 run의 `wandb.group` 값을 그대로 전파받습니다(예: `team_focus_YYYYmmdd_HHMMSS`).

## 7) 보안 주의사항

- `WANDB_API_KEY`는 **절대 Git에 커밋하지 않습니다**.
- 공유 환경(대회 서버 등)에서는 `export` 명령어로 일시적으로 설정하거나, `.gitignore`에 등록된 `.env` 파일을 사용합니다.
