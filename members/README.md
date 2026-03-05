# Members

팀원별 개인 작업 영역은 `members/<github_id>/` 아래에서 관리한다.

- `members/<github_id>/src/` : 개인 실험 코드
- `members/<github_id>/notebooks/` : 개인 노트북
- `members/<github_id>/scripts/` : 개인 실행 스크립트
- `members/<github_id>/configs/` : 개인 실험 설정
- `members/<github_id>/docs/` : 개인 문서
- `members/<github_id>/.env` : 개인 환경변수 파일
- `members/<github_id>/data/` : 개인 대회 데이터(로컬 전용)
- `members/<github_id>/outputs/` : 개인 실행 산출물

규칙:

- 개인 파이프라인 코드는 `members/<github_id>/src/`에서 관리한다.
- 공용 저장소 반영은 멤버 영역에서 검증된 변경만 최소 단위로 이관한다.
## Onboarding (초기 설정)

저장소 클론 후 아래 명령어로 환경을 구축한다.

```bash
MEMBER_ID=<github_id> bash bootstrap.sh
source .venv/bin/activate
```

- 팀 운영 원칙: bootstrap은 1회 실행, 가상환경 활성화는 각 터미널 세션에서 수동으로 수행한다.
- `MEMBER_ID`를 지정하면 `members/<github_id>/.env.template`에서 개인 `.env`를 자동 생성한다.
- 실행 스크립트는 `CV_PRJ_ROOT`(프로젝트 루트)와 `ROOT_DIR`(멤버 워크스페이스 루트)를 지원하므로, 멤버 폴더 내부/외부 어디서 실행해도 동일 동작을 보장한다.
- 기본 `ROOT_DIR`는 `members/<github_id>`이며 `outputs/configs/docs/data/cache`는 모두 이 경로를 기준으로 상대 해석된다.
- 루트 `data/`는 사용하지 않고, 각 멤버가 자신의 `members/<github_id>/data/`에 대회 데이터를 두는 것을 기본 정책으로 한다.

- 실행 스크립트가 `ROOT_DIR`를 기준으로 `PYTHONPATH`를 자동 구성하므로, 멤버 `src`를 별도 설정 없이 임포트할 수 있다.
- 예시: `from src.data.dataset import CVDCDataset`

## Execution (실험 실행)

멤버별 작업 영역에서 실험을 실행하는 방법은 다음과 같다.

```bash
# 1. 멤버별 전용 스크립트 사용 (권장)
bash members/<github_id>/scripts/run_explore.sh

# 2. 직접 CLI 실행 (세부 제어 필요 시)
PYTHONPATH=src python -m experiment scenario=local runner_profile=local_proxy kind=explore
```

## Contribution Guidelines (협업 규칙)

- **Small PRs**: 하나의 PR은 하나의 논리적 변경만 포함한다.
- **Atomic Commits**: 하나의 커밋은 하나의 관심사(concern)만 다룬다.
- **Review First**: `main` 브랜치 직접 푸시는 금지하며, 반드시 PR을 통해 리뷰 후 머지한다.
