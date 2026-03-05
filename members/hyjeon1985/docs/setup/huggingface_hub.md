# Hugging Face Hub Setup

본 문서는 Hugging Face(HF) 토큰/캐시를 설정하여 사전학습 가중치 다운로드를 안정화하고, 실험 시작 지연을 줄이기 위한 운영 가이드입니다.

현재 파이프라인은 `timm.create_model(..., pretrained=True)`를 사용하므로, 네트워크 상태와 캐시 경로 설정에 따라 초기 실행 시간이 크게 달라질 수 있습니다.

## 1) 토큰 발급

1. Hugging Face 계정에 로그인합니다.
2. Settings -> Access Tokens에서 read 권한 토큰을 생성합니다.
3. 생성한 토큰은 안전한 채널에만 보관하고 Git에는 절대 커밋하지 않습니다.

## 2) 환경 변수 설정

`members/<github_id>/.env` 또는 셸 환경에 아래 값을 설정합니다.

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `HF_TOKEN` | (선택) | HF 인증 토큰. private/gated 모델 접근 시 필수 |
| `HUGGINGFACE_HUB_TOKEN` | (선택) | 레거시 호환 변수. 가능하면 `HF_TOKEN` 중심으로 사용 |
| `HF_HOME` | `cache/huggingface` 권장 | HF 캐시 루트(토큰/허브 캐시 포함, `ROOT_DIR` 기준 상대) |
| `HF_HUB_CACHE` | `${HF_HOME}/hub` | 모델/리포지토리 캐시 경로 |
| `TRANSFORMERS_CACHE` | `${HF_HUB_CACHE}` | 일부 라이브러리 호환용 캐시 경로 |
| `HF_XET_HIGH_PERFORMANCE` | `1` 권장 | 고성능 전송 모드 활성화 |

예시:

```bash
HF_TOKEN=hf_xxx_your_token
HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}
HF_HOME=cache/huggingface
HF_HUB_CACHE=cache/huggingface/hub
TRANSFORMERS_CACHE=cache/huggingface/hub
HF_XET_HIGH_PERFORMANCE=1
```

주의:

- 상대 경로(`cache/...`)를 사용하면 실행 시 `ROOT_DIR` 기준 절대 경로로 정규화되어 멤버 워크스페이스 구조와 일관됩니다.
- 캐시는 가능하면 빠른 SSD/NVMe 경로를 사용합니다.

## 3) 적용 확인

```bash
python -c "from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE; print(HF_HOME); print(HF_HUB_CACHE)"
huggingface-cli whoami
```

`whoami`가 정상 응답하면 토큰 인증이 유효합니다.

## 4) 속도 개선 운영 팁

- 첫 실행 전에 자주 쓰는 backbone을 미리 1회 로딩(prewarm)합니다.
- 동일 서버에서 반복 실험 시 `HF_HOME`을 고정해 캐시 재사용률을 높입니다.
- 네트워크가 불안정할 때는 캐시 확보 후 `HF_HUB_OFFLINE=1`로 오프라인 실행을 고려합니다.

prewarm 예시:

```bash
python - <<'PY'
import timm

models = [
    "convnext_small.fb_in22k_ft_in1k",
    "tf_efficientnet_b3",
    "tf_efficientnetv2_s",
]
for name in models:
    timm.create_model(name, pretrained=True, num_classes=17)
print("hf/timm prewarm done")
PY
```

## 5) 보안 가드레일

- `HF_TOKEN`은 절대 코드/문서/커밋에 평문으로 남기지 않습니다.
- `members/<github_id>/.env` 파일 권한은 최소 `600`으로 유지합니다.

```bash
chmod 600 members/<github_id>/.env
```

- shared 서버에서는 토큰을 개인 계정별로 분리하고, 필요 시 즉시 폐기/재발급합니다.

## 6) 트러블슈팅

- **403/401 에러**: 토큰 권한(read) 확인, gated 모델 접근 승인 여부 확인
- **매번 재다운로드**: `HF_HOME`/`HF_HUB_CACHE` 경로가 매 실행마다 바뀌는지 확인
- **다운로드 느림**: `HF_XET_HIGH_PERFORMANCE=1` 적용 여부, 캐시 경로 디스크 성능 확인
