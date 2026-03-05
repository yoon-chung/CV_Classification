# Slack Notifications

이 저장소는 Slack 알림으로 **단계별 하위 작업 시작/완료/실패**를 전송합니다.

- Bot Token 경로(`SLACK_BOT_TOKEN` + `SLACK_CHANNEL_ID`)가 있으면 채널 메시지로 전송
- Bot API 전송 실패 시 Webhook이 있으면 채널 메시지로 자동 폴백
- Bot Token이 없고 Webhook만 있어도 채널 메시지로 전송

## 1) Webhook 만들기

1. Slack 워크스페이스에서 Incoming Webhooks 기능을 활성화합니다.
   - 참고: https://api.slack.com/incoming-webhooks
2. Webhook을 추가하고, 메시지를 받을 채널을 선택합니다.
3. 발급된 Webhook URL을 안전하게 보관합니다.

### 비공개(Secret/Private) 채널 주의

- Webhook을 비공개 채널로 보내려면, Webhook을 생성한 앱(또는 연결된 Slack App)이 해당 채널에 초대되어 있어야 합니다.
- 팀 공용 알림은 별도 비공개 채널을 만들고 해당 채널로만 보내는 방식을 권장합니다.

## 2) 환경 변수

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `SLACK_NOTIFY` | `0` | `1`일 때만 Slack 알림 활성화 (`0`이면 완전 no-op) |
| `SLACK_WEBHOOK_URL` | (없음) | Incoming Webhook URL (절대 커밋 금지) |
| `SLACK_BOT_TOKEN` | (없음) | Bot 토큰(`chat.postMessage`/`files.upload` 사용) |
| `SLACK_CHANNEL_ID` | (없음) | Bot 전송 대상 채널 ID |

## 3) 동작 개요

- `SLACK_NOTIFY=0` 또는 (`SLACK_WEBHOOK_URL` 미설정 && Bot 변수 미설정) 시: Slack 관련 로직은 **아무것도 하지 않습니다**(에러/파일 생성 없음).
- 큐 시작: 큐당 1회 한국어 요약(`탐색 파이프라인 | 조합 전수 탐색 큐 시작`) 전송
- 하위 작업 시작: 각 하위 작업마다 한국어 요약(`탐색 파이프라인 | 조합 전수 탐색 하위 작업 시작`) 전송
- 하위 작업 실패: 예외 발생 시 한국어 오류 요약(`탐색 파이프라인 | 조합 전수 탐색 오류`) 전송
- 하위 작업 완료: 각 하위 작업 종료 시 한국어 완료 요약(`탐색 파이프라인 | 조합 전수 탐색 하위 작업 완료`) 전송

알림 메시지에 포함되는 핵심 정보:

- 큐 ID, 프로필(`explore`/`tune`)
- 단계 진행(`현재 번호/전체 개수`)
- 현재 best run과 Macro F1
- 현재 탐색 조합 요약(`config_summary`)
- W&B 링크(있으면)

## 4) 실행 예시

```bash
export SLACK_NOTIFY=1
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

./scripts/run_explore.sh --dummy-data
```

## 5) 개인정보/보안 가드레일

- `SLACK_WEBHOOK_URL`은 **로그/파일/W&B/커밋에 남기지 않습니다**.
- 메시지에는 **hostname, 하드웨어 스펙, 사용자 홈 경로** 같은 환경 정보가 포함되지 않도록 합니다.
  - 현재 구현은 큐 식별자에 대해 출력 경로의 tail 일부만 사용하며, 하드웨어 정보는 전송하지 않습니다.

## 6) Bot Token 사용 시 장점

Incoming Webhook 대비 Bot Token을 쓰면:
- 채널 선택/권한/메시지 포맷을 더 세밀하게 제어 가능
- 최종 리포트 전송 시 파일 업로드(`files.upload`) 경로 사용 가능

참고: https://docs.slack.dev/reference/methods/chat.postMessage/
