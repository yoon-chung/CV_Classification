# S3 Artifact Storage Setup

모델 체크포인트와 대규모 실험 산출물을 안전하게 보관하기 위해 AWS S3를 사용합니다.

## 1) S3 버킷 생성

1. [AWS S3 콘솔](https://s3.console.aws.amazon.com/)에 접속합니다.
2. **버킷 만들기(Create bucket)**를 클릭합니다.
3. **버킷 이름**을 입력합니다(예: `cv-team-1`).
4. **리전**을 선택합니다(대회 서버와 가까운 `ap-northeast-2` 권장).
5. 나머지 설정은 기본값을 유지하고 버킷을 생성합니다.

## 2) IAM 사용자 및 권한 설정

실험 코드에서 S3에 접근하기 위한 전용 사용자를 생성합니다.

1. [IAM 콘솔](https://console.aws.amazon.com/iam/)에서 **사용자 추가**를 클릭합니다.
2. 사용자 이름을 입력하고 **직접 정책 연결**을 선택합니다.
3. 아래의 **최소 권한 정책**을 JSON 탭에 붙여넣어 생성하고 연결합니다.
4. 사용자 생성을 완료한 후 **액세스 키(Access Key ID)**와 **비밀 액세스 키(Secret Access Key)**를 안전하게 기록합니다.

### 최소 권한 정책 (JSON)

`<your-bucket-name>`을 실제 버킷 이름으로 교체합니다.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::<your-bucket-name>",
                "arn:aws:s3:::<your-bucket-name>/*"
            ]
        }
    ]
}
```

## 3) 환경 변수 설정

S3 관련 변수는 `S3_*`를 단일 소스로 사용합니다.
(`scenario=cloud` 설정에서도 `S3_BUCKET`, `S3_PREFIX`를 그대로 참조합니다.)

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `AWS_ACCESS_KEY_ID` | (필수) | AWS 액세스 키 ID |
| `AWS_SECRET_ACCESS_KEY` | (필수) | AWS 비밀 액세스 키 |
| `AWS_REGION` | `ap-northeast-2` | S3 버킷이 위치한 리전 |
| `S3_BUCKET` | `cv-team-1` | 산출물을 저장할 버킷 이름 |
| `S3_PREFIX` | `cvdc` | 버킷 내 저장 경로 접두사 |
| `S3_DEDUP_BY_HASH` | `1` | SHA256 기반 content-addressable key dedup 활성화 |
| `S3_DRY_RUN` | `0` | `1`이면 실제 업로드 없이 로컬 staging(`exports/s3`)만 수행 |
| `S3_UPLOAD_WHITELIST` | (비어있음) | 업로드 허용 패턴(쉼표 구분). 비어있으면 기본 whitelist 사용 |

## 4) 저장 경로 규칙 (Prefix)

저장소 내에서 산출물은 기본적으로 아래 구조(CAS)로 저장됩니다.
`{S3_PREFIX}/cas/sha256/{sha256}/{filename}`

`S3_DEDUP_BY_HASH=0`으로 비활성화하면 기존 run-id 경로를 사용합니다.
`{S3_PREFIX}/profile={profile}/date={YYYY-MM-DD}/run_id={run_id}/{relative_path}`

- **추천 Prefix**: `cvdc/{repo_name}/{profile}/` 형식을 권장합니다.

## 5) 수명 주기 및 비용 관리

S3 비용을 절감하기 위해 아래 설정을 권장합니다.

- **오래된 체크포인트 삭제**: 버킷 설정의 **수명 주기 규칙**에서 30일 이상 지난 객체를 자동으로 삭제하거나 Glacier로 이동하도록 설정합니다.
- **불완전한 멀티파트 업로드 중단**: 업로드 중 실패한 파일 조각이 용량을 차지하지 않도록 7일 후 삭제 규칙을 추가합니다.

## 6) 비활성화 동작

- `S3_BUCKET` 환경 변수가 설정되지 않으면 S3 업로드 로직은 **동작하지 않는다**.
- 이 경우 산출물은 로컬의 `exports/s3/` 디렉토리에만 저장됩니다.

## 7) 저장 대상 정책 (현재 구현)

- upload stage는 이전 stage contract의 `outputs`에 등록된 파일만 수집합니다.
- 따라서 `logs/events.jsonl`, `meta.json`, `contracts/*.json`, `.hydra/`는 기본적으로 S3 업로드 대상이 아니다.
- 민감/대용량 보호를 위해 아래 항목은 업로드에서 제외됩니다.
  - `data/**`
  - 이미지 파일(`.jpg`, `.jpeg`, `.png`)
  - `.env*`
- S3가 비활성 또는 dry-run이면 `exports/s3/` 하위에 로컬 staging 결과와 manifest만 남는다.
- dedup이 활성화되면 동일 파일은 `uploaded`가 아니라 `reused`로 manifest에 기록됩니다.
