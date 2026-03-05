# GitHub Fine-Grained PAT Setup (Read-Only Clone)

본 문서는 대회 서버에서 이 저장소를 안전하게 clone하기 위한 GitHub fine-grained personal access token(PAT) 생성/사용 방법을 정리합니다.

목표는 "읽기 전용 최소 권한"으로 private 저장소 clone만 가능하게 구성하는 것입니다.

## 1) 토큰 생성

1. GitHub Settings -> Developer settings -> Personal access tokens -> Fine-grained tokens로 이동합니다.
2. **Generate new token**을 선택합니다.
3. Expiration(만료일)을 설정합니다. (권장: 30~90일)
4. Resource owner를 올바른 사용자/조직으로 선택합니다.
5. Repository access는 **Only select repositories**를 선택하고, 필요한 저장소만 지정합니다.

## 2) 최소 권한 설정

아래 권한만 부여합니다.

| 범주 | 권한 |
| --- | --- |
| Repository permissions -> Contents | Read-only |
| Repository permissions -> Metadata | Read-only |

주의:

- 조직 정책에 따라 토큰이 owner 승인 전까지 비활성일 수 있습니다.
- 조직이 SSO/SAML/IP allow list를 강제하면 서버 IP와 SSO 승인 상태를 함께 확인해야 합니다.

## 3) 서버에서 clone하기

### 방법 A: 대화형 clone (권장)

```bash
git clone https://github.com/<owner>/<repo>.git
```

프롬프트가 나오면:

- Username: GitHub 사용자명
- Password: 발급한 fine-grained PAT

### 방법 B: 비대화형(자동화 스크립트)

```bash
export GH_READONLY_TOKEN='github_pat_xxx'
git clone "https://oauth2:${GH_READONLY_TOKEN}@github.com/<owner>/<repo>.git"
```

주의:

- URL에 토큰이 직접 노출되므로 로그/히스토리 노출에 유의합니다.
- 운영 스크립트에서는 secret manager 또는 서버 전용 `.env` 사용을 권장합니다.

## 4) GitHub CLI 사용 (선택)

```bash
export GH_TOKEN='github_pat_xxx'
gh auth status
gh repo clone <owner>/<repo>
```

## 5) 보안 운영 가드레일

- 토큰은 절대 Git에 커밋하지 않습니다.
- 서버 `.env` 파일은 최소 권한(`chmod 600`)으로 관리합니다.
- 토큰은 저장소 단위로 분리하고, 만료 주기를 짧게 유지합니다.
- 서버 작업 종료 후 불필요한 장기 토큰은 폐기/재발급합니다.

## 6) 문제 해결

- **403 발생**: 권한 부족, 조직 승인 미완료, SSO 미승인, IP allow list 미등록 가능성 확인
- **401 발생**: 토큰 만료/오타 여부 확인
- **repo not found**: Resource owner/선택 저장소 범위가 올바른지 확인
