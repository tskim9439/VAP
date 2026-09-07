---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-12-07
created: 2026-09-07
updated: 2026-09-07
summary: 텍스트 정규화 규약 asr-tn-v1.0.0 을 동결 — 구현·golden/tokenizer-ID test·전체 audit·230 h diff 완료, 이후 manifest·정렬은 fingerprint 로 규약을 고정
sources:
  - [[output-asr-tn-v1-spec]]
  - [[source-asr-tn-v1-audit]]
  - [[asr-text-normalization]]
---

# 결정: `asr-tn-v1.0.0` 동결

## 맥락

Stage 2 데이터 확장(LibriSpeech 960 h + KsponSpeech 970 h, 정렬 재생성)을 앞두고 학습 타깃·채점 규약을 먼저 고정해야 한다
([[output-stage1-mono-pilot]] 의 권장, 사용자 지시 2026-09-07). 규범 문서 [[output-asr-tn-v1-spec]] 은 동결 후보였고,
구현·전체 transcript audit·기존 230 h diff 가 남아 있었다.

## 검토한 선택지

| 선택지 | 장점 | 단점 |
|---|---|---|
| A. 기존 구현(2026-09-05 판) 그대로 확장 | 즉시 진행 | num2words silent fallback, LibriSpeech digit 를 추측 변환, 부분 변환 허용, fingerprint 없음 — 환경·시점에 따라 타깃이 달라질 수 있음 |
| B. spec 대로 구현·audit 후 동결 (채택) | 재현성·quarantine·버전 관리 | 반나절 소요, 정렬 루트 새로 생성 |
| C. display view(대소문자·구두점)까지 포함해 동결 | 최종 출력 요구 반영 | pseudo-label 출처 정책 미확정 → v1 범위 밖으로 미룸 |

## 결정

`vapasr/data/textnorm.py` 의 `asr-tn-v1.0.0` 을 동결한다. 이후 모든 manifest 는 fingerprint 를 기록하고, 정렬기·학습기는 fingerprint 가
없거나 다르면 시작 전에 실패한다(동결 전 산출물 재현은 `VAPASR_ALLOW_LEGACY_TN=1` 로만). 정렬 루트는 `align-asr-tn-v1/`.

## 근거

- 관문 8 개 중 7 개 통과([[source-asr-tn-v1-audit]]): 의존성 즉시 실패, golden + tokenizer-ID snapshot 8/8, LibriSpeech 잔존 digit 0,
  Kspon quarantine 68/620,000(전부 digit, malformed 0), empty·idempotence 0, 230 h diff 19 건 전부 설명, fingerprint 8 필드.
- 관문 6(목적 표본 사람 검토)은 TSV 로 준비됐고 사용자 검토가 남았다. 검토에서 규칙 변경이 필요하면 v1.1.0 으로 올리고 manifest·정렬을 새 경로에 만든다.

## 결과 / 파급

- 강제: 새 코퍼스는 corpus parser + audit 를 거쳐 minor 버전으로 편입. 타깃 문자열·token ID 가 바뀌는 변경은 새 정렬 필수.
- 배제: EN 숫자 추측 변환(연도·시각·전화·분수·소수 통화·단위 결합), KO 독립 Latin 의 한글 변환·대소문자 통일(TV/tv 혼재 1.14 % 는 원형 유지).
- 기존 run A/B(230 h, align2)는 동결 전 타깃이며 EN 14 건·KO 5 건만 다르다. 재현 시 legacy 플래그.
- 영향 받는 페이지: [[asr-text-normalization]], [[output-asr-tn-v1-spec]], [[task-uslm-u1-interleaved-asr]]

## 재검토

`review` 날짜에 유효성을 다시 확인한다. 뒤집히면 이 페이지를 `decision_status: superseded` 로 바꾸고 새 결정 페이지에서 링크한다.
