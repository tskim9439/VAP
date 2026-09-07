---
type: source
status: active
created: 2026-09-07
updated: 2026-09-07
summary: asr-tn-v1.0.0 전체 transcript audit — LibriSpeech 960 h 잔존 digit 0, KsponSpeech 620k 중 68 quarantine, 230 h diff 19 건 전부 설명
raw_path: raw/sources/experiments/2026-09-07-asr-tn-v1-audit/README.md
observed: 2026-09-07
raw_authors:
  - tskim
---

# asr-tn-v1.0.0 전체 transcript audit

## 무엇인가

[[output-asr-tn-v1-spec]] 의 동결 관문을 채우기 위해 `experiments/tn_audit.py` 로 LibriSpeech 7 split 전체와
KsponSpeech 01–05·dev·eval 전체 transcript(920,912 행)를 정규화·검사한 결과다. 오디오는 읽지 않는다.

## 핵심 수치

| 코퍼스 | 행 | quarantine | 사유 | 비고 |
|---|---:|---:|---|---|
| LibriSpeech train 960 h + dev/test | 292,367 | 0 | — | 잔존 digit 0, collision 0 |
| KsponSpeech train 01–05 | 620,000 | 68 | 이중표기 밖 digit | malformed 0, empty 0 |
| KsponSpeech dev / eval_clean / eval_other | 2,545 / 3,000 / 3,000 | 0 / 1 / 0 | digit | 평가 세트는 제외하지 않음 |

- 이중표기 선택: numeric 87,616, latin 1,091, plain 12. standalone Latin 행 7,080 (1.14%), 상위 TV·PC·A·pc·B·LG·SNS — 대소문자 혼재는 규약대로 원형 유지.
- idempotence 위반 0, tokenizer unk 0 (전 partition).
- 동결 전 230 h manifest 와의 diff: LibriSpeech-100 14/28,539 (단어 끝 `'` 제거), KsponSpeech-100 5/62,000 quarantine, 문자열 변경 0.
- fingerprint: `textnorm_sha256 62e11231a2c4…`, `tokenizer_json_sha256 66915f2a66d1…`, num2words 0.5.14.

## 한계

- 관문 6(목적 표본 사람 검토)은 TSV 5 개(각 40 행)로 준비만 됐고 아직 검토되지 않았다.
- 시간(h)은 nominal 값이며 오디오 길이를 재지 않았다. target collision 은 표지·구두점만 다른 원문에서 나오는 정상 현상이라 원인별 분류는 하지 않았다.
