# asr-tn-v1.0.0 전체 transcript audit (2026-09-07)

도구: `experiments/tn_audit.py` (mxc 컨테이너, 225 s). 대상: LibriSpeech 7 split 전체 `*.trans.txt`, KsponSpeech `train.trn`(01–05 partition)·`dev.trn`·`eval_clean.trn`·`eval_other.trn`.
오디오는 읽지 않았다(시간은 nominal). 원본 산출물: mxc `$MXC_DATA_LOG_DIR/tn-audit/asr-tn-v1.0.0/` (이 디렉토리에 전부 복사).

## fingerprint (코드 쪽)
```json
{
 "textnorm_version": "asr-tn-v1.0.0",
 "textnorm_sha256": "62e11231a2c4e409c22a92bbe605389afca63ff96bb0a00d3e3ca9c49c1a093e",
 "numeric_backend_version": "num2words 0.5.14",
 "tokenizer_json_sha256": "66915f2a66d1e6688e8672cd3df1c74c23c658689fab9798706d62a4a490f447"
}
```

## 결과 요약
| partition | lang | rows | kept | quarantined | reasons | idem-viol | unk | collisions | dual num/lat/plain | Latin rows |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| train-clean-100 | En | 28539 | 28539 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| train-clean-360 | En | 104014 | 104014 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| train-other-500 | En | 148688 | 148688 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| dev-clean | En | 2703 | 2703 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| dev-other | En | 2864 | 2864 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| test-clean | En | 2620 | 2620 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| test-other | En | 2939 | 2939 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| KsponSpeech_01 | Ko | 124000 | 123988 | 12 | {'charset': 12, 'digit': 12} | 0 | 0 | 859 | 17755/234/3 | 1452 (0.0117) |
| KsponSpeech_02 | Ko | 124000 | 123983 | 17 | {'charset': 17, 'digit': 17} | 0 | 0 | 829 | 16937/214/2 | 1364 (0.011) |
| KsponSpeech_03 | Ko | 124000 | 123991 | 9 | {'charset': 9, 'digit': 9} | 0 | 0 | 904 | 17876/209/2 | 1420 (0.0115) |
| KsponSpeech_04 | Ko | 124000 | 123985 | 15 | {'charset': 15, 'digit': 15} | 0 | 0 | 866 | 17584/227/1 | 1459 (0.0118) |
| KsponSpeech_05 | Ko | 124000 | 123985 | 15 | {'charset': 15, 'digit': 15} | 0 | 0 | 831 | 17464/207/4 | 1385 (0.0112) |
| dev | Ko | 2545 | 2545 | 0 | {} | 0 | 0 | 12 | 412/4/- | 26 (0.0102) |
| eval_clean | Ko | 3000 | 2999 | 1 | {'charset': 1, 'digit': 1} | 0 | 0 | 33 | 287/8/293 | 18 (0.006) |
| eval_other | Ko | 3000 | 3000 | 0 | {} | 0 | 0 | 1 | 625/30/497 | 15 (0.005) |

230 h diff: {'librispeech-100': {'segments': 28539, 'same': 28525, 'text_changed': 14}, 'kspon-100': {'segments': 62000, 'same': 61995, 'now_quarantined': 5}}


### 합계
- LibriSpeech 7 split 292,367 행: quarantine 0, idempotence 위반 0, tokenizer unk 0, target collision 0. **raw 잔존 digit 0** (관문 3).
- KsponSpeech train 01–05 620,000 행: quarantine 68 (전부 이중표기 밖 digit, malformed 0, empty 0). 이중표기 선택 numeric 87,616 / latin 1,091 / plain 12.
  standalone Latin 행 7,080 (1.14%), 상위 표현 TV 368·PC 357·A 300·pc 225·B 223·LG 200·SNS 184 (`latin-top100.json`). 대소문자 혼재(PC/pc, TV/tv)는 v1 규약대로 원형 유지.
  target collision(서로 다른 원문 → 같은 target) partition 별 01: 859, 02: 829, 03: 904, 04: 866, 05: 831 — 표지·구두점·+/* 만 다른 원문이라 정상.
- dev 2,545 행 quarantine 0 · eval_clean 3,000 행 중 1 행 digit 잔존(평가 세트는 제외하지 않고 채점 불일치로 센다) · eval_other 0.

### 기존 230 h(동결 전 manifest) 와의 diff
- librispeech-100: 28,539 세그먼트 중 14 건 문자열 변경 — 모두 단어 끝/앞의 `'`(인용·생략 부호: `honesty'`, `an'`, `settin'`) 제거. 규약의 "단어 내부 아포스트로피만 유지" 에 따른 것.
- kspon-100: 62,000 중 5 건 quarantine(이중표기 밖 digit), 문자열 변경 0. (첫 실행에서 3 건 띄어쓰기 차이가 났고 원인은 구현의 괄호 안 strip → 수정 후 0.)

## 동결 관문 (spec 'Audit 산출물과 동결 관문')
| # | 관문 | 결과 |
|---|---|---|
| 1 | num2words 미설치 시 즉시 실패 | ✅ `import vapasr.data.textnorm` → ImportError (실측) |
| 2 | golden 문자열 + tokenizer-ID test | ✅ `tests/test_textnorm.py` 8/8, snapshot `tests/golden/asr-tn-v1.0.0.json` |
| 3 | LibriSpeech raw 잔존 digit 0 | ✅ 7 split 0 |
| 4 | Kspon 이중표기 밖 digit·malformed 전부 quarantine, 누락 0 | ✅ 68 (digit) / malformed 0 — `quarantine-*.jsonl` |
| 5 | 빈 target·idempotence 위반 0 | ✅ 0 / 0 |
| 6 | Kspon 01–05 목적 표본 사람 검토 | ⏳ `review-KsponSpeech_0{1..5}.tsv` (각 40 행: numeric dual 12·latin dual 8·standalone Latin 8·quarantine 6·plain 6) — **사용자 검토 필요** |
| 7 | 230 h diff 전부 설명 | ✅ 위 |
| 8 | manifest fingerprint | ✅ `s1_build_manifest.py` 가 stats.json 에 8 필드 기록, 정렬기·학습기가 검사 |
