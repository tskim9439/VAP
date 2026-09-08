---
type: output
status: active
created: 2026-09-08
updated: 2026-09-08
summary: VAP-ASR 학습 데이터 스키마 v1 — manifest(streams.jsonl)·데이터 카드(dataset.json)·정렬 레코드·정렬 카드(align.json)의 필드·규칙·검증기·로더 관문 정의
sources:
  - [[output-asr-tn-v1-spec]]
  - [[output-vapasr-model-and-sequence]]
  - [[decision-asr-tn-v1-freeze]]
---

# 데이터 스키마 v1 (`vapasr-ds-1.0` / `vapasr-align-1.0`)

정본 코드: `vapasr/data/schema.py`(검증기·카드 생성·로더 관문), `experiments/ds_cards.py`(기존 산출물 카드 생성), JSON Schema 문서 `vapasr/data/schemas/*.json`.
원칙: **행은 그대로, 버전은 카드에.** 기존 산출물의 행을 다시 쓰지 않고 디렉토리마다 카드 파일만 추가한다. 파생 캐시(`_items-*.json.gz`)는 스키마 밖(언제든 재생성).

## 1. 계층

```text
<manifest>/                     예: /soundai/users/tskim/VAPKT-data/data/manifests/kspon-full
  dataset.json                  데이터 카드 (schema_version, 행 수·시간·subset·mode·언어 분포, textnorm fingerprint, quarantine 수, 빌드 커밋·시각, 검증 결과)
  streams.jsonl                 정본: 스트림(= 학습 샘플) 1 행
  quarantine.jsonl              제외된 발화와 사유
  stats.json                    빌더 통계(하위 호환)
<align-root>/<manifest>/        예: …/manifests/align-asr-tn-v1/kspon-full  (규약이 바뀌면 새 루트)
  align.json                    정렬 카드 (schema_version, textnorm fingerprint, tokenizer sha, 정렬 모델, 레코드 수·빈 레코드·중복, 검증 결과)
  fingerprint.json              textnorm·tokenizer·manifest 해시 (관문)
  parts/*.jsonl                 정렬 레코드(추가 전용, 청크당 1 파일)
  stats-<mode>.json
```

## 2. manifest 행 (`streams.jsonl`)

| 필드 | 타입 | 규칙 |
|---|---|---|
| id | str | 전역 유일. 정렬·캐시·평가가 이 키로 조인 |
| corpus / split / subset | str | split ∈ {train, dev, test, eval} |
| mode | enum | `stream`(발화 연결 스트림 또는 발화=스트림) / `utt`(단일 발화 평가) |
| lang | enum | `English` / `Korean` (prefix `language …` 에 그대로 사용) |
| speaker / chapter | str·null | 출처 메타 |
| duration_s | number > 0 | 스트림 길이(무음 포함). K = round(duration_s × 12.5) |
| n_utts | int ≥ 1 | segments 수 |
| silence_after_s | number ≥ 0 | 끝 무음 |
| segments[] | ≥ 1 | 아래 |
| ↳ utt_id / path | str | path 는 파일 경로 또는 `archive.tar::member` |
| ↳ offset_s / dur_s / silence_before_s | number | **스트림 시작 기준**, 겹침 없음, 마지막 끝 ≤ duration_s |
| ↳ lexical_text | str | asr-tn 규약 학습·정렬 타깃([[output-asr-tn-v1-spec]]) |
| ↳ raw_text / text / display_source | str | 원문 / lexical 과 동일(하위 호환) / display 출처(`none` 등) |

## 3. 정렬 레코드 (`parts/*.jsonl`)

```json
{"id": "<manifest 행 id>", "utts": [{"speaker": 0, "start": 0.40, "end": 3.35, "text": "…", "tokens": [{"id": 130271, "text": "바", "end_time": 0.964}, …]}]}
```
- `tokens[].id` 는 **모델 tokenizer 의 토큰 id**(Qwen3-ASR), `end_time` 은 스트림 기준 초, 발화 안에서 단조 비감소. 청크 배정은 `⌊end_time/0.08⌋ + δ`.
- `utts: []` 는 "정렬 대상 없음(빈 스트림) 또는 오디오 누락" 레코드 — 완료 판정을 위해 남긴다.
- 같은 id 가 두 줄이면 첫 줄만 유효(카드가 duplicate_lines 로 집계).

## 4. 관문과 검증

- 빌더(`s1_build_manifest.py`) 는 끝에 `dataset.json` 을 쓰며 전 행을 검증한다. 정렬기는 `--card` 로(후처리 1 회) `align.json` 을 쓴다. `slurm/s2_prep.sbatch` 는 정렬 완료 후 `ds_cards.py` 를 돌린다.
- 로더(`MonoStreamDataset`) 는 시작 시 카드를 읽어 **schema major** 와 **정렬 tokenizer sha ≠ 현재 tokenizer** 를 검사한다. 기본은 경고, `VAPASR_STRICT_SCHEMA=1` 이면 실패. textnorm fingerprint 관문은 기존대로 별도 동작.
- 검증 항목: 필수 필드·타입·열거값, 길이/오프셋 부호, 세그먼트 겹침·초과, 토큰 시각 단조성, JSON 파손, id 중복.

## 5. 이관 결과 (2026-09-08)

`experiments/ds_cards.py --all` 로 기존 manifest 전부에 카드를 생성했다(행 수정 없음). 결과 요약은 `raw/sources/experiments/2026-09-08-hf-trainer-migration/ds-cards.log` 참고.
주의: `align-asr-tn-v1/kspon-full/stats-all.json` 은 requeue 루프 prep(65774)이 0 으로 덮어썼다. 카드는 parts 를 직접 세므로 영향 없다.

## 6. 확장 지점

- 2-화자/VAP: 같은 id 에 `vap/<version>/` 층을 추가(턴·끼어들기 시각). segments 에 `speaker` 는 이미 있다.
- display 텍스트: segments 의 `display_text` 필드와 카드의 `text_fields` 로 선언(learn 여부는 config).
- HF `datasets`: `load_dataset("json", data_files=streams.jsonl)` 로 그대로 열린다. 60 만 행 이상은 parquet 캐시(예정).
