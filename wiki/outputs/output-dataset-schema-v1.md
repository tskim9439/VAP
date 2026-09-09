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

`experiments/ds_cards.py --all` 로 기존 manifest 전부에 카드를 생성했다(행 수정 없음, 전 행·전 레코드 검증). 로그: `raw/sources/experiments/2026-09-08-hf-trainer-migration/ds-cards.log`.

| manifest | 행 | 시간 | 행 오류 | 정렬 고유 스트림 / 빈 / 중복 줄 | 정렬 오류 |
|---|---|---|---|---|---|
| librispeech-960 | 126,513 | 1,033 h | 0 | 126,513 / 0 / 13,169 | 0 |
| kspon-full | 619,932 | 1,189 h | 0 | 619,932 / 40 / 302,491 | 0 |
| swbd-train | 179,427 | 290 h | 0 | 179,463 / 1 / 791 | 0 |
| nikl-1000 | 1,249,471 | 1,451 h | 0 | 1,249,471 / 2,256 / 1,053,631 | 0 |
| mnsc-1000 (2026-09-09 재빌드) | 74,215 | 135 h(발화 108 h) | 0 | 74,215 / **68,047 빈 레코드(오디오 없음)** / 0 · 정렬된 발화 6,168 | 0 |
| librispeech-dev / test | 7,037 / 7,042 | 24.5 / 24.9 h | 0 | 7,037 / 7,042 (레거시 파일) | 0 |
| kspon-dev / eval | 2,545 / 5,687 | 4.9 / 8.1 h | 0 | 2,545 / 5,687 (레거시 파일) | 0 |
| librispeech-100, kspon-100 (동결 전 파일럿) | 13,182 / 62,000 | 108 / 119 h | `lexical_text` 없음(예상) | – | – |

**MNSC 판정(2026-09-09)**: 재빌드한 manifest 의 오디오 경로 400 개 표본 중 39 개만 존재(서버의 ASR-PART1-Train wav 는 CSV 가 가리키는 파일의 약 10 % 뿐) → 정렬 job 66524 는 74,215 중 6,168 발화만 정렬(나머지 audio_missing). 정렬된 발화도 offset 오차 중앙값 530–570 ms 로 다른 코퍼스보다 커 전사–오디오 대응이 의심된다. **MNSC 는 학습에서 제외**(D2 TRAIN 에 넣지 않음).

**id 중복 검사 추가(2026-09-09)**: 초기 mnsc-1000 manifest 는 676,864 행 중 고유 id 가 77,095 개(CSV 가 같은 wav 를 약 30 번 반복)였는데 v1 검증기가 이를 놓쳤다. 검증기에 id 중복 항목을 넣고 카드에 `unique_ids/duplicate_ids` 를 기록하며, manifest 를 중복 제거로 재빌드했다(74,215 발화 108 h). 정렬 "중복 줄" 은 정렬 job 재시작·양방향 워커가 같은 스트림을 다시 쓴 흔적이다. 로더는 첫 레코드만 쓰므로 학습에 영향은 없고 디스크만 낭비한다(정리 스크립트는 사용자 승인 후).
주의: `align-asr-tn-v1/kspon-full/stats-all.json` 은 requeue 루프 prep(65774)이 0 으로 덮어썼다. 카드는 parts 를 직접 세므로 영향 없다.

## 5b. 신규 manifest (2026-09-09, asr-tn-v1.2.0 / v1.3.0)

빌드 직후 `dataset.json` 카드가 자동 생성·검증된다(오류 0). 정렬(`slurm/s2_prep.sbatch`)은 아직 제출 전.

| manifest | 발화 | 발화 시간 / 스트림 시간 | quarantine | 비고 |
|---|---|---|---|---|
| voxpopuli-train | 177,422 | 520.5 h | – | EN, `train_part_k.tar::member`(36 tar) |
| yodas-en129 | 133,853 | 333.8 h | – | EN Granary-YODAS, num2words 숫자 정규화 |
| aihub-bc-train | 448,867 | 578.2 h / 740.2 h | 3,132 | KO 031+033 방송 원음(zip 멤버, stem 중복 제거). 첫 빌드(385,647 / 473 h)는 간투사 표지 `/` 를 quarantine 했던 것 → `/` 만 떼도록 고쳐 재빌드 |
| aihub71631-train / dev | 231,647 / 66,365 | 159.4 h / 44.4 h(스트림 243 / 68 h) | 6,884 / 2,087 | KO 자유대화 stereo, 화자=채널(에너지 VAD 로 채널 판정), `path#chN` + `src_offset_s`. 라벨 8,306 대화 중 서버 보유 wav 는 TS_01.실내_5 757 · VS_02.실외 186 |

aihub71631 quarantine 은 대부분 0.3 s 미만 맞장구(`bad_time` 4,062: '네', '예')와 익명화 `#@이름#`(`anon` 2,542)이다. 채널이 라벨과 뒤바뀐 대화 train 26 / dev 3(에너지 VAD 겹침으로 판정해 바로잡음). 단일 스레드 빌드가 45 분간 무출력이라 리더를 프로세스 풀로 병렬화(757 대화 11 분).

aihub-bc-train 장르 분포(발화): 예능오락 120,279 · 연예공연 94,500 · 교양 77,867 · 인터뷰 62,140 · 다큐 61,311 · 영화드라마 35,902. 길이 p50 5.4 s.
제외: MNSC(오디오 부재, §5), AI Hub 98(원천 m4a zip 15 GB 만 있고 라벨 없음), NIKL 추가분(오디오 없음).

## 6. 확장 지점

- 2-화자/VAP: 같은 id 에 `vap/<version>/` 층을 추가(턴·끼어들기 시각). segments 에 `speaker` 는 이미 있다.
- display 텍스트: segments 의 `display_text` 필드와 카드의 `text_fields` 로 선언(learn 여부는 config).
- HF `datasets`: `load_dataset("json", data_files=streams.jsonl)` 로 그대로 열린다. 60 만 행 이상은 parquet 캐시(예정).
