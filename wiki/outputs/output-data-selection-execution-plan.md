---
type: output
status: active
created: 2026-09-18
updated: 2026-09-19
summary: 15개 DB 선별 실행안. MNSC 제외 426만 행 음향 QC 완료 후 886개 시간축·채널 검수 표본과 3교사 대조 진행.
sources:
  - "[[source-data-selection-plan]]"
  - "[[output-asr-tn-v1-spec]]"
  - "[[output-phase2-d1b-sample-inference]]"
---

# 고품질 데이터 선별: 실행 계획과 결과

최신 후속 실행(2026-09-19): [[output-data-selection-review-v02]]. CPU 전수 QC는
MNSC 제외 14개 DB에서 완료됐고, 현재 시간축·채널 검수 및 886개 3교사 calibration 단계다.
아래 §8.5까지의 실행 상태는 9월 18일 당시 기록이다.

## 1. 결론

원안의 **원전사 + 독립 계열 ASR 두 개 + 음향 QC + 사람 검수**는 타당하다.
다만 이를 단일 점수로 줄여 GOLD/REJECT를 결정하지 않는다. 정확한 전사, 정확한
시각, 정확한 화자 대응, 확실한 턴 종료는 서로 다른 증거가 필요하다.

이번 구현은 기존 학습기를 바꾸지 않는 선별 sidecar이다. **Phase 1/2 공통 학습
직렬화까지 완료했다는 뜻이 아니다.** 원본·기존 manifest·정렬·학습 작업을 보존하고,
새 경로에 전수 집계와 검수 후보를 생성한다. 자동 재전사·라벨 교체·학습 재개는 하지 않는다.

원문: [DataSelectionPlan.md](../../raw/inbox/DataSelectionPlan.md).
실행 버전: `data-selection-v0.1`; TN: 기존 `asr-tn-v1.3.0` 타깃을 보존.

## 2. 원안을 수정한 지점

| 원안/위험 | 실행 규칙 |
|---|---|
| Qwen·Whisper가 같으면 정답 | 합의는 `AGREEMENT_CANDIDATE`일 뿐. 공통 오류·숫자 표기·유창화·맞장구 누락을 검수 |
| 정렬이 잘 되면 전사가 맞음 | 정렬은 시각 품질 보조 증거. 전사의 독립 심판으로 사용하지 않음 |
| overlap·짧은 음성·무음 비율로 제거 | 별도 표본층과 보존량을 보고. 품질 결함과 대화 현상 구분 |
| teacher와 다르면 나쁜 데이터 | 어려운 음향·방언·고유명사·disfluency일 수 있으므로 검수. 불일치만으로 폐기 금지 |
| v3 또는 v2가 항상 우수 | 초기 독립 교사는 v2를 잠정 채택. train 표본에서 v3 대조 후 고정; heldout 예제로 선택 최적화 금지 |
| 숫자 의미를 일괄 통일 | 기존 TN 동결 유지. 문맥 의존 숫자·약어는 검수; 비교 결과를 학습 타깃에 역반영하지 않음 |
| 바로 Parquet/새 저장소로 전환 | 기존 JSONL·Dialogue와 호환되는 JSONL sidecar. 원본 저장 구조 변경 불필요 |

Qwen 공식 forced aligner는 결과로 `text/start_time/end_time`을 돌려주며,
`fix_timestamp`에서 시각을 단조롭게 보정한다. 따라서 **보정된 시각의 단조성 자체를
전사 정확도 증명으로 쓸 수 없다**. 토큰 posterior 등 존재가 확인되지 않은 점수를
confidence로 가정하지 않는다.
출처: [공식 forced aligner 구현](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py)
(확인 2026-09-18).

## 3. DB별 범위와 주의점

이번 전수 감사의 모집단은 **현재 서버에 있는 기존 unrefined manifest**다.
이미 이전 빌더가 제외한 데이터는 포함되지 않는다. 특히 NIKL의 기존 overlap 제거분,
TN quarantine, 제공되지 않은 원음은 원본 DB를 다시 읽는 별도 회수 작업이 필요하다.
`*-1000` 등의 디렉터리명을 실제 시간으로 해석하지 않는다.

| DB / 입력 | 먼저 확인할 문제 | 공통 포맷에서의 역할 |
|---|---|---|
| LibriSpeech 960 | 고유명사·장문·발화 잘림; 100 h 중복 제외 | mono ASR; chapter 연결은 합성 시간축이라고 표시 |
| KsponSpeech full | 이중표기·독립 Latin·짧은 맞장구·PCM 형식 | mono ASR; 파일 간 화자 동일성/대화 복원 금지 |
| Switchboard | yeah/uh-huh 등 짧은 반응·축약·전화 음질 | 우선 발화 ASR; 원대화 복원 전 턴 정답 없음 |
| MNSC | 현재 CSV의 동일 경로↔다른 전사 충돌·대량 경로 결손 | **source mapping 복구 전 전체 보류** (§8.3) |
| NIKL 2021–2025 | 익명화·원전사와 표준형·잘린 PCM | 기존 mono subset부터; 원대화 복원은 별도 |
| VoxPopuli | archive 멤버·긴 문장·이름·시각 대응 | mono ASR |
| YODAS en129 | 기존 의사라벨의 교사 상관·숫자·자막 오프셋 | 엄격 검수/낮은 신뢰 트랙, 자동 GOLD 금지 |
| AI Hub 방송 031/033 | 번역문 혼입·발화형/표기형·음악·archive | 원음과 한국어 전사의 실제 대응 우선 |
| AI Hub 71631 | stereo 채널 대응·누설·겹침·끝 잘림 | 원대화 시간축 유지; P1 사본은 통합 후보에서 제외 |
| AI Hub 134-1 | 실제 crop 결손·71631과 세션 중복 | 정확한 crop index만 허용, 이웃 crop 대체 금지 |
| AI Hub 134-2 | crop 결손·청소년 발화/짧은 반응 | 같은 규칙, 연령/발화 특성으로 자동 제외 금지 |
| otoSpeech | 배우/session split·주석/무음/overlap | 분리 채널 대화 |
| AMI | headset 누설·고유명사·주석 경계·회의 split | overlap과 다화자 검증을 위한 보존 트랙 |
| NOTSOFAR | close-talk 대응·회의 split·겹침 | 원대화·단어 시각을 독립적으로 검증 |
| ICSI | channel/participant 매핑·SPH/FLAC·긴 회의 | 같은 기준으로 검증, corpus-specific timestamp 오차 확인 |

`nikl2020`, `chime6`, `turnbench`, 모든 dev/test/eval은 학습 선별에서 제외한다.
P2는 `phase2/splits/v1.json`을 먼저 적용한다. 71631↔134-1의 동일 stem은
holdout을 전파하고 학습 후보 중복도 방지한다. 이는 **known session/locator dedup**이며
서로 다른 인코딩·파일명의 음성 중복까지 검사한 content dedup이 아니다.
P1 기존 split은 존중하지만, 신규 통합셋 승인 전 speaker/session 및 waveform 중복
감사를 추가해야 한다. 기존 모델이 이미 학습한 corpus를 새 unseen benchmark라 부르지 않는다.

## 4. 판정과 공통 데이터 계약

각 행은 `key, source, manifest_sha256, parent_id, utt_id, raw_text, text,
audio(path/offset/duration), speaker/scope, start/end, timeline, overlap,
selection_state, reasons`를 가진다. **원문·현재 lexical target은 수정하지 않는다.**

품질 축은 `text_quality / timing_quality / speaker_quality / turn_quality`로 분리한다.
현재 이중 교사 표본 점검은 첫 축의 후보 증거만 만든다. 나머지 축은 `unverified`.
교사별 모델 가중치·설정 SHA256, TN·실행 코드·입력 SHA256, 실제 16 kHz 파형 SHA256을
기록한다. 입력/코드가 달라지면 같은 출력으로 재개할 수 없다. CPU metadata 감사는
오디오 디코딩이 아니므로 파일 존재/음질 검사 완료로 오해하지 않는다.

| 상태 | 의미와 처리 |
|---|---|
| `HELDOUT` | 사전 split에 의해 제외; 선별 임계값 맞추기에 쓰지 않음 |
| `DUPLICATE_LOCATOR_OR_SESSION` | 현재 확인 가능한 경로/세션 중복; 통합 후보에서는 한 번만 사용 |
| `QUARANTINE_METADATA` | 원본 flag·빈 타깃·시간축·crop 대응 문제. 원전사 복구 여부를 별도 검토 |
| `PENDING_AUDIO_TEACHERS` / `PENDING` | 필수 증거 미완료/교사 실패. 통과로 계산하지 않음 |
| `QUARANTINE` | 표본 디코딩 실패·파형 불일치 등. 영구 삭제가 아니라 사유별 복구 대기 |
| `AGREEMENT_CANDIDATE` | 두 교사와 원전사가 잠정 기준에 동의. GOLD 아님 |
| `CORRECTION_REVIEW` | 교사끼리는 동의하지만 원전사와 차이. 원문 보존, 자동 교체 금지 |
| `REVIEW` | 혼합 불일치·표기·음향·길이 문제. 어려운 좋은 데이터도 여기에 포함됨 |

잠정 규칙: EN은 단어, KO는 공백 제거 문자. 각 teacher 대 원전사 오류는 참조 길이가
분모이고, teacher 간 오류는 두 출력 중 긴 길이가 분모인 대칭 지표다. 100% 초과
오류도 자르지 않는다. 세 값 모두 ≤5%이면 합의 후보; 참조가 4단위 이하면 완전 일치가
필요하다. teacher 간 ≤5%이며 둘 다 원전사 대비 ≥15%이면 교정 검수 후보다.
빈 교사 출력·디코딩 실패·미확인 confidence를 합의로 승격하지 않는다.

음향 QC는 실제 길이/채널 수/샘플레이트/파형 hash/RMS/clipping을 기록한다.
없는 채널을 0번으로 바꾸거나 잘린 crop을 묵인하지 않는다. 무음 여부는 전체 0 파형 등
명확한 기술 이상만 격리하고, 낮은 에너지·clipping 경고만으로 영구 폐기하지 않는다.
VAD·SNR·화자 누설·정렬 정확도는 이번 표본 worker에서 아직 측정하지 않는다.

## 5. Phase 1 → Phase 2 동일 학습 포맷

선별 후 **같은 Dialogue/window/sequence serializer**로 가는 것이 목표다.
이번 sidecar를 기존 학습기에 그대로 넣을 수 있다는 뜻은 아니다.

1. P1의 독립 mono clip도 lane 1을 사용하는 관측 구간으로 표현한다. 다른 파일들을
   실제 대화처럼 연결하거나 제공되지 않은 speaker ID를 만들지 않는다.
2. `ONSET`은 에너지/주석으로 관측한 시작 뒤에만 감독한다. 모든 clip 첫 block에
   ONSET을 강제하지 않는다. pre-roll·지속 발화로 시작·실제 무음도 포함한다.
3. clip 끝은 기본적으로 right-censored다. 파일 끝을 의미론적 `EOT` 정답으로 바꾸지 않는다.
4. P2는 발화가 선별에서 탈락해도 원음과 절대 시간축은 유지한다. 전사 loss만 마스크한다.
   잘못된 text 때문에 activity/turn을 자동으로 0으로 만들지 않는다.
5. 결손 오디오는 별도의 observability mask를 두고 해당 구간의 activity/turn/텍스트
   감독을 끈다. 단순히 loss를 끄는 기능만으로 결손 채널 복원이 완료되는 것은 아니다.
6. 텍스트 품질 승인은 ONSET/EOT 승인과 별개다. 정렬·채널·턴 검수 후 축별 마스크를
   부여하고, 무음/mono/overlap/재진입/clip 경계 golden sequence를 테스트한다.

주의: 이번 발화 manifest 선별에는 **발화 사이의 무음 창 자체가 행으로 들어 있지 않다**.
무음 보존은 실제 parent dialogue를 유지한다는 뜻이며, 무음 학습셋 생성 완료를 뜻하지
않는다. 후속 window builder에서 채널별 관측 가능 구간·검증 VAD로 실제 무음 창을
추출하고, ASR이 아무것도 출력하지 않았다는 이유만으로 무음 정답을 만들지 않는다.
`ASR-clean`, `dialogue-verified`, `hard/overlap-preserved` 풀을 구분하고, 후자의 보존량과
미확인 감독 마스크를 별도로 보고해야 첫 블록 ONSET 편향을 다시 만들지 않는다.

## 6. 실행 단계와 검수 관문

1. 전수 metadata census: 현재 15개 입력을 읽고 제외 이유·발화 시간·split·중복을 기록.
2. DB별 층화 표본: short(<2 s), overlap(주석), long(>15 s), regular에서 각각 최대 16개.
   short가 overlap보다 우선인 배타적 strata다. 실제 overlap 비율 추정에는 가중치가 필요하다.
3. 같은 파형을 Qwen3-ASR-0.6B와 Whisper-large-v2에 넣어 독립 전사. 기존 `asrcheck`
   결과는 파형/모델 지문이 없고 crop 선택 위험이 있어 이번 합의 증거로 재사용하지 않는다.
4. 두 출력과 원전사 비교, 샘플별 이유를 기록. **층화 표본의 후보율을 전체 DB 통과율로
   보고하지 않는다.** 처리 속도도 길이·NFS 영향과 초기 모델 로딩 시간을 구분한다.
5. 표본에서 길이/채널 결함이 확인되어 CPU 전수 오디오 검사를 먼저 진행한다(§8.3).
   각 DB에서 합의 후보와 불일치 후보를 모두 청취. 숫자/고유명사/겹침/맞장구/긴 무음
   전후를 포함해 우선 20–50개씩 보정하고, 교사/규칙을 고정한 뒤 전량 teacher screening.
6. 최소 acceptance audit 후 학습용 selection version을 동결. 50개 무오류 검수는
   오류율 1% 미만을 증명하지 못한다. 독립 무작위 표본 0오류 기준의 단측 95% 상한은
   `1 - 0.05^(1/n)`이므로 1% 기준에는 약 299개 이상이 필요하다. 층화·반복 세션 표본은
   그대로 독립 표본으로 계산하지 않는다.
7. 전사 통과분의 timing/channel QC와 원본 복원, quotas, leakage 검사 뒤 공통 serializer
   smoke → 동일 compute/동일 heldout의 품질 선별 ablation. 어려운 데이터 제거 때문에
   WER이 좋아 보이는지 전체/overlap/short/도메인별 성능을 같이 본다.

아직 사람 청취·임계값 보정 전이므로 **자동 학습 승인 수는 0으로 명시**한다.
이는 좋은 데이터가 없다는 뜻이 아니라 승인 단계를 생략하지 않는다는 뜻이다.

## 7. 구현·자원·재현

- `vapasr/data/selection.py`: 공통 행, 정확한 crop 대응, overlap strata, 잠정 판정.
- `vapasr/data/selection_audio.py`: 채널/구간을 엄격히 검사하는 디코더.
- `experiments/select_data.py`: 전수 census, 압축 JSONL sidecar, 결정론적 hash 표본.
- `experiments/selection_teachers.py`: 모델·코드·입력 지문 기반 재개, 이중 교사 각각의 worker.
- `experiments/selection_report.py`: 동일 파형·TN 검증 후 합의 결과 병합.
- `experiments/selection_audit_mnsc.py`: 현행 CSV의 ID/전사 충돌 및 원음 경로 전수 감사.
- `experiments/selection_audio_census.py`: 최초 8 CPU worker로 원음을 실제 디코딩한
  전수 검사. DB별 완료 marker와 중단된 부분 산출물을 보존하며 학습 승인하지 않음.
- `experiments/selection_audio_census_resume.py`: 동일 검사 함수를 사용하는 32-worker 재개 실행기.
  완료 결과 checksum과 코드 지문을 검증하고, 부분 gzip의 유효 행을 후보 순서와 대조해 복구한다.
- `tests/test_data_selection.py`: 결손 crop/채널 이웃 대체 금지, source offset, overlap,
  후보와 GOLD 구분, holdout alias 전파, 중복, 원본 불변성.

서버 산출물: `/soundai/users/tskim/VAPKT-data/data/selection/v0.1-20260918/`.
최신 사용자 허용량은 **mxc의 GPU 동시 최대 4장**이다(기존 표본 실행은 최대 2장 사용).
현재 CPU QC는 GPU를 사용하지 않는다. 교사 실행은 GPU별 작업과 길이/언어별 배치로
나눈다. GPU 메모리 상한 85%, OOM 시 배치를 반으로
재시도한다. 메모리를 의미 없이 채우는 대신 처리량을 보고 배치를 늘린다.
추론 오류는 누락하지 않고 실패 행으로 기록한다.

CPU 실행:

```bash
python experiments/select_data.py \
  --data-root /soundai/users/tskim/VAPKT-data/data \
  --out /soundai/users/tskim/VAPKT-data/data/selection/v0.1-20260918
```

출력 디렉터리가 이미 있으면 census는 중단한다. 새 버전/새 경로에서 재실행해야 한다.
teacher는 동일 fingerprint의 중단된 JSONL만 재개한다. 행이 잘린 파일은 조용히 무시하지
않고 실패하므로 복구 시 원본 결과를 보존하고 잘린 마지막 행을 명시적으로 처리해야 한다.

## 8. 실행 결과

### 8.1 전수 metadata 감사 완료

근거: [census.json](../../raw/sources/experiments/2026-09-18-data-selection/census.json),
[고정 표본 806개](../../raw/sources/experiments/2026-09-18-data-selection/pilot.jsonl).
원본을 변경하지 않고 15개 기존 manifest, **4,634,601개 발화 행**을 감사했다.

| 입력 | 추가 검사 대상 발화 | 명목 발화시간 h | metadata 격리 | heldout 제외 | 교사 표본 |
|---|---:|---:|---:|---:|---:|
| LibriSpeech 960 | 281,241 | 961.054 | 0 | 0 | 48 |
| KsponSpeech full | 619,932 | 964.941 | 0 | 0 | 48 |
| Switchboard | 179,427 | 225.276 | 0 | 0 | 48 |
| MNSC `mnsc-1000` | 74,215 | 108.411 | 0 | 0 | 38 |
| NIKL `nikl-1000` | 1,249,471 | 1,000.127 | 0 | 0 | 48 |
| VoxPopuli | 177,422 | 520.512 | 0 | 0 | 48 |
| YODAS en129 | 133,853 | 333.782 | 0 | 0 | 48 |
| AI Hub 방송 | 448,867 | 578.183 | 0 | 0 | 48 |
| AI Hub 71631 | 216,954 | 148.976 | 3,236 | 18,341 | 64 |
| AI Hub 134-1 | 337,837 | 242.079 | 45,523 | 43,420 | 64 |
| AI Hub 134-2 | 359,792 | 175.188 | 57,524 | 60,348 | 64 |
| otoSpeech | 78,313 | 63.145 | 416 | 8,164 | 64 |
| AMI | 74,144 | 89.075 | 39 | 9,295 | 64 |
| NOTSOFAR | 17,193 | 9.458 | 5 | 37,452 | 48 |
| ICSI | 90,303 | 56.674 | 4,494 | 7,380 | 64 |
| **합계** | **4,338,964** | **5,476.881** | **111,237** | **184,400** | **806** |

해석 주의:

- 시간은 **manifest가 선언한 발화 길이의 합**이다. overlap은 화자별로 중복 합산되며
  synthetic silence는 제외된다. §8.3에서 일부 길이의 오류를 실측했으므로
  **5,476.881 h를 실제 가용 원음 시간으로 사용하면 안 된다.**
- 격리 111,237개(71.329 발화시간)는 학습 후보 안의 metadata 문제다. heldout과는
  배타적이며, 오디오 디코딩 검사는 아직 전수로 수행하지 않았다.
- P1의 격리 0은 기존 빌더에서 이미 제외한 행을 이번 입력에 포함하지 않았기 때문이다.
  원본 DB에 오류가 없다는 뜻이 아니다. dev/test는 애초 P1 입력에서 제외했다.
- 이번 입력에서 추가 locator/session 중복은 0이었다. P1 71631·100 h 사본을
  catalog 단계에서 빼 두었으며, 재인코딩 중복·전체 speaker leakage는 여전히 미검증이다.
- AI Hub 134-1/2의 원 metadata상 missing crop은 각각 **46,020 / 60,494개**다.
  이 수치는 heldout/다른 사유와 겹치므로 격리 열에 더하면 안 된다.
- `mnsc-1000`의 기존 stats도 실제 발화시간 108.41 h를 기록한다. 원 빌더는
  basename ID 중복 2,181,201행을 제외했다. 이를 전부 중복 음성이라고 단정하지 말고,
  원 CSV에서 같은 ID의 path·전사 충돌 여부를 확인한 뒤 원본 규모 확장을 결정해야 한다.

### 8.2 이중 교사 표본 점검

시작: mxc GPU **2·3**, Qwen3-ASR-0.6B / Whisper-large-v2, 각 배치 128,
프로세스당 메모리 상한 85%. 서버 worker PID는 Qwen `562968`, Whisper `562969`
(시작 당시 값). 이 PID는 재시작 시 바뀌므로 로그를 기준으로 상태를 확인한다.

로그:

```text
/soundai/users/tskim/VAPKT-data/data-selection-v01-census.log
/soundai/users/tskim/VAPKT-data/data-selection-v01-pilot.log
/soundai/users/tskim/VAPKT-data/data/selection/v0.1-20260918/qwen.log
/soundai/users/tskim/VAPKT-data/data/selection/v0.1-20260918/whisper-v2.log
```

806개 모두 결과 행 생성 완료. 두 교사 각각 771개 추론 성공, 35개 오디오 오류이며
추론 자체 오류/결과 누락은 없었다. Qwen peak allocated 14.29 GB, Whisper-v2
44.77 GB; 모델 로딩/지문 계산을 제외한 처리 시간은 각각 102.9 / 116.1 s였다.
읽은 성공 표본은 1.655 h이며 이 속도를 원본 전량/NFS 콜드 상태에 그대로 외삽하지 않는다.

Whisper-v2 판정에서는 MNSC source hold를 추가 반영한 결과가 정본이다:
[pilot-v2-audited-summary.json](../../raw/sources/experiments/2026-09-18-data-selection/pilot-v2-audited-summary.json),
[발화별 결과](../../raw/sources/experiments/2026-09-18-data-selection/pilot-v2-audited-decisions.jsonl).
선행 `pilot-summary.json`은 MNSC 추가 감사 전의 이력으로 보존한다.

| DB | 합의 후보 | 원전사 교정 검수 후보 | 기타 검수 | 격리/보류 |
|---|---:|---:|---:|---:|
| LibriSpeech | 36 | 1 | 11 | 0 |
| KsponSpeech | 8 | 0 | 40 | 0 |
| Switchboard | 12 | 9 | 27 | 0 |
| MNSC | 0 | 0 | 0 | 38 |
| NIKL | 9 | 5 | 34 | 0 |
| VoxPopuli | 11 | 4 | 33 | 0 |
| YODAS | 22 | 2 | 24 | 0 |
| AI Hub 방송 | 11 | 2 | 35 | 0 |
| 71631 | 14 | 5 | 44 | 1 |
| 134-1 | 11 | 3 | 50 | 0 |
| 134-2 | 8 | 2 | 54 | 0 |
| otoSpeech | 23 | 6 | 35 | 0 |
| AMI | 13 | 10 | 41 | 0 |
| NOTSOFAR | 20 | 2 | 26 | 0 |
| ICSI | 10 | 13 | 41 | 0 |
| **합계** | **208** | **64** | **495** | **39** |

합의율이 낮다는 이유로 한국어/회의 DB 대부분을 버려서는 안 된다. strata와 전사 관습,
crop 대응, 채널 문제를 포함한 후보 목록이다. 예컨대 AMI의 `mm hmm`을 두 모델이
모두 `Hmm`으로 출력해도 제공자 전사가 틀렸다고 단정할 수 없다.

**전체 433.9만 발화의 이중 교사 선별이나 고품질 학습셋 승인이 완료된 상태는 아니다.**
전량 처리 전 표본 청취·규칙 보정이 남는다. 기존 학습 manifest와 데이터는 교체하지 않았다.

검증: `python3 -m unittest discover -s tests -p test_data_selection.py` **11개 통과**,
신규 Python 파일 구문 검사 및 shell `bash -n` 통과.

### 8.3 품질 선별보다 먼저 고쳐야 할 입력 결함

**MNSC — source integrity hold.**
[전수 감사](../../raw/sources/experiments/2026-09-18-data-selection/mnsc-source-audit.json):
현재 manifest 74,215개 중 68,047개(명목 99.354 h)는 지정 디렉터리에 파일이 없고,
목록에서 확인된 파일은 6,168개(명목 9.057 h)다. 현행 `*_train_tn.csv`의 PART1
2,258,301행은 77,100개 ID를 반복하며, **77,100개 ID 모두 같은 경로에 서로 다른
target이 매핑**된다. 단순 중복 제거 후 첫 행 채택으로 해결할 수 없다.
현재 서버의 CSV/파일 대응 문제이지 원본 MNSC 전체에 대한 품질 판정은 아니다.
이 DB는 합의 후보로 보였던 표본까지 source hold가 우선한다.

**AI Hub 방송 — 길이 계산의 16 kHz/mono 가정 오류.**
`vapasr/data/aihub.py:iter_bc`가 `(nbytes - 44) / 32000`으로 길이를 추정한다.
실측 `et_m_002_001_153_0355`는 manifest 15.18 s지만 원음은 48 kHz·2채널,
2.53 s다(6배 차이). 44.1 kHz·2채널 표본도 있었다. WAV header/frames로 길이를
계산하고 채널 downmix/선택 규약을 명시해야 한다. 원본 오디오가 나쁜 것이 아니라
현재 시간축 metadata가 잘못된 경우다. 표본 11개에서 길이와 채널 경고가 함께 나왔다.

**NIKL / VoxPopuli — 주석 길이와 실제 crop 길이 차이.**
NIKL 표본 15개에서 길이 경고가 났고 여러 2024/2025 PCM은 선언값보다 정확히
0.4 s 길었다. 이것만으로 앞뒤 각 0.2 s padding이라고 확정하거나 자르지 않는다.
VoxPopuli는 표본 17개에서 차이가 확인됐다(예: 14.838 s ↔ 실제 13.778 s).
두 빌더 모두 주석의 end−start를 쓰므로, 실제 crop의 시간축과 원문 범위를 재검증해야 한다.

**71631 — 전사 있는 crop의 파형 전체 0.** 표본 1개가 해당했다. 채널·구간·소스 파일을
확인할 문제이며, 이 row를 정상 무음 학습 정답으로 바꾸지 않았다.

이 발견에 따라 `selection_audio_census.py`를 **CPU 8-worker / nice 10 / GPU 0장**으로
전량 검사 실행을 시작했다(완료 아님). 방송 → NIKL → VoxPopuli를 먼저 검사하고, MNSC는 source hold로
건너뛴다. 나머지 DB도 이어 검사한다. 실제 디코딩·실제 길이·채널·RMS·clipping·파형
SHA256을 새 sidecar에 기록하며 기존 manifest/캐시는 수정하지 않는다.

```text
로그: /soundai/users/tskim/VAPKT-data/data-selection-v01-audio.log
산출물: .../data/selection/v0.1-20260918/audio-census/
완료 기준: 각 DB의 <source>.done.json
```

다음 학습용 manifest는 실제 길이와 원음 대응을 고친 **새 버전**으로 만들고,
offset/정렬/특징 캐시의 영향을 확인해야 한다. 이번에 이상을 찾았다고 기존 학습의
WER 악화 원인이 전부 설명됐다고 주장하지 않는다.

### 8.4 Whisper-v3 train 표본 대조도 완료

같은 806개에 Whisper-large-v3를 추가 적용했다. GPU 3 **한 장**, 배치 256,
peak allocated 70.56 GB, 로딩 제외 187.4 s. 정상 추론 771개/오디오 오류 35개로
v2와 같다. v2와 배치가 다르므로 시간 차이를 모델 자체 속도 차이라고 단정하지 않는다.

근거: [v3 요약](../../raw/sources/experiments/2026-09-18-data-selection/pilot-v3-audited-summary.json),
[v3 발화별 판정](../../raw/sources/experiments/2026-09-18-data-selection/pilot-v3-audited-decisions.jsonl).
MNSC hold 반영 후 **합의 후보 220 / 교정 검수 54 / 기타 검수 493 / 격리 39개**다.

| 진단 지표: 현행 원전사 대비 불일치 | Qwen | Whisper-v2 | Whisper-v3 |
|---|---:|---:|---:|
| EN 432개, 9,824단어 기준 WER | 12.04% | 14.20% | 14.33% |
| KO 335개, 10,464문자 기준 공백 제거 CER | 17.33% | 25.57% | 19.77% |

두 표본군은 MNSC와 디코딩 실패만 제외한 동일 행이다. 길이 불일치 등 검수 대상과
오류가 의심되는 원전사도 포함하므로 **정제 gold benchmark 성능이 아니다**.
KO에서 v3의 불일치가 작게 나왔으므로, 과거 NIKL 한 사례로 v2가 한국어 전체에서
우세하다고 일반화할 수 없다. 원안의 v3를 버리지 않고 v2를 보조 검수 증거로 보존한다.
최종 전량 screening 교사는 사람 청취와 시간축 복구 후 동결한다.

이 시점에 Qwen/v2/v3 GPU 작업은 모두 정상 종료했고 GPU 2·3은 반납했다.
CPU 전수 오디오 QC만 진행 중이다. 전량 ASR 추론을 자동으로 이어 붙이지 않았다.

당시 실행 확인(아래 §8.5가 최신): CPU worker PID `643202`, 방송 데이터 **148,480개** 검사 진행,
그중 음향/길이/채널 검수 1,273개. 파일 순서대로 처리 중인 부분 결과이므로 전체
결함률로 일반화하지 않는다. 15개 입력 manifest의 사후 SHA256도 최초 값과 모두 같았다.

### 8.5 CPU worker 확대 및 결과 보존형 재개 — 2026-09-18 23:27 KST

사용자 요청으로 **8 → 32 thread worker**로 늘렸다. mxc CPU 96개, load 약 24,
available 메모리 약 1.3 TiB를 확인했다. 원래 검사 함수는 수정하지 않고 재개 실행기만
추가했다. `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `nice 10`으로 중첩
스레드 과다 생성을 제한한다. 실제 프로세스는 worker 32개 + 주 스레드 1개다.

- 완료된 DB 8개는 SHA256 검증 후 재사용하고, MNSC source hold도 유지한다.
- 진행 중이던 `aihub134-1`은 부분 gzip에서 유효한 **44,282행**을 복구했다.
  종료 직전 로그는 51,200행이지만 gzip 버퍼에 남은 마지막 구간은 다시 검사했다.
  원본 오디오·manifest·이전 산출물은 수정하거나 삭제하지 않았다.
- 23:27 확인 시 해당 DB **56,570행**까지 기록됐다. 학습 적합성은 여전히 미승인이다.
- 기존 마지막 3,072행은 47.9초(약 **64행/초**), 새 실행의 미처리 구간
  52,474→56,570행은 38.6초(약 **106행/초**)였다. 단기 관측상 약 1.7배지만
  서로 다른 파일 구간·스토리지 부하이므로 전체 작업 가속률로 확정하지 않는다.
  재검사 구간의 캐시로 높게 나온 초반 처리율은 이 비교에서 제외했다.
- 기존 데이터 선별 테스트 11개 + gzip 재개 테스트 3개 통과. 새 프로세스 PID **4129725**.

현재 확인할 경로:

```text
로그: /soundai/users/tskim/VAPKT-data/data-selection-v01-audio-w32.log
산출물: /soundai/users/tskim/VAPKT-data/data/selection/v0.1-20260918/audio-census-w32/
실행 지문: 위 디렉터리의 fingerprint.json
```

완료 DB의 새 marker는 검증한 이전 결과 파일을 `output`으로 참조한다. 결과 수집 시
새 디렉터리의 gzip만 glob하지 말고 **marker의 output 경로**를 따라야 한다.
이전 8-worker 프로세스는 종료됐으며 이전 로그 경로는 더 이상 진행하지 않는다.
