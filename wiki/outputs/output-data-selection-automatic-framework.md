---
type: output
status: active
created: 2026-09-20
updated: 2026-09-21
summary: 두 교사의 정규화 편집거리 ≤5%를 유지 기준으로 확대했다. 원전사는 진단용으로 보존하고 GPU 재추론 없이 별도 버전으로 재선별한다.
sources:
  - '[[source-data-selection-plan]]'
  - '[[source-data-selection-human-review-v03]]'
  - '[[source-data-selection-human-diagnostic-v04]]'
---

# 데이터 선별 자동 실행 프레임워크

## 1. 바꾸는 운영 방식

사용자는 모든 오디오를 들을 수 없다. 따라서 **샘플을 듣고 모호한 샘플을 다시 듣는 절차를 종료하고, 규칙을 코드로 고정한 뒤 데이터 전체에 적용한다.** 사용자에게 개별 보류 건의 해결을 계속 요구하지 않는다.

기존 426만 건의 음향 QC와 포맷·채널 복구 근거는 재사용한다. 사람의 기존 실패 판정은 보호한다. 추가 검수는 자동 판정 정책의 오류를 알아보는 제한된 감사이지, 모든 행의 승인 절차가 아니다.

이 프레임워크의 첫 산출물은 **자동 ASR silver 후보 manifest**다. 교사 합의를 정답으로 부르거나, ASR 합의를 EOT·화자·경계 승인으로 확대하지 않는다. 기존 학습 입력을 자동 교체하지도 않는다.

```text
원본 manifest + 완료된 전량 오디오 QC + 검증된 복구 예외
                 ↓ 지문·split·원음 대응 확인
          ┌──────┴──────┐
       추론 가능      구조적 문제/불확실성
          ↓              ↓
   Qwen + Whisper     원인별 HOLD / 명백한 불능 쌍 제외
          ↓
  Qwen–Whisper 동일 TN 비교 (원전사 차이는 기록)
          ├─ 편집거리 비율 ≤5% → KEEP_ASR_SILVER
          └─ >5% 또는 안전 조건 미충족 → 원인별 HOLD
                 ↓
   DB별 건수·시간·유지율·실패율 + 재개 가능한 manifest
```

## 2. 자동 판정 계약: 현재 asr-pair-select-v2

### 2026-09-21 변경: 두 교사 일치까지 유지 범위 확대

사용자가 **Qwen ↔ Whisper 정규화 편집거리 비율 ≤5%인 데이터까지 유지**하도록 지시했다. 원전사와 두 교사 간 오류율은 기록만 남기고 더 이상 유지의 필수 조건으로 쓰지 않는다. 기존 유지126만 건을 포함하는 확장 집합을 만든다.

판정식: `edit_distance(Qwen, Whisper) / max(1, Qwen 단위 수, Whisper 단위 수) ≤ 0.05`. 단위는 기존 TN 후 EN 단어 / KO 공백 제외 문자이며 재정규화하거나 GPU 추론을 반복하지 않고 기존 지문으로 검증한 수치를 재사용한다. 기존의 **원전사 길이 ≤4이면 세 문자열 완전 일치** 조건은 새 유지 기준에서 사용하지 않는다.

- `reference_and_pair`: 기존 v1의 세 비교를 통과한 유지 집합.
- `pair_only`: 원전사 일치 조건 때문에 보류됐지만 두 교사 간 ≤5%를 통과해 추가 유지하는 집합.
- 원음·채널·split·사람 검수·생성 경고·빈 전사 등 기존 안전 보류는 해제하지 않는다. 구체적으로 v1의 `KEEP_ASR_SILVER`와 `HOLD_DISAGREEMENT`만 재선별하고 다른 상태는 유지한다.
- 원전사·두 교사의 원출력·기존 비교 수치 모두 보존한다. 새 학습 타깃 후보는 계속 Qwen 원출력이며 원전사를 덮어쓰지 않는다.
- 교사가 함께 틀린 사례까지 통과할 수 있으므로 gold가 아닌 silver다. 원전사와의 큰 불일치는 후속 분석 가능하도록 남긴다. 화자/시각/EOT 승인은 여전히 별도다.

구현: `vapasr/data/selection_pair.py`, `experiments/selection_reselect_pair.py`. v1 완료 집계·shard·JSONL의 SHA256와 행 수를 검증하고 별도 폴더에 쓴다. shard 단위 원자적 완료·재개, 추가 유지 건수/시간과 기존 상태를 기록한다. 신규 테스트7개 통과. GPU0장, CPU worker8개로 실행한다.

서버 출력: `/soundai/users/tskim/VAPKT-data/data/selection/pair-v0.6-full-20260921/`. 진행은 `summary.json`의 `complete`, `rows`, `added`와 `/soundai/users/tskim/VAPKT-data/selection-pair-v06.log`에서 확인한다. `results/part-*/keep-asr.jsonl`이 확대된 유지 후보이고, `decisions.jsonl`은 이번 버전부터 key·이전/새 상태·이유·수치를 담은 경량 감사 기록이다. 원문을 포함한 전체 행은 `keep-asr.jsonl`/`hold.jsonl`에 보존한다.

재개 명령:

```bash
cd /soundai/users/tskim/VAPKT
/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python experiments/selection_reselect_pair.py \
  --source /soundai/users/tskim/VAPKT-data/data/selection/auto-v0.5-full-20260920 \
  --out /soundai/users/tskim/VAPKT-data/data/selection/pair-v0.6-full-20260921 \
  --workers 8
```

**재선별 완료:** 3,757,805건·459shard 전부 처리, 약153초, GPU0장. 기존 유지1,260,807건을 모두 보존하며510,174건을 추가 유지했다. 원음/TN/기존 교사 추론/원 학습 manifest는 변경하지 않았다.

| 항목 | v1: 세 비교 모두 ≤5% | v2: 교사 간 ≤5% |
|---|---:|---:|
| 유지 후보 | 1,260,807건 | **1,770,981건** |
| 유지 오디오 | 1,637.217시간 | **2,359.703시간** |
| 추론 대상 중 유지 비율 | 33.6% | **47.1%** |
| 보류 | 2,496,998건 | 1,986,824건 |

추가 유지 **510,174건·722.486시간**. 잔여 보류는 교사 불일치1,983,214건 / 빈 전사3,608건 / 생성 경고2건이다. 원음 사전 검사에서 보류된 데이터는 이번375만 건의 모집단 밖이며 새 기준으로 임의 복구하지 않았다.

| DB | v2 유지 후보 | v1 대비 추가 |
|---|---:|---:|
| LibriSpeech | 217,712 | 29,987 |
| KsponSpeech | 230,857 | 77,824 |
| Switchboard | 77,126 | 42,602 |
| NIKL | 447,088 | 118,615 |
| VoxPopuli | 132,687 | 55,943 |
| YODAS | 68,213 | 8,222 |
| AI Hub 방송 | 139,781 | 36,704 |
| AI Hub 71631 | 76,070 | 22,931 |
| AI Hub 134-1 | 135,374 | 39,594 |
| AI Hub 134-2 | 116,889 | 36,244 |
| otoSpeech | 44,179 | 7,907 |
| AMI | 28,608 | 10,075 |
| NOTSOFAR | 10,747 | 2,634 |
| ICSI | 45,650 | 20,892 |

근거: [v1 최종 집계](../../raw/sources/experiments/2026-09-21-data-selection-pair-v2/v1-final-summary.json), [v2 최종 집계](../../raw/sources/experiments/2026-09-21-data-selection-pair-v2/v2-final-summary.json), [코드·입력 지문](../../raw/sources/experiments/2026-09-21-data-selection-pair-v2/config.json). 신규 재선별/재개/변조 감지 테스트7개와 기존 판정 테스트10개 통과. 학습 실행과 Phase2 턴 라벨 승인은 별도다.

### 저장과 학습 타깃: verbatim-asr-v1 (사용자 후속 지시)

**저장 시 TN을 강제하지 않는다.** 원전사와 두 ASR API가 반환한 전사 문자열의 대소문자·구두점·숫자·띄어쓰기를 보존한다. 비교할 때만 동결 TN을 메모리에서 적용한다. 기존 교사 파일의 `hyp`도 이미 이 원형이므로 재추론 없이 새 저장 규약으로 후처리할 수 있다.

| 필드 | 내용 |
|---|---|
| `transcripts.source_raw` | 기존 manifest가 보유한 `raw_text`, 없으면 null; 이미 소실된 표기를 복원했다고 주장하지 않음 |
| `transcripts.source_manifest_target` | 기존 학습용 `text` 원형 보존 |
| `transcripts.qwen_raw` | Qwen 전사 API 반환 문자열 그대로 |
| `transcripts.whisper_raw` | Whisper 전사 API 반환 문자열 그대로 |
| `recommended_training_target.text` | KEEP_ASR_SILVER에 한해 Qwen 원출력, TN 없음 |
| `recommended_training_target.origin` | `qwen3_asr_pseudo_label`; 원 정답이 아닌 교사 생성 타깃 |

기존 `text` 필드를 몰래 다른 값으로 바꾸지 않는다. **새 학습 export의 타깃 선택은 `recommended_training_target.text`를 우선하는 방향**으로 정한다. 아직 학습 export/serializer를 연결하지 않았으며 원 학습 manifest와 TN 모듈은 그대로다. 선택 정책의 `auto_relabel=False`는 원전사 덮어쓰기를 금지한다는 뜻이고, 명시적 출처를 가진 별도 교사 타깃 후보 생성을 금지하는 뜻은 아니다.

Qwen backbone과 표기 양식을 맞추는 방향은 실험할 가치가 있지만, 더 빠른 수렴·더 좋은 WER이 **입증된 것은 아니다.** TN 후 일치는 lexical 내용의 근사 합의이지 구두점·대문자 정답 검증이 아니므로 `display_style_verified=False`를 기록한다. 같은 데이터로 lexical/display 타깃을 비교할 때 WER/CER뿐 아니라 토큰 수·방출 지연도 확인해야 한다. 구두점은 실제 음향 경계나 EOT의 정답으로 자동 사용하지 않는다.

### 이전 v1 규칙 — 재현용 기록, 전사 유지 조건은 위 v2로 대체

| 단계 | v1 자동 처리 | 하지 않는 일 |
|---|---|---|
| 범위 | 기존 train 후보만 처리, MNSC source hold 유지 | dev/test를 학습에 혼입 |
| 원음 | 기존 QC와 실제 추론 파형 SHA256 대응 | 깨진 경로를 비슷한 파일로 대체 |
| 검증된 복구 | NIKL 두 세션 복구 sidecar, 동일 채널의 명시적 ch0, 방송 파일·VoxPopuli 검증 길이 사용 | 전역 PCM sample rate 변경, 원음/원전사 덮어쓰기 |
| 명백한 불능 | 0 파형+비어 있지 않은 전사, 비유한 파형, 빈 crop을 ASR 쌍에서 제외 | 원음 삭제, 무음·overlap 전체 제거 |
| 기술적 보류 | 읽기 실패, 미해결 채널, QC 경고, 30초 초과, 낮은 RMS | 오류를 조용히 누락하거나 무한 재시도 |
| 전사 후보 | 원전사–Qwen, 원전사–Whisper, Qwen–Whisper 오류가 모두 ≤5% | 두 교사의 합의문으로 원전사 자동 교정 |
| 짧은 전사 | EN 4단어 이하 / KO 4문자 이하에서는 완전 일치 | 짧은 응답을 관대한 비율로 통과 |
| 이후 | 원래 텍스트·경로·선별 이유·지문 보존 | EOT/화자 품질을 전사 품질과 동일 취급 |

EN은 동결된 `score_en`의 단어, KO는 `score_ko(..., False)`의 문자로 비교한다. 기존 TN 코드를 변경하지 않는다. 길이·RMS·오류율 임계값은 **초기 보수적 운영값이며 검증된 품질 보증값이 아니다.** RMS <0.005는 조용한 정상 음성도 포함할 수 있으므로 제외가 아닌 HOLD다.

교사는 Qwen3-ASR-0.6B + Whisper-large-v2로 고정한다. v2가 모든 DB에서 우수하다는 결론이 아니라 재현 가능한 첫 실행 조합이다. 다른 조합은 별도 정책/실행으로 비교한다.

## 3. HOLD는 수동 검수 대기열이 아니다

- `HOLD_AUDIO/CHANNEL/TECHNICAL`: DB·포맷·원인별 건수/시간이 큰 묶음부터 규칙을 개선한다. 고친 규칙의 영향 범위만 다시 처리한다.
- `HOLD_DISAGREEMENT`: 원본 그대로 남기고 현재 silver 집합에서 제외한다. 모두 듣거나 교사 전사로 고치지 않는다. 나중에 도메인/길이/overlap별 분석 또는 별도 약지도 집합으로 다룬다.
- `HOLD_LONG`: 장문 구간화·문맥 보존 경로가 구현될 때 재처리한다. 현재 단문 교사 경로에 억지로 잘라 넣지 않는다.
- `HOLD_LOW_LEVEL`: 무음이라고 단정하지 않는다. 향후 발화 존재 검사 등 검증된 규칙이 생길 때 재평가한다.
- 기존 사람이 실패·불확실·메모를 남긴 사례는 자동 합의로 덮어 승인하지 않는다.

처리율이 낮은 DB를 조용히 버려서는 안 된다. **선별 전후 건수와 오디오 시간을 DB별로 함께 기록**한다. 쉬운 단일화자·표준어만 남는 편향, 맞장구·희귀어·겹침 소실은 후속 구성 감사 대상이다. 교사 불일치율을 데이터 오류율로 표현하지 않는다.

## 4. 실행·재개·자원 제한

구현:

- `vapasr/data/selection_auto.py`: 정책과 행별 판정.
- `experiments/selection_auto_prepare.py`: 완료 QC 전량 재사용, 지문 확인, 입력 shard 생성.
- `experiments/selection_auto_run.py`: GPU 큐·고정 코드 snapshot·교사 결합·후처리·집계.
- `experiments/selection_teachers.py`: 두 교사 추론, 배치 OOM 분할, 오디오 병렬 읽기.
- `tests/test_selection_auto.py`, `tests/test_selection_auto_score.py`: 범위·원음·사람 판정·짧은 텍스트·조인 실패 보호.

GPU 1–4 중 사용 메모리가 1 GiB 미만인 장치만 시작 시 사용하고 동시에 최대 4개 worker만 실행한다. 이 확인은 스케줄러의 독점 예약은 아니므로 공유 서버에서 다른 사용자의 동시 할당 가능성은 남는다. 다른 사용자의 프로세스를 종료하지 않는다. worker당 decode thread 8개, 네 worker에서 최대 32개다. torch CPU thread는 별도 worker당 4개다.

첫 검증의 배치128에서 peak allocated는 Qwen 약22.6 GB, Whisper 약44.8 GB였다. 사용자 요청에 따라 **전량 실행은 Qwen512 / Whisper256**, GPU 메모리 한도85%로 늘린다. GPU마다 약140 GiB여서 여유를 활용할 수 있지만 대형 배치의 실제 peak/속도는 전량 로그로 확인해야 한다. OOM 시 예외 traceback과 GPU 임시 텐서를 정리한 뒤 반으로 분할한다. 언어·길이순 묶음은 유지하고, 다음 배치의 오디오를 현재 GPU 추론 중 미리 읽는다. peak allocated/reserved와 오디오 처리 배율을 로그에 남긴다.

입력·모델 가중치·TN·코드·디코딩 옵션의 지문을 보관한다. 실행 코드를 snapshot으로 고정해 작업 중 프로젝트 변경이 재개 결과에 섞이지 않게 한다. 실행 lock으로 같은 큐의 이중 실행을 막는다. 교사 출력의 key·지문·행 수가 맞지 않으면 완료로 표시하지 않는다.

교사 추론은 JSONL의 완료 key부터 재개한다. 잘린 JSONL은 추측하여 읽지 않고 실패한다. 해당 출력의 유효 prefix 복구가 필요하다. 준비 단계는 새 디렉터리에 재실행하는 방식이며, **준비 스캔의 중간 체크포인트 재개는 아직 지원하지 않는다.** 후처리는 준비 완료 표본을 다시 읽어 재생성할 수 있다. 대규모 전량 실행은 shard마다 모델을 재로딩하므로 큰 shard 사용/상주 worker 최적화가 후속 비용 개선점이다.

서버 실행 예시(같은 명령을 다시 실행하면 교사 단계 재개):

```bash
cd /soundai/users/tskim/VAPKT
/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python experiments/selection_auto_run.py \
  --prepared /soundai/users/tskim/VAPKT-data/data/selection/auto-v0.5-20260920 \
  --gpus 1,2,3,4 --batch 128
```

전량 입력은 준비기에 `--per-source 0 --shard-size 8192`와 **새 `--out` 경로**를 주어 만든다. 기본값 128은 DB별 실행 검증 크기이지 매번 사람에게 보낼 표본 수가 아니다.

전량 실행 경로는 `auto-v0.5-full-20260920`이며 실행 옵션은 `--qwen-batch 512 --whisper-batch 256`이다. 준비 프로세스가 끝나면 같은 큐가 자동 실행되도록 `--wait-preparation-pid`로 연결했다. 두 교사가 shard를 마칠 때마다 그 shard의 ASR 후보/보류 manifest를 생성한다. 전량이 끝나기 전에는 집계의 `complete=False`를 유지한다.

## 5. 첫 실행과 다음 실행의 관문

첫 실행은 DB별 최대 128건을 고정 hash로 추출한다. 기존 목적 표본만 반복하지 않고 전체 READY 모집단에서 뽑는다. 총량은 준비 결과로 확정한다.

1. 모든 입력 key가 두 교사 결과 또는 명시적 실패 상태에 대응해야 한다.
2. source·파형·TN 지문 불일치가 있으면 해당 실행을 승인하지 않는다.
3. 추론 오류는 HOLD로 집계하고, 운영 관문에서는 정상 추론 비율도 별도로 확인한다. 단순히 프로세스 exit 0만으로 건강하다고 판단하지 않는다.
4. DB별 유지 건수/시간과 처리 속도를 측정해 전량 GPU 시간을 추정한다. 실패가 집중된 포맷은 bulk 규칙을 고친 뒤 그 묶음만 재시도한다.
5. 다음은 **같은 recipe의 전량 실행**이다. 또 다른 개별 청취 round가 아니다. 전량 소요 시간이 자원 예산을 크게 넘으면 큰 DB를 shard 단위로 점진 처리한다.

출력:

| 파일 | 의미 |
|---|---|
| `preflight.jsonl.gz` | 전량 QC 재판정: key·DB·상태·이유 |
| `summary.json`, `inputs/` | 준비 모집단 통계·지문·교사 입력 shard |
| `teachers/` | 재사용 가능한 원 교사 추론·지문·로그 |
| `results/*/keep-asr.jsonl` | 원전사를 보존한 ASR silver 후보 |
| `results/*/hold.jsonl` | 보류 원인과 전사 비교 수치 |
| `automatic-summary.json` | DB별 자동 후보/보류 건수·시간 |
| `run-status.json` | 실행 중 PID/GPU 또는 완료/실패 |
| `optional-audit.json` | 전 DB 합계 최대 40건, 진행을 막지 않는 감사 목록 |

감사는 유지 최대 32건·보류 최대 8건이며 **사용자에게 즉시 추가 청취를 요청하지 않는다.** 이는 진단용 상한이고, DB별 정밀도를 통계적으로 인증하는 표본 크기가 아니다. 품질 보증이 필요하면 별도 표본 설계와 신뢰구간을 사용해야 한다.

## 6. Phase 1/2 공통 학습 포맷으로 이어지는 경계

ASR silver는 전사 축만 통과한다. 학습 입력 export는 이후 한 번의 별도 단계로 수행하며 원 세션 식별자·split·라벨 출처·중복/평가 데이터 격리도 다시 확인한다. 현재 `training_eligible=False`는 수동으로 모든 행을 승인하라는 뜻이 아니라 **학습 export와 아직 연결하지 않았다는 안전장치**다.

Phase 2에서는 선별한 발화만 이어 붙여 실제 시간축을 만들면 안 된다. 원 세션의 무음·겹침·미승인 구간을 보존하고, 품질 축에 따라 ASR/ONSET/EOT/화자 loss mask를 따로 만든다. 전사 합의만으로 EOT를 생성하거나, 잡음/무음 후보를 이 프레임워크에서 삭제해 턴 학습 데이터를 구성하지 않는다. 경계·활동·화자·턴의 자동 QC는 별도 후속 모듈이며 이번 구현이 완료했다고 주장하지 않는다.

## 7. 실행 기록: 2026-09-20

**2026-09-21 완료 확인:** v1 전량3,757,805건, 유지1,260,807건·1,637.217시간, 보류2,496,998건. 459shard·918교사 작업 모두 완료했으며 최종 로그에 추론 오류·OOM 분할0건. 완료 시각은9월21일02:14 KST. 이 값은 아래 시작 기록을 잇는 최종 v1 결과이며, 위 v2 재선별 집계와 혼동하지 않는다.

전량 QC 재판정 4,264,749건 완료. 근거: [준비 결과 원본](../../raw/sources/experiments/2026-09-20-data-selection-automatic/prepare-summary.json).

| 상태 | 건수 |
|---|---:|
| 교사 추론 준비 완료 | 3,757,805 |
| QC 경고 보류 | 420,870 |
| 낮은 RMS 보류 | 70,117 |
| 30초 초과 등 길이 보류 | 11,633 |
| 채널 보류 | 2,937 |
| 기술적 보류 | 305 |
| 기존 사람 판정 보류 | 21 |
| metadata 보류 | 1 |
| 명백한 ASR 쌍 불능 | 1,060 |

MNSC source hold는 위 전량 음향 검사 모집단 바깥에 별도로 유지한다. READY 약 4,783시간은 **전사 품질 통과량이 아니라 교사 추론 대상의 길이 합계**다. 오류 행의 측정 불가 길이는 0으로 집계되므로 제외 시간 0을 손실이 없다는 뜻으로 해석하지 않는다.

14개 DB에서 각128건, 총1,792건·2.266시간을 두896행 shard로 준비했다. 별도 사람 검수 표본이 아닌 실행/운영 검증 배치다. 서버 root는 `/soundai/users/tskim/VAPKT-data/data/selection/auto-v0.5-20260920`, 로그는 `/soundai/users/tskim/VAPKT-data/auto-v05-run.log`다. 최종 교사 결과는 `automatic-summary.json` 생성 여부로 확인한다.

신규 정책·결합 테스트 11개 통과. GPU 1·2·3·4에서 Qwen 두 worker와 Whisper-v2 두 worker를 실행했다. 이 검증은 전량 두 교사 추론 완료를 뜻하지 않는다. 기존 generated index/log/todo와 원천 자료는 변경하지 않았다.

**첫 자동 배치 완료:** 1,792건 × 두 교사 모두 정상 추론. ASR silver633 / 전사 불일치 보류1,154 / 빈 전사 관련 보류5. 준비/모델 로딩 일부를 포함한 큐 elapsed 약233초. DB별128건 균등 표본이므로 35.3%를 전체 데이터 유지율로 외삽하면 안 된다. 근거: [자동 집계](../../raw/sources/experiments/2026-09-20-data-selection-automatic/automatic-summary.json).

기존 교사 출력을 재사용해 `results-verbatim-v1/`에 원출력 보존 후처리도 별도로 생성한다. 원래 `results/`와 교사 JSONL은 그대로 보존한다. 전량 입력 준비와 확대 배치 실행은 별도 root에서 이어진다.

## 관련 기록

- [[output-data-selection-execution-plan]] — 초기 전량 QC와 선별 계획.
- [[output-data-selection-repair-v03]] — 검증된 포맷·채널·시간축 예외의 근거.
- [[output-data-selection-diagnostic-v04]] — 개별 사례 진단 기록; 앞으로의 반복 청취 관문이 아님.
- [[source-data-selection-human-diagnostic-v04]] — 보존된 사용자 입력.
