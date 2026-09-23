---
type: output
status: active
created: 2026-09-19
updated: 2026-09-19
summary: 포맷·시간축·채널 감사 후 32개 사람 검수 반영: 전사 실패 11개. 기술 일치와 실제 전사 품질을 분리하고 원인 진단을 우선한다.
sources:
  - "[[source-data-selection-human-review-20260919]]"
  - "[[output-data-selection-review-v02]]"
  - "[[source-data-selection-plan]]"
  - "[[source-data-selection-human-review-v03]]"
---

# 데이터 선별 v0.3: 복구 결과와 다음 관문

## 결론

요청한 순서 중 **NIKL 두 세션의 포맷 복구 → 나머지 시간축·채널의 기술 감사**를
완료했다. 원음·원전사·기존 학습 manifest는 수정하지 않았다. 새 학습 승인 수는 0이며,
전량 teacher screening과 Phase 1/2 학습은 시작하지 않았다.

**23:28 KST 후속 검수 수신:** [추가 청취 32개](../../raw/sources/experiments/2026-09-19-data-selection-next-review/index.html)의
key가 모두 일치했다. 전사 실패 11·경계 실패 5·채널/화자 실패 3, 턴은 모두 미검수다.
VoxPopuli의 실제 단어 누락/전사 차이, AMI의 낮은 가청성 의심, 방송 반복 타깃의 생성 이력을
먼저 진단해야 한다. 기술 감사 통과를 학습 승인으로 해석하지 않는다.
상세 근거와 표기 선호의 구분은 [[source-data-selection-human-review-v03]]에 정리했다.
기존 886개에서 추린 목적 표본이며 모든 DB의 품질을 대표하거나 전체 DB를 승인하는 검수는 아니다.

| 작업 | 완료 결과 | 아직 승인하지 않은 것 |
|---|---|---|
| NIKL 2022 두 세션 | 원주석 대응 PCM 681개 검사, 기존 후보 625개를 새 WAV·명시적 포맷 예외표로 복구 | 다른 세션의 48 kHz 일괄 적용, 전체 전사 정답화 |
| 복구 파형 재추론 | Qwen·Whisper-v3 각각 625/625 정상, key·입력 지문·파형 hash 일치 | 교사 합의만으로 GOLD/학습 승인 |
| VoxPopuli | 177,422/177,422개 길이가 공식 VAD 연결 규칙과 0 sample 오차 | VAD가 실제 발화를 전혀 자르지 않았다는 보장, 원대화 턴 복원 |
| 방송·AMI 채널 | 3,657개 전수 비교: 동일 718, 상이 2,939, 검사 오류 0 | 상이 채널의 일괄 downmix 또는 화자 분리 간주 |
| NIKL 2024/25 | +0.4초 차이 유지, 배포 설명서 확인 | 앞뒤 0.2초라고 가정한 이동·자르기 |

## 1. NIKL 포맷 복구

범위는 사람이 진단 음성을 들은 `SDRW2200000598`, `SDRW2200000635`에 한정했다.
각각 347개·334개, 총 681개가 **검수된 48 kHz 해석에서 주석 길이와 30 ms 이내로 일치**했고,
주석에 대응하지 않는 추가 PCM은 없었다. 헤더로 샘플레이트를 확인한 것은 아니며,
사람 청취와 세션 내 파일별 일관성에 근거한 제한적 예외다.

이 중 기존 학습 후보에 있던 **625개만** 48 kHz s16le mono로 해석한 뒤
16 kHz FLOAT WAV로 resample했다. 나머지 56개는 후보로 자동 편입하지 않았다.
샘플레이트를 길이비에서 자동 추정하는 범용 규칙도 추가하지 않았다.

- 원음 SHA256, 원주석 지문, 사람 검수 지문, resampler 버전, 예외 적용 근거를 보존했다.
- 새 key와 `source_key`를 함께 저장했다. 기존 정렬·특징·교사 출력·시퀀스는 재사용 금지로 표시했다.
  실제 기존 캐시 파일은 삭제하지 않았고, 정본 학습 경로도 교체하지 않았다.
- 시간축은 독립 clip의 0초부터 시작한다. 원대화 주석은 별도 보존하되
  `crop_origin_in_dialogue_s=null`로 두었다. 이 복구가 원대화의 정확한 시작점 복원을 뜻하지 않는다.
- 원전사는 유지했다. `긍까 연기를`의 사람 검수 쟁점 1개는 `human_text_review_hold`로 보류했다.

Qwen·Whisper-v3의 잠정 분류는 합의 후보 **247**, 일반 검수 **345**, 교정 검수 **32**,
보류 **1**이다. 두 교사의 입력 파형은 복구 QC와 625개 전부 같았다.
이 집계는 두 세션의 기존 후보만을 대상으로 하며 NIKL 전체 품질 비율이 아니다.
교사 출력이 더 유창하다는 이유로 원전사를 교체하지 않는다.

근거: [복구 요약](../../raw/sources/experiments/2026-09-19-data-selection-repair/summary.json),
[파일별 감사](../../raw/sources/experiments/2026-09-19-data-selection-repair/session-files.jsonl),
[포맷 예외표](../../raw/sources/experiments/2026-09-19-data-selection-repair/rate-overrides.jsonl),
[교사 판정 요약](../../raw/sources/experiments/2026-09-19-data-selection-repair/results-v3/summary.json).

## 2. VoxPopuli: −0.2초는 무엇이었나

공식 전처리는 `start_time/end_time` 전체를 자르는 대신, annotation의 `vad` 구간들을
추출해 순서대로 연결한다. 현재 manifest는 바깥쪽 주석 구간 길이를 사용해 두 규약이 달랐다.
[공식 구현](https://github.com/facebookresearch/voxpopuli/blob/main/voxpopuli/get_asr_data.py)을 확인하고,
서버 annotation과 QC에서 확인한 실제 파일 길이를 전량 대조했다(확인일 2026-09-19).

**177,422개 모두 `sum(int(vad_end × 16000) − int(vad_start × 16000))`와 실제 sample 수가 정확히 같았다.**
따라서 기존 약 −0.2초 차이 자체를 배포 파일의 손상으로 판단하거나, 일괄 padding으로 보충하면 안 된다.
이 검사는 길이·생성 규약의 일치이지, 원 세션 파형과의 샘플별 동일성이나 전사 정확도 검증은 아니다.

- 176,368개: VAD 구간 1개. annotation상의 단일 원점 변환 후보를 기록했다.
- **1,054개: VAD 구간 2~9개 연결.** 하나의 offset으로 원대화 시간축에 대응시킬 수 없다.
- 전량에 대해 연결 clip의 sample 범위 ↔ 원 VAD 구간 대응표를 생성했다.
- ASR 길이는 파일의 실제 길이를 쓰는 방향이 맞다. 다만 이번에는 sidecar만 만들었고 정본을 바꾸지 않았다.
- 원래 무음과 휴지가 제거된 clip은 원대화 EOT/VAP/hazard 감독으로 곧바로 사용하지 않는다.
  공통 serializer로 넣더라도 원대화 턴 라벨과 합성 스트림 라벨을 구분해야 한다.

근거: [전량 시간축 요약](../../raw/sources/experiments/2026-09-19-data-selection-repair/voxpopuli-timebase-summary.json).
전체 대응표는 서버 `selection/voxpopuli-timebase-v0.3-20260919/vad-timebase.jsonl`에 보존했다.
SHA256: `6215ffd43168cf0ad7a5aba007864c3a851b102a2588efee7c26f9e6ed3c6a14`.

## 3. 채널과 남은 padding 문제

| 대상 | 비교 수 | 채널 간 완전 동일 | 상이 — 검수 필요 |
|---|---:|---:|---:|
| AI Hub 방송 | 3,333 | 394 | 2,939 |
| AMI | 324 | 324 | 0 |

현행 QC가 다채널 경고를 낸 **실제 입력 crop 전체**를 비교했다. AMI의 원본 회의 파일 전체나
다른 행까지 동일하다고 일반화하지 않는다. 모든 행에서 재디코딩한 채널 0의 hash가 기존 QC와 같았다.
완전 동일한 718개는 음향 입력을 바꾸지 않고 `#ch0`을 명시할 수 있는 후보이며, sidecar에 기록했다.
상이한 2,939개는 stereo 차이가 있을 뿐 화자별 분리 채널이라고 판정한 것이 아니다.
음질·내용·화자 차이를 청취한 뒤 채널 정책을 정한다.

방송 길이는 WAV header를 기준으로 해야 한다. 기존 `(nbytes-44)/32000` 가정으로 만든
시간축·합성 스트림을 그대로 재사용해서는 안 된다. 실제 길이와 채널별 측정값을 모두 보존했다.

NIKL 2024의 [원음 배포 설명서](../../raw/sources/experiments/2026-09-19-data-selection-repair/nikl-2024-pcm-guide.pdf)
2쪽은 16 kHz·16-bit little-endian PCM을 명시한다. 2024·2025 전사 설명서는 `start/end`를
발화 시작·끝으로 설명하지만, 확인한 문서들은 추가 0.4초의 위치를 설명하지 않는다.
PDF의 본문과 관련 페이지 화면을 확인해 포맷 근거와 미확인 padding을 구분했다.
**총 길이 차이만으로 앞뒤 0.2초라고 확정하지 않는다.** 독립 clip의 청취 검수가 끝나도
원대화상의 정확한 위치는 원음 대응 또는 명시적 배포 규약으로 별도 확인해야 한다.

근거: [채널 요약](../../raw/sources/experiments/2026-09-19-data-selection-repair/channels/summary.json),
[파일별 결과](../../raw/sources/experiments/2026-09-19-data-selection-repair/channels/channels.jsonl),
[2024 전사 설명서](../../raw/sources/experiments/2026-09-19-data-selection-repair/nikl-2024-guide.pdf),
[2025 전사 설명서](../../raw/sources/experiments/2026-09-19-data-selection-repair/nikl-2025-guide.pdf).

## 4. 사람이 확인할 최소 묶음과 이후 순서

**아래 청취는 수신 완료.** 32개 전체를 다시 검수할 필요는 없다. 실패·메모 쟁점의 원인별
진단과 미검수 DB calibration이 다음이며, 구체 순서는 [[source-data-selection-human-review-v03]] §5를 따른다.
아래 표본 구성·안내는 배포 당시 기록이다.

**[청취 페이지 열기](../../raw/sources/experiments/2026-09-19-data-selection-next-review/index.html)**

- NIKL 2024/25 각 8개: 실제 발음과 전사의 일치, 첫·끝 음절 잘림.
- VoxPopuli 8개: 규약상 길이가 맞더라도 VAD 경계에서 발화가 잘렸는지, 단어가 누락됐는지.
- 방송 6개: 다른 채널 4개·동일 채널 2개. 내용·화자·명료도 차이와 선호 채널을 메모.
- AMI 2개: 동일 채널인 사례의 대조 청취.

짧은 것부터 긴 것까지 길이 분위수로 고른 목적 표본 32개다. 연결된 40개 WAV의 파일 복사
무결성을 확인했고, 기본 파형 hash·세 교사의 입력 지문도 대조했다.
**전사 / 경계·잘림 / 채널·화자만 평가**하고, 문맥이 부족한 턴 근거는 미검수로 둔다.
끝나면 `human-review-v03.json`을 내려받아 전달한다. 새로고침 전에 저장해야 한다.

이후 순서:

1. 검수 결과로 해당 DB의 포맷·길이·채널 정책과 전사 판정 기준을 보정한다.
   원대화 원점이 불명확한 자료는 clip ASR과 대화 타이밍 자격을 분리한다.
2. 기존 886개 calibration 중 아직 사람이 검수하지 않은 DB는 DB별 calibration을 진행한다.
   이번 32개나 NIKL 16개만으로 모든 DB의 임계값·교사 조합을 확정하지 않는다.
3. **관문을 통과한 DB부터** 전량 teacher screening을 실행한다. 동시 GPU 최대 4장,
   오디오 CPU worker 최대 32, 메모리 사용량과 OOM 분할을 감시한다.
4. 선별된 샘플로 Phase 1/2 공통 serializer를 smoke test한다. 시각·화자·턴 감독은
   축별 품질 마스크를 적용하고, ASR 합격을 EOT 합격으로 취급하지 않는다.

## 실행 기록

- 신규 구현: `selection_session_repair.py`, `selection_channel_audit.py`,
  `selection_vad_audit.py`, `selection_focus_review.py`, `run_selection_repair_validation.sh`.
- 회귀·신규 단위 테스트 **26개 통과**, Python compile 및 shell 문법 검사 통과.
- GPU 1·2 두 장만 사용했다. Qwen batch 128 / Whisper-v3 batch 256, 후자의 최대 allocated 약 69.6 GB.
  14:12 KST 확인 시 이 작업의 GPU 프로세스는 모두 종료됐다. 다른 사용자의 작업은 변경하지 않았다.
- 원본·기존 산출물 삭제 없음. 기존 학습 코드의 decode 기본값도 일괄 변경하지 않았다.
- 서버 경로 공통 접두사: `/soundai/users/tskim/VAPKT-data/data/`.
  복구 결과는 `selection/repair-v0.3-20260919/`, 채널은 `selection/channels-v0.3-20260919/`.
- [완료 로그](../../raw/sources/experiments/2026-09-19-data-selection-repair/validation.log).
  전량 screening을 자동으로 시작하는 대기 체인은 걸지 않았다. 추가 검수 수신 후 현재 관문은
  실패 사례·표기 선호의 원인 분리와 DB별 calibration이다.
