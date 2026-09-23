---
type: output
status: active
created: 2026-09-19
updated: 2026-09-19
summary: 886개 교사 대조와 NIKL 16개 사람 검수 기록. 세션별 복구·VoxPopuli 시간축·채널 전수 감사 결과는 후속 v0.3 참조.
sources:
  - "[[source-data-selection-plan]]"
  - "[[output-data-selection-execution-plan]]"
  - "[[source-data-selection-human-review-20260919]]"
---

# 데이터 선별 v0.2: 시간축 감사와 청취 검수

**후속 갱신:** 두 세션의 625개 포맷 복구·재추론과 VoxPopuli/채널 전수 감사를 완료했다.
최신 결론과 다음 32개 청취 묶음은 [[output-data-selection-repair-v03]]에 있다.
아래 내용은 v0.2 당시의 관측·미확정 사항을 보존한다.

## 결론

CPU 전수 QC는 **14개 DB, 4,264,749행**을 처리하고 2026-09-19 02:09 KST에
완료됐다. MNSC는 원음·전사 대응 문제로 보류했다. 완료 결과 파일의 SHA256과
후보 목록의 행 순서·key·source·manifest 지문을 다시 대조했다.

음향 검사 통과 3,812,311개, 검수 451,073개, 검사 오류 1,365개다.
**통과는 현행 검사 규칙에서 기술적 경고가 없다는 뜻이지, 전사·시간축의 정확성이나
원음 포맷까지 입증한 것이 아니다.** 기존 기준이 놓친 일정한 길이 차이를 추가로 확인했다.

원본·전사·기존 manifest·정렬·학습 입력은 변경하지 않았다. 학습 승인 수는 계속 0이다.

근거: [전수 재집계·파일 지문](../../raw/sources/experiments/2026-09-19-data-selection-review/audit.json),
[886개 검수 표본](../../raw/sources/experiments/2026-09-19-data-selection-review/review.jsonl).

**[원음·원전사·세 교사 출력을 함께 보는 청취 페이지](../../raw/sources/experiments/2026-09-19-data-selection-review/index.html)**
를 로컬 브라우저에서 연다. 청취 가능한 기본 파형 775개와 다채널 대조 50개, 총 825개
오디오 사본이 연결돼 있다. 기본 파형 775개는 로컬 WAV의 실제 data chunk SHA256까지
서버 QC와 동일함을 검증했다. 오류 표본 111개는 오디오 없이 원인과 기록을 보존한다.

## 1. 새로 확인한 시간축·포맷 위험

아래 길이 차이는 **현재 디코더가 해석한 파일 길이 − manifest 길이**다.
숫자는 반올림된 millisecond histogram 기준이며, 오류로 디코딩하지 못한 행은 분모에서 제외한다.

| 대상 | 전수 관측 | 처리 방향 |
|---|---|---|
| NIKL 2024 | 216,089개 중 215,896개(99.91%)가 +399~401 ms | padding 위치·원대화 기준점을 확인하기 전 자르거나 이동하지 않음 |
| NIKL 2025 | 199,218개 중 198,515개(99.65%)가 +399~401 ms | 같은 규칙. 총 0.4초 차이만으로 앞뒤 0.2초를 확정하지 않음 |
| VoxPopuli | 177,422개 중 166,447개(93.81%)가 −199~201 ms | 제공 crop과 annotation의 시작·끝 규약 차이 검수; 앞 0.1초·뒤 0.1초 손실이라고 추정하지 않음 |
| AI Hub 방송 | 48 kHz stereo 2,237개, 44.1 kHz stereo 1,096개 | WAV header 기반 실제 길이 사용 필요. 채널 선택·downmix는 별도 검증 |
| AMI | stereo 324개가 채널 미지정 | headset 파일 이름만으로 mono라고 가정하지 않음. 채널 비교 후 대응 결정 |
| NIKL 2022 일부 | 선택된 원음 5개가 16 kHz 해석에서 주석 길이의 약 3배 | 48 kHz 등 포맷 문제 또는 잘못된 crop 대응을 의심하되 미확정. 자동 보정 금지 |

다채널 표본 추가 대조: AMI 26개는 채널 0·1의 파형이 완전히 같았다.
방송 24개는 3개만 완전히 같았고, 나머지는 차이가 있었다(최저 상관 약 0.608).
따라서 경고가 곧 화자 매핑 오류라는 뜻도 아니고, 모든 stereo를 동일 mono로 취급할
근거도 아니다. 이 수치는 선택된 표본의 관측이며 전체 다채널 파일에 일반화하지 않는다.
근거: [청취 사본 진단값](../../raw/sources/experiments/2026-09-19-data-selection-review/exports.jsonl).

기존 경고 기준은 `abs(diff) > max(0.1 s, 5% × 선언 길이)`였다. 따라서 VoxPopuli의
긴 발화는 0.2초 차이가 있어도 통과할 수 있다. **ASR용 기술 QC와 타이밍 QC 기준을
분리해야 한다.** 반대로 일정한 padding이 있다는 이유만으로 좋은 전사를 폐기해서도 안 된다.

### 코드와 원주석 대조

- [NIKL 빌더](../../experiments/s1_build_manifest.py)는 원 JSON의 `end-start`를
  manifest 길이로 쓴다. 연도별 2개씩 원 JSON을 재확인했다.
  예: `SDRW2400001165.1.1.33`은 원 JSON 137.1→142.36초(5.26초), 파일은
  현행 해석에서 5.66초다. `SDRW2500000325.1.1.130`은 원 JSON 약 2.82564초,
  파일 약 3.225625초다. 주석을 읽는 산술 오류만으로 이 차이가 생긴 것은 아니다.
- `SDRW2200000635.1.1.159`는 주석 약 1.08651초, 16 kHz 해석 시 파일 3.27초다.
  조사한 2022·2023 표본에 인식 가능한 오디오 헤더는 없었다. 확장자 `.pcm`만으로
  모든 연도·세션의 실제 샘플레이트를 확정할 수 없다.
- [VoxPopuli 리더](../../vapasr/data/voxpopuli.py)는 annotation의 `end_time-start_time`을
  쓴다. crop 파일과 원 annotation의 시간 기준은 아직 확정하지 않았다.
- [방송 리더](../../vapasr/data/aihub.py)의 `(nbytes-44)/32000`은 16 kHz·16-bit·mono
  가정이다. 관측된 길이비 약 6배와 5.5125배는 48 kHz/44.1 kHz stereo와 일치한다.
  이 부분은 원음 불량이라기보다 manifest 생성 가정의 오류다.

## 2. 이번에 실행한 다음 단계

1. `selection_review_audit.py`: 전수 결과 무결성·후보 대응 검증, DB/연도별
   길이 차이·포맷 집계 완료.
2. DB × NIKL 연도 × 음향 판정/사유 × 기존 길이·overlap 층마다 hash 순위로 최대
   8개를 골라 **886개**의 재현 가능한 검수 표본 생성. 오류·경고·정상 사례 모두 포함.
3. `selection_review_bundle.py`: 원래 QC와 동일한 파형 hash를 확인한 오디오를
   float WAV 사본으로 내보내 청취 페이지 생성. 다채널 경고는 최대 4개 채널을
   별도로 제공하지만, 교사 입력이나 학습 채널을 자동 변경하지 않는다.
4. Qwen3-ASR-0.6B / Whisper-large-v2 / Whisper-large-v3를 **GPU 1·2·3**에 제출.
   batch 128 / 128 / 256, 메모리 상한 85%, OOM 시 배치 분할. **전량 screening이
   아니라 886개 calibration 표본의 대조**다. 사용자 허용량인 동시 GPU 최대 4장 이내다.
5. 세 교사와 오디오 내보내기가 끝나면 서버 실행 체인이 v2/v3별 잠정 판정을 만들고
   청취 페이지에 세 전사를 붙인다. 실패 시 완료·승인으로 처리하지 않는다.

**실행 결과:** 세 교사 모두 완료, 각각 886행 = 정상 775 + audio_error 111.
오류 표본은 의도적으로 포함한 진단 층이며 추론 과정에서 조용히 제외하지 않았다.
최대 GPU allocated는 Qwen 20.81 GB / v2 40.22 GB / v3 68.97 GB였다.

| 잠정 규칙 판정 | Qwen + v2 | Qwen + v3 |
|---|---:|---:|
| 합의 후보(GOLD 아님) | 141 | 148 |
| 원전사 교정 검수 후보 | 32 | 25 |
| 기타 검수 | 602 | 602 |
| 오디오 오류 격리/복구 대기 | 111 | 111 |

근거: [v2 조합 요약](../../raw/sources/experiments/2026-09-19-data-selection-review/summary.json),
[v3 조합 요약](../../raw/sources/experiments/2026-09-19-data-selection-review/summary-v3.json).
이 숫자만으로 v3를 최종 교사로 선택하지 않는다.

### NIKL 2022 샘플레이트 가설 진단 — 완료

길이비 약 3배인 16개에서 세 교사의 원전사 불일치가 컸다.
`selection_rate_probe.py`로 **48 kHz int16 PCM 가설의 진단 사본**만 만들어
GPU 1에서 Qwen을 재실행했다. 두 세션(`SDRW2200000598`, `SDRW2200000635`)의
표본으로, 동일 원전사·동일 TN의 공백 제외 156문자 기준 결과는 다음과 같다.

| 해석 가설 | 문자 편집 오류 수 | 현재 원전사 대비 CER |
|---|---:|---:|
| 기존 16 kHz | 146 | 93.59% |
| 48 kHz로 해석 후 16 kHz resample | 23 | 14.74% |

예: `제가 가끔`은 기존 `차.`에서 `제가 가끔`으로, `새벽부터 막`은
`십구구터널`에서 `새벽부터 막.`으로 바뀌었다. **이 표본은 전사 불량보다
샘플레이트 오해석 가능성을 강하게 지지한다.** 사람이 검증한 gold 성능은 아니며,
세션 2개만으로 NIKL 2022 전체를 48 kHz로 바꿀 수는 없다.

[48 kHz 진단 사본 청취](../../raw/sources/experiments/2026-09-19-data-selection-review/diagnostic-48khz.html),
[진단 입력·원음 hash](../../raw/sources/experiments/2026-09-19-data-selection-review/rate-probe.jsonl),
[Qwen 진단 출력](../../raw/sources/experiments/2026-09-19-data-selection-review/rate-probe-qwen.jsonl).
기존 해석은 기본 청취 페이지에서 같은 발화 ID를 검색해 비교한다.

별도 key·원음 hash·`diagnostic_only=true`를 기록했으며 원본·정규 디코더·학습 입력은
바꾸지 않았다. 다음 포맷 복구는 해당 세션의 파일별 규약 확인과 사람 청취 후 **명시적
sample-rate metadata**로 적용해야 한다. 길이비로 샘플레이트를 자동 추정하는
fallback을 정규 디코더에 넣지 않는다. 3교사 calibration과 추가 진단 GPU 작업은 모두 종료됐다.

재개 3개·추가 감사 4개·기존 데이터 선별 11개, 총 **18개 테스트 통과**.

서버 경로:

```text
/soundai/users/tskim/VAPKT-data/data/selection/review-v0.2-20260919/
  audit.json          전수 집계와 무결성 지문
  review.jsonl        고정 검수 표본
  qwen.log / whisper-v2.log / whisper-v3.log
  listening/index.html
  results-v2/ / results-v3/   교사 완료 후 생성
체인 로그: /soundai/users/tskim/VAPKT-data/data-selection-review-v02-teachers.log
```

## 3. 사람이 확인할 내용과 다음 관문

첫 청취는 위험 진단용 **30개**부터 권장한다: NIKL 2024/25 각 5개,
NIKL 2022 길이비 약 3배 5개, VoxPopuli 5개, 방송 stereo 5개, AMI stereo 5개.
이후 각 DB에서 합의·불일치·짧은 반응·겹침을 포함해 20–50개를 검수한다.

- **전사:** 실제 말과 원문/각 교사 출력이 일치하는가? 숫자·고유명사·말더듬·맞장구가 누락됐는가?
- **경계/잘림:** 첫 음절·마지막 음절이 잘리는가? 파일 앞뒤에 추가 발화가 있는가?
- **채널/화자:** 채널별로 같은 말인가, 서로 다른 화자인가, 한 채널이 비어 있는가?
- **턴:** 독립 crop 끝만 보고 EOT를 승인하지 않는다. 원대화 근거 없으면 미검수/판단 보류.

페이지에서 축별로 `확인됨/문제 있음/판단 보류`와 메모를 입력하고 JSON을 내려받는다.
브라우저를 새로고침하면 입력이 사라지므로 **닫기 전에 반드시 내보내기**해야 한다.
경계 200 ms 에너지는 참고값일 뿐 VAD나 음소 경계의 증거로 사용하지 않는다.

사람 청취가 끝나기 전에는 모델·규칙을 최종 동결하거나 기존 라벨을 대체하지 않는다.
정해진 규약으로 수정할 때도 `file_duration`, `annotation_interval`, `crop_origin`,
`speech_boundary`를 구분해 새 버전 sidecar에 남긴다. 파일 길이만 바꿔 기존 정렬·캐시를
그대로 유효하다고 취급하지 않는다.

현재 표본의 교사 합의율은 DB 통과율이 아니다. 특히 경고·오류를 과표집했으므로,
전량 고품질 시간 추정이나 v2/v3의 일반 성능 순위로 해석하지 않는다.

**다음 실행 관문:** 포맷/원음 대응 확인 + 사람 검수 → 교사·규칙 동결 →
DB별 전량 screening(최대 GPU 4장) → 전사·시각·화자·턴을 분리해 승인 →
Phase 1/2 공통 serializer 검증. MNSC source hold는 별도 복구 전 유지한다.

## 4. 9월 19일 13:02 KST 사람 검수 반영

[[source-data-selection-human-review-20260919]]에 원본과 표본별 메모를 보존했다.
전달받은 key는 모두 **48 kHz 진단 표본 16개**에 대응하며, 886개 전체 검수와 구분한다.

- 원래 라벨: 전사 **pass 15 / fail 1**, 경계·화자 각각 **pass 16**, 턴 **pass 5 /
  uncertain 2 / unreviewed 9**. 실패·보류를 통과로 바꾸지 않았다.
- 전사 실패 1개(`SDRW2200000635.1.1.35`)는 “현재 타깃이 발음상 정확하지만 의미는
  Qwen이 적절”하다는 메모다. 현재 lexical ASR 목표와 의미 보정의 판정 기준이 충돌하므로
  **원래 실패 라벨을 유지하면서 타깃 변경은 보류**한다. 의미가 자연스럽다는 이유로
  `연기`를 `용기`로 자동 교체하지 않는다.
- `손석구→손석호` 등 교사가 잘못 고친 사례도 확인됐다. 교사 합의·유창성을 gold나
  자동 전사 교체 조건으로 쓸 수 없다는 기존 원칙을 유지한다.
- 청취 판단은 검수된 사본의 48 kHz 해석을 지지하지만, 전체 NIKL 포맷 승인,
  단어 정렬·원대화 offset·EOT 시각 승인으로 확대하지 않는다. 학습 자동 승인은 여전히 0이다.

다음 조치 순서: **두 세션의 파일별 포맷 확인 → sample-rate를 명시한 새 예외표·재디코딩
검증 → 원전사 유지/쟁점 표본 보류 → 나머지 DB 청취 관문**. 기존 PCM·manifest·캐시는
수정하지 않았고, 이번 검수 반영만으로 전량 추론이나 학습을 새로 시작하지 않았다.
