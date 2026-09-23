---
type: output
status: active
created: 2026-09-20
updated: 2026-09-20
summary: 세 우선 진단 후 청취문 4개 수신. 상태는 모두 unreviewed이며 AMI great·VoxPopuli in 생략 확인 대기. 기존 실패 보류 유지.
sources:
  - "[[output-data-selection-diagnostic-protocol]]"
  - "[[source-data-selection-human-review-v03]]"
  - "[[output-data-selection-repair-v03]]"
  - "[[source-data-selection-human-diagnostic-v04]]"
---

# 데이터 선별 v0.4: 세 우선 진단 결과

## 결론

**00:37 KST 후속 입력 수신:** 8개 key 일치, 청취문 4개 입력, 상태는 모두 `unreviewed`다.
방송 `네 네`, AMI `great`, VoxPopuli의 `And i understand that too`와
`let's keep this perspective`를 정정 후보로 보존했다. AMI의 청취 파일·구간과 마지막 문장의
`in` 생략 의도를 확인하기 전 타깃은 바꾸지 않았다. 상세: [[source-data-selection-human-diagnostic-v04]].
아래는 기술 진단 실행 결과다.

방송 → AMI → VoxPopuli 순서로 **기술 진단을 실행 완료**했다. 사람 검수 실패 11개를
실제 판정에 반영하고, 8개 사례·13개 오디오의 [후속 청취 페이지](../../raw/sources/experiments/2026-09-20-data-selection-diagnostic/index.html)를 만들었다.
이는 전사 정정·학습 승인까지 완료했다는 뜻은 아니다. 원음·원전사·TN·기존 학습 manifest는 유지했다.
GPU는 사용하지 않았다.

| 우선 과제 | 확인한 사실 | 남은 판단 |
|---|---|---|
| 방송 반복 | 원 발음전사부터 `네, 네, 네.`이며 TN→manifest가 재현됨. 라벨 3개·오디오 2개 사본은 각각 동일 | 실제 음성의 정확한 반복 수. TN 버그로 일괄 수정하지 않음 |
| AMI `Right` | agent A→Headset-0, 원 segment·word 시각·기존 crop hash가 모두 일치 | 가청성 문제인지 다른 음향 문제가 있는지 문맥/증폭 청취 |
| VoxPopuli 실패 6개 | 모두 단일 VAD 구간·길이 일치. 알려진 저장소에서 원 세션 미발견. 1개는 현행 lexical 비교상 교사와 원타깃 동일 | 나머지의 정확한 전사 및 일부 경계 원인. 원 세션 없는 상태로 crop 복구를 주장하지 않음 |

## 1. 사람 검수 보호: 구현과 실제 재판정

[selection_review.py](../../vapasr/data/selection_review.py)를 추가하고
[selection_report.py](../../experiments/selection_report.py)에 `--human-review`와
`--review-manifest`를 연결했다.

- key·source·발화 ID·manifest 지문·오디오 참조·원문·타깃이 달라지면 검수 재사용을 거부한다.
- 정상 교사 출력의 파형 hash가 검수 파형과 다르면 거부한다. 중복·누락 key도 오류다.
- 사람 fail은 `human_text_fail` 등 사유로 **QUARANTINE** 처리한다. 여기서 격리는 삭제가 아니라 보류 판정이다.
- pass는 기존 기술 오류나 미확인 타이밍을 지우지 않는다. 메모가 있으면 추가 해석 필요로 남긴다.
- 원형 라벨·메모·검수 파일 지문을 결과 행에 보존하며 `training_eligible=false`를 유지한다.

기존 886개 교사 결과를 재사용해 새 경로에서 판정을 실행했다.
**검수 32개 모두 적용, 그중 실패 11개 전부 보류, 나머지 21개는 REVIEW**다.
이는 21개를 불량으로 판정했다는 뜻이 아니라 기존 기술 경고/검수 메모를 자동 해제하지 않았다는 뜻이다.
전체 886개는 합의 후보 148·교정 검수 25·일반 검수 591·보류 122다.

근거: [검수 예외표](../../raw/sources/experiments/2026-09-20-data-selection-diagnostic/human-overlay.json),
[재판정 요약](../../raw/sources/experiments/2026-09-20-data-selection-diagnostic/adjudicated/summary.json),
[행별 재판정](../../raw/sources/experiments/2026-09-20-data-selection-diagnostic/adjudicated/decisions.jsonl).

이 보호는 **검수 인자를 전달한 판정 실행에서 적용**된다. 기존 학습 입력이나 전량 실행 체인을
자동 교체하지 않았으며, 향후 전량 선별 제출에도 검수 연결을 필수로 포함해야 한다.
원 검수의 semantic/display 쟁점이 해결돼도 기존 fail을 덮지 않고 별도 후속 판정을 남긴다.

## 2. 방송: TN이 아니라 원주석의 두 전사 필드 차이

대상 `vr_m_001_370_049_0348`을 031/033 Training의 라벨 ZIP 21개·원음 ZIP 14개에서 찾았다.
라벨 3개(한-러/한-스/한-영)는 JSON hash까지 동일하고, 발견된 원음 2개(한-러/한-영)도
파일 바이트와 원래 sample 파형이 동일했다. 이 표본에서는 배포 사본 충돌이 발견되지 않았다.

| 단계/필드 | 실제 값 |
|---|---|
| 원 JSON 철자전사 | `네.` |
| 원 JSON 발음전사 | `네, 네, 네.` |
| 원 JSON 전사작업본 | `(네, 네, 네)/(네). ~~` |
| 발음전사를 현행 TN으로 변환 | `네 네 네` |
| 기존 manifest/선별 타깃 | `네 네 네` |

manifest stats의 TN 버전·소스 hash·숫자 backend가 현재 재현 조건과 같고,
원 manifest hash도 기존 선별 기록과 같았다. **반복 수가 TN에서 새로 늘어났다는 가설은
이 표본에서 재현되지 않았다.** 사용자 청취의 `네 네`와 원 발음전사 간 차이는 여전히 전사 검수 문제다.
TN 전체를 변경하거나 철자전사로 일괄 전환하지 않았다.

부수 확인: 실제 WAV는 48 kHz·2채널·14,064 frame, **0.293초**인데 manifest는 **1.758초**다.
이는 이미 발견한 16 kHz mono 크기 환산 오류의 6배 차이와 일치한다. 반복 전사 문제와는 별도다.
원 라벨이 유지되더라도 시간축을 재구축할 때 header 기반 실제 길이를 써야 한다.
짧은 맞장구 자체를 나쁜 데이터로 간주하거나 이번 실행에서 임의 삭제하지 않았다.

근거: [필드·배포본·manifest 추적표](../../raw/sources/experiments/2026-09-20-data-selection-diagnostic/broadcast.json).

## 3. AMI: 메타데이터 불일치는 발견되지 않음

`ami_public_manual_1.6.2.zip`의 원 XML을 대조했다.

- 회의: `ES2010d`, agent A → Headset-0.
- segment: `ES2010d.sync.39`, **234.102–234.834초**.
- word: `ES2010d.A.words194`, **Right, 234.280–234.640초**.
- 현재 crop은 원 segment와 일치하고 원음 재추출 hash도 기존 QC와 같다.

따라서 현재까지는 리더의 단순 시간/채널 변환 오류나 청취 파일 복사 오류의 근거가 없다.
다만 원주석 자체의 정확성이나 실제 가청성까지 입증한 것은 아니다.

별도 청취 사본 6개를 생성했다.

1. 원 crop(0.732초).
2. **+12 dB 증폭 crop**: peak 0.02216→0.08820으로 clipping 없음. 길이·위치는 유지.
3. A/B/C/D Headset 각각 **232.102–236.834초** 문맥(4.732초).
   문맥 파일 안에서 원 crop은 2.000–2.732초, 원 word 주석은 2.178–2.538초다.

문맥 전체 RMS는 다른 Headset에서 더 클 수도 있다. 다른 화자의 발화·누설음이 섞일 수 있으므로
큰 채널을 정답 화자로 재매핑하지 않았다. 사람에게는 후보 문장을 보기 전에 듣도록 안내했다.
**현재 결론은 메타데이터 일치, 가청성/내용 판정 보류**다. 전사 삭제·무음 전환·자동 통과는 하지 않았다.

근거: [AMI 원주석·파형·증폭 기록](../../raw/sources/experiments/2026-09-20-data-selection-diagnostic/ami.json).

## 4. VoxPopuli: 6개 원인 분리와 확보 범위

6개의 공식 annotation을 찾아 원문·정규화문·VAD·시각을 보존하고 현재 crop을 재디코딩했다.
모두 원 QC hash가 같고, **6개 모두 VAD 구간이 하나**다. 이번 실패들을 다중 구간 연결 접합부의
문제라고 설명할 근거는 없다. 실제 길이도 VAD sample 수와 모두 같다.

| 사례 | 이번 진단으로 좁힌 범위 | 현재 후속 판정 |
|---|---|---|
| `getting so much` | 사람·교사는 `so much`, 단일 VAD crop | `getting`의 실제 잘림/원 라벨 오류 원인 미확정 |
| 앞의 `and` | Whisper 둘과 사람 메모는 추가 단어 지지, Qwen은 원타깃과 동일 | 전사 누락/clip 경계 문제 검수 |
| `let us` / `let's` | 세 교사와 사람 메모가 축약형 지지 | 국소 lexical 정정 후보. 원래 경계·화자 fail은 자동 해제하지 않음 |
| `who attend` / `attending` | 세 교사와 사람 메모가 실제 발화 어순 차이 지지 | 국소 정정 후보. 확정 target·승인 이력 필요 |
| ECR 문장 | 현행 `score_en`에서 원타깃과 세 교사 모두 동일 | 내용 오류보다 표시 선호 쟁점. 원형 text fail 유지 |
| fisheries 긴 문장 | 전치사·반복·말미 표현이 서로 다름 | 전체 교사 문장을 자동 채택하지 않고 위치별 정정 필요 |

원 세션 탐색은 다음 알려진 저장소 3곳, 깊이 4까지 수행했다. 확인한 파일은 111개이며,
해당 session ID와 일치하는 독립 ogg/wav/flac/mp3 파일은 발견되지 않았다.

- `/soundai/users/tskim/VAPKT-data/data/audio/voxpopuli`
- `/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/EN/TRAIN/OPEN/voxpopuli`
- `/soundai/DB/raw/voxpopuli` — 경로 없음.

이는 서버 전체에 원 세션이 없다는 증명이 아니다. 원 세션을 새로 다운로드하거나,
배포 조각을 붙여 원래 문맥을 복원한 것처럼 표시하지 않았다.
현재 crop에 충실한 전사 확정은 가능하지만 **원 세션 경계 복구/원대화 턴 검증은 추가 원음이 필요**하다.

근거: [annotation·VAD·lexical 비교·탐색 범위](../../raw/sources/experiments/2026-09-20-data-selection-diagnostic/voxpopuli.json).

## 5. 사용자 확인과 다음 단계

**2026-09-20 운영 갱신:** 아래는 당시 준비한 청취 절차의 기록이다. 사용자의 후속 지시에 따라 개별 재청취 요청을 중단하고 [[output-data-selection-automatic-framework]]의 전량 자동 판정·원인별 HOLD로 전환했다. 아래 확인 사항은 전체 선별의 선행 조건이 아니다.

**[후속 청취·정정 페이지](../../raw/sources/experiments/2026-09-20-data-selection-diagnostic/index.html)**
에서 먼저 방송 반복 수와 AMI 원본/증폭/문맥을 확인한다. VoxPopuli의 기존 판단은 보존돼 있으므로
6개를 무조건 전부 다시 들을 필요는 없다. 정확한 정정문이 필요한 항목만 보완하고,
ECR 문장은 표시 선호였는지만 메모하면 된다. 페이지의 후보 문장은 펼치기 전까지 숨겨져 있다.

내보내기 `human-diagnostic-v04.json`은 `heard_text`, 후속 판단, 메모를 받는다.
`resolved` 선택만으로 학습 승인이 되지 않으며, 원형 검수와 별도의 후속 증거다.
이후 원음에 충실한 정정문을 확정하고 새 override/manifest·재정렬 필요 범위를 정한다.
원 세션이 필요한 경계 문제는 그 자료 확보 전까지 보류하되 다른 DB를 모두 멈출 이유는 아니다.

## 검증·실행 기록

- 신규 진단 구현: [selection_diagnose_v04.py](../../experiments/selection_diagnose_v04.py).
- 신규 보호 테스트 6개 포함 회귀 테스트 **32개 통과**, Python compile·diff 공백 검사 통과.
- 청취 페이지 8개 카드·13개 오디오 링크 및 파형 hash 모두 확인했다.
- 방송 단계 CPU thread 8개, GPU 0장. 원본·기존 산출물 삭제 없음.
- 서버: `/soundai/users/tskim/VAPKT-data/data/selection/diagnostic-v0.4-20260920/` 및
  `human-adjudicated-v0.4-20260920/`. 두 실행 완료. 전량 screening·학습은 제출하지 않았다.
- [산출물 지문 요약](../../raw/sources/experiments/2026-09-20-data-selection-diagnostic/summary.json).
