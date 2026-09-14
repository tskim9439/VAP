---
type: output
status: active
created: 2026-09-14
updated: 2026-09-14
summary: 시퀀스 8개 비판 검증 — Q1 C-mode·기존 A/B ID 재사용·KO 실물 Dataset 우선·경량 QC 채택, 3초 지연과 미구현 명시
sources:
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[output-phase2-eot-review-framework]]'
  - '[[output-phase2-real-sequence-probe]]'
  - '[[output-phase2-sequence-replay]]'
  - '[[output-phase2-training-db]]'
---

# Phase 2 시퀀스 비판에 대한 검증과 반영

## 결론

**주요 비판을 수용한다. 지금 결과는 실데이터 직렬화 진단이지 Phase 2 학습 파이프라인 완성의 근거가 아니다.** 정본을 Q1 C-mode EOT와 공용 Dataset 경로 우선으로 바꾸고 검수 인프라를 축소했다. P-mode probe/replay는 당시 실험 기록으로 보존한다.

다만 C-mode도 관측 조건 자체가 잘못되면 오라벨이 된다. 특히 “3초 내 본인 재개 없음”을 쓰면 실제 3초 대기가 필요하다. 이를 숨기거나 offset+δ 타깃을 C로 이름만 바꾸지 않는다. **Q1 C는 저지연 제품 목표를 달성하는 해법이 아니라 학습·라벨 불확실성을 분리하는 기준선**이다.

## 1. 8개 지적의 처리

| 지적 | 검증·판단 | 정본 반영 |
|---|---|---|
| 1. 미래 P 타깃 잡음 | 동의. 기존 `next_other < min(resumed,horizon)`는 상대가 먼저면 본인 재개가 있어도 shift. 다만 재개 자체가 실제 floor EOT 오답을 뜻하지는 않음 | 둘 다 horizon 안이면 uncertain. Q1 토큰 C, Q3 head/P ablation. C에서 조건 전체의 관측 시각 사용 |
| 2. registry 미결 | 부분 동의. 정본은 A/B legacy 보존 방향이 있었지만 ID·padding·migration 실행 계약이 부족했음 | 슬롯 1/2 기존 A/B 재사용, 3..K·ONSET/EOT만 신규. 기존 행 보존·새 행 초기화·비등록 logits 차단·round-trip |
| 3. KO 주 경로 부재 | 동의. AMI 원 주석 재현은 71631 파이프라인 검증이 아님 | 실외 1대화로 TN→조각 정렬→2채널 복원→VAD→480블록→Dataset 및 head targets를 먼저 확인 |
| 4. TN 우회 | 동의. 전용 norm_word는 지문·quarantine 계약 없음 | v1.3.0 target+flags 필수. 기존 표본의 텍스트 일치 여부는 별도 확인 결과로만 기록 |
| 5. 독립 JSON·테스트 없음 | 동의. 공용 dialogue 모듈/16 fixtures 없음 | 정본에 정렬 키·의존성 계약, §4.4 16행 case_01..16, 실제 HF forward/backward·save/load 완료 조건 |
| 6. cap 근거 부족 | 동의. 38.4초 최대 3토큰은 상한 근거가 아님 | KO/EN·동시 0/1/2/3+명·전체/출력있는 chunk 별 밀도·cap/삭제/tick 검사 |
| 7. 검수 인프라 과대 | 동의. 전용 UI·전수 이중 검수는 Q1 선행 조건에서 제외 | TurnBench dev + AMI 200경계로 축소. 추가 의미 gold 구축·릴리스 플랫폼 이월 |
| 8. 규약 중복·미커밋 | 동의. 기존 명세는 이미 superseded였으나 본문에 현재 테스트 규약처럼 읽히는 문장 잔존 | 정본만 규약, 나머지 실험/과거안 명시. 본 작업의 코드·문서·텍스트 산출물만 커밋 |

규약 본문은 [[output-phase2-streaming-asr-diarization-plan]] §4.2·4.4·5.3·6.3·9에만 둔다. 이 표는 변경 근거이며 별도 serializer 규격이 아니다.

## 2. 확인한 사실과 아직 확인하지 못한 것

### EOT와 헤드

`experiments/p2_sequence_probe.py`의 조건과 기존 이벤트 JSON을 확인했다. AMI D의 offset=84.576초, 본인 재개=86.048초, 예전 P 방출=84.96초, horizon 끝=87.576초다. 새 C-rule의 positive에서는 제외할 uncertain 사례다. **기존 raw 결과의 EOT를 삭제하거나 새 판정으로 덮어쓰지 않았다.**

“예측은 이미 헤드가 한다”는 현재 구현 사실이 아니라 앞으로의 역할 분담으로 정정한다. 현재 `vapasr/data/targets.py`에 dyadic VAD/VAP/hazard 함수는 있지만 Phase 2 K×4 target·head와 probe의 연결은 없다. 헤드 경로의 유효성은 Q3에서 따로 검증해야 한다.

### Tokenizer와 행 수

2026-09-14 로컬 E2 MLX export의 tokenizer와 safetensors **헤더**를 읽었다. 전체 가중치를 로드하거나 수정하지 않았다.

| 항목 | 확인 값 |
|---|---|
| tokenizer 유효 ID | 151,717개, 최대 ID 151716 |
| `<SPK_A>` / `<SPK_B>` | 151707 / 151708 |
| `<NEXT_AUDIO>` | 151705 |
| `model.embed_tokens.weight` | `[151936,1024]`, F16 |
| 독립 `lm_head` 텐서 | 이 MLX 파일에는 없음 |
| 코드 weight tying | `vapasr/hf/modeling_vapasr.py`는 tied head 명시 |
| 미등록 행 개수 | 151936−151717 = 219 |

`config.json`은 dataless 상태였고 제한 시간 내 읽지 못했다. HF config 클래스의 기본값 151936은 **실제 E2 config 확인을 대체하지 않는다.** 따라서 “리사이즈 없이 가능”은 유력한 경로지만, 원 HF checkpoint의 config·shape·tied 상태·저장 후 parity 확인 전 확정 구현 사실로 쓰지 않는다. E2는 full FT였으므로 미등록 행이 미학습/초기값 그대로였다고 가정하지 않는다.

임시 probe ID를 재사용하지 않고 논리 슬롯과 실제 token ID를 분리한다. 토큰 등록 수와 padded 모델 vocab 크기를 구분하며 무심코 embedding을 축소하지 않는다. 기존 A/B의 decoding block도 Phase 2에서 수정해야 실제 출력된다.

### TN

코드의 현재 값은 `asr-tn-v1.3.0`, numeric backend `num2words 0.5.14`다. 보존된 AMI **92개 단어 묶음**을 `target_en(raw, corpus="ami")`와 대조한 결과, 기존 제한 함수 출력과 차이 0·target_flags 0이었다. 이는 이 영어 표본의 surface 검사일 뿐 TN 지문·전체 문맥 정규화·정렬 재생성·KO 경로를 검증한 것은 아니다.

특히 현재 TN은 EN 모든 corpus의 숫자를 자동 발음형으로 바꾸지 않는다. AMI는 숫자 허용 corpus가 아니므로 남은 digit을 flags로 격리해야 한다. `target_en` 단독 호출을 학습 승인으로 사용하지 않는다.

### 한국어 대화 경로

서버 데이터 상태를 이번에 재조회하지 않았다. 저장된 DB 보고서 기준으로 71631 실외 원본(dev 186개)과 134-1 실외 조각(train 후보 1,492개)은 별도 풀이다. 대응 대화 ID가 존재하는지 확인한 뒤 원본 대비 복원 검증을 해야 한다. 없는 대응 쌍을 가정해 경로를 만들지 않는다. [[output-phase2-training-db]]

원본 없는 조각의 빈 구간은 실제 무음 원음이 아니다. FA가 잃어버린 대화 gap을 복구하지도 않는다. 에너지 VAD·head 라벨은 관측/결손 mask와 함께 검사하고, KO 1대화 QA만으로 한국어 EOT 의미 품질 전체가 확보됐다고 쓰지 않는다.

## 3. 축소 검수의 한계도 함께 수용한다

TurnBench dev 결과로 규칙을 고치면 같은 dev 재평가는 독립 검증이 아니다. 처음 고정 규칙 결과와 튜닝 후 결과를 나눠 기록하고 unseen 일반화는 별도 held-out에서 확인한다. AMI 200경계는 후보 오라벨/모호성 진단에 적합하지만 후보 생성 단계부터 빠진 사건을 모두 찾지 못하므로 전체 recall을 보고하지 않는다.

검수한 경계만으로 주변 창을 event-complete로 승격하지 않는 안전장치는 유지한다. 전용 플랫폼 구축을 줄이는 것이 라벨 누락을 negative로 학습시키는 허가가 되지는 않는다. [[output-phase2-eot-review-framework]]

## 4. 이번 작업의 완료 범위

정본과 경량 QC 문서를 수정하고 실험 기록·legacy 코드를 구분했다. registry 변경·C 라벨러·공용 serializer·KO FA/VAD·HF Trainer 연결은 **구현하지 않았다**. 독립 JSON을 학습용으로 승인하지 않았으며 모델/데이터 전량 작업도 제출하지 않았다.

다음 구현은 **registry/TN → 공용 serializer·16 tests → 71631 실외 실물 Dataset → 경량 QC/밀도 → 32창 overfit** 순서다. 이전 probe/replay를 다시 돌리는 것은 이 구현을 대체하지 않는다.

검증: 문서 YAML·링크·코드 fence와 diff 공백 검사, probe/replay 컴파일, 임시 출력 경로에서 480블록 replay 복원 검사를 통과했다. `pytest`는 로컬 환경에 없어 설치하지 않고 제공된 `python tests/test_textnorm.py` 경로로 실행했다. 문자열/규칙 검사 10개는 통과했고 tokenizer-ID golden 검사 1개는 해당 설정 부재로 생략됐다. AMI 92개 재토큰화 검사는 replay에서 별도로 통과했다.

커밋 범위는 관련 코드·문서·로그·텍스트 실험 산출물이다. 오디오 바이너리는 저장 정책이 확정되지 않아 로컬 보존본으로 남기며, 새 clone에서 음성을 재현하려면 원본/별도 저장소가 필요하다. 기존 DB 정리·무관한 테스트 변경은 이번 커밋에 포함하지 않는다. generated index/log/todo는 직접 수정하지 않았고 pre-merge 재생성 대상으로 남긴다.

원 `blocks.tsv`는 CSV writer의 CRLF 줄끝이므로 기본 Git 공백 검사에서 경고한다. raw 바이트 보존을 위해 줄끝을 바꾸지 않고 `core.whitespace=cr-at-eol`을 해당 검사 명령에만 적용한다. 저장소 설정이나 원천 파일은 변경하지 않는다.
