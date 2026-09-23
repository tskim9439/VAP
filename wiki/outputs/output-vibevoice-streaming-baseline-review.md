---
type: output
status: active
created: 2026-09-18
updated: 2026-09-18
summary: VibeVoice-ASR-Streaming-1.5B를 전사·화자 귀속 대조군으로 권고하되 지연·EOT·timestamp·세션 길이 조건을 분리하는 평가안
sources:
  - '[[output-phase2-lane-plan]]'
  - '[[output-phase2-independent-evaluation-plan]]'
---

# VibeVoice-ASR-Streaming-1.5B baseline 검토

## 판단

**Phase 2 전사·화자 귀속의 주요 외부 baseline으로 권고한다.** RNN-T 대조군을 없애거나 현재 모델을 교체한다는 결정은 아니다. 아래는 공식 자료 검토와 평가 제안이며 모델 설치·다운로드·추론·학습은 실행하지 않았다.

공식 모델 카드는 한국어·영어를 포함한 10개 언어, streaming speaker-attributed transcription, 공개 가중치, MIT 라이선스를 명시한다. [모델 카드](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B), 확인일 2026-09-18.

## 비교 전에 고정할 사실

공개 구성은 22-frame 청크(약 2.93초)와 4-frame lookahead(약 0.53초)다. 평균 2.00초는 알고리즘 기대 지연이고, 첫 출력에는 약 3.5초의 오디오가 필요하다. 연산·대기 지연은 별도다. 공개 모델은 최대 8분을 지원하며, 장시간 겹침은 한계로 명시된다. 출력은 `Speaker k:` 전사이고 timestamp는 없다. `<|text_chunk_end|>`는 청크 종료이지 의미적 EOT가 아니다. [기술 보고서 §3·6·부록 A/B](https://arxiv.org/html/2609.02812v1).

1.5B 공개 청크 구성의 저자 측 측정값:

| 데이터 | WER | cpWER |
|---|---:|---:|
| AMI-IHM | 22.85% | 45.15% |
| AMI-SDM | 34.43% | 56.83% |

출처: [논문 Table 5](https://arxiv.org/html/2609.02812v1#S5). 7B의 대표 결과를 1.5B의 성능으로 인용하지 않는다. 현재 D1 계열 점수와는 입력·split·정규화·채점이 달라 직접 순위를 매기지 않는다.

## 프로젝트에서 맡길 역할

| 축 | 제안 |
|---|---|
| 전사 | 동일 TN의 ORC-WER/CER 및 S/D/I로 비교 |
| 사람별 전사 | 세션 전체 cpWER/cpCER로 비교. VAPKT의 재사용 lane을 영구 사람 ID로 오인하지 않음 |
| 지연 | 최초 텍스트·화자 귀속 가용 시각, 첫 출력, p50/p90/p99, RTF·VRAM 별도 측정 |
| 활동·음향 경계 | timestamp가 없어 native DER·onset/offset 정확도는 N/A. 별도 정렬기를 붙이면 복합 시스템으로 명명 |
| 의미적 턴 종료 | native EOT baseline으로 사용하지 않음. 필요 시 동일한 causal EOT 모듈을 붙인 cascade를 별도 실험 |

현재 목표인 80ms audio clock·약 320ms 설계 EOT와는 지연 조건이 다르다([[output-phase2-lane-plan]]). 이것만으로 VAPKT가 더 빠르다고 결론 내릴 수도 없다. 실제 추론 적체까지 포함해 비교해야 한다.

## 최소 실행 제안

1. **설치 격리:** 학습 환경을 변경하지 않고 별도 환경에서 공식 파일 추론 경로를 사용한다. 모델 revision·코드 commit·전처리 설정을 동결한다. 청크 크기를 임의로 줄여 다른 조건을 공식 모델 성능으로 보고하지 않는다. 공식 문서는 checkpoint의 `preprocessor_config.json`을 따르도록 한다. [공식 실행 문서](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr-streaming.md).
2. **개발용 smoke:** KO/EN 각각 무음·단일화자·교대·짧은 겹침·긴 겹침·세 번째 화자 재등장 사례를 고른다. silent chunk와 다음 청크의 speaker label 생략 시 직전 사람 ID를 유지하는 parser를 검사한다. 정답을 hotword로 제공하지 않는다.
3. **공통 평가:** 노출 감사한 동일 mono 세션·동일 마이크 조건을 사용한다. 모델별 필요 sample rate로만 변환한다. 8분 이하 세션을 1차 공통 트랙으로 하고, 초과 세션은 잘라 숨기지 말고 reset/재연결 정책이 있는 별도 장시간 트랙으로 둔다. 사람 ID 대응은 창마다 새로 최적화하지 않는다.
4. **지연 측정:** 파일을 가능한 한 빨리 처리하는 throughput과 실시간 속도로 공급하는 latency 실험을 분리한다. 관측한 오디오 끝, 토큰 방출 wall time, speaker ID 확정 시각을 기록한다. 오프라인 ForcedAligner로 뒤늦게 구한 시각을 온라인 모델의 timestamp 출력으로 간주하지 않는다.
5. **순서:** 1.5B zero-shot → D1b와 공통 dev 비교 → 오류 분석 후 필요하면 7B 추가. baseline fine-tuning과 모델 구조 변경은 그 이후 별도 의사결정이다. 최종 locked test는 parser·TN·정책 동결 전 튜닝에 사용하지 않는다.

데이터 독립성은 우리 학습 이력과 외부 모델 사전학습 이력을 따로 표시한다. 외부 모델의 미노출을 증명하지 못하면 미확인으로 둔다. 같은 데이터·예산으로 학습한 구조 통제 실험이 아니라 **공개 사전학습 시스템 비교**라는 점도 명시한다.

## 연구적 의미

내 판단으로는 단순한 추가 baseline보다 **Phase 2의 차별점을 검증할 직접 비교 대상**에 가깝다. 전사와 화자 귀속을 함께 스트리밍하는 것만으로 기여를 주장하기보다는, 같은 인식 품질에서의 낮은 실측 지연, 겹침 처리, 음향 경계, 의미적 EOT, 긴 대화의 재매핑 중 실제로 입증한 차이를 제시해야 한다.

권고 구성은 RNN-T(lexical 기준) + VibeVoice 1.5B(통합 전사·화자 귀속 기준) + 별도 턴 예측 baseline이다. 상세 채점 계약은 [[output-phase2-independent-evaluation-plan]]을 따른다.
