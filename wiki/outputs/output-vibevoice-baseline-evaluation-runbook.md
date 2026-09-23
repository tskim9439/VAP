---
type: output
status: active
created: 2026-09-18
updated: 2026-09-18
summary: VibeVoice 1.5B 평가 준비 — 격리 환경·고정 revision·KO/EN smoke manifest·청크별 추론·ORC/cp 채점·apex 제출 방법
sources:
  - '[[output-vibevoice-streaming-baseline-review]]'
  - '[[output-phase2-independent-evaluation-plan]]'
---

# VibeVoice 1.5B 평가 실행 안내

## 범위

사용자 요청에 따라 **평가 실행 준비**를 구현했다. 기존 학습 코드·환경·잡 설정은 변경하지 않는다. 공식 소스·모델을 전용 경로에 준비하고, GPU 잡 제출과 실제 모델 성능 보고는 구분한다. 이번 smoke는 최종 locked test가 아니다.

## 파일

| 파일 | 역할 |
|---|---|
| `experiments/p2_setup_vibevoice.sh` | 읽기 전용 학습 env 상속 venv, 공식 소스·가중치 준비 |
| `experiments/p2_prepare_baseline.py` | 동일 mono WAV·참조 manifest·선택 감사 기록 |
| `experiments/p2_eval_vibevoice.py` | 공식 step API 추론, 청크 로그·시간·토큰 상한 기록 |
| `experiments/p2_score_baseline.py` | 공통 segments → 정확 ORC/cp, SegLST, corpus별 micro 집계 |
| `vapasr/data/baseline_eval.py` | 파서·TN·hash·채점 공통부 |
| `tests/test_baseline_eval.py` | 순열·중간 귀속 변경·문자 CER·무음 삽입·manifest 안전 검사 등 11개 |
| `slurm/p2_eval_vibevoice.sbatch` | apex GPU 1장, 추론 후 CPU 채점 |

## 데이터와 환경

- 환경: `/soundai/users/tskim/VAPKT-data/baselines/vibevoice-1.5b-v1/`
- 모델 revision: `4262d23d8a539a6530cf64fbd0b1751ef9a30853`
- 공식 코드 commit: `1541f590c7099820f10ea012f48d2399282df69f`
- 주 의존성: `transformers==4.57.1`, `meeteval==0.4.3`, `num2words==0.5.14`. 설치 결과 전체는 `environment.freeze.txt`로 기록한다.
- venv는 `--system-site-packages`로 기존 torch/CUDA를 상속한다. 기존 env에 쓰지 않지만 완전히 독립된 환경 복제는 아니므로, 상위 env 업데이트 시 다시 검증한다.
- 상속된 `qwen-asr==0.0.6`는 Transformers 4.57.6을 요구하므로 전용 env의 4.57.1과 의존성 경고가 발생한다. 이 env는 VibeVoice·MeetEval 전용으로 사용하고 Qwen 평가는 기존 env에서 실행한다. 기존 학습 env의 Transformers는 변경하지 않았다.
- 사용할 manifest: `/soundai/users/tskim/VAPKT-data/eval/baseline-smoke-v3/manifest.jsonl`
- 내용: NIKL 2020 KO 1개 + CHiME-6 S02 EN 1개. 최대 60초씩, 첫 주석 발화 3초 전부터 시작하고 모든 화자의 발화 중간을 피해서 끝낸다. crop 시작·길이·원 세션·참조·WAV hash·TN fingerprint를 남긴다.
- 기존 프로젝트 mono 캐시를 재사용한다. 이는 공식 CHiME 마이크 조건을 재현했다는 뜻이 아니다. D1b와의 내부 공통 입력 비교용이다.
- `baseline-smoke-v1`은 세션 시작 0초 표본이라 S09가 전사 없는 60초였다. `v2`는 시작점을 바꿨지만 S09의 unintelligible/빈 전사로 엄격 검증이 실패했고 manifest를 발행하지 않았다. 두 산출물은 삭제하지 않고 감사용으로 보존한다. **제출에는 v3만 사용한다.**

| v3 표본 | 길이 | 참조 발화 | 채점 단위 |
|---|---:|---:|---:|
| `nikl2020:SDRW2000001990` | 57.0000625초 | 29 | 226문자 |
| `chime6:S02` | 57.46초 | 10 | 76단어 |

합계 114.4600625초. 실제 manifest 참조를 그대로 가설로 넣은 CPU round-trip에서 두 표본 모두 ORC/cp 오류 0을 확인했다. 이는 **채점기 연결 검사**이지 모델 성능이 아니다.

## 실행

새 환경을 처음 준비할 때만 실행한다. 기존 경로가 있으면 자동 덮어쓰지 않는다.

```bash
cd /soundai/users/tskim/VAPKT
bash experiments/p2_setup_vibevoice.sh
```

환경·가중치 준비와 import 검증이 끝나면 `SETUP_READY`가 생성된다. 아래 제출은 준비 완료 후 사용한다.

```bash
cd /soundai/users/tskim/VAPKT
sbatch --partition=apex \
  --export=ALL,MANIFEST=/soundai/users/tskim/VAPKT-data/eval/baseline-smoke-v3/manifest.jsonl,OUT=/soundai/users/tskim/VAPKT-data/eval/vibevoice-1.5b-smoke-v3,SPEED=throughput \
  slurm/p2_eval_vibevoice.sbatch
```

실시간 속도 공급 시험은 **별도 OUT**에 `SPEED=realtime`로 제출한다. 완료 세션 JSON은 재사용하며 다른 run fingerprint가 있으면 중단한다. 실패·선점 중 쓰던 `.tmp`는 완성 결과로 간주하지 않는다. 완성 세션 내부의 모델 KV는 재개하지 않고 해당 세션 처음부터 다시 실행한다.

## 채점 계약과 산출물

- `run.json`: 코드·모델 revision, 실행 코드 hash, 설정·의존성·manifest hash.
- `session-0000.json` 등: 원 chunk 텍스트·speaker segments·관측 오디오 끝·방출 경과 시각·연산 시간·상한 도달·VRAM.
- `session-0000.{ref,hyp}.seglst.json`: 시각을 발명하지 않는 공통 전사 채점 입력.
- `scores.json`: EN WER / KO CER-nospace의 ORC·cp·S/D/I, 코퍼스별 합산, 미완료 세션 목록·완전성, 화자 미할당 문자/단어 수·cap hit.
- `INFERENCE_DONE`은 추론만 완료한 표식. 전체 완료는 SLURM 로그의 `EVALUATION_DONE`과 `scores.json.complete=true`를 함께 확인한다.

가설에 첫 speaker label이 없으면 텍스트를 버리지 않고 `__unassigned__`에 남긴다. 청크 사이 단어·화자 continuation을 연결하고, KO는 BPE나 어절이 아닌 공백 제외 문자로 MeetEval에 전달한다. 전사 없는 참조의 삽입도 합산 분자에 포함한다. 일부 세션 실패를 조용히 건너뛰어 성공한 표본만 대표 점수로 삼지 않는다.

## 타이밍 해석의 제한

공식 `streaming_generate` 파일 경로는 특징 청크를 먼저 준비하는 구현이다. 여기서는 `init_streaming_state` → 각 청크의 `encode_speech` → `streaming_generate_step`를 호출한다. 모델은 해당 청크+lookahead만 받고 KV는 clip 전체에서 유지한다. [고정 revision 공식 코드](https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/vibevoice/modular/modeling_vibevoice_asr.py), 확인일 2026-09-18.

`throughput`의 elapsed/RTF는 파일 처리 비용이며 실시간 word latency가 아니다. `realtime`은 청크가 관측 가능해지는 시점까지 기다린다. 출력 시각은 **청크 전사 완료 시각**이고, 첫 토큰도 speaker 태그일 수 있다. sample-rate 변환은 파일 입력 준비 단계라 capture/resampler까지 포함한 live device latency 검증은 아니다. prefix-causality·공식 파일 경로와의 출력 parity·정렬 기반 단어 지연 검증은 GPU smoke 이후 추가한다.

Native DER·onset/offset·EOT 및 word latency는 `null`로 남긴다. 의미적 EOT나 실제 음향 경계로 청크 종료를 재해석하지 않는다. D1b의 기존 30초 초기화 평가 점수와 직접 비교하지 말고, 같은 manifest를 소비하는 D1b 출력 어댑터와 외부 사람 ID 연결을 추가한 후 종합 비교한다.

## 검증 상태

2026-09-18 준비 완료:

- 공식 코드 설치·고정 revision 모델 가중치(약 5.63 GB) 다운로드 완료. VibeVoice 모델 클래스·프로세서 import 통과.
- MeetEval `0.4.3`은 기존 학습 env에 이미 설치돼 있었고, 사용자 추가 요청에 따라 전용 venv에도 별도로 고정 설치했다. 실제 import 경로는 `baselines/vibevoice-1.5b-v1/venv/lib/python3.11/site-packages/meeteval/__init__.py`이다. 기존 학습 env는 수정하지 않았다.
- 서버 기존 env와 전용 venv 각각 CPU 검사 **11/11 통과**. 화자 순열·중간 귀속 변경·KO 문자 CER·누락·무음 삽입을 검증했다. 로컬 Mac은 7개 통과·MeetEval 미설치로 4개 skip으로, 서버 설치와 구분한다.
- Python AST·셸 문법 검사 통과. 설치 후 `environment.freeze.txt` 갱신.
- **GPU 잡 미제출**. 실제 모델 GPU 로드·출력 parity·WER/CER 측정은 아직 실행하지 않았다. 다음 단계는 위 명령으로 GPU 1장 smoke를 제출하는 것이다.
