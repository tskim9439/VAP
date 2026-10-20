# V2-a 두 스트림 입력 — 인코더 아래층 탭 (2026-10-20)

- 질문(사용자): 슬롯 기억은 잘 될 것 같지 않고, LLM 에 화자 인식 과제를 더하기도 그렇다. VibeVoice 는 음성 인코더를 2 개 써서 귀속 문제가 덜한 것 같다. → "그래 학습 전 점검까지 준비해."
- 근거: V2 M0([[2026-10-20-v2-speaker-m0]]) — 같은 창 두 화자를 가르는 정확도가 Nemotron 첫 층 96 %, 마지막 층 66 %, LLM 입력(adapter) 69 % 다.
- VibeVoice-ASR(`modeling_vibevoice_asr.py` `encode_speech`): acoustic tokenizer(σ-VAE 64 차원, 복원용) + semantic tokenizer(128 차원) → 각자 connector → 더해서 LLM 입력. 화자 과제는 없다.

## 구현
- `vapasr/features/online.py` `NemotronOnline.forward(…, taps=(L,))`: 마지막 층과 함께 `enc.layers[L]` 출력(forward hook, 호출마다 뗌)을 같은 프레임 수로 잘라 `(B, 1 + len(taps), K, D)` 로 낸다. taps 를 안 주면 예전과 같다.
- `vapasr/hf/modeling_vapasr.py`:
  - `add_enc_tap(L, hidden)`: `adapter_tap`(adapter 와 같은 LN·Linear·GELU·Linear)을 붙이고 **마지막 Linear 를 0 으로 초기화**한다 → 붙인 직후 LLM 입력은 V1 과 같다. config `enc_tap_layer`·`enc_tap_hidden`(−1 = 끔, V1 체크포인트는 키가 없어 끔).
  - `encode` 는 탭이 켜져 있으면 두 스트림 `(B, 2, K, Din)`, `chunk_embed` = `adapter(마지막 층) + adapter_tap(탭 층)`.
  - 이름이 `adapter` 로 시작해 trainer 의 adapter lr(1e-3) 그룹에 든다.
  - 한 스트림 특징을 두 스트림 모델에 넣으면 AssertionError(특징 캐시·실시간 경로가 조용히 V1 입력을 쓰지 않게).
- 특징 텐서의 스트림 축을 그대로 넘기도록 고쳤다:
  - `offline_decode` 가 `(K, D)`·`(S, K, D)` 둘 다 받는다.
  - `eval_single_turn_asr` final 모드가 스트림 축째 넘긴다.
  - `stream_decode`·`decode_batch`·TAXI·Stage 3/4 평가는 원래 `encode(…)[0]` 을 그대로 넘겨 바꿀 것이 없다.
- 실시간 세션(`live.py` StreamingEncoder, cache-aware)은 아직 탭 프레임을 내지 않는다. 두 스트림 모델이면 시작에서 막는다.
- 학습: `stage3_train.py --enc-tap-layer L [--enc-tap-hidden H]`, 지문 키 `enc_tap`. init 에 이미 탭이 있으면 같은 층이어야 한다. `submit-stage3-train.sh --enc-tap-layer L`.
- `experiments/spk_probe.py` 의 `adapter` 특징은 이제 `chunk_embed`(두 스트림이면 합) — LLM 입력 화자 분리도를 V2-a 에서도 같은 정의로 잰다.
- 탭 층 = L00: M0 에서 모든 단계(원본·Stage 1–4)에 걸쳐 화자 분리가 가장 높다(cen 81.8 %, L03 80.7 %).

## 점검
- 단위 시험 `tests/test_enc_tap.py`(CPU, 가짜 인코더·thinker):
  - 탭 출력 = 층 출력, 자르기·0 패딩, hook 제거
  - 0 초기화 = V1, 재호출·다른 층 거부, 한 스트림 특징 거부, adapter lr 그룹
  - 0 초기화에서 둘째 step 부터 탭 가지 전체가 배운다
  - config·state_dict 왕복
- `tests/test_offline_mode.py` 에 스트림 축 형태 디코드가 같은지 더했다.
- mxc `v2a_check.sh`: 단위 시험 → 학습 dry-run(CPU) → `v2a_check.py`(V1 = st4-m1 ck1500) → 스모크 20 step → 탭 가중치 확인 → 평가 경로 스모크.
- `v2a_check.py`(실제 Nemotron·V1):
  1. 탭 스트림 = L 층 hook 출력
  2. 0 초기화 LLM 입력·배치 디코드 방출 토큰이 V1 과 같다, 직렬 = 배치
  3. 인과성: 앞부분만 넣은 인코딩의 두 스트림이 전체 인코딩과 같다
- `cmp_v2a.py`: 파일럿 대조(대화 단위 짝 bootstrap — cp·agn·귀속 몫·턴 통째·연속 뒤바뀜, 언어별·겹침 창)

## 결과 (2026-10-20, mxc 호스트)
- 단위 시험: mxc 54 통과·3 건너뜀(varlen packing 포함)
- 학습 dry-run(st4-m1 설정 + `--enc-tap-layer 0`): 11 셋 352 항목 × δ 7 개 불일치 0, 지문 `enc_tap {layer 0}`
- 스모크 20 step(Stage 3 최종에서, 197 s):
  - 학습 1,213.9 M(탭 +4.2 M), 손실 정상, 끝 재로드 PARITY OK(누락·추가 키 0 — `adapter_tap` 저장·재로드)
  - 0 초기화 마지막 층이 |w| 평균 0.0023 으로 학습됐다(탭 가지가 쓰이기 시작)
- 평가 경로(스모크 final): Stage 3 평가 직렬 = 배치(한·영 δ10 PARITY PASS), 영↔독 합성 평가 실행
- `v2a_check.py`(V1 = st4-m1 ck1500, 창 12 개, `v2a-check.json`):
  - 탭 스트림 = L00 hook 출력(최대 차이 0), 마지막 층 스트림 = V1(0)
  - 0 초기화 LLM 입력 = V1(0), 배치 디코드 방출 토큰 12/12 같음, 직렬 = 배치
  - 인과성: 앞부분만 넣은 인코딩과 전체 인코딩의 상대 차이가 두 스트림 모두 1e-7–6e-7(fp32 잡음, L00 값 크기 약 350), 최대 차이 프레임이 창 곳곳에 흩어져 있다(자른 끝에 몰리지 않음) → 탭 가지로 미래가 새지 않는다
  - 처음 두 번은 점검 코드 문제로 다시 돌렸다: 언어가 다른 창을 한 배치로 디코드(배치는 한 prefix 만 받는다), 인과성 판정을 절대 차이 1e-4 로 둔 것(L00 값 크기를 몰랐다) → 상대 차이로 고침
- 다음: [[decision-v2-run-scope-20261020]] — V2-a 단독 파일럿 대신 4 화자·창 변경과 한 run 으로 묶는다([[output-v2-run-plan-20261020]])
