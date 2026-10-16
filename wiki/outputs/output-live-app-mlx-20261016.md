---
type: output
status: active
created: 2026-10-16
updated: 2026-10-16
summary: q06-s4-ast 실시간 마이크 전사·번역 앱(App/, MLX 전용·브라우저 UI) — torch 기준과 골든 39/39 토큰 일치, 기본 M4 에서 fp16+q4 초안으로 번역 RTF 0.29·청크 p95 198 ms
sources:
  - '[[output-q06-s4-ast-eval-20261013]]'
  - '[[output-simulst-baselines-20261012]]'
related:
  - '[[decision-ast-monotonic-data]]'
  - '[[decision-stage1-lookahead-320-1040ms]]'
  - '[[decision-cst-training-stages-and-speaker-slots]]'
raw_authors:
  - tskim
---

# 실시간 전사·번역 앱 (App/, MLX, 2026-10-16)

## 무엇인가
- Stage 2 AST 체크포인트 q06-s4-ast 로 마이크(또는 파일) 음성을 실시간으로 받아 적고 번역하는 로컬 앱이다. 사용자 요청(2026-10-15): MacBook Pro 에서 마이크로 직접 시험, 실시간을 위해 추론 최적화.
- 위치: 저장소 `App/` (사용법·측정 `App/README.md`, 설계 문서 `App/docs/`).
- 실행: `bash App/setup.sh` (한 번, `~/.vapkt-app/venv`) → `bash App/run.sh` → 브라우저 `http://127.0.0.1:8796/`.
- 모델 폴더: `~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync/final/` (`.nosync` 라 iCloud 로 안 넘어가므로 Mac 마다 따로 복사).
- 지원: 전사 한·영·독(δ4–10, 기본 δ6), 번역 영→독·영→한·독→영·한→영(학습된 4 방향만, 기본 δ10, 한→영 δ8). 선택적으로 세그먼트마다 final(오프라인) 재디코드.

## 구조
- **추론은 MLX 만 쓴다**(torch·transformers·NeMo 없음). 로그멜 → Nemotron FastConformer 캐시 스트리밍 인코더([56,3], 320 ms 청크) → 어댑터 → Qwen3 thinker(고정 KV).
  - 체크포인트(fp32 safetensors)를 직접 읽어 적재 때 정밀도를 바꾼다. 변환 단계가 없다.
  - 인코더 설정은 공개 Nemotron yaml 에서 뽑은 `App/assets/nemotron_encoder.json`, mel filterbank·window 는 체크포인트에 들어 있다.
- **세그먼트 = 평가 발화 하나.** 에너지 VAD 로 끊고(앞 480 ms, 0.8 s 쉼에서 끝, 최대 25 s), 세그먼트마다 인코더 캐시·KV·prefix 를 새로 하고 끝에 0 꼬리를 붙인다. 평가와 같은 조건이라 토큰이 같다.
- **속도 최적화(모두 토큰 동일):**
  - 투기적 오디오 묶음 — 320 ms 의 4 프레임을 한 forward 로 넣고 첫 방출 프레임에서 KV 포인터를 되돌린다.
  - q4 자기 초안 — 초안 4 토큰을 thinker 가 한 번에 검증한다.
  - fp32 에서는 2단계 lm_head(q8 상한 + 정확성 인증서)를 쓰고, prefix KV 를 캐시한다.
  - final 재디코드는 스트리밍 출력을 초안으로 써서 검증한다.
- **정밀도 자동 선택:** 시작할 때 각 구성의 속도를 재서 실시간 여유 기준을 통과하는 가장 정확한 조합을 고른다(fp32/fp32 → … → fp16/fp16). fp16 thinker 는 잔차 fp32 와 상위 8 후보 fp32 재채점을 쓴다.
- 서버는 aiohttp 이고, 워커 스레드 하나가 MLX 를 전부 맡는다. 토큰이 확정되는 즉시 글자를 보낸다. UI 는 정적 HTML/JS 한 장이며 CDN 을 쓰지 않는다.

## 검증 (Mac mini M4 16 GB)
- **골든:** torch fp32 CPU 로 만든 기준 39 케이스를 썼다(TTS 6 클립, KsponSpeech 3, Europarl-ST 4, K 경계 길이 3; 전사와 번역 4 방향, δ4–10, final 모드).
  - 서버 WebSocket 경로 전체에서 스트리밍 (k,id)·원문·번역·K·final id 가 39/39 일치했다. fp32/fp32, fp16/fp16, fp16/fp32, fp32/fp16 모두 같다.
  - 리뷰 수정 뒤에도 auto(fp16/fp16 + q4 초안)와 fp32 가 39/39 였다.
  - 기준 데이터는 `/Volumes/Samsung_T5/vapkt-models/app-work/golden-s4/`, 생성기는 `App/tools/make_reference.py`.
- **1× 실시간(48 kHz 마이크 경로, auto = fp16/fp16 + q4 초안):**
  - 한→영 번역 60 s 대화: RTF 0.29, 청크 p95 198 ms, 320 ms 를 넘은 청크 0/197.
  - 영어 전사: RTF 0.24, 청크 p95 126 ms.
  - 짧은 클립(한→영, 영→한, 독 전사): 처리 지연 p50 75–119 ms, 세그먼트 마무리 186–385 ms, 첫 글자 0.97–1.32 s.
- **fp32/fp32 (기본 M4):** 정확하지만 번역 몰림 구간에서 청크 p95 가 기준(256 ms)을 넘는다. 그래서 auto 가 fp16 을 고른다. M4 Pro·Max 에서는 fp32 가 뽑힐 것으로 예상한다(미측정).
- 테스트 820 개가 통과한다(개발 env·패키지 venv).
- 적대적 리뷰: 7 관점에서 52 건을 찾아 62/66 검증 판정이 확인했고, 영역별로 고친 뒤 회귀 점검을 했다.

## 알려진 한계
- MacBook Pro, Safari, 실제 방 소음·블루투스 마이크, 배터리·저전력 모드는 측정하지 않았다.
- 에너지 VAD 라 시끄러운 곳(웅성거림 −40 dBFS 등)에서는 발화가 합쳐지거나 길게 이어진다.
- 한→영 번역 모드에서 한국어 전사가 앞부분에서 끊기는 s4 약점은 그대로다([[output-q06-s4-ast-eval-20261013]]).
- `App/` 은 아직 git 에 커밋되지 않았다(2026-10-16). iCloud 데스크탑 동기화로는 넘어간다.
