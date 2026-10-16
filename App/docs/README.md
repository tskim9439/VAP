# App 설계 문서 (2026-10-16 스냅숏)

`App/` 을 만들 때 쓴 명세와 통합 측정 보고서다. 영어 원문 그대로 둔다. **정본은 코드다** — 아래 "달라진 점"과 어긋나면 코드가 맞다.

| 문서 | 내용 |
|---|---|
| `decode-contract.md` | s4 디코드 규칙(특수 토큰·차단 id·prefix·라운드·cap·flush·K·꼬리·final 모드·출력 분리)과 torch 기준 하네스 만드는 법 |
| `encoder-spec.md` | 로그멜 + Nemotron FastConformer 캐시 스트리밍(`[56,3]`) MLX 포팅 명세 |
| `thinker-spec.md` | 어댑터 + Qwen3 thinker MLX 포팅, 고정 KV, 투기적 오디오 묶음(정확성 논증), 수치 정밀도 |
| `audio-ui-spec.md` | 세그먼트 분할·VAD·이득, 지연 회계, 브라우저 캡처, WebSocket 프로토콜, 서버 스레드, UI, 패키징 |
| `app-design.md` | 모듈 배치·API 계약(구현 분담용) |
| `integration-report-20261016.md` | 통합 측정(골든 대조, 1× 실시간, 정밀도 자동 선택 근거, 메모리 캐시 버그) — 리뷰 수정 **전** 수치 |

`[critic]` 표시는 적대적 검토에서 바로잡은 항목이다. 문서 안의 `scratchpad/…` 경로(검증 스크립트·로그)는 만들 때의 임시 폴더라 지금은 없다.

## 검증 데이터

- 골든(torch fp32 CPU 기준) 39 케이스: `/Volumes/Samsung_T5/vapkt-models/app-work/golden-s4/` (`index.json`, 케이스마다 `meta.json`·`wav.npy`·`feats.npy`·`mel.npy`).
  다시 만들기: `App/tools/make_reference.py` (연구 env `vapasr-local` 전용 — torch·NeMo·vapasr 사용).
- 대조: `App/tools/verify_encoder.py`, `App/tools/verify_engine.py`, 서버 경로 전체는 `App/tools/client_file.py --golden <폴더> --speed 0 --final-redecode`.

## 리뷰 수정 뒤 달라진 점 (명세에 아직 반영 안 됨)

- **thinker / decoder**
  - q4 자기 초안(`thinker.Draft`) + 라운드 끝 투기(`decoder._finish_round`): 초안 4 토큰 → 검증 forward 1 회. 내는 토큰은 thinker argmax 그대로.
    `--draft auto|q4|off`, auto 는 `4·t_d + t_v ≤ 3.5·t1` 일 때만 켠다.
  - fp32 thinker 는 2단계 lm_head(`TwoStageHead`: q8 상한 → top-64 fp32 재채점 + 정확성 인증서, 실패 시 전체 lm_head)를 쓴다.
    "forward 마다 동기 1 회"에 인증서 실패 때 1 회가 더해질 수 있다.
  - prefix KV 캐시(`PrefixCache`, (src,tgt,δ) 별 LRU 16).
  - final 재디코드(`FinalJob`)는 스트리밍 출력을 초안으로 한 번에 최대 16 행을 검증한다(`final_lookahead`, fp32 thinker 기본 11, fp16 은 0).
  - 토큰 확정 즉시 콜백(`SegmentDecoder.on_token`) → 서버가 `delta{partial:true}` 로 바로 보낸다.
  - auto 정밀도 순서: fp32/fp32 → fp32·fp16인코더 → **fp16/fp16 → fp16/fp32** (fp16 thinker 에서는 인코더 정밀도가 정확도에 차이 없음).
- **audio_pipeline** (`audio-ui-spec` §1.3–1.5 와 다름)
  - VAD 특징: 프레임 간 이어지는 1차 80 Hz IIR 고역 통과(`HighPassDb`) — rfft 근사는 50/60 Hz 험을 못 막았다.
  - 끝 판정 바닥 N = 최근 비영(非零) 94 프레임의 10 백분위, `T_off = N+6`; 시작 `T_on = min(중앙값+6, N+10)`;
    유성 = 최근 5 프레임 중 3 이상(4 ms 하위 블록 중앙값 검사 포함). 디지털 무음(≥ 64 표본 |x| ≤ 1e-6) 프레임은 바닥·문턱을 얼린다.
  - 이득: 양(+)의 이득은 발화 증거(유성 ≥ 30 프레임, 발화 레벨 − 조용한 바닥 ≥ 15 dB)가 있을 때만, 첫 세그먼트는 조용한 시작일 때만.
  - `hold_from_s` 기본 = `hard_max − 0.64 − 0.064` (24.296 s; 이전 23 s).
- **server / web** (`audio-ui-spec` §3–§4, §8 과 다름)
  - 모든 세션 이벤트에 `sid`; `delta.partial`; 처리 지연(`proc_lag`)은 delta 를 **보내는 순간** 기준.
  - 텍스트 없는 세그먼트는 숨긴다(final 재디코드 대기 중이면서 유성 ≥ 6 프레임이면 표시).
  - `quiet_input`(최대 이득) 경고는 세그먼트 열림·발화 상태·원문 2 단어 이상일 때만.
  - 빠른/평가 파일 경로는 브라우저가 WAV 를 직접 읽어 원래 샘플레이트로 보낸다(리샘플은 서버 soxr 한 번). 숨은 탭에서도 `pacer.js` 워커로 속도 유지.
  - 이미 떠 있는 같은 설정의 VAPKT 를 재사용(`--new-instance` 로 하나 더), SIGHUP 정상 종료.
  - Python 3.11–3.14, macOS 14+ (`setup.sh`·`server.py` 둘 다 검사).
