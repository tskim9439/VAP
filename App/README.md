# VAPKT 실시간 전사·번역 앱 (q06-s4-ast, MLX)

마이크(또는 파일) 음성을 실시간으로 받아 적고 번역하는 로컬 앱이다. 모델은 VapAsr Stage-2 AST 체크포인트
`semcommit-q06-s4-ast` 이고, 추론은 Apple Silicon 의 **MLX 만** 쓴다 (torch·transformers·NeMo 없음). 화면은 브라우저,
서버는 aiohttp 하나다. 네트워크 요청 없이 이 Mac 안에서만 돈다.

- 전사: 한국어·영어·독일어 (δ 4–10, 기본 δ6)
- 번역: 영→독, 영→한, 독→영, 한→영 (학습된 4방향만, 기본 δ10 · 한→영 δ8)
- 세그먼트마다 선택적으로 final(오프라인) 재디코드 — 같은 특징으로 품질 상한을 보여 준다

## 빠른 시작

0. 저장소가 iCloud 로 동기화되는 `~/Desktop`·`~/Documents` 아래에 있으면 먼저 [MacBook 으로 옮기기](#macbook-으로-옮기기)를 본다
   (내려받지 않은 dataless 파일을 읽으면 아무 메시지 없이 멈출 수 있다).
1. 준비물: Apple Silicon Mac (M3/M4 Pro·Max 권장, 메모리 16 GB 이상), **macOS 14 이상**, **arm64 Python 3.11–3.14**
   (`brew install python@3.12` 권장; arm64 conda 3.11+ 도 된다). 모델 폴더
   `~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync/final/` (`config.json`, `model.safetensors`, `tokenizer.json`).
   다른 곳에 있으면 `--model <폴더>` 또는 `VAPKT_APP_MODEL=<폴더>`.
2. 설치 (한 번): `bash App/setup.sh` — `~/.vapkt-app/venv` 를 만들고 고정 버전(`requirements.lock`)을 설치한 뒤 테스트를 돌린다.
   - 사내 CA 가 필요하면 `~/.vapkt-ca-bundle.pem` 을 두면 자동으로 쓴다 (전체 CA 번들이어야 한다 — [문제 해결](#문제-해결)).
   - `--recreate`: venv 다시 만들기 · `--no-test`: 테스트 건너뛰기 · `PYTHON=<arm64 python>`: Python 직접 지정.
   - 새 Mac 에서는 테스트 오디오·토크나이저가 필요한 테스트가 `skipped` 로 나온다. `failed` 만 없으면 정상이다.
3. 실행: `bash App/run.sh` → 브라우저가 `http://127.0.0.1:8796/` (8796–8805 중 첫 빈 포트) 을 연다.
   처음 8–10 초는 "모델 로딩 중" (가중치 적재 + 정밀도·q4 초안 자동 측정 ≈ 8 s + 워밍업 ≈ 1 s; fp32/fp32 고정은 ≈ 4.5 s + 워밍업 3 s; Mac 이 바쁘면 더 걸린다).
   이미 같은 설정의 VAPKT 가 떠 있으면 새로 띄우지 않고 그 주소를 연다 (`--new-instance` 로 하나 더). 끝내기: 터미널에서 Ctrl-C.

모델 없이 화면만 보려면 `bash App/run.sh --engine fake`. `run.sh`·`setup.sh` 는 어느 폴더에서 불러도 된다.

## MacBook 으로 옮기기

**앱 (`App/`, 소스 약 0.6 MB)** — 셋 중 하나:

- **git**: `App/` 가 커밋·푸시된 브랜치를 clone 한다. `git ls-files App` 이 비어 있으면 아직 커밋되지 않은 것이고, clone 에 `App/` 가 없다.
  clone 은 가능하면 iCloud 밖(예: `~/Developer/VAPKT`)에 둔다.
- **iCloud 데스크탑 동기화** (같은 `~/Desktop/VAPKT`): 파일이 dataless(이름만 있고 내용은 iCloud 에)로 올 수 있다.
  사내망에서는 iCloud 다운로드가 실패하므로 **iCloud 가 되는 네트워크(핫스팟 등)에서** Finder "지금 다운로드" 또는
  `brctl download ~/Desktop/VAPKT/App` 로 먼저 내려받는다. 그다음 Finder 에서 `VAPKT` 폴더 우클릭 › **"다운로드된 상태로 유지"**
  (macOS 14.4+)로 고정해 두면 디스크가 모자랄 때 다시 퇴출되지 않는다 ("Mac 저장 공간 최적화"를 통째로 끄면 퇴출된 다른 파일까지
  모두 내려받으므로 디스크 여유를 먼저 본다).
- **직접 복사** (예: T5 경유): `rsync -a --exclude __pycache__ --exclude .pytest_cache App/ <대상>/App/` (또는 Finder 복사).
  폴더 이름 `App` 만 유지하면 위치는 어디든 된다.

dataless 점검: `ls -lOd App App/setup.sh App/run.sh` 에 `dataless` 가 보이면 먼저 내려받는다. `setup.sh`·`run.sh` 는
시작할 때 `App/` 전체를 검사해 dataless 가 있으면 목록을 보이고 멈추지만, **`setup.sh`·`run.sh` 자체가 dataless 면 bash 가 읽다가
멈추므로** 스크립트가 막아 줄 수 없다. 점검에 `ls -R`·`find` 처럼 폴더를 여는 명령은 쓰지 않는다 (dataless 폴더를 열면 멈출 수 있다).

**모델 (약 4.9 GB)** — `semcommit-q06-s4-ast.nosync` 는 이름의 `.nosync` 때문에 iCloud 로 동기화되지 않는다. 따로 복사한다:
`~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync/final/` 또는 `/Volumes/Samsung_T5/vapkt-models/semcommit-q06-s4-ast/` 에 두면
자동으로 찾고, 다른 곳이면 `--model <폴더>` 또는 `VAPKT_APP_MODEL=<폴더>` (상대 경로는 `run.sh` 를 부른 폴더 기준).

## 사용법

- **원문 언어 / 번역 / 지연 δ** 를 고르고 **● 마이크 시작** (Space). **■ 중단** (Esc) 은 남은 처리를 버린다.
  δ 옆에 알고리즘 지연(≈ 80δ + 96 ms)과 번역 LAAL(평가값)이 보인다.
- 말을 멈춘 뒤 마지막 구절(특히 번역)은 끝 멈춤(기본 0.8 s) + 약 0.3–0.45 s 뒤에 확정된다 (기본 M4 측정).
  더 빨리 보려면 고급 › 끝 멈춤을 줄인다 (대신 세그먼트가 더 잘게 나뉜다).
- **파일**: 실시간 재생(1×) · 2× · 4× · 최대. "평가 방식(분할 없음)" 은 30 s 이하 파일을 평가와 똑같이(무분할, 이득 끔, 끝에 1 s 0)
  디코드한다. 16 kHz WAV 는 브라우저가 보낸 표본으로 골든 39/39 가 오프라인 기준과 토큰까지 같았다.
  다른 샘플레이트는 리샘플이, WAV 가 아닌 형식은 브라우저 디코더가 끼므로 기준과 다를 수 있다.
- **고급**: 끝 멈춤(기본 800 ms), 최대 길이(기본 25 s), 서버 음량 자동 맞춤(기본 켬), 세그먼트 끝 final 재디코드(기본 끔),
  토큰 로그, 세션 오디오 저장, 브라우저 에코 제거·잡음 억제·자동 이득(기본 끔).
- 상태 줄: 처리 지연 p50/p95, 밀림(backlog), RTF, 청크 ms / 320 ms, KV %, 메모리. 결과는 **텍스트 복사 / JSON 저장**.
- 세션 기록(설정·세그먼트·텍스트, 오디오 없음): `~/.vapkt-app/sessions/<id>.jsonl`, 서버 로그: `~/.vapkt-app/logs/`.

## 실행 옵션 (`run.sh` 인자는 `server.py` 로 그대로 간다)

| 옵션 | 뜻 |
|---|---|
| `--precision auto\|fp32\|fp16` | thinker 정밀도 (기본 auto, 아래) |
| `--encoder-precision auto\|fp32\|fp16` | 인코더 정밀도 (기본 auto) |
| `--model DIR` | 체크포인트 폴더 (`<dir>` 또는 `<dir>/final`; 상대 경로는 부른 폴더 기준) |
| `--port N` / `--no-browser` | 포트 고정 (8765·8790 금지) / 브라우저 열지 않기 |
| `--kv-capacity 2048` | KV 위치 수 (세그먼트 ≤ 31 s 에 충분) |
| `--no-speculative` | 투기적 오디오 묶음 끄기 (느려지기만 하고 결과는 같다) |
| `--draft auto\|q4\|off` | q4 자기 초안 검증 디코드 (기본 auto: 시작 때 재서 이득일 때만 켠다; 내는 토큰은 thinker argmax 그대로) |
| `--new-instance` | 이미 실행 중인 VAPKT 가 있어도 새로 띄운다 (모델 사본이 하나 더 메모리에 올라간다) |
| `--save-audio` | 입력 오디오를 `~/.vapkt-app/sessions/<id>.wav` 로 |
| `--debug` | 로그를 DEBUG 수준으로 (100 ms 넘는 워커 호출 등) |
| `--host ADDR` | 기본 `127.0.0.1`. 바꾸지 않는다 — localhost 가 아니면 브라우저가 마이크를 막고(보안 컨텍스트), 앱이 이 Mac 밖으로 열린다 |
| `--engine fake` | 모델 없는 가짜 엔진 (UI·프로토콜 시험용) |

환경 변수: `VAPKT_APP_MODEL` (= `--model`), `VAPKT_APP_HOME` (기본 `~/.vapkt-app`: venv·로그·세션·pycache 위치 —
`setup.sh`·`run.sh` 에 같은 값을 준다), `PYTHON` (`setup.sh` 만: 쓸 Python).

## 정밀도 자동 선택 (`auto`)

시작할 때 thinker(fp32·fp16)와 인코더(fp32·fp16)를 하나씩 재고(q=1 토큰 forward, 투기 q=8 forward, q4 초안 한 스텝·검증 forward,
인코더 320 ms 스텝), **정확한 순서** (fp32/fp32 → fp32 thinker·fp16 인코더 → fp16/fp16 → fp16/fp32) 로 다음 실시간 여유 기준을
처음 통과하는 조합을 고른다 (fp16 thinker 에서는 인코더 정밀도가 정확도에 차이가 없어 더 빠른 fp16 인코더가 먼저다).
청크(320 ms) 비용 예측은 골든 번역 통계(청크당 토큰 p95 13, 평균 2.85)를 쓴다:

- p95 = 인코더 + 2·t8 + 13·t1 ≤ 256 ms (320 의 80 %)
- 평균 = 인코더 + 1.73·t8 + 2.85·t1 ≤ 160 ms (RTF 0.5)
- 몰림 = 인코더 + t8 + 23·t1 ≤ 450 ms
- q4 초안이 켜지는 조합은 토큰 비용을 검증 라운드(4·t_d + t_v, 한 번에 최대 5 토큰)로 센다. 초안은 4·t_d + t_v ≤ 3.5·t1 일 때만 켠다.

하나도 통과하지 못하면 가장 빠른 조합을 고른다. 결정과 측정값은 터미널·서버 로그와 화면 모델 칩의 툴팁에 남는다
(예: `자동 정밀도: thinker fp16 · 인코더 fp16 (실시간 여유 기준 통과; 예측 청크 p95 206 ms·평균 89 ms / 320 ms (q4 초안); 측정 7698 ms) · q4 초안 켬`).
측정에 약 7–8 s 가 더 걸린다. 고정하려면 `--precision fp32 --encoder-precision fp32` 처럼 준다.

- 기본 M4 (Mac mini 16 GB): fp32 조합은 기준 미달, **fp16/fp16 + q4 초안** 이 통과한다 (예측 p95 206 ms). fp32 는 초안을 켜도
  번역 청크 p95 가 기준(256 ms)을 넘는다 (아래 표).
- M4 Pro·Max 급: fp32/fp32 가 통과할 것으로 예상 (측정 안 함 — 시작 로그로 확인).
- 선택은 시작할 때 한 번만 한다. 배터리·저전력 모드에서는 측정하지 않았다 — 전원 연결 상태에서 시작한 뒤 배터리로 바꿨을 때
  상태 줄의 "320 ms 넘은 청크" 가 늘면 `--precision fp16 --encoder-precision fp16` 으로 다시 시작한다.
- 정확도: 골든 39 케이스에서 fp32 는 torch 기준과 토큰이 완전히 같고, fp16 조합도 스트리밍 39/39 가 같다.
  fp16 은 결정 마진을 중앙값 2e-3 정도 흔들 뿐이라 기준 마진이 그보다 작은 근소한 경우에만 갈릴 수 있다.

## 구조

```
브라우저 (web/: index.html app.js worklet.js pacer.js style.css) ── WebSocket /ws (f32 PCM ↑, JSON 이벤트 ↓)
server.py  이벤트 루프: 리샘플(soxr) → VAD·분할·이득 (audio_pipeline.py) → 워커 큐 / 텍스트 조립 (text_stream.py), 통계 4 Hz
           워커 스레드 1개 (MLX 전부): engine/ — 로그멜(frontend.py) → FastConformer 스트리밍 인코더(encoder.py)
           → 어댑터 + Qwen3 thinker 고정 KV (thinker.py) → 세그먼트 디코더 (decoder.py: 라운드·투기 묶음·q4 초안 검증·cap·flush·final)
           디코더가 토큰을 확정하는 즉시 글자를 보낸다 (delta partial:true); 처리 지연은 그 delta 를 보내는 순간 기준
```

세그먼트 = 평가 발화 하나 (새 인코더 캐시·KV·prefix, 끝에 0 꼬리). 자세한 규칙은 설계 문서(decode-contract, encoder/thinker/audio-ui spec).

## 측정 (Mac mini M4 16 GB, mlx 0.32.2, 2026-10-16)

정확도 — 골든 39 케이스 (torch fp32 CPU 기준, 무분할·16 kHz·끝 1 s 0), 서버 WebSocket 경로 전체:

| thinker/인코더 | 스트리밍 (k,id) 같음 | 원문·번역 텍스트 | K | final 재디코드 id |
|---|---|---|---|---|
| fp32/fp32 | 39/39 | 39/39 | 39/39 | 39/39 |
| fp16/fp16 (auto) | 39/39 | 39/39 | 39/39 | 39/39 |
| fp16/fp32 | 39/39 | 39/39 | 39/39 | 39/39 |
| fp32/fp16 | 39/39 | 39/39 | 39/39 | 39/39 |

실시간 1× — 통합 측정 W6 (q4 초안·토큰 스트리밍 **전**; 48 kHz 마이크 경로 → soxr → VAD·이득 → 엔진; TTS 클립 6개 × 전사·번역,
60 s 대화 2개 × 전사·번역, 쉼 없는 44 s 연속 × 전사·번역, 설정마다 전사 9 세션 217 s + 번역 9 세션 217 s):

| 설정 | 모드 | RTF | 청크 p95 (ms) | 320 ms 넘은 청크 | 처리 지연 p50 / p95 (ms) | 밀림 끝 |
|---|---|---|---|---|---|---|
| fp16/fp16 (auto) | 전사 | 0.23–0.28 | 121–154 | 0 / 680 | 73–93 / 132–162 | 0 |
| fp16/fp16 (auto) | 번역 | 0.33–0.38 | 168–350 | 26 / 680 (3.8 %) | 79–113 / 201–333 | 0 |
| fp32/fp32 | 전사 | 0.38–0.44 | 194–238 | 0 / 680 | 99–139 / 198–244 | 0 |
| fp32/fp32 | 번역 | 0.53–0.63 | 303–620 | 93 / 680 (13.7 %) | 101–203 / 366–634 | 0 |

- 두 설정의 세그먼트 경계·원문·번역은 20 세션 58 세그먼트 모두 같았고, final 재디코드 결과도 14/14 같았다.
- 세그먼트 끝 마무리(finalize) p50: auto 전사 60–188 ms · 번역 185–414 ms. 이벤트 루프 지연 p99 ≤ 4.1 ms (auto).
- final 재디코드를 켜도 auto 전사의 처리 지연 p95 는 그대로 (152 vs 156 ms), 19 s 세그먼트 final 1.37 s (fp32 2.5 s).
- 적재: auto 4.9–5.9 s (측정 4.3–5.2 s 포함) + 워밍업 0.8–1.0 s, 고정 정밀도 1.8–1.9 s + 워밍업. 다른 무거운 프로세스와 겹치면 7.7 s 까지 봤다.
- 메모리 (MLX): fp16/fp16 상주 3.25 GB · 실행 중 최대 3.85 GB · 프로세스 3.4–4.4 GB, fp32/fp32 상주 5.22 GB · 프로세스 5.3–6.4 GB.
  적재 순간 최대 6.65 GB (auto) / 5.64 GB (fp32). MLX 빈 버퍼 캐시는 1 GB 로 묶는다 (안 묶으면 fp32 프로세스가 11 GB 까지 커졌다).

회귀 점검 (q4 초안·토큰 스트리밍·VAD 끝 판정 수정 **뒤**, 같은 Mac mini, 바쁜 상태):

- 골든: auto(fp16/fp16 + 초안)·fp32/fp32 모두 스트리밍 (k,id)·원문·번역·K·final id 39/39. 39 케이스 전체 auto 108 s (전 122 s),
  fp32 145 s (전 248 s); fp32 final 재디코드 중앙값 277 ms (전 1477 ms, lookahead 11).
- 1× conv-ko 번역 (한→영 δ8): RTF 0.29, 청크 p95 198 ms, 320 ms 넘은 청크 0/197 (전 0.36 · 312 ms · 8/199).
  conv-en 전사: RTF 0.24, 청크 p95 126 ms, 0/188 (전 0.23 · 121 ms · 0/188).
- 처리 지연은 이제 결과를 **보내는 순간** 기준이라 위 표(계산 끝 추정)보다 p50 이 20–30 ms 크게 나온다 (conv-en 전사 101/154 ms,
  conv-ko 번역 112/232 ms). 글자는 토큰마다 바로 가므로 실제 도착은 더 빠르다: 같은 세그먼트에서 번역 글자 p50 171 ms, 원문 31–72 ms 일찍.
- 세그먼트 경계: 끝 판정(80 Hz 고역 통과·바닥 10 백분위·5 프레임 중 3 유성)을 고쳐 0.8 s 이상 쉼에서 나뉜다. conv-ko 의
  soft_max 문장 중간 절단이 없어지고, conv-en 의 마지막 두 문장이 따로 나뉜다.
- 메모리: auto 적재 최대 5.79 GB · 상주 3.61 GB (초안 0.335 GB 포함), fp32/fp32 적재 최대 6.23 GB · 상주 5.79 GB
  (초안 0.372 GB + q8 헤드 0.176 GB). 적재: auto 8.2–8.9 s + 워밍업 1.1 s, 고정 fp32 4.5 s + 워밍업 3.3 s.

## MacBook Pro 에서 확인하기

```bash
bash App/setup.sh && bash App/run.sh            # 시작 로그의 "자동 정밀도: …" 줄과 주소(포트)를 본다
# 도구는 venv Python 으로 돌리고, .pyc 가 저장소(iCloud) 안에 생기지 않게 PYTHONPYCACHEPREFIX 를 준다 (터미널마다 한 번)
H="${VAPKT_APP_HOME:-$HOME/.vapkt-app}"; export PYTHONPYCACHEPREFIX="$H/pycache"
# 엔진만 따로 (서버 없이): 마이크로 벤치 + 파일 실시간 재생
"$H/venv/bin/python" App/tools/bench.py --micro
"$H/venv/bin/python" App/tools/bench.py --wav <파일.wav> --src English --tgt German --pace 1 --precision fp32
# 서버 경로 실측: 다른 터미널에서 (서버가 떠 있는 동안). --port 는 run.sh 가 찍은 주소의 포트 (기본 8796)
"$H/venv/bin/python" App/tools/client_file.py <파일.wav> --port 8796 --src Korean --source mic --speed 1 --json out.json
```

골든 대조(torch 기준 파일이 있을 때): `client_file.py --golden <golden-s4 폴더> --speed 0 --final-redecode --port <N>`.

## 문제 해결

- **마이크가 켜지지 않는다 / "마이크를 시작하지 못했습니다" / "마이크 입력이 0 입니다"**: macOS 시스템 설정 › 개인정보 보호 및 보안 ›
  마이크에서 브라우저를 허용한 뒤 **브라우저를 완전히 종료했다가 다시 연다** (허용은 다시 연 뒤에 적용된다).
  Chrome 은 사이트마다 따로 묻는다: 주소창 왼쪽 사이트 정보 › 마이크 › 허용 (`http://127.0.0.1:<포트>` 별로; 포트가 바뀌면 다시 묻는다).
- **`setup.sh` 가 Python 을 못 찾거나 원하지 않는 Python 을 고른다**: "건너뜀" 줄에 이유가 나온다.
  `PYTHON=/opt/homebrew/bin/python3.12 bash App/setup.sh --recreate`.
- **설치(pip) 실패**: pip 오류에 `Requires-Python` / `No matching distribution` 이 있으면 Python(3.11–3.14)·macOS(14+) 버전 문제다.
  `SSL`·`CERTIFICATE` 오류면 사내 TLS 가로채기다 — `~/.vapkt-ca-bundle.pem` 에 **전체 CA 번들**(일반 루트 인증서 + 사내 루트)을 둔다;
  프록시 인증서 하나만 두면 다른 사이트 검증이 깨진다. conda Python 으로 만든 venv 의 pip(23.x)는 macOS 키체인 인증서를 쓰지 않으므로
  Homebrew Python(pip 24.2+)으로 `--recreate` 하는 편이 낫다.
- **"기존 venv 를 쓸 수 없습니다" / "venv 의 기본 Python 이 없어졌습니다"**: venv 를 만든 Python 이 지워졌거나(예: `brew upgrade`·`brew uninstall`)
  버전이 맞지 않는다 → `bash App/setup.sh --recreate`.
- **"dataless 파일이 있어 실행하지 않습니다"**: [MacBook 으로 옮기기](#macbook-으로-옮기기) — iCloud 가 되는 네트워크에서 내려받은 뒤 다시 실행.
- **포트**: 8796–8805 중 첫 빈 포트를 쓴다 (8765·8790 은 쓰지 않는다). 실제 주소는 `run.sh` 출력 "VAPKT 실시간 전사·번역: http://…" 에 있다.
  도구(`client_file.py`)에는 그 포트를 `--port <N>` 으로 준다. 모두 차 있으면 `--port N` 으로 직접 고른다.
- **모델을 못 찾는다**: `--model <폴더>` (`<dir>` 또는 `<dir>/final`, `config.json`·`tokenizer.json`·`model.safetensors` 필요).
- **느리다 / "320 ms 넘은 청크" 가 많다**: 시작 로그의 "자동 정밀도" 줄을 본다. `--precision fp16 --encoder-precision fp16` 으로 고정하고,
  final 재디코드를 끈다.
- **로그**: `~/.vapkt-app/logs/server-*.log` (`VAPKT_APP_HOME` 을 바꿨으면 그 아래). 더 자세히: `--debug`.

## 알려진 한계

- 기본 M4 에서 번역은 몰림 구간(청크당 토큰 13개 이상)에서 한 청크가 320 ms 를 넘기도 한다 (밀림은 곧 0 으로 돌아온다).
  fp32 는 초안을 켜도 기본 M4 의 번역 여유 기준(청크 p95 256 ms)을 넘는다 — auto 가 fp16 을 고르는 이유.
- 세그먼트 시작 때 pre-roll(≈0.5–1 s)을 한 번에 처리하므로 밀림이 잠깐 튄다 (지연 백분위에서는 뺀다).
- final 재디코드는 기본 끔. 켜면 세그먼트마다 prefill 한 번 + 토큰 forward 가 더 든다 (fp16: 19 s 세그먼트 ≈ 1.4 s).
  스트리밍 일이 없는 틈에 한 걸음씩 끼워 넣으므로 전사 지연은 그대로지만, 기본 M4 의 fp32 번역과 함께 쓰면 밀린다.
- Safari·블루투스 마이크·실제 방 소음, 배터리·저전력 모드는 이 Mac 에서 확인하지 못했다.
- 세그먼트는 최대 25 s(설정 10–30 s); 넘으면 조용한 지점에서 자른다.
