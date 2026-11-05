---
author_id: tskim
created: 2026-11-05
source_type: rough-note
---

# Qwen3.8-LiveTranslate 를 우리 AMI·TAXI 채점으로 재기 (맥 로컬 전용)

- 사용자 요청(2026-11-05): "Qwen 3.8 live translate 도 동일하게 평가해보고, TAXI 에 대해서도 번역 성능을 평가해보고 싶은데, 어떻게 해야할까? 이 부분에 대해서는 절대 mxc 서버에 올리지말고 로컬에서만 진행해야해."
- TAXI 전송 판단(사용자, 2026-11-05): "추론만 하는거고 공개적으로 배포하는 것이 아니니 상관없을거 같은데?" → TAXI 오디오도 API 추론에 보낸다. BAS 학술 라이선스·Alibaba 보관 정책(학습 미사용·보관 기간은 출처마다 다름)은 확인 전이라고 알렸다.
- 모델: API 전용 `qwen3.8-livetranslate-flash-realtime`(가중치 비공개 — 추론은 그쪽 서버에서만 돈다). 원천 노트 [[source-qwen38-livetranslate]].

## 지키는 것
- 이 폴더의 코드와 Qwen 로그·결과는 mxc 에 올리지 않는다(동기화 대상에서 뺀다). mxc 는 평가 입력을 맥으로 **읽어 오기만** 했다(rsync pull, 15:16 시작).
- 데이터·로그·결과는 T5 `/Volumes/Samsung_T5/vapkt-qwen-lt/` 에만 둔다. TAXI 전사·번역 글자는 저장소에 넣지 않는다(집계만).
- API 키는 저장소 `.env.local` 의 `DASHSCOPE_API_KEY` — 출력하지 않는다.

## 입력(T5 `data/`, mxc 원본과 같은 파일)
- `ami-bench-v1/`: `manifest.json`, `ami-{ihmmix,sdm}.refined.dialogues.jsonl`, `mono/<코퍼스>/<코퍼스>_<회의>.npy`(공식 eval 16 회의 × 2 — 우리 회의 통째 평가가 디코드하는 mono 혼합 그대로).
- `tok-q17-e13/`: E13 final 의 토크나이저 파일만(참조 단어를 우리 평가와 똑같이 만들 때).
- `taxi-natural/`: cstbench-v0.1 TAXI 세션(`sessions.jsonl`, `SES*/mix.wav` — 우리 TAXI 평가 입력).

## 파일
- `qlt_client.py`: 실시간 WebSocket 클라이언트. 100 ms 씩 실시간 속도로 보내고 받은 이벤트를 전부 원본 로그로 남긴다(재채점은 API 없이). 끊기면 보낸 데부터 새 연결.
- `qlt_events.py`: 로그 → 원문 item(화자·구간·원문·감지 언어)·번역 item(조각·받은 시각). 받은 시각은 오디오 시각으로 바꾼다(연결 준비·재연결 공백 제외).
- `qlt_ami.py`: AMI job 목록·채점. `experiments/baseline_ami_ext.py`(Parakeet·VibeVoice 를 우리 틀로 잰 경로)의 참조·마스크·score_ext 와 `meeteval_cpwer.py` 를 그대로 쓴다.
- `qlt_taxi.py`: TAXI job 목록(세션마다 목표 de·en 두 번 — API 는 세션 목표 언어가 하나)·채점(`experiments/cst_stage4_taxi.py score` 그대로).
- `qlt_selftest.py`: 가짜 서버(배선·끊김 점검)와 정답 로그(참조를 Qwen 로그 꼴로 — 채점 경로가 0 오류를 내는지).

## 돌리는 법(맥, vapasr-local)
```bash
P=~/anaconda3/envs/vapasr-local/bin/python; X=raw/sources/experiments/2026-11-05-qwen-livetranslate; T=/Volumes/Samsung_T5/vapkt-qwen-lt
$P $X/qlt_ami.py jobs --data $T/data/ami-bench-v1 --out $T/ami-jobs.jsonl                    # 32 job(eval 16 × IHM-mix·SDM)
$P $X/qlt_client.py --jobs $T/ami-jobs.jsonl --out $T/logs-ami --concurrency 8 --dry-run      # 규모·벽시계 어림
$P $X/qlt_client.py --jobs $T/ami-jobs.jsonl --out $T/logs-ami --limit 1                      # 한 회의 시험
$P $X/qlt_client.py --jobs $T/ami-jobs.jsonl --out $T/logs-ami --concurrency 8                # 전체(끝난 job 은 건너뜀)
$P $X/qlt_ami.py score --data $T/data/ami-bench-v1 --tok $T/data/tok-q17-e13 --logs $T/logs-ami --out $T/res-ami
$P $X/qlt_taxi.py jobs --sessions $T/data/taxi-natural/sessions.jsonl --out $T/taxi-jobs.jsonl
$P $X/qlt_client.py --jobs $T/taxi-jobs.jsonl --out $T/logs-taxi --concurrency 8
$P $X/qlt_taxi.py score --sessions $T/data/taxi-natural/sessions.jsonl --logs $T/logs-taxi --out $T/res-taxi
```

## 같은 것·다른 것(해석 주의)
- 같은 것: 회의·오디오·참조·전사 없는 구간 마스크·정규화·채점 코드(AMI cpWER 표준 = Whisper + meeteval, agn = Whisper 단어마다, DER = 우리 LLM DER 와 같은 80 ms 프레임 셈·마스크; TAXI = cstbench BLEU·chrF·StreamLAAL·빈 턴, 받아쓰기 WER·cp/agn/귀속 몫).
- 다른 것
  - 단어 시각: API 가 주지 않는다 → 원문 item 구간 안에 고르게 놓는다(외부 기준 VibeVoice·Parakeet 과 같은 근사, 마스크·정렬에만 쓴다).
  - 지연: 받은 시각(오디오 기준)이라 네트워크 왕복이 섞인다 → RTT 를 같이 적는다. 우리 δ10 은 모델 방출 시각이다.
  - TAXI: Qwen 은 세션당 두 번(목표 de·en) 돈다. 방향은 Qwen 이 감지한 원문 언어로 고른다(`qwen`), 정답 방향 상한(`qwenref`)도 따로.
  - 화자 번호는 연결마다 새로 매겨진다 — 연결이 끊긴 회의는 화자가 갈라진다(서비스 한계로 그대로 두고 끊김 수를 적는다).
  - API 모델은 바뀔 수 있다 — 수치에 실행 날짜를 붙인다.

- `qlt_tc.py`: meeteval tcpWER·tcORC-WER(collar 5 s, CHiME-8 방식)를 Qwen 과 우리 모델에 똑같이 — 화자 무시 비교는 이것으로 한다(아래 점검 2).

## 점검 기록(2026-11-05, API 없이)
1. 가짜 서버(`qlt_selftest.py mock`): 실시간 4× 속도에서 받은 시각 × 4 = 보낸 오디오(오차 0.1 s 안), 7 s 마다 끊는 서버에서 3 연결로 끝까지 가고 item 시각이 오디오 기준으로 이어진다(재연결 오프셋 `_seg`, 흘리기 시작 `_stream` 을 고친 뒤).
2. AMI 정답 로그(참조 발화를 Qwen 로그 꼴로, eval 16 회의): 표준 cpWER 0.40 · DER 0.61 · tcpWER·tcORC-WER 0.43 — 경로 확인. 그런데 우리 agn(Whisper)은 9.48 이 나왔다.
   단어 시각을 item 구간에 고르게 놓으면 겹친 구간의 화자 사이 단어 순서가 어긋나기 때문이다 → Qwen 의 화자 무시 비교는 agn 이 아니라 tcORC-WER 로 한다.
   (처음 정답 로그는 다듬은 AMI 의 나뉜 조각 text 가 비어 cp 79 % 였다 — 토큰에서 풀어 고침, 점검 도구 쪽 문제.)
3. 우리 E13 + G2b(mxc 결과 행을 읽어 옴, 읽기만): 맥 `meeteval_cpwer.py` 가 eval 24.48 / 36.10(IHM-mix / SDM)을 그대로 재현, 참조 단어 87,010 = Qwen 경로와 같다.
   `qlt_tc.py --kind ours`(같은 슬롯 2 s 묶음): cpWER 24.51 / 36.11 · tcpWER 26.88 / 40.47 · **tcORC-WER 22.34 / 29.38**(Whisper agn 21.92 / 28.57 과 같은 수준).
   단어마다 정규화하면 cpWER 이 24.96 으로 0.5 pt 높아져(숫자 등) 묶음으로 바꿨다.
4. TAXI 정답 로그(86 세션 × 목표 2): BLEU·chrF 100, 받아쓰기 WER 0, 턴 교대 100 % — 같은 언어 출력은 방향 고르기가 걸렀다.
5. 규모: AMI 32 job 오디오 18.1 h(입력 약 0.46 M 토큰, 동시 8 이면 약 2.3 h), TAXI 172 job 1.5 h(약 0.04 M, 약 0.2 h).
