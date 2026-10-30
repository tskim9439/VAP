## [2026-10-30] task | VibeVoice-Streaming 7B Flash Attention 대조 — 같은 조각 18 개 cp 42.07(FA2) 대 36.60·41.33(sdpa), 논문 차이를 설명하지 못함

- Changed:
  - 위키: [[output-ami-external-comparison-20261027]](Flash Attention 대조 절), [[output-ami-results-summary-20261030]](2 절)
  - 서버: `VAPKT-data/baselines/vibevoice-fa-v1/venv`(별도 env — 학습 env 는 그대로), 결과 `VAPKT-data/tmp/e11-check/ext/vv-seg480-t28-{sdpa,fa2}`
- Reason:
  - 사용자 "flash-attn 은 서버에 바로 설치 가능하지 않아?" → "별도 env 를 만들어 대조해보자". 저자 측이 Flash Attention 경로를 언급해, 논문 IHM 수치와의 차이가 커널 때문인지 보았다.
  - torch 2.8.0 + 공식 flash-attn 2.8.3.post2 휠을 vapasr env 위 venv 에 깔았다(`setup_vibevoice_fa_env.sh`). 커널 점검에서 sdpa 와의 최대 차는 0.002(bf16)였다.
  - IHM-mix 480 s 조각 18 개(4 회의): cp 2.9 + sdpa 41.33, 2.8 + sdpa 36.60, 2.8 + FA2 42.07. agn 은 24.05–24.37 로 그대로다.
  - 차이는 설정 쌍마다 2–4 조각이 "화자 가름 ↔ 한 명으로 몰림"을 오가는 데서 나온다(방향 제각각). 앞서 본 화자 표지의 불안정과 같은 꼴이다.
- Next:
  - 전 조각(318 개)을 FA2 로 다시 재도 결론이 바뀔 가능성은 낮다고 본다. 할지는 사용자가 정한다(하면 묶음으로 낸다).
- By: tskim
