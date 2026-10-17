## [2026-10-17] task | Stage 3 청취·열람 묶음 — r1 ck2000 vs p9 ck3000 사례 24 개(개선·악화·둘 다 실패·둘 다 잘 됨, 오디오·정답·두 출력)

- Changed:
  - `experiments/cst_listen_bundle.py`(새): 두 run 의 평가 원출력·오류 분해에서 사례를 골라 HTML(오디오 mono·화자별 스테레오, 정답·출력 시간 줄, 화자 틀림·전사 틀림 표시, 지표)을 만든다.
  - raw [[2026-10-17-stage3-spk-head]]: "청취·열람 묶음" 절, `listen-cases.json`
- Reason: 사용자 요청 "잘 된 케이스와 안 좋은 케이스를 직접 보고 듣고 하고 싶어"
- 결과:
  - mxc `results/stage3-listen-r1ck2000-vs-p9ck3000/`(72 MB) → 로컬 T5 `vapkt-review/` 로 받아 브라우저로 열었다. 외부에는 올리지 않았다.
  - 극단 사례는 두 형태다. 한 화자의 긴 말을 두 슬롯으로 쪼개거나, 소수 화자를 다수 화자 슬롯으로 흡수한다.
- Next:
  - 사용자 청취 소감을 받아 오류 원인(음색 변화·채널 누설·웃음 등)을 정리한다.
  - Stage 3 최종 교체 여부를 정한다.
- By: tskim
