## [2026-10-19] task | Stage 3 최종을 st3-p11-revive checkpoint-3000 으로 교체 — 화자 헤드 + 합성 대화 + 격리 발화 되살리기, δ10 cp 25.28·한국어 귀속 몫 6.9(p10 ck3000 25.62·8.0)

- Changed:
  - [[output-stage3-prep-plan-20261014]]: 결정 블록, P6 표, p10 결정 문장에 교체 표시, summary
  - raw [[2026-10-18-stage3-ko-revive]]: 결정 절. raw [[2026-10-18-stage3-sim-dialogue]]: 교체 표시
  - [[status]]: 2026-10-19 절
- Reason: 사용자 "1 번은 너의 권장대로 하고, 2번 seed 는 굳이 하지 말자"(권장은 교체)
- 결과:
  - `/soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000` 점검:
    - 인코더 키 638, `spk_head` 저장
    - `<EOT>` 등록·차단 해제, tokenizer 있음
    - `config.stage3` cst-v1(지문 `revive`), global step 3,000
  - 감수한 것: EOT 정밀도 −0.03, agn +0.3. seed 반복은 하지 않는다.
- Next:
  - EOT 헛신호는 추론 문턱으로 다시 볼 수 있다.
  - 한국어 화자 귀속을 더 줄이려면 깨끗한 한국어 대화 데이터가 필요하다.
- By: tskim
