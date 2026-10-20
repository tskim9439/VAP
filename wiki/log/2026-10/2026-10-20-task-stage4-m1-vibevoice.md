## [2026-10-20] task | st4-m1 결과(ck1500 최선: TAXI cpWER 40.9·영→독 27.6)와 VibeVoice 외부 기준선 — 같은 영역 대화는 같거나 낫고 TAXI 는 16–19 로 뒤진다

- Changed:
  - raw [[2026-10-19-stage4-dialogue-translation]]: "본 학습 st4-m1 결과"·"VibeVoice-ASR 외부 기준선" 절, `st4-m1/`·`st4m1-vs-p1.txt`·`vibevoice/`(집계만), `stage4_chde_dry.sh`
  - `experiments/baseline_vibevoice.py`(taxi 모드, 잘린 JSON 복구, 반복 붕괴 집계)
  - [[output-stage4-prep-plan-20261019]] S6·summary
- Reason: st4-m1 평가 큐 완료, 사용자 "VibeVoice 와 비교해서 평가해보고 싶어."
- 결과:
  - st4-m1: ck1500 이 TAXI 받아쓰기 cpWER 40.9(st4-p1 ck2000 43.2)·영→독 BLEU 27.6·빈 턴 26/15·합성 영어 WER 26.1 로 가장 좋다. 영어 반복 생성은 ck1500 뒤로 다시 는다(강제 진행 창 9 → 21–23).
  - VibeVoice 대화(δ10 cpWER, 우리 st4-m1 final): 한국어 우리 19.9–32.4 대 오프라인 28.6–73.8·스트리밍 52–105, 독일어 kec 30.2 대 24.0(오프라인)·nsc 22.0 대 27.7·ztc 21.4 대 20.4, 영어 25.8 대 21.1(오프라인)·26.3(스트리밍)
  - VibeVoice TAXI cpWER: 스트리밍 19.2·오프라인 16.4, 우리 st4-m1 ck1500 40.9
  - mxc SSH 가 00:2x – 09:1x 끊겼다(점프 호스트 → mxc 포트 거부). 서버 작업은 그사이 모두 끝났다.
- Next: 체크포인트 선택(사용자, 권고 st4-m1 ck1500), 전화 영역 데이터(CallHome rep_chde)·전화 대역 증강, 영어 반복 생성 원인
- By: tskim
