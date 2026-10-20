## [2026-10-20] task | st4-m2 final 이어 돌리기 끝 — 모든 칸을 채웠다, Stage 4 최종 후보는 그대로 m2a ck2000

- Changed:
  - raw [[2026-10-19-stage4-dialogue-translation]]: "st4-m2a·m2b 결과" 절. 제목에서 "중간"을 뺐다. 재평가 칸과 4–6 단계 표를 채웠다(m2a ck1500·final, m2b ck1500 열).
  - raw `st4-m2a/final-{dlg,xl,stage3}-summary.json`, `st4-m2b/ck1500-stage3-summary.json`·`final-{dlg,xl}-summary.json`, `vibevoice/dialogue-{stream,offline}-score-m2-final.json`
- Reason: final 의 대화 번역·합성·Stage 3 회귀와 m2b ck1500 의 Stage 3 회귀가 OOM 으로 끊겨 `stage4_eval_resume.sh` 로 이어 돌렸다(m2b final 18:31, m2a final 19:21 끝). `vibevoice_cmp_m2.sh` 의 final 채점도 19:23 에 끝났다.
- 결과(δ10):
  - m2a final: Stage 3 회귀 cp 25.47 · agn 18.46, EOT 0.730 · 0.610. 대화 번역 BLEU 37.7 · 41.8 · 46.4 · 28.0, 합성 영어 WER 30.2(δ4 27.2)·강제 진행 창 12, FLEURS 한→영 CER 12.06.
  - m2b final: 합성 영어 WER 42.1(δ4 32.5)·강제 진행 창 22 로 반복 생성이 늘었다. m2b ck1500 의 Stage 3 회귀는 cp 26.55 · agn 18.55 다.
  - VibeVoice 스트리밍 대비(같은 창): m2a final 도 otoSpeech 26.05 대 26.28 로 같다. 한·독은 ck2000 과 ±1 pt 안이다.
  - 체크포인트 고르기는 바뀌지 않는다. m2a ck2000 이 가장 고르다(사용자 결정 대기).
- Next: Stage 4 최종 교체 여부(st4-m1 ck1500 → st4-m2a ck2000)는 사용자가 정한다.
- By: tskim
