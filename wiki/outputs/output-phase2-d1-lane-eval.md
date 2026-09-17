---
title: Phase 2 D1 lane 디코드 평가 — 창 단위 free-running 결과·규약 제약 디코딩·held-out
summary: D1(잡 71599) 모델을 창 단위로 스스로 디코드해 참조와 대조. argmax 는 soft EOT 때문에 lane 이 안 닫혀 구조가 무너지고, lane 규약 제약 디코딩(v2)으로 71631 pooled CER 44→20 %, EOT 재현율 26→72 %. held-out NIKL 2020 pooled CER 33 %, CHiME-6 dev WER 83 %. 뷰어(오디오·타임라인·전사 대조) 링크.
type: output
created: 2026-09-17
updated: 2026-09-17
sources: [raw/sources/experiments/2026-09-17-phase2-d1-lane-eval/]
related: [output-phase2-lane-plan, task-phase2-data-prep, decision-eot-immediate-soft-label]
---

# Phase 2 D1 lane 디코드 평가

**뷰어**: https://claude.ai/code/artifact/416d0f85-a784-4ac3-aa38-c932be938c25 — 창 42 개(seen 24 · held-out 18)의 오디오, lane 별 참조/가설 타임라인(▲ ONSET ◆ EOT ✕ 활동 닫힘, 활동 헤드 확률 띠), lane 별 전사 diff, 청크별 출력 토큰열(EOT 확률 포함), 코퍼스별 지표와 디코드 방식 비교.

## 1. 설정
- 모델 `/soundai/Model/VAPASR/p2-D1/final`(잡 71599, 10 epoch·5,050 step, [[task-phase2-data-prep]]). δ_text=4, R=6, 창 20–40 s, 창 시작은 무음, 마스크 청크가 있는 창 제외, 화자 ≥2.
- 디코더 `vapasr/hf/lane_state.py`: `decode_p2`(stream_decode 규약 + audio 위치 활동 헤드) → `LaneParser`(태그·ONSET·텍스트·EOT → lane 구간). 평가 `experiments/p2_eval_lanes.py`.
- 데이터: seen = 71631·134-1·AMI·ICSI(학습 대화). held-out = **NIKL 일상대화 2020**(Phase 1 은 2021–2025 만 사용, 40 대화 9.1 h, 발화 PCM 을 화자 채널에 배치, 시각은 라벨·토큰은 proxy)과 **CHiME-6 dev**(S02·S09 바이노럴 4 인, 4.46 h). 두 held-out 은 정렬이 없어 토큰 시각이 근사 → 지연·ONSET/EOT 지표는 참고만.
- 지표: 텍스트 CER(KO)/WER(EN) 세 가지(lane 번호 그대로 / 가설→참조 lane 겹침 최대 순열 매핑 / 화자 무시 pooled), ONSET·EOT ±0.4 s 매칭 P/R, lane 정확도(그대로/매핑), 활동 헤드 F1, 일치 토큰 비율, 지연.

## 2. 결과 (창 수: 71631 8 · 134-1 6 · AMI 6 · ICSI 4 · NIKL 10 · CHiME-6 8)

| 코퍼스 | 구분 | 디코드 | 텍스트 pooled | 텍스트 lane 매핑 | ONSET P/R | EOT P/R | lane 정확도(매핑) | 일치 토큰 | 지연 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 71631 KO | seen | argmax | CER 44.2 % | – | 22/19 % | 25/26 % | 77 % | 60 % | 0.78 s |
| 71631 KO | seen | 제약 v2 | **CER 20.3 %** | 36.1 % | 28/56 % | 33/72 % | 89 % | 90 % | 0.49 s |
| 134-1 KO | seen | argmax | CER 51.3 % | – | 26/20 % | 15/17 % | 53 % | 46 % | 0.52 s |
| 134-1 KO | seen | 제약 v2 | **CER 25.0 %** | 40.3 % | 30/57 % | 39/77 % | 76 % | 80 % | 0.45 s |
| AMI EN | seen | argmax | WER 68.1 % | – | 37/20 % | 30/21 % | 32 % | 34 % | 0.59 s |
| AMI EN | seen | 제약 v2 | **WER 32.7 %** | 67.6 % | 40/46 % | 40/52 % | 61 % | 71 % | 0.51 s |
| ICSI EN | seen | argmax | WER 74.2 % | – | 46/22 % | 29/18 % | 52 % | 34 % | 0.83 s |
| ICSI EN | seen | 제약 v2 | **WER 49.8 %** | 60.0 % | 46/51 % | 42/49 % | 47 % | 62 % | 0.57 s |
| NIKL 2020 KO | held-out | 제약 v2 | **CER 33.2 %** | 43.4 % | (1/9 %)* | (1/17 %)* | 81 % | 67 % | 0.53 s |
| CHiME-6 dev EN | held-out | argmax | WER 98.5 % | – | 21/11 % | 12/9 % | 35 % | 15 % | 3.0 s |
| CHiME-6 dev EN | held-out | 제약 v2 | **WER 82.9 %** | 89.6 % | 20/23 % | 21/28 % | 57 % | 28 % | 0.61 s |

\* NIKL 참조는 발화 사이 간격이 0.25 s 미만이라 episode 가 길게 병합돼 ONSET/EOT 수가 적고, 모델은 실제 쉼에서 구간을 나눠 이벤트 정밀도가 낮게 나온다(전사 지표만 유효). 제약 v1(활동 닫힘 3 청크·0.5, 닫힌 lane 은 새 lane 하나만 허용)은 같은 화자 복귀를 막아 텍스트가 끊겼다(71631 pooled CER 84 %) — 뷰어 비교표.

## 3. 무엇이 문제였고 무엇을 고쳤나
1. **soft EOT × greedy**: EOT 위치의 학습 목표가 (EOT: p_end, NEXT: 1−p_end) 라 p_end<0.5 인 곳에서 argmax 가 NEXT 를 골라 lane 이 안 닫힌다. 그러면 다음 화자 ONSET 이 "이미 열린 lane" 에 얹히고(재개 무시), 태그만 있고 payload 가 없는 잡음 태그가 쌓인다(71631 창당 `<SPK_B>` 40 개 vs ONSET 14). → 제약 디코딩: **태그 직후에는 NEXT 를 금지하고 {EOT, 텍스트} 중 고른다**. EOT 재현율 26→72 %.
2. **시작 놓침(δ_onset=0)**: ONSET 은 시작 청크에서 80 ms 증거만으로 내야 해 태그 확률이 0.5 를 못 넘고, 늦게 내는 법은 학습에 없어 발화를 통째로 놓쳤다(71631 일치 토큰 60 %). → 허용 태그의 최대 확률 ≥0.25 면 태그를 낸다. 일치 토큰 90 %, 대신 허위 구간 증가(ONSET 정밀도 28 %).
3. **lane 규약 제약**: 텍스트는 OPEN lane 에서만, ONSET/EOT 는 태그 뒤에서만, 새 화자용 lane 은 lazy-free 가 줄 lane 하나, HELD lane 은 주인이 복귀 가능. 구조 오류(stray EOT·암묵 구간·미닫힘)가 0 이 됐다.
4. **활동 헤드**: 참조 대비 F1 은 0.4–0.66 이지만, 확률을 보면 *모델이 연 lane* 을 따라간다(태그를 내기 전에는 발화 중에도 ≈0.05, ONSET 뒤 0.9, 진짜 끝에서 급락). 즉 독립 VAD 가 아니라 자기 문맥의 lane 상태 추정이다. 닫힘 판정(6 청크·0.3)에는 쓸 만하고, 시작 검출에는 못 쓴다.
5. **lane 번호**: 창 안 등장 순서로 번호가 붙어 첫 발화를 놓치면 이후 번호가 모두 어긋난다. 순열 매핑 지표를 함께 본다(71631 매핑 lane 정확도 89 %). lane 3–6 은 다자 창에서도 거의 안 나온다(AMI/ICSI 4–5 인 창에서 `<SPK_3>` 0 회) — 학습 빈도가 낮다.
6. **텍스트**: 구조가 맞은 창은 전사가 좋다(71631 창 0: pooled CER 4.8 %). held-out KO(NIKL 2020) 33 % 는 seen 71631(20 %) 보다 나쁘지만 같은 자릿수. CHiME-6 는 배경소음·원거리 대화라 83 % 로 별개 문제.

## 4. 다음
- **D1b 학습**: δ_onset 2–3 청크(시작 증거 확보), 태그·ONSET 위치 가중, lane 3–6 노출 확대(stitching 합성 §8), EOT 임계값 대신 hard/soft 혼합 검토. held-out split(7 코퍼스 세션 단위) 구성 후 학습.
- **디코더**: onset_thr·act 닫힘 파라미터 sweep(P/R 곡선), 태그 뒤 EOT 확률을 그대로 p(EOT) 로 내보내 TurnBench 채점, N>R 재배정 세대 표기.
- **평가**: NIKL 2020·CHiME-6 정렬(p2_align) 후 지연·이벤트 지표 재계산, cpWER(외부 재매핑) 연결.
