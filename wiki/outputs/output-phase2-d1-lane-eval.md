---
title: Phase 2 D1 lane 디코드 평가 — 인코더 저장 사고 수정, 규약 제약 디코딩, held-out 결과와 문제 분석
summary: D1 free-running 평가에서 텍스트가 무너진 1차 원인은 동결한 E2 학습 인코더가 체크포인트에 저장되지 않아 재로드 시 .nemo 원본이 붙은 사고(teacher-forced text 손실 0.08→1.5). 수정 후 seen 71631 pooled CER 18 %(치환 1 %), held-out NIKL 2020 12.6 %, AMI 39 %, CHiME-6 92 %. 남은 문제는 구조(시작 검출 prior·soft EOT·lane 3–6)이며 보완안을 정리.
type: output
created: 2026-09-17
updated: 2026-09-17
sources: [raw/sources/experiments/2026-09-17-phase2-d1-lane-eval/]
related: [output-phase2-lane-plan, task-phase2-data-prep, decision-eot-immediate-soft-label]
---

# Phase 2 D1 lane 디코드 평가

**뷰어**: https://claude.ai/code/artifact/416d0f85-a784-4ac3-aa38-c932be938c25 — 창 42 개(seen 24 · held-out 18)의 오디오, lane 별 참조/가설 타임라인(▲ ONSET ◆ EOT ✕ 활동 닫힘, 활동 헤드 확률 띠), lane 별 전사 diff, 청크별 출력 토큰열(EOT 확률), 코퍼스별 지표, 디코드 방식·수정 전후 비교표. 본문 수치는 **인코더 수정본(`p2-D1/final-enc`)** 기준이다.

## 1. 사고: 동결 인코더가 저장되지 않았다
- E2 는 인코더를 학습했고(`encoder_trainable=True`, safetensors 에 638 키), D1 은 그 인코더를 **동결**해 썼다. `p2_train_hf.py` 가 `config.encoder_trainable=False` 로 두자 `save_pretrained` 가 인코더를 버렸고, 재로드 시 `.nemo` 원본 인코더가 붙었다. thinker 는 E2 인코더 특징에 맞춰 학습됐으므로 재로드 모델은 다른 특징을 받는다.
- 증거: `--check-save`(메모리 모델 vs final 재로드, 같은 배치) — thinker·adapter·활동 헤드 321 키 값 차이 0, **인코더 636/638 키 차이**; overfit32 모델의 teacher-forced text 손실이 메모리 0.34 → 재로드 1.20, D1 seen 창 top-1 0.60 → 수정 후 0.97(손실 3.1 → 0.07).
- 수정: `VapAsrConfig.encoder_saved`(동결이어도 인코더 저장·재로드), `p2_train_hf.py` 는 초기화 체크포인트가 인코더를 갖고 있으면 자동으로 켠다, `experiments/p2_fix_encoder.py` 로 기존 D1 에 E2 인코더를 붙여 `final-enc` 생성(4.8 GB). 재학습 불필요.

## 2. 설정
- 디코더 `vapasr/hf/lane_state.py`: `decode_p2`(stream_decode 규약 + audio 위치 활동 헤드) + `LaneParser`. **규약 제약 디코딩 v2**: 태그 직후 {EOT, 텍스트}(NEXT 금지 — soft EOT 라 p<0.5 여도 닫히게), 텍스트는 OPEN lane 만, ONSET/EOT 는 태그 뒤, 새 화자 lane 은 lazy-free 가 줄 lane 하나, HELD lane 은 주인 복귀 허용, 허용 태그 최대 확률 ≥ onset_thr(0.25) 면 태그, 활동 헤드 6 청크 연속 <0.3 이면 닫음.
- 데이터: seen = 71631·134-1·AMI·ICSI(학습 대화 그대로). held-out = NIKL 일상대화 2020(Phase 1 은 2021–2025 만 사용; 40 대화 9.1 h, 발화 PCM 을 화자 채널에 배치, 토큰 시각 proxy) · CHiME-6 dev(S02·S09 바이노럴 4 인, 4.46 h). 창 20–40 s, 시작은 무음, 마스크 창 제외, 화자 ≥2. 평가 `experiments/p2_eval_lanes.py`.

## 3. 결과 (인코더 수정본, 제약 v2; 괄호는 수정 전)

| 코퍼스 | 구분 | pooled | S / D / I (pooled, 정규화 전) | lane 매핑 | ONSET P/R | EOT P/R | lane 정확도(매핑) | 일치 토큰 | 지연 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 71631 KO | seen | CER 17.9 % (20.3) | 1.2 / 9.8 / 4.6 | 16.5 % | 27/65 % | 36/93 % | 95 % | 88 % | 0.36 s |
| 134-1 KO | seen | CER 11.2 % (25.0) | 2.4 / 0.6 / 5.9 | 18.6 % | 43/90 % | 42/94 % | 77 % | 97 % | 0.35 s |
| AMI EN | seen | WER 39.4 % (32.7) | 17.3 / 13.8 / 9.8 | 67.9 % | 39/71 % | 37/75 % | 47 % | 73 % | 0.39 s |
| ICSI EN | seen | WER 42.9 % (49.8) | 17.0 / 5.4 / 13.2 | 79.4 % | 53/87 % | 50/85 % | 39 % | 79 % | 0.33 s |
| NIKL 2020 KO | held-out | **CER 12.6 %** (33.2) | 5.2 / 3.3 / 3.8 | 18.3 % | (10/68 %)* | (1/17 %)* | 85 % | 86 % | 0.33 s |
| CHiME-6 dev EN | held-out | WER 92.3 % (82.9) | 29.3 / 51.2 / 4.9 | 101 % | 28/33 % | 19/25 % | 43 % | 21 % | – |

같은 모델의 argmax(제약 없음): 71631 32.3 %, 134-1 16.5 %, AMI 38.8 %, ICSI 63.6 %, NIKL 13.1 %, CHiME-6 95.9 %. \* NIKL 참조는 발화 간격이 0.25 s 미만이라 episode 가 길게 병합돼 이벤트 지표는 참고만. teacher-forced(정답 문맥) text top-1: seen 4 코퍼스 0.996–1.0(암기), 태그 0.79–0.92, EOT 0.48–0.64(soft 목표라 상한 ≈ p_end).

## 4. 두 문제의 분석 (사용자 지적 2026-09-17)
### 4.1 첫 블록의 `<SPK_A><ONSET>`
- 실측: 수정 전 모델은 42 창 전부 청크 1 에서 `<SPK_A><ONSET>`(태그 확률 0.27–0.29 로 창·음성과 무관하게 일정). 수정 후 KO 창은 참조 시작을 따라간다(134-1 27→27, 24→24; NIKL 29→30, 33→34 …)지만 EN 회의 창은 여전히 청크 0 에서 시작하고 71631 은 8 창 중 4 창이 이르다.
- 원인 세 겹: (1) **학습 창의 prior** — 창 시작이 hop 격자상의 무음 시점이라 첫 ONSET 이 청크 ≤1 인 창이 11–25 %(NOTSOFAR 37 %), ≤5 가 약 1/3, 중앙값 5–15 청크(`first_onset_train.json`). 모델의 청크 1 태그 확률 0.245(KO)/0.27–0.29(EN) 는 정확히 이 prior 다. (2) **δ_onset=0** — 시작 청크에는 새 화자의 증거가 80 ms 뿐이라 teacher-forced 로도 참조 시작 청크의 태그 확률이 0.04–0.97 로 들쭉날쭉(절반이 0.5 미만). 늦게 내는 법은 학습에 없어 argmax 는 시작을 통째로 놓친다. (3) **디코더 임계값 0.25 가 prior 와 같은 높이** — EN prior(0.27–0.29) 는 넘고 KO(0.245) 는 못 넘어 EN 만 청크 0 에서 발화가 없어도 발사했다. 인코더 사고 때는 음성 증거가 사라져 prior 만 남아 42 창 전부 발사했다.
- 보완안: **디코더** — 임계값을 prior 위로(0.35–0.4; c3 실험 진행) 또는 "무음 prior 대비 상승분"으로 판정, 첫 N 청크는 활동 헤드·에너지 게이트. **데이터** — 창 시작을 무음 안 임의 시점 + 0–5 s 선행 여백으로 뽑아 첫 ONSET 위치 분포를 평탄화(현재 `_windows` 의 hop 격자 규칙 교체), 발화 중간 시작 창은 lane 을 이미 OPEN 으로 두는 규약 추가 검토. **학습** — δ_onset 2–3 청크(160–240 ms 증거; 지연 비용 작음), 태그·ONSET 위치 가중(EOT 가중 2 와 같은 방식), audio 위치에 "δ 청크 안 시작" BCE 헤드(활동 헤드와 같은 구조)를 두어 디코더 트리거를 LM argmax 에서 분리, ONSET 을 1–5 청크 늦게 놓는 증강(p≈0.3)으로 늦은 시작·복구를 학습.

### 4.2 WER/CER
- 1차 원인은 §1 의 인코더 사고였다(NIKL 33→12.6 %, 134-1 25→11 %). 수정 후 seen KO 의 치환은 1–2 %, 나머지는 **누락(71631 9.8 %)·삽입(4.6–5.9 %)** 즉 구조 오류다: 놓친 시작(누락), 허위 구간·같은 말 반복(삽입, 71631 허위 구간 81 개). EN 회의는 치환 17 % 가 추가된다 — 겹침 14–18 %, 1 s 미만 backchannel 이 episode 의 절반, mono 혼합의 SNR. CHiME-6 는 누락 51 %·치환 29 % 로 도메인(원거리·46 % 겹침·소음) 문제이며 E2 인코더가 본 적 없는 조건이다.
- 보완안: (a) 구조부터 — §4.1 의 시작 검출 개선이 누락·삽입을 같이 줄인다; ONSET 정밀도/재현율 곡선으로 임계값 선택. (b) soft EOT 를 디코더에서 확률로 그대로 내보내되(TurnBench 용) 닫힘 판정은 제약 규칙 유지; hard/soft 혼합(p_end<0.3 은 mask) 학습 비교. (c) lane 3–6 노출 — 다자 창에서도 `<SPK_3>` 이 0 회. AMI/ICSI/NOTSOFAR 비중 상향 또는 2 인 대화 stitching 합성(정본 §8)으로 lane 3+ 사례 공급. (d) 겹침·backchannel — 활동 헤드에 겹침 구간 가중, 짧은 episode 의 최소 길이 규칙(0.3 s 미만은 병합) 검토. (e) 인코더 동결 유지 시 held-out KO 12.6 % 는 seen 18 % 와 같은 자릿수라 일반화는 되고 있다; 원거리 도메인은 인코더 미세조정(낮은 lr) 또는 far-field 자료 없이는 어렵다. (f) 세션 단위 held-out split 을 만들고 D1b 부터는 seen 수치 대신 held-out 으로 판단.

## 5. onset_thr 0.35 (c3) 와 다음
- onset_thr 0.25 → 0.35(양쪽 언어 prior 0.245/0.27–0.29 위): pooled 71631 17.9→18.8 %, 134-1 11.2→10.9 %, **AMI 39.4→25.6 %**, ICSI 42.9→42.6 %, NIKL 12.6→14.8 %, CHiME-6 92.3→79.7 %; 허위 구간 71631 81→61, AMI 46→34; ONSET 정밀도 전반 상승(AMI 39→52 %). EN 이 크게 좋아지고 KO 는 1–2 %p 나빠져 0.35 를 기본값으로 둔다(`lane_state.decode_p2`).
- D1b(2026-09-17 사용자 결정: **인코더 해동 학습**, SLURM 제출): `--train-encoder --lr-encoder 1e-5 --delay-onset 2 --tag-weight 2 --start-mode mixed`(임의 시점 시작 창 50 % 추가) + 학습 종료 시 final 재로드 parity 검사 기본. 예산은 인코더 학습 포함 메모리 실측으로 정한다.
- 이후: NIKL 2020·CHiME-6 정렬 후 지연·이벤트 지표 재계산, 세션 단위 held-out split, D1b 는 held-out 으로 판단.
