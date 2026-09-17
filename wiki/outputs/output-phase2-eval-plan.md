---
title: Phase 2 평가 계획 v2 — 인식·화자 귀속·턴 구간을 분리한 지표와 객관 held-out 평가셋
summary: lane 번호로 CER/WER 를 재던 1차 평가의 결함(화자 오배정·초반 순서 어긋남이 전사 점수에 전가) 을 고쳐, 인식(화자 무관 utterance-assigned WER/ORC-WER)·화자 귀속(cpWER, DER, lane 전환)·턴 구간(ONSET/EOT/턴 교대 P/R·지연·turn 내 오방출)·지연의 4 축으로 나누고, 학습에 쓰지 않은 NIKL 2020·TurnBench dev·CHiME-6/DiPCo 와 다음 학습부터 떼어 둘 세션 split 로 객관 평가셋을 만든다.
type: output
created: 2026-09-17
updated: 2026-09-17
sources: [raw/sources/experiments/2026-09-17-phase2-d1-lane-eval/]
related: [output-phase2-lane-plan, output-phase2-d1-lane-eval, task-phase2-data-prep]
---

# Phase 2 평가 계획 v2

## 0. 왜 바꾸나 (사용자 지적 2026-09-17)
1차 평가(`p2_eval_lanes.py`)는 lane 번호가 같은 참조·가설 텍스트를 대조했다. 그래서 (a) 전사는 맞는데 lane 을 잘못 붙이면 그 발화가 누락+삽입으로 두 번 벌점을 받고, (b) 창 초반에 화자 순서가 한 번 어긋나면 이후 전부가 뒤바뀌어 CER/WER 가 무너진다. 순열 매핑(diarization 식)을 넣어도 세션 중간의 lane 전환은 여전히 텍스트 점수로 새어 들어간다. 발화 시작·끝·턴 교대는 ±0.4 s 매칭 하나뿐이고, "학습에 없던 자료" 는 NIKL 2020·CHiME-6 두 개에 창 표본도 손으로 골랐다. 평가 축을 분리하고 평가셋을 고정한다.

## 1. 지표 — 네 축
가설의 단위는 디코더가 내는 **구간**(lane, 시작 청크, 끝 청크, 텍스트) 이고 참조 단위는 보정된 **발화**(화자, 시작, 끝, 텍스트) 다. 텍스트 정규화는 asr-tn v1(KO 글자, EN 단어).

| 축 | 지표 | 정의 | 무엇에 둔감한가 |
|---|---|---|---|
| A. 인식(화자 무관) | **utterance-assigned WER/CER** | 가설 구간의 단어를 시간 겹침이 가장 큰 참조 발화에 붙인다(lane 무시). 참조 발화별 편집거리 합 / 참조 길이. 겹치는 참조가 없으면 삽입 | lane 오배정·순서 어긋남 |
| | ORC-WER (MeetEval) | 참조 발화를 가설 스트림에 최적 배정한 WER — 표준 구현과 대조용 | 위와 같음 |
| | pooled WER/CER | 화자 무시·시간순 결합(현재 지표; 참고) | lane, 그러나 겹침 구간 순서에 민감 |
| B. 화자 귀속 | **cpWER** | 세션 단위로 lane→화자 최적 순열을 한 번 정한 뒤 화자별 결합 텍스트 대조. `cpWER − ORC-WER` = 화자 귀속 비용 | 전역 번호 뒤바뀜(초반 한 번 어긋남은 그 구간만 벌점) |
| | lane-DER | 같은 순열로 lane 구간을 화자 구간과 대조: 누락·오경보·혼동 시간 비율(collar 0.25 s) | 텍스트 |
| | lane 전환율 | 한 화자가 세션 안에서 lane 을 바꾼 횟수 / 그 화자 발화 수; N>R 재배정 세대 표기 | — |
| C. 구간·턴 | ONSET / EOT P·R | ±0.2 / 0.4 / 0.8 s 1:1 매칭, 시각 오차 중앙값·p90 | 텍스트·lane |
| | 턴 교대 검출 | 참조에서 화자가 바뀌는 시점(앞 화자 끝→다음 화자 시작, 간격 또는 겹침)을 가설이 같은 순서로 내는가: F1(±0.5 s)·검출 지연 | — |
| | turn 내 EOT 오방출 | 참조 발화가 이어지는 동안(유지 구간) 낸 EOT 수 / 유지 구간 시간 — barge-in 위험 | — |
| | 활동 F1 | 청크 단위 lane 활동(순열 매핑 후) 정밀도·재현율 | — |
| | p(EOT) 품질 | 디코더가 낸 EOT 확률의 AUC·calibration(양성 교대∪침묵, 음성 유지, 혼재 제외 — 정본 §10) | 임계값 |
| | backchannel | TurnBench 라벨(Normal Turn / Reaction Backchannel) 별로 ONSET/EOT 지표 분리 | — |
| D. 지연 | 토큰·ONSET·EOT 지연 | 일치 토큰 방출 청크 끝 − 참조 시각, ONSET/EOT 시각 차(참조 δ 보정) | — |

보고 형식: 축별 표를 언어(KO/EN)·화자 수(2 / 3–4 / ≥5)·seen/held-out 으로 나누고, 세션 단위 bootstrap 95 % CI 를 붙인다. 뷰어에는 창별 네 축 값을 chip 으로, 요약에 축별 표를 둔다.

## 2. 객관 held-out 평가셋

| 이름 | 언어·N | 규모 | 상태 | 용도 |
|---|---|---|---|---|
| **NIKL 일상대화 2020** | KO 2인 | 40 대화 9.1 h → 100 대화로 확장 | Phase 1·2 미사용. 발화 PCM + 시각. 토큰 시각은 proxy → 정렬(p2_align) 필요 | A·B·C·D 전부 |
| **TurnBench dev** | EN 2인 full-duplex | ≈48 대화(parquet 3 개), 화자별 FLAC 채널 + 발화 라벨(Normal Turn·Reaction Backchannel 등)·텍스트, 어노테이터 3 트랙 | 서버 `/soundai/DB/raw/turnbench/dev`. 미사용. 로더·정렬 필요 | C(턴·backchannel 의 기준 셋)·A·B |
| **CHiME-6 dev** | EN 4인 원거리 | S02·S09 4.46 h | 빌드 완료, proxy 토큰 | 도메인 외 참고(A·B) |
| **DiPCo** | EN 4인 저녁식사 | ≈5 h(tgz) | 미추출. 로더 필요 | 도메인 외 참고 |
| **7 코퍼스 세션 split** (다음 학습부터) | KO/EN 2–10인 | AMI·ICSI·NOTSOFAR 공식 test/eval, 71631·134-1·134-2 세션 5 %(화자 id 기준), otoSpeech 배우 기준 | D1·D1b 는 이미 전량 학습 → seen 으로만 보고. D1c 부터 제외 | 도메인 내 held-out |

규약: 세션 전체를 0 초부터 고정 격자(30 s, hop 0)로 자른 창 전부를 평가한다(창 골라내기 없음). 마스크 청크가 있는 창은 그 청크만 채점 제외. 세션 단위로 지표를 모아 CI 를 낸다. seen 수치는 별도 열로만 남긴다.

## 3. 구현 순서
1. **지표 모듈** `vapasr/hf/lane_metrics.py`: utterance-assigned WER/CER, cpWER(순열 ≤6! 전수), lane-DER, lane 전환율, ONSET/EOT/턴 교대 매칭(허용오차 다중), turn 내 EOT 오방출, 활동 F1, 지연. 단위 테스트(합성 참조·가설로 각 축이 서로 새지 않는지: lane 만 바꾼 가설은 A 축 불변·B 축만 악화, 텍스트만 바꾼 가설은 A 만 악화). MeetEval 은 서버 env 에 설치되면 ORC-WER/cpWER/tcpWER 를 대조 열로 붙인다(pip 가능 여부 확인; 없으면 자체 구현만).
2. **SegLST 내보내기**: 평가 스크립트가 세션·창별 가설 구간(lane→speaker, 절대 시각, 텍스트)과 참조를 JSON 으로 남겨 어떤 채점기든 다시 돌릴 수 있게 한다.
3. **held-out 자료**: NIKL 2020 100 대화 빌드·정렬(GPU 1 장, ≈30 분), TurnBench 로더(parquet → 화자별 FLAC·라벨; 트랙 a 기본, 3 트랙 일치 구간만 턴 채점) + 정렬, DiPCo 추출·로더(선택). CHiME-6 정렬.
4. **세션 split 파일** `data/phase2/splits/v1.json`: 코퍼스별 held-out 세션 목록 + 근거. 학습 스크립트에 `--exclude-split` 추가. D1c 부터 적용.
5. **`experiments/p2_eval_sessions.py`**: 고정 격자 창 → 디코드(δ_onset 반영) → 지표 모듈 → 세션 집계·CI → report.json; 뷰어 갱신(축별 표·창 chip).
6. **기준값**: D1e(인코더 수정본)·D1b 를 같은 셋으로 채점해 D1→D1b 변화를 축별로 본다.

## 4. 결정이 필요한 것
- TurnBench 어노테이터 트랙 사용 규칙(단일 트랙 vs 3 트랙 합의 구간만).
- 7 코퍼스 split 은 재학습이 있어야 held-out 이 된다 — D1c 부터 적용할지, 지금 D1b 결과는 seen 으로만 볼지.
- 서버 conda env 에 meeteval 설치 허용 여부(순수 파이썬, 사내망 pip 가능하면 즉시).
