---
type: output
status: active
created: 2026-09-15
updated: 2026-09-15
summary: Phase 2 Q0 — 7개 DB 실제 샘플(각 2대화)로 dialogues.jsonl → Qwen 정렬 → mono 혼합 창 → lane 태그·ONSET·EOT(soft) 시퀀스를 생성하고 8가지 검사를 통과시킨 기록; DB별 관찰(quarantine·창 수율·ICSI SPH 변환)
sources:
  - '[[output-phase2-lane-plan]]'
  - '[[task-phase2-data-prep]]'
  - '[[decision-eot-immediate-soft-label]]'
---

# Phase 2 실제 샘플 시퀀스 생성·검증 (DB 별)

## 질문
사용자 요청(2026-09-15): "각 DB 마다 실제 샘플로 시퀀스를 짜보고 검증까지 진행해봐." 정본 [[output-phase2-lane-plan]] §4–§5 규약이 보유 DB 7종의 실물에서 그대로 성립하는지, 파이프라인(로더 → TN → 정렬 → 혼합 → 창 → 직렬화)이 끝까지 도는지 확인한다.

## 요약
- 7개 DB 모두 실물 2대화씩으로 `dialogues.jsonl` 을 만들고, Qwen3-ForcedAligner 로 화자 채널에서 실제 정렬해, 창 3개씩 시퀀스를 생성했다. ICSI 는 SPH(shorten) → FLAC 변환(75회의, 32 GB) 을 마친 뒤 정렬했다.
- **8가지 검사 전부 통과(ALL_OK).** 텍스트 라운드트립, never_free ≡ lazy_free(N≤6), ONSET<lexical<EOT 순서와 EOT 청크 규칙, soft label 위치·가중치, 활동 타깃=참조 구간, 오디오 길이=K·0.08 s, 태그 규칙, lane 부족 0.
- 발견·수정: AMI 단어 없는 segment(웃음)와 NOTSOFAR `<ST/>` 태그가 quarantine 으로 창을 제외시키던 것을 로더에서 제거; EN 대화 코퍼스는 숫자 표기가 있어 TN 숫자 규칙(`yodas` 키)을 적용; 대화 안 발화는 앞 공백 포함 토큰화(`" " + text`)로 "tv"+"and" 가 붙지 않게 함; ICSI 는 sph2pipe 를 서버에 컴파일해 FLAC 변환(`VAPKT-data/data/audio/icsi/`).
- 남은 관찰: KO 71631 은 숫자·익명화(`#@이름#`) quarantine 이 발화의 20 % 라 창 수율이 낮다(§4). 산출물: `raw/sources/experiments/2026-09-15-phase2-sequence-samples/`(코퍼스별 report.json·window-*.txt·stats.json), 서버 `VAPKT-data/data/phase2-samples/`(wav 포함).

## 1. 파이프라인과 실행

| 단계 | 스크립트 | 실행 위치 |
|---|---|---|
| 코퍼스 → Dialogue(TN 포함) | `experiments/p2_build_dialogues.py --limit 2` | mxc 로그인 노드(CPU, conda vapasr) |
| 화자 채널 정렬 | `experiments/p2_align.py` | 로그인 노드 GPU 2(소량) — 전량은 `slurm/p2_align.sbatch` |
| 창·직렬화·검증·렌더 | `experiments/p2_show_sequence.py --windows 3` | CPU |

창 규약: 길이 20–40 s, 시작은 아무도 말하지 않는 시각(hop 10 s 격자), 끝은 잘릴 수 있음(잘린 episode 는 창 안 토큰만, EOT mask). δ_text=4, δ_on=0, R=6.

## 2. DB 별 결과

| DB | 샘플 | 화자 수 | 발화 | quarantine | 정렬 발화 | 창 후보 / 제외(전사 없는 음성) | 검증 창 | 결과 |
|---|---|---|---:|---:|---:|---:|---:|---|
| otoSpeech | 2대화 0.5 h | 2 | 498 | 3 (digit·charset) | 460 | 19 / 10 | 3 | ALL_OK |
| AMI (EN2001a·b) | 2회의 2.4 h | 5·4 | 1,565 | 1 | 1,557 | 108 / 6 | 3 | ALL_OK |
| NOTSOFAR-1 (MTG_30860·61) | 2회의 0.2 h | 5·5 | 367 | 0 | 365 | 5 / 0 | 3 | ALL_OK |
| 134-1 실외 조각 | 2대화 0.36 h | 2 | 735 | 6 (anon), 결손 5 | 691 | 12 / 16 | 3 | ALL_OK |
| 134-2 실외 조각 | 2대화 0.46 h | 2 | 583 | 3 | 565 | 39 / 18 | 3 | ALL_OK |
| 71631 실내 stereo | 2대화 0.67 h | 2 | 249 | 50 (digit 31·anon 21) | 199 | 5 / 3 | 3 | ALL_OK |
| ICSI (Bdb001·Bed002, FLAC) | 2회의 1.9 h | 6·6 | 2,319 | 78 (empty) | 2,240 (+proxy 1) | 102 / 19 (+무음 창 41 제외) | 3 | ALL_OK |

"창 후보"는 시작 무음 조건을 만족한 창, "제외"는 quarantine·미정렬·결손 발화가 걸쳐 있어 뺀 창이다. 창당 텍스트 토큰 41–175, lane 태그 2–98, 청크당 최대 토큰 1–11(KO 71631 이 가장 큼), overflow 0, lane 재배정 0(모든 샘플이 N≤6), lane 부족 0.

## 3. 예시: NOTSOFAR-1 MTG_30860, 240–275 s (5명, 창 하나)

```text
ep000 spk=Ernie  lane=1  0.26-1.43  outcome=both  p_end=0.5 :: reclining seat
ep001 spk=Peter  lane=2  1.26-1.96  outcome=shift p_end=1.0 :: yes
ep002 spk=Olivia lane=3  2.20-6.76  outcome=both  p_end=0.5 :: big tv and some movie some some uh internet for
ep003 spk=Ernie  lane=1  3.32-4.17  outcome=shift p_end=1.0 :: big tv
…
ep008 spk=Bert   lane=5 10.88-13.98 outcome=both  p_end=0.5 :: you sold me on charging station as long as i could charge my phone
ep012 spk=Bert   lane=5 14.78-22.38 outcome=shift p_end=1.0 :: i mean that's the problem in the current buses …
ep016 spk=Peter  lane=2 23.39-35.16 outcome=truncated p_end=None (창 끝에서 잘림 → EOT mask)
```

22개 episode, soft EOT 21개(교대 12·혼재 9·침묵 1), 태그 90개, 텍스트 133토큰. 5명이 도착순으로 lane 1–5 를 받고 재배정은 없다. 블록 문자열 전체는 `seq/notsofar/window-2.txt`.

## 4. 관찰과 조치

1. **KO 71631 quarantine 20 %.** 숫자(`digit` 31)와 익명화 표지(`anon` 21)가 발화의 20 % 를 quarantine 으로 만들고, 그 발화가 걸친 창은 통째로 제외된다(음성은 있는데 전사가 없는 창을 학습시키지 않기 위해). 2대화 20분에서 창 5개뿐이다. 선택지: (a) 숫자 발화의 원문(숫자 표기)을 그대로 target 으로 허용(TN 동결 규칙과 충돌), (b) 창 시작 격자를 10 s → 2 s 로 촘촘히 해 수율을 올림, (c) quarantine 발화만 lexical 손실을 빼고 창은 유지(음성-무전사 학습 위험). **(b) 는 바로 적용 가능**하고 (a) 는 사용자 결정 항목이다.
2. **창 시작 조건(무음)** 때문에 회의·KO 대화에서 격자점의 대부분이 탈락한다(`no_start` AMI 754/868, 71631 231/239). hop 을 줄이면 후보가 늘지만 창 간 중복도 늘어 epoch 정의를 함께 손봐야 한다.
3. **거의 무음인 창**(otoSpeech window-9: 30 s 에 90 ms 발화)이 생겼다 → Dataset 에 `min_text_tokens=8`(기본) 을 넣어 제외(ICSI 재실행에서 41개 제외, `skipped_sparse`).
4. **정렬 범위.** 처음 `--min-dur 0.3/--max-dur 60` 은 ICSI 맞장구(0.3 s 미만) 242개를 건너뛰어 창 71개를 제외시켰다 → 범위를 0.1–120 s 로 넓히고 범위 밖 발화는 균등 배치 proxy 로 채워(parts 에 `proxy` 표시) 창을 잃지 않게 했다. ICSI 재실행: 미정렬 0, proxy 1.
5. **ICSI.** 원본 SPH 가 `pcm,embedded-shorten-v2.00` 이라 libsndfile·sox 로 못 읽는다. sph2pipe(robd003 미러)를 `VAPKT-data/tmp/sph2pipe-master/` 에 컴파일해 회의별 `chanN.flac` 로 변환 중(75회의). 로더는 flac 를 우선 찾는다. 참가자↔채널은 `.mrt` 의 `<Participant Name Channel>` 로 매핑했고, 단어는 `ICSI/Words/<회의>.<agent>.words.xml` 이다.
6. **NOTSOFAR-1 word_timing** 은 정렬 QC 대조군으로 보존했다(Utterance.word_timing).

## 불확실성
- 검사는 규약 정합성이지 라벨 품질이 아니다. 정렬 시각의 정확도(특히 KO 긴 발화·조각 경계)는 Q0 QC(`p2_qc.py`)와 청취 spot-check 로 따로 본다.
- 샘플 2대화는 분포 추정이 아니다. 창 수율·quarantine 비율은 전량 빌드 뒤 코퍼스별로 다시 센다.

## 근거
- `raw/sources/experiments/2026-09-15-phase2-sequence-samples/` — 코퍼스별 `seq/<corpus>/report.json`·`window-*.txt`, `<corpus>.stats.json`
- 서버 `/soundai/users/tskim/VAPKT-data/data/phase2-samples/` — dialogues.jsonl, align parts, seq(wav 포함), build 로그
- [[output-phase2-lane-plan]] §3–§5·§8, [[task-phase2-data-prep]]
