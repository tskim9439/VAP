---
author_id: tskim
created: 2026-10-14
source_type: source-drop
---

# Stage 3 P1 독일어 대화 데이터(KEC·ZTC·NSC) 준비 (2026-10-14 00:5x – 02:2x, mxc 호스트 GPU 0·4·6)

## 산출물 (mxc `VAPKT-data/data/phase2/`)
| 코퍼스 | 대화 파일 | 정렬 | 정제본 | 창 |
|---|---|---|---|---|
| KEC | `kec.dialogues.jsonl`(표준 철자) · `kec.verbatim.dialogues.jsonl`(축약 표기 원본) | `align-asr-tn-de-v1/kec` | `kec.refined.dialogues.jsonl` | `windows-de/kec.windows.jsonl` |
| ZTC | `ztc.dialogues.jsonl` · `ztc.verbatim.dialogues.jsonl` | `align-asr-tn-de-v1/ztc` | `ztc.refined.dialogues.jsonl` | `windows-de/ztc.windows.jsonl` |
| NSC | `nsc.dialogues.jsonl`(교사 의사 전사) | `align-asr-tn-de-v1/nsc` | `nsc.refined.dialogues.jsonl` | `windows-de/nsc.windows.jsonl` |

- 그 밖의 산출물:
  - ASR 대조: `asrcheck/{kec,ztc}.utts.jsonl`(축약 표기 기준) · `asrcheck/{kec,ztc}.canon.utts.jsonl`(표준 철자 기준, 간투사 제외 WER)
  - mono 캐시: `mono/{KEC,ZTC,NSC}/`
  - NSC 중간 산출: `nsc-work/`(구간·교사 ASR 샤드)
  - split: `splits/v2.json`
  - 로그: `logs-de/`
- 쓰지 않지만 지우지 않고 남겨 둔 것:
  - `align-asr-tn-v1/{kec,ztc}`: 축약 표기본 정렬
  - `nsc-work/nsc-segments.merge0.3.jsonl`: 병합 0.3 s 판
  - `/tmp/sa_tskim/nsc-smoke{,2}`: 스모크 결과
- 이 폴더의 파일:
  - `*.stats.json`: 빌드·정제·창 통계
  - `logs-status`: 단계별 시각과 ASR 대조 요약
  - `logs-run-*.sh`: 실행 스크립트
  - `split-v2-german.json`: v2 의 독일어 held-out
  - `kec_err.py`: 오류 쌍 분해
  - `nsc_novad.py`: NSC VAD 미검출 점검
  - `de_probe.py`: 로더 경로 표본 점검

## 순서
1. `p2_build_dialogues.py --corpus kec|ztc` (1 차, 축약 표기 그대로)
2. `p2_asr_check.py`(Qwen3-ASR-1.7B, 9 샤드) → `asrcheck/<c>.utts.jsonl`
3. `p2_build_dialogues.py ... --asr-guide asrcheck/<c>.utts.jsonl` (2 차)
   - 구어 축약 표기를 표준 철자로 바꾼다(`dialogue_corpora.canon_de`).
   - 바뀐 전사로 오류율을 다시 재 `*.canon.utts.jsonl` 로 쓴다.
4. NSC 의사 전사(`nsc_pseudo_label.py segment → asr → build --dest phase2`)
5. `p2_align.py --out-root align-asr-tn-de-v1`
   - 정렬기: Qwen3-ForcedAligner-0.6B, 토크나이저: Qwen3-ASR-0.6B
6. `p2_refine.py`
   - KEC·ZTC: `--asr-flags *.canon.utts.jsonl --asr-thr 0.5 --asr-spk-thr 0.8`
   - NSC: ASR 플래그 없음. 교사 불일치는 이미 flags 에 있다.
7. `p2_build_mono.py --corpora kec,ztc` / `nsc`
8. `p2_build_windows.py --corpora kec,ztc,nsc --out windows-de`
9. `p2_make_split.py --base splits/v1.json --corpora kec,ztc,nsc --frac 0.10 --version v2`

## KEC — 축약 표기와 익명화
- **전사는 축약 표기다.**
  - KEC 는 강제 정렬기가 깨지지 않게 축약형을 그대로 적었다(`KEC_Information.pdf` §3.4: "gibts", "nen", "hats" 등).
  - 문서는 축약형↔표준형 사전을 제공한다고 했지만 배포본(CLARINDocu.zip)에는 없다.
- **ASR 대조 오류 상위(`kec_err.py`, 축약 표기본):**
  - 치환: nich→nicht 2,331 · is→ist 2,141 · hab→habe 1,909 · des→das 1,360 · ok→okay 1,182 · äh→mhm 1,067 · glaub→glaube 976 · n→ein 908 · ne→eine 690 · s→es 656
  - 삭제: äh 4,395 · NANA 1,131 · NANANA 595
- **`NANA`·`NANANA`·`NA` 는 익명화된 이름·지명이다.**
  - 18 개 지점을 재 보니 그 구간은 RMS −20 dB 일정 음이었다. 예: "die Katze vom NANA".
  - 처리: 단어를 전사에서 빼고 발화 구간은 유지한다. 이름만 남은 발화 45 개는 사라졌다. 발화 수 38,358 → 38,313, 번호는 그대로다.
- **`<HÄSITATION>` 은 "äh" 로 적는다.**
  - 문서상 모든 머뭇거림을 포함한다(mhm·hm 도).
  - ASR 가설이 간투사 후보이면 그것을 쓴다.

## 표준 철자 변환 (`canon_de`)
- 단어마다 ASR 가설과 편집 거리로 정렬하고 아래 순서로 정한다.
  1. **ASR 이 고른 후보(모호한 축약형만):** 정렬된 가설 단어가 허용 후보 안에 있으면 그것을 쓴다.
     - des→{das, es}
     - s→{es, das, ist}
     - n→{ein, einen, eine, den}
     - ne→{eine, nein, nee}
     - m→{dem, einem}
     - ma→{mal, man}
     - mer→{wir, man}
     - äh→{ähm, öh, hm, mhm…}
     - das↔dass
  2. **고정 대응:** 약 60 개
     - nich·net→nicht, is·isch→ist, hab→habe, ham→haben, gibts→gibt's, naja→na ja, weißte→weißt du 등
     - ZTC 전용: ehm→ähm, eh→äh, hmh→mhm, kuckt→guckt, parallelogram→parallelogramm
  3. **가설과 형태 차이만 나는 경우:** 가설을 쓴다.
     - 끝 e 탈락(find→finde)
     - 접어 's(wenns→wenn's)
     - 스위스식 ss → ß(gross→groß). 반대 방향(ß → ss)은 하지 않는다.
- 가설이 없거나 후보가 아니면 모호한 축약형은 그대로 둔다. 예: 속격 "des" 는 보존된다.

| | KEC | ZTC |
|---|---|---|
| 대조 발화(0.3 s 이상) | 34,076 | 71,555 |
| 바뀐 발화 | 15,735 | 13,623 |
| 바뀐 단어(고정·ASR 선택·형태) | 17,244 · 7,148 · 2,128 | 11,082 · 809 · 2,436 |
| 오류율 중앙값 (축약 표기 → 표준) | 0.25 → 0.083 | 0.111 → 0.0 |
| ≥ 0.2 비율 | 59.4 % → 33.3 % | 43.0 % → 30.9 % |
| ≥ 0.5 비율(표준) | 13.7 %(시간 6.4 %) | 16.7 %(시간 11.2 %) |

- 축약 표기 쪽 오류율은 간투사를 포함한 값이다.
- 표준 쪽은 간투사를 뺀 WER(`de_word_err`)이다. KEC 에서 간투사만 빼면 중앙값 0.192, ≥ 0.2 비율 49.9 % 였다.
- **남은 불일치는 대부분 ASR 오류다.**
  - 표본을 보면 0.2–0.6 구간의 사람 전사가 맞다.
  - 구어·스위스 억양에서 ASR 이 틀린다. 예: "ein teekrug" → "Taco", "so wie eine giraffe" → "zu wenig jeder".
- **비교:** Phase 2 한국어 AI Hub(군중 전사)는 0.2 에서 38–52 % 를 뺐고, 영어 사람 전사 코퍼스(AMI·otoSpeech)는 ASR 대조를 하지 않았다.
- **결정:** 사람이 단어 단위로 고친 KEC·ZTC 는 0.5 이상만 전사 감독에서 뺀다.
- **화자 통째 어긋남(`--asr-spk-thr 0.8`, 대조 발화 20 개 이상):**
  - KEC `rec_026_p4` IS_051: 124 발화 중앙값 1.0. 전사와 오디오가 전혀 다르다.
  - ZTC `MMZS1c`·`MMZS2c` ZS: 각 24 발화. 전사는 맞아 보이고 ASR 이 실패한 경우라 손실을 감수한다.
- **dialogue 단위 오류율 중앙값의 p90:** KEC 0.154, ZTC 0.054.

## NSC — 교사 의사 전사
- **구간(`segment`):**
  - 50 fps 에너지. 자기 채널이 바닥 + 15 dB 이상이고, 상대 채널보다 16 dB 이상 작지 않은 프레임만 쓴다.
  - 100 ms 다수결. gap < 0.5 s 병합, 0.3 s 미만은 버린다. 20 s 를 넘으면 가장 조용한 프레임에서 자른다.
  - 병합 간격을 0.3 s 로 하면 구간 중앙값이 1.2–1.7 s 로 짧아 0.5 s 로 올렸다.
- **구간 통계:** 276 세션 · 34.8 h
  - 화자: 19,945 구간 · 10.6 h
  - 상대: 20,598 구간 · 16.4 h
  - 겹침: 말한 시간의 중앙값 4.0 %
  - 상대역: amsterdam 254 · aden 20 · 둘 다 2
- **교사:**
  - Qwen3-ASR-1.7B(독일어, 배치)
  - Whisper large-v3: 번들 SimulStreaming 코드, 탐욕 디코드, 이전 문맥 없음, 배치 decode
  - hibiki env, 9 샤드, 28 분
- **Whisper 결함 두 가지(스모크에서 발견):**
  - 번들 `transcribe()` 는 mel 80 고정이라 large-v3(128)에서 오류가 난다. → mel 을 직접 만들어 `decode` 를 쓴다.
  - 번들 `DecodingTask` 는 언어 수 99 로 토크나이저를 만든다. large-v3(100 개 언어)에서는 언어·과제 토큰 id 가 하나씩 밀려 영어 번역을 냈다.
    - 예: "Guten Tag hier ist die Stadtbücherei" → ", what can i do for you"
    - → `num_languages` 를 넘기도록 패치했다.
    - SimulStreaming 베이스라인 평가는 자체 토크나이저(`num_languages` 전달)를 써서 영향이 없다.
- **일치 판정:**
  - Qwen 기준으로 Whisper 가설의 WER ≤ 0.2 면 일치로 본다. 간투사는 무시한다.
  - Whisper 의 숫자는 독일어 수사로 편다. 0 으로 시작하거나 7 자리 이상이면 한 자리씩 읽는다.
  - 전사 = Qwen 결과의 asr-tn-de 정규화. 불일치·빈 결과는 `asr_disagree`(시각 유지, 전사 감독 제외)다.
- **결과:**
  - 40,543 발화 중 일치 27,411(67.6 %)
  - 불일치 12,678, digit 478, abbrev 318
- **VAD 미검출 점검(`nsc_novad.py`, 30 세션 4,613 발화):**
  - Phase 2 에너지 VAD 가 못 잡은 발화는 대부분 조용한 맞장구("ja")다. 바닥 위 16.9 dB 로, VAD 가 잡은 발화의 26.4 dB 보다 낮다.
  - 상대 채널이 0 으로 편집된 구간의 발화(5.4 %)는 "okay"·"alles klar" 같은 실제 발화다. 0 편집은 그 채널의 비음성 구간으로 보인다(KEC 문서와 같은 관행).
  - 누설이 들어간 발화라면 상대의 말이 전사됐을 텐데 그런 예는 보지 못했다.

## 정제·감독 비율
| | KEC | ZTC | NSC |
|---|---|---|---|
| 대화 · 시간 | 171 · 29.2 h | 373 · 41.9 h | 276 · 34.8 h |
| 발화(정제 후) | 43,155 | 89,209 | 56,406 |
| 발화 시간 | 24.5 h | 27.1 h | 23.1 h |
| 전사 감독 시간 | 21.8 h (89.2 %) | 24.1 h (89.0 %) | 16.4 h (70.8 %) |
| 감독 제외 | asr_disagree 1.43 h · 말 끊김 0.82 h · 알아듣기 어려움 0.34 h | asr_disagree 2.78 h | 교사 불일치 6.02 h · digit 0.56 h · abbrev 0.35 h |
| 분할 · 당김 · VAD 미검출 | 4,842 · 219 s · 90 | 4,974 · 1,863 s · 3,912 | 15,863 · 8,608 s · 6,143 |

## 창 (`windows-de/`, 20–40 s, hop 10 s)
| | 창 · 시간 | 마스크 비율 평균(> 50 % 창) | 창당 텍스트 토큰 | 창당 EOT | 2 화자 창 |
|---|---|---|---|---|---|
| KEC | 2,444 · 20.3 h | 11.8 % (1.5 %) | 111 | 11.9 | 94.7 % |
| ZTC | 5,877 · 48.8 h | 12.2 % (0.1 %) | 74 | 14.9 | 94.1 % |
| NSC | 4,261 · 35.7 h | 23.1 % (5.4 %) | 69 | 10.5 | 99.2 % |

- 창 시간은 hop 10 s 로 겹쳐 센 값이다.
- `windows-de/overfit32.windows.jsonl` 은 비어 있다. overfit 선택이 기존 코퍼스 목록만 본다. 독일어에는 쓰지 않는다.

## 로더 경로 표본 점검 (`de_probe.py`)
- 경로: `DialogueWindowDataset` → mono 캐시 → crop → 시퀀스(Phase 2 lane 형식)
- 코퍼스마다 40 창을 뽑았다. 길이 오차·RMS < −50 dB·텍스트 토큰 < 8 인 창은 0 개였다.
- RMS 중앙값: KEC −24.2, ZTC −29.2, NSC −32.2 dB
- 정렬 토큰 시각이 발화 구간 ±0.3 s 안에 드는 비율: KEC 99.67 %, ZTC 98.72 %, NSC 96.78 %
- 청취 점검은 하지 않았다.

## split v2 (`splits/v2.json`)
- v1 의 7 코퍼스는 그대로다(대조 확인).
- 독일어 held-out:
  - **KEC:** 녹음 쌍 4/39(rec_003·004·012·028). 화자가 한 쌍에만 나온다. 21 대화 · 3.5 h
  - **ZTC:** 화자를 공유하는 dyad 를 묶은 27 덩어리 중 3 개(BS·JL·LS 덩어리). 39 대화 · 4.5 h
  - **NSC:** 상대역이 aden 인 세션 22 개(처음 듣는 상대 확인용). 3.3 h. 평가 정답이 교사 전사라 전사 지표는 참고용이다.
