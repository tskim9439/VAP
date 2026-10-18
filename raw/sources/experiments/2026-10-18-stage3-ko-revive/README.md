---
author_id: tskim
created: 2026-10-18
source_type: source-drop
---

# Stage 3 한국어 화자 귀속 — 마스크 분석과 격리 발화 되살리기(revive) 구현·파일럿 p11 준비 (2026-10-18, mxc 호스트)

## 배경
- 사용자(2026-10-18): "한국어 화자 귀속은 어떻게 개선할 수 있을까?" → 분석 뒤 "1번 구현과 파일럿 준비 진행하자"
- p10 ck3000(Stage 3 최종) δ10 귀속 몫(cp − agn, 단위 가중, `2026-10-18-stage3-sim-dialogue/p10-attr-errors.txt`):

  | 언어 | r1 ck2000 | p9 ck3000(화자 헤드) | p10 ck3000(+ 합성 대화) |
  |---|---|---|---|
  | 한국어 | 10.7 | 7.5 | 8.0 |
  | 영어 | 10.0 | 7.9 | 6.4 |
  | 독일어 | 12.7 | 9.9 | 8.4 |

  - 화자 헤드는 세 언어 모두 줄였다. 합성 대화는 한국어만 효과가 없었다(p10 − p9 δ10 cp +0.46 [−1.42, +2.28]).

## 분석 1 — 학습 데이터의 마스크 (`mask-by-len.txt`·`mask_by_len.py`, `flag-kinds.txt`·`flag_kinds.py`)
- 규칙(`DialogueWindowDataset._bad_spans`): flags(격리) · 토큰 없음 · `meta.missing`(결손 조각) 발화가 덮는 청크(+δ+1 청크 + 0.24 s)는 출력 label 이 전부 −100 이다. 화자 태그·`<EOT>`·겹친 상대 화자 전사도 포함된다.
- 격리 발화는 정제(`p2_refine`)가 `text`·`tokens` 를 지워 텍스트에서 빠진다. 그래서 A → B(격리) → A 대화는 태그 없이 A 가 이어지는 시퀀스가 된다.

  | 코퍼스 | 마스크된 발화 시간 | 한쪽이 마스크인 화자 교대 | 교대 수 |
  |---|---|---|---|
  | aihub71631 | 48 % | 74 % | 132,242 |
  | aihub134-1 | 46 % | 69 % | 249,565 |
  | aihub134-2 | 58 % | 80 % | 257,674 |
  | KEC · ZTC · NSC | 11 · 11 · 29 % | 27 · 28 · 50 % | 25,238 · 37,855 · 31,918 |
  | otoSpeech | 2 % | 2 % | 65,760 |

  - 발화 길이와 무관하게 한국어는 38–74 % 가 마스크다(짧은 발화만의 문제가 아니다).
  - 사유(발화 수): 대부분 `asr_disagree`(라벨 ↔ Qwen3-ASR-0.6B CER ≥ 0.2) — 71631 89,916 · 134-1 133,767 · 134-2 198,294. 그다음 `implausible_rate`(2.0–2.7 만), 134-1·134-2 의 결손 조각(4.6 만·6.0 만).
  - 격리 발화는 원 전사(`raw`)는 있지만 정렬 토큰이 없다.

## 분석 2 — 평가 창의 마스크와 귀속 오류 (`mask-vs-attr-p10ck3000.txt`·`mask_vs_attr.py`)
- p10 ck3000, δ10, 2 화자 창. 평가 원출력의 `masked_chunks`/K 와 `cst_attr_errors` 창 행을 (id, δ)로 붙였다.

  | 한국어 창 마스크 비율 | 창 | 귀속 몫 | agn | 참조 교대당 오배정(글자) |
  |---|---|---|---|---|
  | 0–5 % | 29 | 2.2 | 6.9 | 0.64 |
  | 5–20 % | 124 | 8.1 | 12.6 | 1.54 |
  | 20–40 % | 190 | 9.2 | 17.7 | 1.82 |
  | 40–60 % | 133 | 9.9 | 20.9 | 1.74 |
  | 60 % 이상 | 84 | 8.8 | 34.4 | 1.30 |

  - 독일어는 교대당 오배정이 마스크 비율과 무관하다(0.50–0.87). 영어(마스크 거의 없음)는 0.57 이다.
  - 평가는 마스크 청크를 참조·가설 양쪽에서 뺀다. 태그 상태는 마스크 청크의 태그로도 이어 센다(`cst_eval.parse_emitted`). 따라서 채점 방식 탓은 아니다.
- **주의:** 마스크가 거의 없는 한국어 창은 29 개뿐이고 받아쓰기도 쉬운 창이다(agn 6.9). 마스크 없는 합성 한국어 대화가 효과가 없었던 것과도 맞물린다. 인과는 파일럿으로 확인한다.

## 구현 — 격리 발화 되살리기(revive)
- **재료(`experiments/stage3_revive_tokens.py`):** 정제는 격리 발화를 나누지 않아 `utt_id` 가 그대로다. 그 발화의 정렬 토큰(p2_align parts: 한국어 `align-asr-tn-v1`, 독일어 `align-asr-tn-de-v1`)을 다시 뽑아 `<corpus>.revive.jsonl` 로 쓴다. 정제 파일은 고치지 않는다.
- **`vapasr/data/cst_dialogue.py`:**
  - `REVIVABLE = {asr_disagree}`: 시각·화자는 믿을 수 있고 전사만 의심되는 사유다. `implausible_rate`·TN 사유·여러 사유가 섞인 발화·`meta.missing` 결손 조각은 그대로 마스크한다.
  - `CstDialogueDataset(revive=load_revive(...))`: 창을 고른 뒤에 토큰을 붙이므로 창 목록은 끈 것과 같다. 되살린 발화는 `_untranscribed` 에서 빠져 마스크 구간이 아니다.
  - `build_cst_sequence(lex_mask, lex_chunks)`: 되살린 발화의 전사 토큰 label 과 방출 구간(청크 ⌊start/0.08⌋+δ … ⌊end/0.08⌋+δ, 창 끝을 넘으면 flush)의 `<NEXT_AUDIO>` label 만 −100 이다. 전사는 틀릴 수 있어 입력 문맥으로만 둔다. 화자 태그·`<EOT>` 결정·겹친 상대 화자 전사는 학습한다. 태그 뒤집기(recover) 후보에서도 뺀다.
  - 검사 `revive:unmasked`(되살린 전사 label 이 −100 인지), 행·배치 카운터 `n_rev`.
- **`vapasr/data/dialogue_dataset.py`:** `_bad_spans` 의 발화 판정을 `_untranscribed` 훅으로 뗐다(기본 동작 그대로).
- **`experiments/stage3_train.py`:** `--revive-dir`(실제 대화 셋에만, 지문 키 `revive` 는 켰을 때만), 로그 `rev_per_dlg_step`
- **`slurm/submit-stage3-train.sh`:** `--revive-dir DIR`
- **테스트(`tests/test_cst_dialogue.py`):**
  - 시퀀스: 전사·방출 구간 `<NEXT_AUDIO>` 만 가리고 교대 태그·`<EOT>` 는 학습한다. 끄면 시퀀스가 같다. 뒤집기 지점은 되살린 원소가 아니다.
  - 데이터셋: 창 목록이 같다. 마스크가 줄고, 끈 쪽에서 사라진 교대 태그(B, 다시 A)가 되살아난다. 다른 사유가 섞인 발화·정렬 없음·결손 조각은 되살리지 않는다. `est_lens`·`collate_cst` `n_rev`·`load_revive` 를 확인한다.
  - Stage 3 관련 테스트 56 통과·2 건너뜀(로컬 transformers 버전). mxc 에 올린 파일은 md5 로 대조했다.
- **결손 조각:** 첫 빌드(`build.log`)는 결손 조각을 거르지 않아 134-1 에서 정렬을 가진 결손 조각 5,384 건이 되살아날 뻔했다(오디오 없음). 빌더와 데이터셋 둘 다에서 빼고 테스트에 넣었다.

## 재료 (`revive-asr-v1`, mxc `VAPKT-data/data/phase2/revive-asr-v1/`, `build2.log`)

| 코퍼스 | 격리(토큰 없음) | 결손 조각 | 다른 사유 | 정렬 없음 | **되살림** | 토큰 | 시간 |
|---|---|---|---|---|---|---|---|
| aihub71631 | 117,920 | 0 | 28,004 | 0 | **89,916** | 104.7 만 | 56.0 h |
| aihub134-1 | 194,806 | 29,381 | 31,658 | 19,848 | **113,919** | 120.6 만 | 69.1 h |
| aihub134-2 | 276,273 | 52,988 | 24,991 | 9,963 | **188,331** | 160.6 만 | 79.7 h |
| KEC | 5,640 | 0 | 1,043 | 0 | **4,597** | 2.5 만 | 1.4 h |
| ZTC | 12,446 | 0 | 513 | 0 | **11,933** | 5.0 만 | 2.8 h |

- 발화 구간 ±0.3 s 밖에 끝나는 토큰은 0.5–2.4 % 다(정제의 VAD 당김). `build_episodes` 가 발화 구간으로 자른다.
- NSC 는 격리 발화(교사 불일치)에 정렬이 없어(12,309) 되살리지 않는다. `nsc.revive.jsonl` 은 첫 빌드가 남긴 빈 파일이다.

## 로더 경로 점검 (`probe.json`·`revive_probe.py`)
- 방법:
  - 학습과 같은 창 규칙(20–40 s, hop 10, grid, 맞장구 1.0·3, held-out 제외)으로 `CstDialogueDataset` 을 끈 것·켠 것 둘 다 만들었다. 두 창 목록은 같았다.
  - 같은 창 300 개를 δ10 으로 직렬화해 창당 평균을 비교했다. 결과 파일에는 숫자만 쓴다.

  | 코퍼스 | 되살린 발화(학습 split) | 학습되는 교대 태그/창 | 학습되는 `<EOT>`/창 | 학습되는 전사 토큰/창 | 출력 label 마스크 | 불변식 위반 |
  |---|---|---|---|---|---|---|
  | aihub71631 | 82,524 | 2.74 → **6.74** | 3.76 → 6.92 | 48.4 → 53.9 | 32.5 → 27.4 % | 0 |
  | aihub134-1 | 102,644 | 3.37 → **7.86** | 3.82 → 6.69 | 56.5 → 63.5 | 35.5 → 30.5 % | 0 |
  | aihub134-2 | 163,951 | 1.72 → **5.89** | 2.82 → 6.51 | 36.0 → 40.9 | 35.2 → 27.9 % | 0 |
  | KEC | 4,079 | 11.06 → 13.92 | 5.66 → 6.82 | 104.5 → 109.2 | 11.7 → 8.5 % | 0 |
  | ZTC | 10,987 | 6.27 → 9.09 | 5.79 → 7.67 | 70.2 → 75.1 | 14.5 → 8.6 % | 0 |

- 한국어 창은 학습되는 교대가 독일어보다 훨씬 적었다(1.7–3.4 대 6.3–11.1). 되살리면 5.9–7.9 가 된다.
- 학습되는 전사 토큰이 느는 것은 되살린 발화와 겹친 상대 화자의 말이다. 되살린 전사 자체는 늘 −100 이다.
- 출력 label 마스크가 크게 줄지 않는 것은 되살린 전사·`<NEXT_AUDIO>` 를 여전히 가리기 때문이다.
- 정렬 없는 asr_disagree 발화(134-1 17,989 · 134-2 8,775)는 그대로 마스크다.
- 확인용 예시(표준오류, 저장하지 않음)에서 되살린 단어·구간 `<NEXT_AUDIO>` 만 `[m]` 이고, 앞뒤 화자 태그와 겹친 상대 화자 말("…고 `<EOT>`")은 학습되는 것을 봤다.

## dry-run·스모크 (`stage3_dry_revive.sh`·`stage3_smoke_revive.sh`, 체인 `revive_chain.sh`)
- **dry-run(CPU):** p10 레시피 + `--revive-dir`, 셋마다 300 창, 점검 64 항목 × δ4–10
  - `dlg_ko` 되살림 349,119 발화(정렬 없는 26,764 는 마스크 그대로), `dlg_de` 15,066(정렬 없는 10,863 은 대부분 NSC)
  - 6 셋(dlg 3 · sim 3) 모두 불일치 0
  - 지문: `revive` 8261d406876b5b94(재료 파일 경로·크기), `sim_data` 0f7ac2794329d0f2, `spk_act_weight` 0.5
- **스모크(호스트 GPU 6, 20 step, replay 포함):**
  - GPU 6 에는 다른 사용자 프로세스(4.8 GB, 사용률 0 %)가 있었다. 그 프로세스는 건드리지 않았다.
  - 학습 20/20(76 s). 섞기 비중은 p10 과 같다(`dlg_ko` 0.221). `rev_per_dlg_step` 은 대화 배치 구성에 따라 0–535 다.
  - 손실 2.21 → 1.10, 화자 헤드 `spk_acc` 0.51 → 0.62–0.70
  - 저장 뒤 재로드 대조: 누락·추가·shape·값 차이 0, 인코더 키 638 — **PARITY OK**
  - run 기록 지문에 `revive` 8261d406876b5b94 가 있다.

## 파일럿 p11 제출 명령 (사용자, 스케줄러 셸에서 `/soundai/users/tskim/VAPKT`)
```bash
(MAX_RESTART=5 bash slurm/submit-stage3-train.sh --run st3-p11-revive --partition apex --nodes 2 --backchannel 1.0,3 --weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,sim_ko:0.5,sim_en:0.8,sim_de:0.8,rep_ko:1.2,rep_en:1.2,rep_de:1.2 --lr 4e-5 --max-steps 4000 --spk-act-weight 0.5 --sim-data /soundai/users/tskim/VAPKT-data/data/sim-dialogue-v2 --revive-dir /soundai/users/tskim/VAPKT-data/data/phase2/revive-asr-v1)
```
- p10 명령과 다른 것은 `--revive-dir` 하나다. 제출 스크립트 `--dry-run` 으로 인자가 그대로 넘어가는 것을 확인했다.
- 평가: 제출 뒤 평가 큐(`stage3-regress/ckpt_eval_queue.sh`)를 ck2000·ck3000·final 에 건다.
  - 대화 평가: 맞장구 참조, δ4·10
  - 오류 분해와 단일 화자 축약 평가
- 판정:
  - 같은 step 의 p10 대비 한국어 귀속 몫을 대화 단위 짝 bootstrap 으로 본다.
  - 마스크 구간별 참조 교대당 오배정을 `mask_vs_attr.py` 로 다시 잰다. 마스크 있는 창이 줄면 원인이 확인된다.
  - EOT·영어·독일어·단일 화자 퇴행도 함께 본다.
  - 줄지 않으면 창 난이도 탓으로 보고 측정 보강(NIKL 2020 held-out)으로 넘어간다.

## 파일
- 분석: `mask_by_len.py`·`mask-by-len.txt`, `flag_kinds.py`·`flag-kinds.txt`, `mask_vs_attr.py`·`mask-vs-attr-p10ck3000.txt`
- 재료·점검: `revive_probe.py`, `revive_chain.sh`, `stage3_dry_revive.sh`, `stage3_smoke_revive.sh`
