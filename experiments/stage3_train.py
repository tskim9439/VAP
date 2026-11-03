#!/usr/bin/env python
"""Stage 3 학습 — 2 인 같은 언어 대화(화자 귀속 + 스트리밍 ASR + 턴 끝 <EOT>) + Stage 2 단일 화자 replay
(wiki output-stage3-prep-plan-20261014 P5, decision-cst-training-stages-and-speaker-slots). 사용자 결정(2026-10-15): semcommit_train.py 와 별도 스크립트,
섞는 비율은 파일럿에서 비교해 정한다(--weights).

  torchrun ... experiments/stage3_train.py --init <Stage 2 final> --nemotron-dir <.nemo 디렉토리> --out-dir <run> \\
      --data <phase2 dir> --split <phase2>/splits/v2.json [--dlg-corpora …] \\
      --replay-words @<EN·KO words 목록> --replay-de-words @<독일어 목록> --ast-targets <번역 타깃> [--weights dlg_ko:1,…,rep_en:0.5] [--max-steps N]
  --dry-run: 모델 없이 tokenizer + 데이터셋만 만들어 시퀀스 불변식을 검사하고 샘플 하나를 찍는다(CPU).

데이터(셋 이름 = 가중치 키):
  dlg_ko·dlg_en·dlg_de  vapasr/data/cst_dialogue.CstDialogueDataset — Phase 2 정제 대화(split held-out 제외)의 20–40 s 창,
                        prefix 'language <SPK_A>{L} <SPK_B>{L}<asr_text><DELAY_d>', 화자 태그, 턴 끝 soft <EOT>(--eot-mode·--backchannel·--decision-weight).
                        --recover-p > 0: 정체 회복 학습(창 일부에서 문맥 슬롯 한 구간을 뒤집고 회복 지점만 정답 — cst_dialogue 모듈 설명).
                        --spk-act-weight > 0: 화자 슬롯 활동 보조 손실(model.add_spk_head — 청크마다 슬롯 A/B 가 말하는 중인지 BCE, 라벨 cst_dialogue.slot_activity).
                        --revive-dir: asr_disagree 로 격리된 발화를 정렬 토큰째 입력 문맥으로 되살려 화자 태그·<EOT> 를 학습(전사 label −100, 창 목록 그대로).
                        --dlg-trans(Stage 4, decision-stage4-prep-20261019): 대화 발화 번역 타깃이 있으면 창을 --dlg-ast-frac 확률로 번역 모드(두 화자 전사 +
                        번역 네 흐름, prefix 에 translate 줄 — cst_dialogue.build_cst4_sequence)로 낸다. 목표 언어는 한·독 → 영, 영 → 독·한(무작위).
                        --dlg-trans-input: 되살린 발화의 번역(입력 문맥으로만). 끄면(기본) Stage 3 와 같고 지문에 키가 없다.
  sim_<l1>-<l2>[-접미사] 교차 언어 합성 대화(Stage 4 — sim_build_dialogues --lang2, 예 sim_ko-en·sim_en-de·sim_en-de-tel): --sim-data 의 <키>.refined.dialogues.jsonl 과
                        targets-<키>/ 번역 타깃. 슬롯은 언어로 고정(A/B 무작위), --sim-trans-frac 확률로 번역 모드(각 슬롯이 상대 언어로), 나머지는 받아쓰기만.
  V3-p1(기본 끔 — 끄면 데이터·지문이 예전 그대로): --change-tag-weight·--hold-tag-weight(교대 태그·<EOT> 뒤 머무름 결정 pos_weight, --decision-weight-sets all|xl,
    같은 언어 셋 따로 --same-lang-change-weight·--same-lang-hold-weight — V3 R2m),
                        --eot-drop-synth P(합성 창 EOT-문맥 dropout, --eot-drop-sets xl|synth), --set-phone KEY:P(셋별 창 단위 전화 대역 확률) — cst_dialogue 모듈 설명.
  sim_ko·sim_en·sim_de  --sim-data 의 합성 2 인 대화(vapasr/data/sim_dialogue — 다른 화자의 실제 발화를 턴으로 번갈아 놓음). 같은 CstDialogueDataset,
                        <EOT> 감독 없음(meta.mask_eot). epoch 기준(anchor)은 실제 대화 셋(dlg_*)만.
  rep_ko·rep_en·rep_de  vapasr/data/ast_dataset.AstStreamDataset — Stage 2 와 같은 단일 화자 ASR·번역 항목(--ast-frac 번역 모드, --offline-frac final 모드).
                        <SEM_END>·<EOT> 둘 다 중립(sem_mask) — 단일 화자 스트림 끝은 턴 끝 라벨이 아니다.
  섞기: vapasr/hf/stage3.WeightedRoundRobinLoader — step 마다 셋을 가중치로 고른다(결정적). epoch = 대화 셋 배치를 한 번 도는 길이.
모델: Stage 2 체크포인트 + add_semantic_tokens(turn=True, reinit_turn=True) — 예약만 돼 있던 <EOT> 행을 다시 초기화하고 차단을 푼다,
  config.turn_weight = --turn-weight. 대화 행은 sem_only_mask(<SEM_END> 만 중립), forward 가 labels_alt/soft_w 로 두 점 soft CE.
  --enc-tap-layer L(V2-a, wiki output-v2-speaker-memory-plan-20261020): 인코더 L 층 출력을 두 번째 connector(adapter_tap, 마지막 Linear 0 초기화 → 붙인 직후 = V1)로
  LLM 입력에 더한다(model.add_enc_tap). init 에 이미 탭이 있으면 같은 층이어야 한다. 지문 키 enc_tap(끄면 키 없음).
  --slot-blocks(E9, 2026-10-28): 대화·합성 창을 슬롯 블록 형식으로(cst_dialogue 모듈 설명 E9 — 청크마다 선언 슬롯 태그를 모두 입력, <BEGIN>·전사·<EOT> 출력,
  멈춤 = 다음 입력 위치 label <NEXT_AUDIO>), model.add_slot_blocks(<ONSET> 행·adapter_tagact). 지문 키 slot_blocks(끄면 키 없음).
  E13(2026-11-02 사용자 "번역을 다시 넣어서 학습할 수 있도록 … 이 고정 슬롯도 해제해보자"): --slot-blocks 도 번역(--dlg-trans·--sim-trans-frac)을 받는다 — 슬롯 칸마다
  [태그] 전사 뒤 [<TRANS>] 번역(cst_dialogue 모듈 설명 E13). --slot-gate N: 라운드마다 정답 활동이 최근 N 청크 안에 있는 슬롯만 태그(model.config.slot_gate — 디코드는
  예측 활동 창 ∪ 열린 발화), --gate-fa: 헛 구간. 지문 키 slot_gate(끄면 키 없음).
  --slot-audio(E10, 2026-10-29): 슬롯 추출기(model.add_slot_audio)가 슬롯별 소리 토큰을 태그 자리에 더하고, --mix-drop 확률로 섞인 소리 토큰을 지운다. 지문 키 slot_audio.
  --n3d PATH(E11, 2026-10-29): 고정 화자 분리 전문가 Nemotron 3 Diarization(model.add_n3d)의 활동이 슬롯 활동 입력이 된다(학습 청크 --n3d-train-chunk, 추론 --n3d-chunk).
    가중치는 체크포인트 밖(config 에 경로). 지문 키 n3d.
나머지(인코더 학습·att_context 고정·SpecAugment·속도/잡음/음량 증강·packing·선점 저장·재개 지문·final 재로드 parity)는 semcommit_train.py 와 같은 규약.
지문: config.stage3(형식·코퍼스·split·창·EOT 규칙·가중치·δ·replay 입력·tokenizer 어휘) — 재개는 같은 지문일 때만(--force-resume 로만 진행).
저장: checkpoint-N 을 지우지 않는다(--save-total-limit 0). 완료 시 out-dir/final + parity.json(PARITY OK/FAIL) + DONE."""
import os, sys, json, time, argparse, random, hashlib, re, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, help="Stage 2 HF 체크포인트(config.json model_type=vapasr, tokenizer 포함)")
ap.add_argument("--nemotron-dir", default=None, help="*.nemo 가 든 디렉토리 — 체크포인트에 인코더가 없을 때만 쓰인다")
ap.add_argument("--out-dir", required=True)
g = ap.add_argument_group("대화(Stage 3)")
g.add_argument("--data", required=True, help="Phase 2 폴더(<corpus>.refined.dialogues.jsonl, mono/)")
g.add_argument("--dlg-corpora", default="aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc")
g.add_argument("--dlg-group-map", default="", help="코퍼스를 새 대화 그룹으로 따로 가중('corpus:dlg_이름', 쉼표 — 2026-10-26 F1: swbd:dlg_sw. 그 코퍼스는 기본 그룹"
               "(dlg_<언어>·dlg_mt·dlg_far)에서 빠지고 --weights 의 그 그룹 가중치를 쓴다 — 같은 그룹에 넣은 기존 셋이 묽어지지 않게). 기본 없음(예전과 같다)")
g.add_argument("--split", required=True, help="held-out split(splits/v2.json) — 그 conv_id 는 학습에서 뺀다")
g.add_argument("--mono-cache", default=None, help="mono 캐시(기본 <data>/mono, p2_build_mono.py)")
g.add_argument("--window", default="20,40", help="창 길이 범위 lo,hi(s)"); g.add_argument("--hop", type=float, default=10.0)
g.add_argument("--start-mode", choices=("grid", "mixed"), default="grid", help="mixed = 무음 격자 창 + 임의 시작 창(--random-frac)")
g.add_argument("--random-frac", type=float, default=0.5)
g.add_argument("--eot-mode", choices=("soft", "hard"), default="soft", help="soft = p_end 그대로, hard = p_end ≥ 0.5 → 1 아니면 0")
g.add_argument("--backchannel", default="off", help="맞장구 선택 'max_s,max_tokens'(예 1.0,3) | off(Phase 2 규칙)")
g.add_argument("--decision-weight", type=float, default=2.0, help="EOT 결정 위치 가중(pos_weight 덮어쓰기; Phase 2 eot_weight 2)")
g.add_argument("--turn-weight", type=float, default=2.0, help="config.turn_weight(<EOT> 타깃 가중 — 결정 위치는 --decision-weight 가 덮는다)")
g.add_argument("--max-windows", type=int, default=0, help="대화 셋마다 창 상한(0 = 전부; 스모크)")
g.add_argument("--sim-data", default=None, help="합성 2 인 대화 폴더(쉼표로 여럿 — 셋마다 그 파일이 있는 폴더를 쓴다; experiments/sim_build_dialogues.py 산출: sim_<ko|en|de>.refined.dialogues.jsonl + mono/) — "
               "가중치 키 sim_ko·sim_en·sim_de 로 쓴다(<EOT> 감독 없음, 전사·화자 태그·화자 활동만)")
g.add_argument("--revive-dir", default=None, help="격리 발화 되살리기(experiments/stage3_revive_tokens.py 산출 <corpus>.revive.jsonl 폴더) — 실제 대화에서 "
               "asr_disagree 로 격리된 발화를 입력 문맥으로 되살려 화자 태그·<EOT> 를 학습한다(전사 label 은 −100; cst_dialogue 모듈 설명)")
g.add_argument("--recover-p", type=float, default=0.0, help="정체 회복 학습: 대화 창마다 문맥 슬롯 한 구간을 뒤집을 확률(0 = 끔; cst_dialogue 모듈 설명)")
g.add_argument("--recover-boundary-frac", type=float, default=0.5, help="회복 지점을 다음 교대 경계(태그 없이 이어 씀)로 둘 비율, 나머지는 run 안(바로잡는 태그)")
g.add_argument("--recover-max-lag", type=int, default=8, help="run 안 회복의 최대 거리(방출 원소 수)")
g.add_argument("--recover-max-boundaries", type=int, default=2, help="경계 회복을 몇 번째 교대 경계까지 고를지")
g.add_argument("--dlg-trans", action="append", default=[], help="Stage 4 대화 발화 번역 타깃(stage4_dialogue_units → ast_mono_generate → ast_build_targets --pack; "
               "반복·쉼표·@목록) — 주면 대화 창을 --dlg-ast-frac 확률로 번역 모드로")
g.add_argument("--dlg-trans-input", action="append", default=[], help="되살린 격리 발화의 번역 타깃(--revive-dir 과 함께; stage4_dialogue_units --revive-dir) — 입력 문맥으로만")
g.add_argument("--dlg-ast-frac", type=float, default=0.5, help="번역 타깃이 있을 때 대화 창을 번역 모드로 둘 확률(나머지는 Stage 3 형식)")
g.add_argument("--sim-trans-frac", type=float, default=0.8, help="교차 언어 합성 셋(sim_<l1>-<l2>) 창을 번역 모드로 둘 확률")
g.add_argument("--trans-lead-space", action="store_true", help="번역 행마다 첫 단어를 앞 공백 토큰으로(cst_dialogue.lead_space_row — 행을 이은 번역 흐름에서 단어가 붙지 않게, Stage 4 본 학습부터). 대화·교차 언어 합성 모두에 적용, 지문에 들어간다")
g.add_argument("--spk-act-weight", type=float, default=0.0, help="화자 슬롯 활동 보조 손실 가중(0 = 끔): LLM audio 위치에 2 슬롯 헤드, 청크마다 'A/B 가 말하는 중' BCE")
g.add_argument("--max-speakers", type=int, default=2, help="V2: 창 화자 상한(최대 4 — 슬롯 C·D = <SPK_3>·<SPK_4>, model.enable_slots·화자 헤드 4 출력). 모든 대화 셋에 같은 슬롯 수")
g.add_argument("--declare-all-p", type=float, default=0.5, help="V2(--max-speakers > 2): 학습 창을 이 확률로 슬롯 전부 선언(나머지는 max(2, 화자 수)) — 추론 때 화자 수를 모르면 최대를 선언")
g.add_argument("--window-draws", type=int, default=0, help="V2: 창 draw 수 R(> 0 이면 격자 대신 대화를 U(--window) 창으로 이어 덮고 epoch e 에 draw e mod R — 매 epoch 다른 창). 화자 상한을 넘는 창은 줄인다")
g.add_argument("--shrink-speakers", action="store_true", help="2026-10-28 E9b: 격자 창도 화자 상한을 넘으면 버리지 않고 (상한+1) 번째 화자가 처음 말하기 직전까지 줄인다(길이 ≥ --window 하한일 때만 — "
               "--window-draws 는 늘 줄인다). 긴 창(20–90 s)에서 회의 창이 버려지지 않게. 지문 키 shrink_speakers")
g.add_argument("--enc-tap-layer", type=int, default=-1, help="V2-a 두 스트림 입력: 인코더 이 층(0 부터) 출력을 adapter_tap 으로 LLM 입력에 더한다(−1 = 끔, V1). 새로 붙이면 마지막 Linear 0 초기화")
g.add_argument("--enc-tap-hidden", type=int, default=0, help="adapter_tap 은닉 차원(0 = adapter 와 같게)")
g.add_argument("--change-tag-weight", type=float, default=0.0, help="V3-p1 C1: 교대 태그 결정 가중(다른 슬롯 태그 + 단어 첫 글자 위치의 pos_weight, 0 = 끔; cst_dialogue.decision_positions)")
g.add_argument("--hold-tag-weight", type=float, default=0.0, help="V3-p1 C1: <EOT> 뒤 머무름 결정 가중(같은 슬롯 재개 태그 | 태그 없는 단어 첫 글자, 0 = 끔)")
g.add_argument("--decision-weight-sets", choices=("all", "xl"), default="all", help="교대·머무름 가중을 걸 셋: all = 대화·합성 셋 전부, xl = 교차 언어 합성(sim_<l1>-<l2>[-접미사])만")
g.add_argument("--same-lang-change-weight", type=float, default=None, help="V3 R2m: 같은 언어 셋(dlg_*·sim_<언어>)의 교대 태그 가중 — 이것이나 --same-lang-hold-weight 를 주면 "
                                                                       "같은 언어 셋은 이 두 값만 쓴다(빠진 쪽 0 = 끔). 교차 언어 셋은 --change-tag-weight·--decision-weight-sets 그대로")
g.add_argument("--same-lang-hold-weight", type=float, default=None, help="V3 R2m: 같은 언어 셋의 <EOT> 뒤 머무름 결정 가중(--same-lang-change-weight 참고)")
g.add_argument("--eot-drop-synth", type=float, default=0.0, help="V3-p1 C5: 합성(meta.mask_eot) 창을 이 확률로 골라 감독 없는 <EOT> 를 문맥에서 뺀다(그 라운드 <NEXT_AUDIO> 는 가림, 0 = 끔)")
g.add_argument("--eot-drop-sets", choices=("xl", "synth"), default="xl", help="EOT-문맥 dropout 을 걸 셋: xl = 교차 언어 합성만, synth = 합성 셋(sim_*) 전부")
g.add_argument("--xl-slot-order", choices=("lang", "arrival"), default="lang", help="V3 R4: 교차 언어 창 슬롯 — lang = 언어 고정(prefix 가 슬롯마다 언어), arrival = 먼저 말한 화자가 A(prefix 는 두 언어만, cst_head_text_xl)")
g.add_argument("--no-eot", action="store_true", help="<EOT> 없는 대화 학습(2026-10-23 — 턴 예측은 주 목표가 아니고 화자 귀속이 중요하다): 대화·합성 셋 모두에서 <EOT> 를 정답·문맥 "
               "모두에서 뺀다. C1 머무름 가중은 뺀 <EOT> 가 놓였을 청크 끝을 기준점으로(cst_dialogue no_eot). <EOT> 행은 켜 두되 정답이 없다. 지문 키 no_eot(끄면 없음)")
g.add_argument("--target-spk", action="store_true", help="E2 대상 화자 모드(2026-10-25, wiki output-diarization-integration-20261025): 대화·합성 창마다 말하는 화자 하나만 "
               "한 슬롯 정답으로 쓰고, 그 화자 80 ms 활동을 인코더 pre-encode 출력에 주입한다(adapter_inject — SSA). 번역 타깃과 함께 쓰지 않는다. 지문 키 target_spk")
g.add_argument("--inject-hidden", type=int, default=512, help="--target-spk 주입 FFN 은닉 차원(새로 붙이면 마지막 Linear 0 초기화 → 붙인 직후 = 원래 모델)")
g.add_argument("--slot-input", action="store_true", help="E5 슬롯 활동 입력(2026-10-26, wiki output-diarization-integration-20261025 6 절): 창의 슬롯 활동(정답을 --act-noise 로 흔든 것)을 "
               "오디오 토큰에 Σ a_s·e_s 로 더한다(adapter_slot — 0 초기화 → 붙인 직후 = 원래 모델). 출력 형식은 그대로(한 스트림). 지문 키 slot_input")
g.add_argument("--act-noise", default="", help="--slot-input 흔들기 'clean=0.3,drop=0.05,short=0.4,short_frac=0.05,jitter=3,ovl=0.3,fa=0.002'(빈 값 = 기본, vapasr/data/act_noise)")
g.add_argument("--diar-head", default=None, help="E6(2026-10-26): 인과 화자 분리 헤드 시작점(E1c diar.pt — cfg·taps·state). 인코더 탭 위에 붙여 Sort·PIT 손실을 함께 걸고, "
               "슬롯 활동 입력을 표본마다 확률 ρ 로 헤드 예측(아니면 흔든 정답)으로 준다. --slot-input 과 함께. 지문 키 diar_head")
g.add_argument("--diar-weight", type=float, default=0.5); g.add_argument("--act-pred-ramp", default="0.3,0.9", help="예측 활동을 쓸 확률 ρ 의 시작,끝(step 따라 선형)")
g.add_argument("--act-threshold", type=float, default=0.45, help="헤드 예측 활동 임계(E1 AMI dev 에서 고른 값)")
g.add_argument("--act-shift", type=int, default=0, help="E8(2026-10-27): 헤드 위치 p 의 출력이 프레임 p − R 의 활동을 맞히게(헤드 손실 정답·흔든 정답 입력을 R 만큼 민다) — "
               "인과 헤드가 R 프레임(80 ms) 뒤를 보고 판단하고, 받아쓰기 지연 δ ≥ R 이면 지연이 늘지 않는다(R ≤ 가장 짧은 --delays). --diar-head 와 함께. 지문 키 act_shift")
g.add_argument("--global-slots", type=float, default=0.0, help="E7(2026-10-27): 학습 창을 이 확률로 대화 전체에서 먼저 말한 순서 슬롯으로(회의 통째·실시간 디코드는 AOSC 로 "
               "회의 내내 같은 슬롯 — 창 안 첫 화자가 A 가 아닐 수 있다). 헤드(--diar-head)가 있으면 헤드 손실은 창 도착 순서, 예측 입력은 정답 자리로 맞춘다(diar_align). 지문 키 global_slots")
g.add_argument("--slot-blocks", action="store_true", help="E9(2026-10-28, wiki output-diarization-integration-20261025 9 절): 대화·합성 창을 슬롯 블록 형식으로 — 청크마다 오디오 뒤 "
               "선언 슬롯 태그를 모두 입력으로 넣고 각 태그 뒤에 그 슬롯 출력(<BEGIN>·전사·<EOT>), 멈춤은 다음 입력 위치 label <NEXT_AUDIO>(cst_dialogue 모듈 설명 E9). "
               "<BEGIN> = <ONSET>(episode 시작 + 4 청크), 태그 임베딩에 그 슬롯 최근 활동(adapter_tagact). 정체 회복·교대/머무름 가중·EOT-문맥 dropout·"
               "no_eot·대상 화자·act_shift 없이(번역은 E13 — 슬롯 칸마다 <TRANS> 번역 칸), --slot-input 과 함께. 지문 키 slot_blocks")
g.add_argument("--slot-audio", action="store_true", help="E10(2026-10-29, wiki output-diarization-integration-20261025 10 절): 슬롯 추출기(vapasr/hf/slot_audio — 인코더 마지막 층 + "
               "화자 단서 임베딩 + 슬롯 활동 → 슬롯별 소리 토큰을 태그 자리에, 출력 0 초기화). --slot-blocks·--diar-head 와 함께. 지문 키 slot_audio")
g.add_argument("--slot-audio-dim", type=int, default=512); g.add_argument("--slot-audio-blocks", type=int, default=4); g.add_argument("--slot-audio-kernel", type=int, default=15)
g.add_argument("--mix-drop", type=float, default=0.5, help="--slot-audio: 학습 표본마다 이 확률로 섞인 소리 토큰을 빈 임베딩으로 바꾼다(슬롯 소리에 기대게 — 평가는 --mix-off 로 켜고 끈 둘 다 잰다)")
g.add_argument("--n3d", default=None, help="E11(2026-10-29, wiki output-n3d-branch-20261029): 고정 화자 분리 전문가 Nemotron 3 Diarization(.nemo — vapasr/hf/n3d)의 "
               "활동을 슬롯 활동 입력으로(헤드 예측 대신 — 표본마다 확률 ρ 로 전문가 활동을 정답 슬롯 자리로 맞춰, 아니면 흔든 정답). 가중치는 체크포인트에 넣지 않고 경로만 "
               "config 에 남긴다. --slot-input 과 함께. 지문 키 n3d")
g.add_argument("--n3d-train-chunk", type=int, default=16, help="--n3d 학습 스트리밍 청크(80 ms 프레임 — 고리 수를 줄인다, 모델 카드의 지연별 DER 차이는 작다)")
g.add_argument("--n3d-chunk", type=int, default=4, help="--n3d 추론 청크(80 ms 프레임, 4 = 인코더 블록 [56,3] 경계 — 지연을 더하지 않는다)")
g.add_argument("--n3d-threshold", type=float, default=0.5, help="--n3d 활동 임계(Nemotron 3.5 Transcribe 와 같은 0.5)")
g.add_argument("--declare-range", action="store_true", help="2026-10-28: 같은 언어 창의 선언 슬롯 수(<SPK_*> 개수)를 [창 화자 수, --max-speakers] 에서 고르게 뽑는다"
               "(--declare-all-p 대신 — 추론 때 선언 수를 화자 수 이상 아무 값으로나). 지문 키 declare_range")
g.add_argument("--slot-shuffle", type=float, default=0.0, help="2026-10-30 활동 추종 학습(wiki output-n3d-branch-20261029 6.8 절): 같은 언어 학습 창을 이 확률로 선언 슬롯 안에서 "
               "화자 자리를 무작위로(빈 자리는 말 없는 가짜 화자) — 태그·활동 타깃·슬롯 활동 입력이 모두 따른다. 먼저 말한 순서라는 단서를 지워 슬롯을 활동 입력으로만 고르게 한다"
               "(회의 통째는 슬롯이 회의 단위 도착 순서). --slot-input 과 함께, 헤드 손실은 창 도착 순서(diar_align). 지문 키 slot_shuffle")
g.add_argument("--slot-remap", type=float, default=0.0, help="2026-10-30 활동 추종 학습(wiki output-n3d-branch-20261029 6.8 절): 학습 창을 이 확률로 창 안 무음 틈(0.4 s 이상)에서 "
               "슬롯 대응을 다른 순열로 바꾼다 — 같은 목소리라도 바꾼 뒤에는 활동이 가리키는 새 슬롯에 써야 맞다(LLM 이 자기 화자 추적·이력보다 활동을 따르게). "
               "이벤트열·활동 타깃·흔든 입력은 나눠 따르고, 헤드 손실·전문가 정렬은 바꾸기 전 순서로(remap_k·remap_perm). --slot-input·--slot-blocks 와 함께. 지문 키 slot_remap")
g.add_argument("--begin-weight", type=float, default=2.0, help="--slot-blocks: <BEGIN> label 위치 가중(pos_weight)")
g.add_argument("--tag-act-lags", type=int, default=12, help="--slot-blocks: 태그 임베딩에 더할 슬롯 활동 청크 수(a_s(k) … a_s(k − n + 1))")
g.add_argument("--slot-gate", type=int, default=0, help="E13(2026-11-02 사용자 \"<SPK_A~D> 까지 활성화할 필요는 없을거 같아. 이 고정 슬롯도 해제해보자\"): 라운드마다 "
               "정답 활동(전사 없는 발화 포함)이 최근 N 청크 안에 있는 슬롯과 그 라운드 출력이 있는 슬롯의 태그만 넣는다(cst_dialogue.gate_rounds — 0 = 끔, "
               "1 ≤ N ≤ --tag-act-lags, 보통 같게). 디코드는 예측 활동 창 ∪ 열린 발화(<ONSET> 뒤 <EOT> 전 — model.config.slot_gate). --slot-blocks 와 함께. 지문 키 slot_gate")
g.add_argument("--gate-fa", type=float, default=0.0, help="--slot-gate: 학습 창의 슬롯마다 이 확률로 말 없는 0.3–3.2 s 구간 하나를 더 연다(예측 활동의 헛 검출 — 태그 뒤 멈춤을 배운다)")
g.add_argument("--x-slot", action="store_true", help="E14(2026-11-03 사용자 \"N3D 가 놓친 말을 받아쓰도록 게이트를 고칠 방법\" → 비귀속 칸 + 확률 활동): 라운드마다 슬롯 칸 뒤 X 칸"
               "(<SPK_5>). 학습 게이트는 정답 활동에서 N3D 놓침을 흉내 내고(--miss-*), 닫힌 슬롯의 글자는 X 칸이 낸다(cst_dialogue x_slot — 칸 열림은 디코더와 같다). "
               "--slot-gate 와 함께. 지문 키 x_slot")
g.add_argument("--miss-drop", type=float, default=0.0, help="--x-slot: 1 s 이상 말을 학습 게이트에서 통째로 뺄 확률(전문가 활동 입력도 같은 자리를 지운다)")
g.add_argument("--miss-short", type=float, default=0.0, help="--x-slot: 1 s 보다 짧은 말을 통째로 뺄 확률")
g.add_argument("--miss-late", type=float, default=0.0, help="--x-slot: 남은 말의 앞 0.3–2 s(말 길이 절반까지)를 뺄 확률")
g.add_argument("--n3d-soft", action="store_true", help="E14: 고정 전문가 활동을 0/1 대신 확률 그대로(학습·추론 — model.config.n3d_soft, 게이트는 > 0.5). --n3d 와 함께. 지문 키 n3d_soft")
g.add_argument("--set-phone", default="", help="셋별 창 단위 전화 대역 확률 'KEY:P[,KEY:P]'(대화·합성 셋만, 그 셋만 --phone-aug 를 덮는다 — P = 0 이면 끔, 예 sim_en-de-tel:0)")
g = ap.add_argument_group("replay(Stage 2 단일 화자)")
g.add_argument("--replay-words", action="append", default=[], help="EN·KO words.jsonl(반복·쉼표·@목록) — 행의 lang 으로 rep_en/rep_ko 에 나뉜다")
g.add_argument("--replay-de-words", action="append", default=[], help="독일어 words.jsonl(반복·쉼표·@목록) → rep_de")
g.add_argument("--replay-set", action="append", default=[], help="따로 가중하는 단일 화자 셋 '이름:언어:words(쉼표·@목록)' — 이름은 rep_ 로 시작, --weights 에 그 이름으로 비율을 준다. --replay-files-frac 를 적용하지 않는다(작은 셋이 파일 해시로 통째로 빠지지 않게). 예: rep_chde:German:@callhome-de-words-train.list")
g.add_argument("--replay-files-frac", type=float, default=1.0, help="words 파일 중 이 비율만 읽는다(경로 해시로 고정 선택 — 초기화 시간 절약)")
g.add_argument("--replay-max-items", type=int, default=0, help="replay 셋마다 항목 상한(0 = 전부)")
g.add_argument("--ast-targets", action="append", default=[], help="번역 타깃(ast_build_targets.py 산출, pack/ 권장; 반복·쉼표·@목록)")
g.add_argument("--ast-frac", type=float, default=0.8, help="번역 타깃이 있는 replay 항목을 이 확률로 번역 모드로")
g.add_argument("--offline-frac", type=float, default=0.0, help="replay 항목의 final(오프라인) 모드 비율")
g.add_argument("--gain-augment", default="off", help="replay 음량 증강 'p,lo_db,hi_db' | off"); g.add_argument("--gain-augment-langs", default="English")
g.add_argument("--speed-perturb", default="", help="replay 속도 변형 배율 목록(예 0.9,1.0,1.1)")
g.add_argument("--noise-bank", default=None, help="replay 잡음·잔향 뱅크"); g.add_argument("--noise-p", type=float, default=0.4)
g.add_argument("--noise-snr", default="5,30"); g.add_argument("--music-snr", default="10,30"); g.add_argument("--rir-p", type=float, default=0.2)
g.add_argument("--phone-aug", type=float, default=0.0, help="전화 대역 증강 확률(vapasr/data/phone_aug — 8 kHz·300–3400 Hz 대역·μ-law, 영위상). 대화 창(실제·합성)과 단일 화자 셋 모두에, 단일 화자는 잡음·잔향 뒤에 건다. 0 = 끔(지문 그대로)")
g.add_argument("--phone-aug-mulaw", type=float, default=0.5, help="전화 대역 증강 안에서 μ-law 8 비트 양자화 확률")
g.add_argument("--far-aug", type=float, default=0.0, help="원거리 증강 확률(vapasr/data/far_aug, 2026-10-26 — 화자별 신호 캐시가 있으면 화자마다 같은 방의 다른 RIR, "
               "없으면 mono 에 RIR 하나, 그 뒤 잡음). 0 = 끔(지문 그대로)")
g.add_argument("--far-sets", default="dlg_ko,dlg_en,dlg_de,dlg_mt", help="원거리 증강을 걸 셋 이름 패턴(쉼표, fnmatch — 예 'dlg_*,sim_*'). 원거리 녹음 셋(dlg_far)은 이미 원거리라 기본에 없다")
g.add_argument("--far-snr", default="5,25"); g.add_argument("--far-noise-p", type=float, default=0.8); g.add_argument("--far-gain-db", type=float, default=3.0)
g.add_argument("--far-real-p", type=float, default=0.3, help="실제 RIR 풀에서 고를 확률(나머지는 모의 방 RoomNNN)"); g.add_argument("--far-bank", default=None, help="RIR·잡음 뱅크(기본 --noise-bank)")
g = ap.add_argument_group("섞기·학습")
g.add_argument("--weights", default="dlg_ko:1,dlg_en:1,dlg_de:1,rep_ko:0.5,rep_en:0.5,rep_de:0.5", help="셋 가중치 'name:w,…' — step 비율(없는 셋 = 0)")
g.add_argument("--delays", default="4,5,6,7,8,9,10"); g.add_argument("--next-weight", type=float, default=0.3); g.add_argument("--next-weight-ko", type=float, default=0.15)
g.add_argument("--lr", type=float, default=6e-5); g.add_argument("--lr-adapter", type=float, default=1e-3); g.add_argument("--lr-encoder", type=float, default=1e-5)
g.add_argument("--warmup", type=int, default=500); g.add_argument("--wd", type=float, default=0.01)
g.add_argument("--epochs", type=float, default=1.0); g.add_argument("--max-steps", type=int, default=0)
enc_g = ap.add_mutually_exclusive_group()
enc_g.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true"); enc_g.add_argument("--train-encoder", dest="freeze_encoder", action="store_false")
ap.set_defaults(freeze_encoder=False)
ap.add_argument("--encoder-right-context", type=int, default=None); ap.add_argument("--encoder-context-sampling", choices=["multi", "fixed"], default="fixed")
ap.add_argument("--spec-augment", default="off")
gc_g = ap.add_mutually_exclusive_group()
gc_g.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true"); gc_g.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
ap.set_defaults(grad_ckpt=True)
lg_g = ap.add_mutually_exclusive_group()
lg_g.add_argument("--no-liger", dest="liger", action="store_false"); lg_g.add_argument("--liger", dest="liger", action="store_true")
ap.set_defaults(liger=True)
ap.add_argument("--save-every", type=int, default=500); ap.add_argument("--save-total-limit", type=int, default=0, help="0 = checkpoint 를 지우지 않는다(원격 삭제 금지)")
ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--num-workers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", default="auto", help="auto | none | <checkpoint dir>"); ap.add_argument("--force-resume", action="store_true"); ap.add_argument("--gpu", default=None)
ap.add_argument("--no-check-save", action="store_true"); ap.add_argument("--check-items", type=int, default=64, help="학습 전 셋마다 시퀀스 불변식 검사 항목 수")
ap.add_argument("--report-to", default="tensorboard"); ap.add_argument("--dry-run", action="store_true")
from vapasr.speedup import add_speed_args, apply_speedups, training_args_kwargs
add_speed_args(ap)
a = ap.parse_args()

DLG_LANG = {"aihub71631": "Korean", "aihub134-1": "Korean", "aihub134-2": "Korean", "nikl2020": "Korean", "otoSpeech": "English", "kec": "German", "ztc": "German", "nsc": "German", "callhome-de": "German", "ami": "English", "icsi": "English", "notsofar": "English"}
DLG_LANG.update({"ami-far": "English", "icsi-far": "English", "notsofar-far": "English", "chime6tr": "English", "chime6tr-far": "English",   # 2026-10-26 데이터 확장
                 "swbd": "English", "nia23mt": "Korean"})
# CHiME-6 train 세션의 키는 chime6tr(사용자 2026-10-26 "train 세션만 학습에 넣기" — wiki decision-chime6-train-sessions-20261026): phase2/chime6.dialogues.jsonl 은 dev 평가 셋이고
# 예전 평가 스크립트(p2_eval_lanes·p2_eval_sessions·s3_eval_multispeaker_mono)가 <corpus>.refined 를 먼저 찾으므로 학습 파일은 이름이 겹치면 안 된다. "chime6" 은 받지 않는다.
MEETING = {"ami", "icsi", "notsofar", "chime6tr", "nia23mt"}                                   # V2: 회의(3–9 인)는 따로 가중한다 — 영어 dlg_mt, 한국어 dlg_mtko(NIA23, 2026-10-26)
FAR = {"ami-far", "icsi-far", "notsofar-far", "chime6tr-far"}                                  # 같은 회의의 원거리 녹음(p2_build_far) — dlg_far 로 따로 가중한다
SHORT = {"Korean": "ko", "English": "en", "German": "de"}
delays = tuple(int(x) for x in a.delays.split(",") if x.strip())
if not delays or not all(1 <= d <= 10 for d in delays): sys.exit(f"--delays {a.delays}: 1–10 의 쉼표 목록")
if a.act_shift > min(delays): sys.exit(f"--act-shift {a.act_shift} > 가장 짧은 지연 δ{min(delays)} — 그 δ 에서는 받아쓸 때 아직 활동이 오지 않는다")
corpora = [c for c in a.dlg_corpora.split(",") if c]
if not all(c in DLG_LANG for c in corpora): sys.exit(f"--dlg-corpora: 알 수 없는 코퍼스 {[c for c in corpora if c not in DLG_LANG]} (알려진 것 {sorted(DLG_LANG)})")
BASE_GROUPS = (("dlg_ko", "Korean"), ("dlg_en", "English"), ("dlg_de", "German"), ("dlg_mt", "English"), ("dlg_far", "English"), ("dlg_mtko", "Korean"))
GMAP = {}                                                                                    # --dlg-group-map: 코퍼스 → 새 그룹
for x in (y.strip() for y in a.dlg_group_map.split(",")):
    if not x: continue
    c, _, gname = (z.strip() for z in x.partition(":"))
    if c not in corpora or c in GMAP or not re.fullmatch(r"dlg_[a-z0-9]+", gname) or gname in dict(BASE_GROUPS):
        sys.exit(f"--dlg-group-map 'corpus:dlg_이름' — --dlg-corpora 의 코퍼스 하나에 기본 그룹({', '.join(dict(BASE_GROUPS))})이 아닌 새 이름: {x!r}")
    GMAP[c] = gname
EXTRA_GROUPS = []
for gname in sorted(set(GMAP.values())):
    ls = {DLG_LANG[c] for c, g in GMAP.items() if g == gname}
    if len(ls) != 1: sys.exit(f"--dlg-group-map: {gname} 에 언어가 섞였다 {sorted(ls)}")
    EXTRA_GROUPS.append((gname, ls.pop()))
win = tuple(float(x) for x in a.window.split(","))
if len(win) != 2 or not 0 < win[0] <= win[1]: sys.exit(f"--window lo,hi — {a.window}")
if a.declare_range and a.max_speakers < 3: sys.exit("--declare-range 는 --max-speakers 3 이상과 함께(선언 수를 고를 범위)")
if not 2 <= a.max_speakers <= 4 or not 0 <= a.declare_all_p <= 1 or a.window_draws < 0: sys.exit(f"--max-speakers 2–4 · --declare-all-p [0, 1] · --window-draws ≥ 0 — {a.max_speakers} {a.declare_all_p} {a.window_draws}")
if any(c in MEETING | FAR for c in corpora) and a.max_speakers < 3: sys.exit("회의 코퍼스(ami·icsi·notsofar·chime6tr·nia23mt, 원거리 포함)는 --max-speakers 3 이상과 함께")
VKW = dict(max_speakers=a.max_speakers, window_draws=a.window_draws, shrink_speakers=a.window_draws > 0 or a.shrink_speakers, **({"declare_all_p": a.declare_all_p} if a.max_speakers > 2 else {}),
           **({"global_slots": a.global_slots} if a.global_slots > 0 else {}),                 # E7(2026-10-27): 끄면 키 없음(데이터셋 기본 0 — 난수 그대로)
           **({"slot_blocks": True, "begin_weight": a.begin_weight} if a.slot_blocks else {}),   # E9(2026-10-28): 끄면 키 없음
           **({"declare_range": True} if a.declare_range else {}),                               # 2026-10-28: 끄면 키 없음
           **({"slot_shuffle": a.slot_shuffle} if a.slot_shuffle > 0 else {}),                   # 2026-10-30 활동 추종 학습: 끄면 키 없음
           **({"slot_remap": a.slot_remap} if a.slot_remap > 0 else {}),                         # 2026-10-30 활동 추종 학습: 끄면 키 없음
           **({"slot_gate": a.slot_gate, "gate_fa": a.gate_fa} if a.slot_gate else {}),          # 2026-11-02 E13 활동 게이트: 끄면 키 없음
           **({"x_slot": True, "miss_drop": a.miss_drop, "miss_short": a.miss_short, "miss_late": a.miss_late} if a.x_slot else {}),   # 2026-11-03 E14: 끄면 키 없음
           **({"xl_order": a.xl_slot_order} if a.xl_slot_order != "lang" else {}))   # 모든 대화 셋 공통(V2, V3 R4)
bc = None if a.backchannel == "off" else tuple(float(x) for x in a.backchannel.split(","))
if bc is not None and len(bc) != 2: sys.exit(f"--backchannel max_s,max_tokens | off — {a.backchannel}")
bc = (bc[0], int(bc[1])) if bc else None
if a.recover_p and not (0 < a.recover_p <= 1 and 0 <= a.recover_boundary_frac <= 1 and a.recover_max_lag >= 1 and a.recover_max_boundaries >= 1):
    sys.exit(f"--recover-*: 범위 밖 p {a.recover_p} · boundary_frac {a.recover_boundary_frac} · max_lag {a.recover_max_lag} · max_boundaries {a.recover_max_boundaries}")
from vapasr.hf.stage3 import parse_weights, WeightedRoundRobinLoader, collate_stage3, file_digest, fingerprint_diff, set_kind, parse_set_phone, decision_weights
try: weights = parse_weights(a.weights)
except ValueError as e: sys.exit(f"--weights: {e}")
if a.change_tag_weight < 0 or a.hold_tag_weight < 0 or not 0 <= a.eot_drop_synth <= 1: sys.exit(f"--change-tag-weight·--hold-tag-weight ≥ 0 · --eot-drop-synth [0, 1] — {a.change_tag_weight} {a.hold_tag_weight} {a.eot_drop_synth}")
if any(x is not None and x < 0 for x in (a.same_lang_change_weight, a.same_lang_hold_weight)): sys.exit(f"--same-lang-change-weight·--same-lang-hold-weight ≥ 0 — {a.same_lang_change_weight} {a.same_lang_hold_weight}")
SL = a.same_lang_change_weight is not None or a.same_lang_hold_weight is not None                 # V3 R2m: 같은 언어 셋 따로 가중
try: set_phone = parse_set_phone(a.set_phone)
except ValueError as e: sys.exit(str(e))
if any(weights.get(k, 0) <= 0 for k in set_phone): sys.exit(f"--set-phone 셋이 --weights 에 없다: {sorted(k for k in set_phone if weights.get(k, 0) <= 0)}")
gain = None if a.gain_augment == "off" else tuple(float(x) for x in a.gain_augment.split(","))
gain_langs = tuple(x for x in a.gain_augment_langs.split(",") if x)
speeds = tuple(float(x) for x in a.speed_perturb.split(",") if x.strip())
if not 0 < a.replay_files_frac <= 1: sys.exit("--replay-files-frac 은 (0, 1]")
if a.target_spk and a.dlg_trans: sys.exit("--target-spk 는 번역 타깃(--dlg-trans)과 함께 쓰지 않는다(E2 는 받아쓰기 상한 시험)")
if a.target_spk and a.inject_hidden <= 0: sys.exit("--inject-hidden > 0")
if a.slot_input and a.target_spk: sys.exit("--slot-input 은 --target-spk 와 함께 쓰지 않는다(E5 는 한 스트림 형식)")
if a.diar_head and not a.slot_input: sys.exit("--diar-head 는 --slot-input 과 함께(E6 — 헤드 예측을 슬롯 활동 입력으로 준다)")
if a.act_shift < 0 or (a.act_shift and not a.diar_head): sys.exit("--act-shift R(≥ 0)은 --diar-head 와 함께(E8)")
if not 0.0 <= a.global_slots <= 1.0 or (a.global_slots > 0 and a.target_spk): sys.exit("--global-slots ∈ [0, 1], --target-spk(E2)와 함께 쓰지 않는다")
if not 0.0 <= a.slot_shuffle <= 1.0 or (a.slot_shuffle > 0 and (a.target_spk or not a.slot_input)): sys.exit("--slot-shuffle ∈ [0, 1], --slot-input 과 함께, --target-spk(E2)와 함께 쓰지 않는다")
if not 0.0 <= a.slot_remap <= 1.0 or (a.slot_remap > 0 and (a.target_spk or a.act_shift or not (a.slot_input and a.slot_blocks))):
    sys.exit("--slot-remap ∈ [0, 1], --slot-input·--slot-blocks 와 함께, --target-spk(E2)·--act-shift(E8)와 함께 쓰지 않는다")
if a.slot_blocks:                                                                               # E9: 슬롯 블록 빌더가 다루지 않는 형식·가중은 켜지 않는다(번역은 E13 이 다룬다)
    bad = [n for n, on in (("--recover-p", a.recover_p > 0), ("--change/hold-tag-weight", a.change_tag_weight > 0 or a.hold_tag_weight > 0),
                           ("--same-lang-*-weight", bool(a.same_lang_change_weight or a.same_lang_hold_weight)), ("--eot-drop-synth", a.eot_drop_synth > 0), ("--no-eot", a.no_eot),
                           ("--target-spk", a.target_spk), ("--act-shift", a.act_shift > 0)) if on]
    if bad: sys.exit(f"--slot-blocks(E9)는 {', '.join(bad)} 와 함께 쓰지 않는다")
    if not a.slot_input: sys.exit("--slot-blocks 는 --slot-input 과 함께(태그 임베딩에 슬롯 활동을 더한다)")
    if min(delays) < 4 or a.begin_weight < 0 or a.tag_act_lags < 1: sys.exit(f"--slot-blocks: 가장 짧은 δ ≥ 4(<BEGIN> 지연) · --begin-weight ≥ 0 · --tag-act-lags ≥ 1 — {delays} {a.begin_weight} {a.tag_act_lags}")
if (a.slot_gate or a.gate_fa) and not a.slot_blocks: sys.exit("--slot-gate·--gate-fa(E13)는 --slot-blocks 와 함께")
if a.slot_gate and not 1 <= a.slot_gate <= a.tag_act_lags: sys.exit(f"--slot-gate 는 1 … --tag-act-lags({a.tag_act_lags}) — {a.slot_gate}")
if not 0.0 <= a.gate_fa <= 1.0 or (a.gate_fa and not a.slot_gate): sys.exit(f"--gate-fa ∈ [0, 1], --slot-gate 와 함께 — {a.gate_fa}")
if a.x_slot and not a.slot_gate: sys.exit("--x-slot(E14)은 --slot-gate 와 함께")
if any(not 0.0 <= p <= 1.0 for p in (a.miss_drop, a.miss_short, a.miss_late)) or ((a.miss_drop or a.miss_short or a.miss_late) and not a.x_slot):
    sys.exit(f"--miss-* ∈ [0, 1], --x-slot 과 함께 — {a.miss_drop} {a.miss_short} {a.miss_late}")
if a.n3d_soft and not a.n3d: sys.exit("--n3d-soft 는 --n3d 와 함께")
if a.slot_audio and not (a.slot_blocks and a.diar_head): sys.exit("--slot-audio(E10)는 --slot-blocks 와 --diar-head 와 함께")
if a.n3d and not (a.slot_input and os.path.exists(a.n3d) and a.n3d_train_chunk >= 1 and a.n3d_chunk >= 1 and 0.0 < a.n3d_threshold < 1.0):
    sys.exit(f"--n3d 는 --slot-input 과 함께, .nemo 가 있어야 하고 청크 ≥ 1·임계 (0, 1) — {a.n3d} {a.n3d_train_chunk} {a.n3d_chunk} {a.n3d_threshold}")
if a.n3d and (a.act_shift or a.target_spk or a.enc_tap_layer >= 0): sys.exit("--n3d 는 --act-shift·--target-spk·--enc-tap-layer 와 함께 쓰지 않는다")
if a.slot_audio and not (0.0 <= a.mix_drop <= 1.0 and a.slot_audio_dim > 0 and a.slot_audio_blocks > 0 and a.slot_audio_kernel > 1): sys.exit("--mix-drop [0, 1] · --slot-audio-dim·blocks > 0 · kernel > 1")
ramp = tuple(float(x) for x in a.act_pred_ramp.split(","))
if len(ramp) != 2 or not all(0.0 <= x <= 1.0 for x in ramp): sys.exit(f"--act-pred-ramp 'ρ0,ρ1'(각 0–1): {a.act_pred_ramp}")
want_act = a.spk_act_weight > 0 or bool(a.diar_head)                                           # 정답 슬롯 활동 타깃(LLM 쪽 화자 헤드 또는 E6 화자 분리 손실)
from vapasr.data.act_noise import from_spec as act_noise_from_spec
try: act_noise = act_noise_from_spec(a.act_noise) if a.slot_input else None
except ValueError as e: sys.exit(f"--act-noise: {e}")
if not (a.batch_max_tokens or a.pack_max_tokens) and not a.dry_run: sys.exit("--batch-max-tokens 또는 --pack-max-tokens 가 필요하다(대화 창은 고정 크기 버킷 배치를 쓰지 않는다)")
from vapasr.features.online import SpecAugment
try: SpecAugment.from_spec(a.spec_augment)
except (ValueError, TypeError) as e: sys.exit(f"--spec-augment {a.spec_augment!r}: {e}")
from vapasr.data.noise_aug import from_args as noise_aug_from_args
try: naug = noise_aug_from_args(a.noise_bank, a.noise_p, a.noise_snr, a.rir_p, a.music_snr)
except (ValueError, OSError) as e: sys.exit(f"--noise-bank {a.noise_bank}: {e}")
from vapasr.data.phone_aug import PhoneAugment, AugChain
phone = PhoneAugment(p=a.phone_aug, mulaw_p=a.phone_aug_mulaw) if a.phone_aug > 0 else None
from vapasr.data.far_aug import from_args as far_from_args
try: faug = far_from_args(a.far_bank or a.noise_bank, a.far_aug, a.far_noise_p, a.far_snr, a.far_gain_db, a.far_real_p)
except (ValueError, OSError) as e: sys.exit(f"--far-aug: {e}")
if a.far_aug > 0 and faug is None: sys.exit("--far-aug 는 --far-bank 또는 --noise-bank 와 함께")
raug = AugChain(naug, phone) if phone is not None else naug                                       # 단일 화자: 잡음·잔향 → 전화 대역

rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
if a.gpu is not None and world == 1: os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def _writable_root():
    """Triton/inductor 캐시·TMPDIR: VAPASR_LOCAL_CACHE → /tmp → out-dir/.cache/<host> (semcommit_train.py 와 같은 이유 — 일부 노드는 job step 안 /tmp 쓰기 불가)."""
    import socket
    name = f"vapasr-{os.getuid()}-stage3-{os.environ.get('SLURM_JOB_ID', 'local')}"
    for base in (os.environ.get("VAPASR_LOCAL_CACHE"), "/tmp", os.path.join(os.path.abspath(a.out_dir), ".cache", socket.gethostname())):
        if not base: continue
        root = os.path.join(base, name)
        try:
            os.makedirs(root, exist_ok=True); probe = os.path.join(root, f".probe-{os.getpid()}"); open(probe, "w").close(); os.remove(probe); return root
        except OSError: continue
    raise SystemExit("쓸 수 있는 캐시 디렉토리가 없다(VAPASR_LOCAL_CACHE·/tmp·out-dir)")
_cache_root = _writable_root()
for k_, d_ in (("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor")): os.environ.setdefault(k_, os.path.join(_cache_root, f"{d_}-{local}")); os.makedirs(os.environ[k_], exist_ok=True)
if not os.access(os.environ.get("TMPDIR", "/tmp"), os.W_OK):
    os.environ["TMPDIR"] = os.path.join(_cache_root, f"tmp-{local}"); os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    import tempfile; tempfile.tempdir = None
import torch, torch.distributed as dist
gloo_pg = None
if world > 1 and not a.dry_run:
    from datetime import timedelta
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    torch.cuda.set_device(local); dist.init_process_group("nccl", timeout=timedelta(hours=3), device_id=torch.device("cuda", local))
    gloo_pg = dist.new_group(backend="gloo", timeout=timedelta(hours=1))
main = rank == 0
def log(*s):
    if main: print(*s, flush=True)
torch.manual_seed(a.seed); random.seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True
out = a.out_dir; os.makedirs(out, exist_ok=True)
if not a.dry_run and world == 1 and torch.cuda.is_available() and torch.cuda.device_count() > 1: sys.exit("GPU 가 여러 장 보인다 — --gpu N 또는 torchrun")
IN_KEYS = ("wav", "wav_len", "K", "ids", "is_audio", "chunk_of", "labels", "mask", "pos_weight", "position_ids", "labels_alt", "sem_mask", "sem_only_mask", "activity", "activity_mask", "dec_kind",
           "tag_chunk", "remap_k", "remap_perm")
LOSS_KEYS = ("loss_next", "loss_text", "top1_text", "loss_turn", "top1_turn", "loss_act")

# ── 모델 + tokenizer
from transformers import AutoTokenizer
from vapasr.data.semcommit_dataset import add_semcommit_specials
from vapasr.data.semcommit_tokens import SEM_SPECIALS, TURN_TOKEN, add_delay_ext_specials, add_trans_special
cfg_path = os.path.join(a.init, "config.json")
assert os.path.exists(cfg_path) and json.load(open(cfg_path)).get("model_type") == "vapasr", f"--init 은 HF vapasr 디렉토리: {a.init}"
assert os.path.exists(os.path.join(a.init, "tokenizer_config.json")), f"{a.init} 에 tokenizer 가 없다 — Stage 2 final 을 준다"
init_tap = int(json.load(open(cfg_path)).get("enc_tap_layer", -1))
if init_tap >= 0 and a.enc_tap_layer != init_tap: sys.exit(f"--init 에 이미 인코더 탭(층 {init_tap})이 있다 — --enc-tap-layer {init_tap} 로 이어서 학습한다(지금 {a.enc_tap_layer})")
t0 = time.time(); tok = AutoTokenizer.from_pretrained(a.init); model = None; speed = {}
if not a.dry_run:
    from vapasr.hf import VapAsrForStreamingASR
    from vapasr.data.semcommit_tokens import add_delay_tokens, add_ast_tokens
    model = VapAsrForStreamingASR.from_pretrained(a.init, encoder_path=a.nemotron_dir); cfg = model.config; old_sp = dict(cfg.sp_ids)
    assert cfg.lanes == 0, "Stage 3 는 mono(lanes=0) Stage 2 체크포인트에서 시작한다"
    model.add_semantic_tokens(tok, turn=True, reinit_turn=True)                              # <EOT> 를 턴 종료로 켠다(예약 행 재초기화·차단 해제)
    if max(delays) > 8: add_delay_tokens(model, tok)                                         # Stage 2 에서 이미 붙었으면 그대로
    add_ast_tokens(model, tok)                                                                # <TRANS>·<SPK_A/B> 차단 해제(이미 있으면 그대로)
    if a.max_speakers > 2: model.enable_slots(tok, a.max_speakers)                            # V2: 슬롯 C·D(<SPK_3>·<SPK_4>) 예약 행 재초기화·차단 해제
    if a.spk_act_weight > 0: model.add_spk_head(max(2, a.max_speakers), a.spk_act_weight)     # 화자 슬롯 활동 헤드(새로 붙이면 adapter lr — trainer.create_optimizer; 2 → 4 는 A·B 행 유지)
    if a.enc_tap_layer >= 0: model.add_enc_tap(a.enc_tap_layer, a.enc_tap_hidden or None)      # V2-a 두 스트림 입력(adapter_tap 도 adapter lr)
    if a.target_spk: model.add_spk_inject(a.inject_hidden)                                      # E2 대상 화자 활동 주입(adapter_inject 도 adapter lr)
    if a.slot_input: model.add_slot_input(max(2, a.max_speakers))                               # E5 슬롯 활동 입력(adapter_slot 도 adapter lr)
    if a.slot_blocks: model.add_slot_blocks(tok, a.tag_act_lags)                                # E9 <BEGIN>(<ONSET>) 행·차단 해제 + adapter_tagact(adapter lr)
    g0 = int(getattr(cfg, "slot_gate", 0) or 0)                                                 # E13 활동 게이트 — 디코더가 읽는다(init 이 게이트 모델이면 같은 값이어야 한다)
    if g0 and g0 != a.slot_gate: sys.exit(f"init 이 활동 게이트 {g0} 모델이다 — --slot-gate {g0} 로 이어 학습한다(지금 {a.slot_gate})")
    if a.slot_gate: cfg.slot_gate = int(a.slot_gate)
    if getattr(cfg, "x_slot", False) and not a.x_slot: sys.exit("init 이 X 칸 모델이다(config.x_slot) — --x-slot 으로 이어 학습한다")
    if a.x_slot: model.add_x_slot(tok)                                                          # E14 X 칸(<SPK_5> 행 = 슬롯 태그 평균 — 처음 켤 때만)
    if getattr(cfg, "n3d_soft", False) and not a.n3d_soft: sys.exit("init 이 전문가 확률 활동 모델이다(config.n3d_soft) — --n3d-soft 로 이어 학습한다")
    if a.n3d_soft: cfg.n3d_soft = True
    if a.diar_head:                                                                             # E6 인과 화자 분리 헤드(이름이 adapter 가 아니라 thinker lr)
        xb = torch.load(a.diar_head, map_location="cpu"); assert xb["cfg"].get("lookahead") is not None, f"인과 헤드(lookahead)가 아니다: {a.diar_head}"
        model.add_diar_head(xb["cfg"], xb["taps"], state=xb["state"], weight=a.diar_weight, threshold=a.act_threshold, ramp=ramp)
        if a.global_slots > 0: model.config.diar_align = True                                      # E7: 헤드 손실은 창 도착 순서, 예측 입력은 정답(대화 전체 순서) 자리로
        if a.act_shift: model.config.act_shift = int(a.act_shift)                                  # E8: 헤드 출력 p = 프레임 p − R(재개 checkpoint 의 config 와 같은 값)
        if xb.get("aosc"): model.config.diar_aosc = dict(xb["aosc"])                                 # AOSC 로 학습한 헤드에서 시작하면 그 설정을 남긴다(회의 통째 디코드)
        if model.spk_head is not None and a.spk_act_weight <= 0: model.config.spk_weight = 0.0   # 활동 타깃이 오므로 LLM 쪽 보조 헤드는 끈다(가중 0)
        if a.slot_audio: model.add_slot_audio(d=a.slot_audio_dim, blocks=a.slot_audio_blocks, kernel=a.slot_audio_kernel, mix_drop=a.mix_drop)   # E10(헤드 뒤 — 헤드 embed 로 시작)
    if a.n3d: model.add_n3d(a.n3d, chunk=a.n3d_chunk, rc=0, train_chunk=a.n3d_train_chunk, train_rc=0, threshold=a.n3d_threshold)   # E11 고정 화자 분리 전문가(체크포인트 밖)
    if a.slot_shuffle > 0 or a.slot_remap > 0: model.config.diar_align = True                    # 2026-10-30: 섞은·바꾼 슬롯 순서 — 헤드 손실은 창 도착 순서, 예측 입력은 정답 자리로(헤드가 체크포인트에서 와도)
    cfg.turn_weight = a.turn_weight; cfg.next_weight, cfg.next_weight_ko, cfg.delays = a.next_weight, a.next_weight_ko, list(delays)
    bad = {k: (v, tok.convert_tokens_to_ids(k)) for k, v in {**old_sp, **cfg.sp_ids}.items() if tok.convert_tokens_to_ids(k) != v}
    assert not bad, f"tokenizer 특수 토큰 id 가 config 와 다름: {bad}"
sp_ids = add_semcommit_specials(tok)
if max(delays) > 8: sp_ids = {**sp_ids, **add_delay_ext_specials(tok)}
trans_id = add_trans_special(tok)["<TRANS>"]
EOT = sp_ids[TURN_TOKEN]
if model is not None:
    reg = dict(cfg.sem_registry or {})
    assert reg == {k: sp_ids[k] for k in SEM_SPECIALS + [TURN_TOKEN]}, f"config.sem_registry 가 <SEM_END>·<EOT> 가 아니다: {reg}"
    assert EOT not in cfg.blocked_ids and sp_ids["<SPK_A>"] not in cfg.blocked_ids and sp_ids["<SPK_B>"] not in cfg.blocked_ids, "<EOT>/<SPK_A>/<SPK_B> 가 decode 차단 목록에 있다"
    init_had_encoder = bool(cfg.encoder_trainable or getattr(cfg, "encoder_saved", False))
    if a.freeze_encoder:
        model.encoder.set_trainable(False)
        for p_ in model.encoder.parameters(): p_.requires_grad_(False)
        cfg.encoder_trainable = False; cfg.encoder_saved = True
    else:
        model.encoder.set_trainable(True); cfg.encoder_trainable = True
    if a.encoder_right_context is not None and a.encoder_right_context != cfg.encoder_right_context:
        ctx = [int(cfg.encoder_left_context), int(a.encoder_right_context)]; allowed = [list(map(int, c)) for c in (getattr(model.encoder.enc, "att_context_size_all", None) or [])]
        assert not allowed or ctx in allowed, f"--encoder-right-context {a.encoder_right_context}: 학습한 att_context 가 아니다 {allowed}"
        model.encoder.enc.set_default_att_context_size(ctx); cfg.encoder_right_context = ctx[1]
    if not a.freeze_encoder and a.encoder_context_sampling == "fixed":
        ctx = [int(cfg.encoder_left_context), int(cfg.encoder_right_context)]; ee = model.encoder.enc
        ee.att_context_size_all, ee.att_context_probs = [ctx], [1.0]; ee.set_default_att_context_size(ctx)
        log(f"인코더 학습 att_context 고정 {ctx}")
    model._keys_to_ignore_on_save = None
    sa = model.encoder.set_spec_augment(a.spec_augment)
    if sa is not None: log(f"SpecAugment {sa.extra_repr()}")
    if a.grad_ckpt: model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.thinker.config.use_cache = False
    speed = apply_speedups(model, a, log, liger=a.liger); a.liger = bool(speed["liger"]); cfg.use_liger = a.liger
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model ← {a.init} ({time.time() - t0:.0f}s) · 학습 {n_tr / 1e6:.1f}M · 인코더 {'동결' if a.freeze_encoder else '학습'}(init 인코더 저장됨={init_had_encoder}) · "
        f"<EOT>={EOT} {'정답·문맥 없음(--no-eot)' if a.no_eot else f'켬(turn_weight {cfg.turn_weight})'} · grad ckpt {a.grad_ckpt} · liger {a.liger} · 화자 슬롯 헤드 {f'켬(가중 {cfg.spk_weight})' if model.spk_head is not None else '끔'} · "
        f"인코더 탭 {f'L{cfg.enc_tap_layer:02d}(은닉 {cfg.enc_tap_hidden})' if model.adapter_tap is not None else '끔'}")

# ── 데이터
def _paths(vals):
    return [q for x in vals for q in ([l.strip() for l in open(x[1:]) if l.strip()] if x.startswith("@") else x.split(",")) if q]
def _subset(paths, frac):
    """경로 해시로 고정 선택(--replay-files-frac) — 같은 frac 이면 늘 같은 파일."""
    if frac >= 1.0: return list(paths)
    keep = [p for p in paths if int(hashlib.sha1(p.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac]
    return keep or list(paths[:1])
from vapasr.data.cst_dialogue import CstDialogueDataset, Recover, load_revive
rc = Recover(p=a.recover_p, boundary_frac=a.recover_boundary_frac, max_lag=a.recover_max_lag, max_boundaries=a.recover_max_boundaries) if a.recover_p > 0 else None
def wav_aug_of(key):
    """대화·합성 셋의 창 단위 증강: --set-phone 에 있으면 그 확률(0 = 끔), 없으면 --phone-aug 그대로."""
    p = set_phone.get(key)
    return phone if p is None else PhoneAugment(p=p, mulaw_p=a.phone_aug_mulaw) if p > 0 else None
def far_of(key):
    """--far-sets 패턴에 드는 셋이면 원거리 증강(아니면 None — 그 셋은 난수를 더 뽑지 않는다)."""
    import fnmatch
    return faug if faug is not None and any(fnmatch.fnmatch(key, x) for x in a.far_sets.split(",") if x) else None
def dec_kw(key):
    """V3-p1 C1·C5 데이터셋 인자(교대·머무름 가중 — --decision-weight-sets, 합성 EOT-문맥 dropout — --eot-drop-sets). 끄거나 이 셋에 걸지 않으면 빈 dict(데이터셋 기본값)."""
    kind = set_kind(key); kw = decision_weights(kind, a.change_tag_weight, a.hold_tag_weight, a.decision_weight_sets, a.same_lang_change_weight, a.same_lang_hold_weight)
    if a.eot_drop_synth > 0 and (kind == "xl" or (a.eot_drop_sets == "synth" and kind == "sim")): kw.update(eot_drop_synth=a.eot_drop_synth)
    if a.no_eot: kw.update(no_eot=True)                                                       # 모든 대화·합성 셋(켤 때만 키 — 끄면 데이터셋 기본값)
    if a.target_spk: kw.update(target_spk=True)                                               # E2 대상 화자 모드(켤 때만 키)
    if act_noise is not None: kw.update(act_input=act_noise)                                    # E5 슬롯 활동 입력(켤 때만 키)
    return kw
split = json.load(open(a.split, encoding="utf-8")); mono = a.mono_cache or os.path.join(a.data, "mono")
train_sets, dlg_files, revive_files = {}, [], []
DT, DTI, dt_fp, dti_fp = {}, {}, None, None
if a.dlg_trans:                                                                               # Stage 4: 대화 발화 번역 타깃(행 id '<conv>|<화자>|<utt>')
    from vapasr.data.ast_dataset import load_ast_targets
    if not 0 < a.dlg_ast_frac <= 1: sys.exit(f"--dlg-ast-frac 는 (0, 1] — {a.dlg_ast_frac}")
    DT, dt_st, dt_fp = load_ast_targets(_paths(a.dlg_trans))
    if a.dlg_trans_input: DTI, dti_st, dti_fp = load_ast_targets(_paths(a.dlg_trans_input))
    log(f"  대화 번역 타깃 {dict(dt_st)} · 입력만(되살림) {dict(dti_st) if a.dlg_trans_input else '없음'} · 번역 모드 {a.dlg_ast_frac}")
elif a.dlg_trans_input: sys.exit("--dlg-trans-input 은 --dlg-trans 와 함께")
def cap(ds):
    """--max-windows: 창 표본(draw 표시와 함께 거른다)."""
    if a.max_windows and len(ds) > a.max_windows: keep = set(random.Random(a.seed).sample(range(len(ds)), a.max_windows)); ds.keep_items([i in keep for i in range(len(ds))])
def group_of(c): return GMAP[c] if c in GMAP else "dlg_far" if c in FAR else ("dlg_mt" if DLG_LANG[c] == "English" else f"dlg_mt{SHORT[DLG_LANG[c]]}") if c in MEETING else f"dlg_{SHORT[DLG_LANG[c]]}"
for key, lang in BASE_GROUPS + tuple(EXTRA_GROUPS):
    cs = [c for c in corpora if group_of(c) == key]
    if key in dict(EXTRA_GROUPS) and weights.get(key, 0) <= 0: sys.exit(f"--dlg-group-map 의 그룹 {key} 에 --weights 가중치가 없다(그 코퍼스 {cs} 가 빠진다)")
    if not cs or weights.get(key, 0) <= 0: continue
    files = [os.path.join(a.data, f"{c}.refined.dialogues.jsonl") for c in cs]; dlg_files += files
    held = {cid for c in cs for cid in split["corpora"].get(c, {}).get("heldout", [])}
    rv = [f for f in (os.path.join(a.revive_dir, f"{c}.revive.jsonl") for c in cs) if os.path.exists(f)] if a.revive_dir else []; revive_files += rv
    ds = CstDialogueDataset(files, tok, delays=delays, exclude=held, window_s=win, hop_s=a.hop, start_mode=a.start_mode, random_frac=a.random_frac,
                            seed=a.seed, mono_cache_dir=mono, decision_weight=a.decision_weight, backchannel=bc, eot_mode=a.eot_mode, recover=rc,
                            spk_activity=want_act, revive=load_revive(rv) if rv else None,
                            **(dict(trans=DT, trans_input=DTI, ast_frac=a.dlg_ast_frac, trans_lead_space=a.trans_lead_space) if DT else {}), wav_aug=wav_aug_of(key), far_aug=far_of(key), **VKW, **dec_kw(key))
    cap(ds); ds.lang = lang; train_sets[key] = ds
    if DT:
        rows = {(c, w, b) for c in ds.dlgs for (w, b) in ds._rows(c)}; got = sum(1 for c, w, b in rows if f"{ds._tcid(c)}|{w}|{b}" in DT or f"{ds._tcid(c)}|{w}|{b}" in DTI)   # 원거리 버전은 원본 id(far_of)
        ds.stats.update(trans_rows=len(rows), trans_rows_with_target=got, trans_dialogues_without_target=sum(1 for c in ds.dlgs if not ds.has_trans(c)))   # 전사 모드만(번역 모드로 뽑지 않는다)
    log(f"  {key} {cs}: {len(ds)} 창 · held-out 제외 {ds.stats['cst_excluded']} 대화 · stats {dict(ds.stats)}")
sim_files = []
def sim_dir(key):
    """--sim-data 폴더들 중 <key>.refined.dialogues.jsonl 이 있는 첫 폴더(없으면 첫 폴더 — 뒤에서 없음으로 끝낸다)."""
    ds_ = [x for x in (a.sim_data or "").split(",") if x]
    return next((x for x in ds_ if os.path.exists(os.path.join(x, f"{key}.refined.dialogues.jsonl"))), ds_[0] if ds_ else "")
for lang, key in [(l, k) for l in ("Korean", "English", "German") for k in [f"sim_{SHORT[l]}"] + sorted(x for x in weights if re.fullmatch(fr"sim_{SHORT[l]}-[a-z0-9]{{3,}}", x))]:
    # 합성 대화(sim_dialogue): held-out 과 무관한 학습 재료로만 만들었다. sim_<l>-<접미사>(세 글자 이상, 2026-10-26 — 예 sim_en-meet)는 같은 언어 합성의 다른 셋
    if weights.get(key, 0) <= 0: continue
    if not a.sim_data: sys.exit(f"--weights 에 {key} 가 있는데 --sim-data 가 없다")
    sd = sim_dir(key); f = os.path.join(sd, f"{key}.refined.dialogues.jsonl")
    if not os.path.exists(f): sys.exit(f"합성 대화 없음: {f}")
    sim_files.append(f)
    ds = CstDialogueDataset([f], tok, delays=delays, exclude=None, window_s=win, hop_s=a.hop, start_mode=a.start_mode, random_frac=a.random_frac, seed=a.seed,
                            mono_cache_dir=os.path.join(sd, "mono"), decision_weight=a.decision_weight, backchannel=bc, eot_mode=a.eot_mode, recover=rc,
                            spk_activity=want_act, wav_aug=wav_aug_of(key), far_aug=far_of(key), **VKW, **dec_kw(key))
    cap(ds); ds.lang = lang; train_sets[key] = ds
    log(f"  {key}(합성): {len(ds)} 창 · 대화 {ds.stats['cst_dialogues']} · stats {dict(ds.stats)}")
sim_trans_fp = {}
for key in sorted(k for k, w in weights.items() if w > 0 and set_kind(k) == "xl"):                # 교차 언어 합성(Stage 4) — sim_<l1>-<l2>[-접미사]
    if not a.sim_data: sys.exit(f"--weights 에 {key} 가 있는데 --sim-data 가 없다")
    sd = sim_dir(key); f, td = os.path.join(sd, f"{key}.refined.dialogues.jsonl"), os.path.join(sd, f"targets-{key}")
    if not os.path.exists(f) or not os.path.isdir(td): sys.exit(f"교차 언어 합성 대화·번역 타깃 없음: {f} · {td}")
    from vapasr.data.ast_dataset import load_ast_targets
    XT, xt_st, sim_trans_fp[key] = load_ast_targets([td]); sim_files.append(f)
    ds = CstDialogueDataset([f], tok, delays=delays, exclude=None, window_s=win, hop_s=a.hop, start_mode=a.start_mode, random_frac=a.random_frac, seed=a.seed,
                            mono_cache_dir=os.path.join(sd, "mono"), decision_weight=a.decision_weight, backchannel=bc, eot_mode=a.eot_mode, recover=rc,
                            spk_activity=want_act, trans=XT, ast_frac=a.sim_trans_frac, trans_lead_space=a.trans_lead_space, wav_aug=wav_aug_of(key), far_aug=far_of(key), **VKW, **dec_kw(key))
    if not ds.has_xl: sys.exit(f"{key}: 교차 언어 대화(meta.spk_lang)가 아니다")
    cap(ds)
    ds.lang = next(iter(ds.dlgs.values())).lang; train_sets[key] = ds
    log(f"  {key}(교차 언어 합성): {len(ds)} 창 · 대화 {ds.stats['cst_dialogues']} · 번역 타깃 {dict(xt_st)} · 번역 모드 {a.sim_trans_frac}")
RW = _subset(_paths(a.replay_words), a.replay_files_frac); RD = _subset(_paths(a.replay_de_words), a.replay_files_frac); ast_fp = None
if (RW or RD) and any(weights.get(k, 0) > 0 for k in ("rep_ko", "rep_en", "rep_de")):
    from vapasr.data.ast_dataset import AstStreamDataset, load_ast_targets
    AST, ast_st, ast_fp = load_ast_targets(_paths(a.ast_targets)) if a.ast_targets else ({}, {}, None)
    log(f"  replay 번역 타깃 {dict(ast_st)} · 번역 모드 {a.ast_frac} · words 파일 EN·KO {len(RW)} · DE {len(RD)} (frac {a.replay_files_frac})")
    for key, lang, ww in (("rep_en", "English", RW), ("rep_ko", "Korean", RW), ("rep_de", "German", RD)):
        if not ww or weights.get(key, 0) <= 0: continue
        ds = AstStreamDataset(ww, [], tok, sp_ids, delays=delays, turn_end=False, hangover_s=0.48, hardneg_weight=1.0, max_items=a.replay_max_items or None,
                              seed=a.seed, online=True, allow_unlabeled=False, max_per_chunk=0, langs=[lang], path_map={}, tail_margin=2, source_pool="main",
                              ko_textnorm=True, written_frac=0.0, sem_free_unlabeled=True, sem_free_max_ratio=None, offline_frac=a.offline_frac, ignore_labels=True,
                              speed_perturb=speeds, noise_aug=raug, ast_targets=AST, ast_frac=a.ast_frac, trans_id=trans_id, gain_aug=gain, gain_aug_langs=gain_langs)
        if not len(ds): log(f"  {key}: 항목 0 — 건너뜀"); continue
        ds.lang = lang; train_sets[key] = ds; log(f"  {key}: {len(ds)} 항목 · 번역 타깃 있는 항목 {ds.stats.get('ast_items', 0)}")
RS = {}
for spec in a.replay_set:                                                                       # 따로 가중하는 단일 화자 셋(예: CallHome 독일어 감독 발화)
    name, lang, ww = spec.split(":", 2)
    if not name.startswith("rep_") or name in ("rep_ko", "rep_en", "rep_de") or name in RS: sys.exit(f"--replay-set 이름은 rep_ 로 시작하고 기본 셋·다른 셋과 달라야 한다: {name}")
    if weights.get(name, 0) <= 0: sys.exit(f"--replay-set {name} 이 --weights 에 없다")
    RS[name] = _paths([ww])
    from vapasr.data.ast_dataset import AstStreamDataset, load_ast_targets
    if "AST" not in globals(): AST, ast_st, ast_fp = load_ast_targets(_paths(a.ast_targets)) if a.ast_targets else ({}, {}, None)
    ds = AstStreamDataset(RS[name], [], tok, sp_ids, delays=delays, turn_end=False, hangover_s=0.48, hardneg_weight=1.0, max_items=a.replay_max_items or None,
                          seed=a.seed, online=True, allow_unlabeled=False, max_per_chunk=0, langs=[lang], path_map={}, tail_margin=2, source_pool="main",
                          ko_textnorm=True, written_frac=0.0, sem_free_unlabeled=True, sem_free_max_ratio=None, offline_frac=a.offline_frac, ignore_labels=True,
                          speed_perturb=speeds, noise_aug=raug, ast_targets=AST, ast_frac=a.ast_frac, trans_id=trans_id, gain_aug=gain, gain_aug_langs=gain_langs)
    if not len(ds): sys.exit(f"--replay-set {name}: 항목 0")
    ds.lang = lang; train_sets[name] = ds; log(f"  {name}(따로 가중): {len(ds)} 항목 · 번역 타깃 있는 항목 {ds.stats.get('ast_items', 0)} · bad {dict(ds.bad)}")
if a.revive_dir and not revive_files: sys.exit(f"--revive-dir 에 대화 코퍼스의 <corpus>.revive.jsonl 이 없다: {a.revive_dir}")
missing = sorted(k for k, w in weights.items() if w > 0 and k not in train_sets)
if missing: sys.exit(f"가중치 > 0 인데 만들어지지 않은 셋: {missing} (입력·코퍼스를 확인하거나 --weights 에서 뺀다)")
assert any(k.startswith("dlg_") for k in train_sets), "대화 셋이 없다"
check = {}
for key, ds in train_sets.items():                                                             # 시퀀스 불변식(오디오 없이)
    n = min(len(ds), a.check_items); rng = random.Random(a.seed); idx = sorted(rng.sample(range(len(ds)), n))
    if key.startswith(("dlg_", "sim_")):
        bad = [(ds.items[i], d, P[:3]) for i in idx for d in ds.delays for P in [ds.target_problems(i, d)] if P]
        if rc: bad += [(ds.items[i], d, "flip", P[:3]) for i in idx for d in ds.delays for P in [ds.target_problems(i, d, rng=random.Random(a.seed * 7919 + i * 31 + d), recover=rc._replace(p=1.0))] if P]
        if getattr(ds, "trans_on", False):                                                     # 번역 모드(목표 언어마다)
            bad += [(ds.items[i], d, tg, P[:3]) for i in idx for d in ds.delays for tg in ds.tgt_options(i) for P in [ds.problems_of(ds.cst_sequence(i, d, tgt=tg))] if P]
        if getattr(ds, "target_spk", False):                                                   # E2 대상 화자 시퀀스(창의 화자마다)
            from vapasr.data.cst_dialogue import window_speakers
            bad += [(ds.items[i], d, "target", t, P[:3]) for i in idx for d in ds.delays for t in window_speakers(ds.eps[ds.items[i][0]], ds.items[i][1], ds.items[i][2])
                    for P in [ds.problems_of(ds.cst_sequence(i, d, target=t))] if P]
        if getattr(ds, "gate_fa", 0) > 0 or getattr(ds, "x_slot", False):                      # E13 학습 rng 경로(헛 구간·번역 모드 추첨·E14 놓침 흉내 — __getitem__ 과 같은 꼴)
            bad += [(ds.items[i], d, "rng", P[:3]) for i in idx for d in ds.delays for P in [ds.problems_of(ds.cst_sequence(i, d, rng=random.Random(a.seed * 104729 + i * 31 + d)))] if P]
        if getattr(ds, "eot_drop_synth", 0) > 0:                                               # EOT-문맥 dropout 경로(V3-p1 C5 — 받아쓰기·번역 모드)
            bad += [(ds.items[i], d, "drop", tg, P[:3]) for i in idx for d in ds.delays for tg in [None] + (ds.tgt_options(i) if ds.trans_on else [])
                    for P in [ds.problems_of(ds.cst_sequence(i, d, tgt=tg, drop_eot=True))] if P]
    else: bad = [(ds.items[i]["id"], d, P[:3]) for i in idx for d in ds.delays for P in [ds.target_problems(i, d)] if P]
    check[key] = dict(n=n, bad=len(bad), examples=[str(b)[:200] for b in bad[:5]]); log(f"  check {key}: {n} 항목 × δ{list(delays)} → 불일치 {len(bad)} {bad[:2]}")
assert all(v["bad"] == 0 for v in check.values()), f"시퀀스 불변식 위반: {check}"
vocab_sha = hashlib.sha256(json.dumps(sorted(tok.get_vocab().items())).encode()).hexdigest()
fp = dict(format="cst-v1", corpora=corpora, split_sha256=hashlib.sha256(open(a.split, "rb").read()).hexdigest(), dialogues=file_digest(dlg_files),
          window=list(win), hop=a.hop, start_mode=a.start_mode, random_frac=a.random_frac if a.start_mode == "mixed" else None, eot_mode=a.eot_mode,
          backchannel=list(bc) if bc else None, decision_weight=a.decision_weight, turn_weight=a.turn_weight, weights=weights, delays=list(delays),
          next_weight=a.next_weight, next_weight_ko=a.next_weight_ko, replay_words=file_digest(RW) if RW else None, replay_de_words=file_digest(RD) if RD else None,
          replay_files_frac=a.replay_files_frac, replay_max_items=a.replay_max_items, ast_targets_sha256=ast_fp, ast_frac=a.ast_frac, offline_frac=a.offline_frac,
          gain_augment=list(gain) if gain else None, speed_perturb=list(speeds) or None, noise_aug=naug.settings() if naug else None, spec_augment=a.spec_augment,
          max_windows=a.max_windows, seed=a.seed, vocab_sha256=vocab_sha, init=os.path.abspath(a.init), **({"recover": rc._asdict()} if rc else {}), **({"spk_act_weight": a.spk_act_weight} if a.spk_act_weight > 0 else {}),
          **({"sim_data": file_digest(sim_files)} if sim_files else {}), **({"revive": file_digest(revive_files)} if revive_files else {}),
          **({"dlg_trans": dt_fp, "dlg_trans_input": dti_fp, "dlg_ast_frac": a.dlg_ast_frac} if DT else {}),
          **({"sim_trans": sim_trans_fp, "sim_trans_frac": a.sim_trans_frac} if sim_trans_fp else {}),
          **({"trans_lead_space": True} if a.trans_lead_space and (DT or sim_trans_fp) else {}),
          **({"replay_sets": {k: file_digest(v) for k, v in sorted(RS.items())}} if RS else {}),
          **({"phone_aug": phone.settings()} if phone is not None else {}),   # 끄면 키 없음(이전 run 지문 그대로)
          **({"far_aug": dict(faug.settings(), sets=a.far_sets)} if faug is not None else {}),   # 2026-10-26: 끄면 키 없음
          **({"dlg_group_map": GMAP} if GMAP else {}),                                        # 2026-10-26: 끄면 키 없음
          **({"global_slots": a.global_slots} if a.global_slots > 0 else {}),                  # 2026-10-27 E7: 끄면 키 없음
          **({"enc_tap": dict(layer=a.enc_tap_layer, hidden=a.enc_tap_hidden or None)} if a.enc_tap_layer >= 0 else {}),
          **({"max_speakers": a.max_speakers, "declare_all_p": a.declare_all_p} if a.max_speakers > 2 else {}), **({"window_draws": a.window_draws} if a.window_draws else {}),
          **({"change_tag_weight": a.change_tag_weight} if a.change_tag_weight > 0 else {}), **({"hold_tag_weight": a.hold_tag_weight} if a.hold_tag_weight > 0 else {}),   # V3-p1: 끄면 키 없음
          **({"decision_weight_sets": a.decision_weight_sets} if (a.change_tag_weight > 0 or a.hold_tag_weight > 0) and a.decision_weight_sets != "all" else {}),
          **({"same_lang_change_weight": a.same_lang_change_weight or 0.0, "same_lang_hold_weight": a.same_lang_hold_weight or 0.0} if SL else {}),   # V3 R2m: 주지 않으면 키 없음
          **({"eot_drop_synth": a.eot_drop_synth} if a.eot_drop_synth > 0 else {}), **({"eot_drop_sets": a.eot_drop_sets} if a.eot_drop_synth > 0 and a.eot_drop_sets != "xl" else {}),
          **({"set_phone": {k: (aug.settings() if aug is not None else None) for k in sorted(set_phone) for aug in [wav_aug_of(k)]}} if set_phone else {}),
          **({"xl_slot_order": a.xl_slot_order} if a.xl_slot_order != "lang" else {}), **({"no_eot": True} if a.no_eot else {}),
          **({"target_spk": dict(inject_hidden=a.inject_hidden)} if a.target_spk else {}),
          **({"slot_input": dict(slots=max(2, a.max_speakers), noise=act_noise.settings())} if act_noise is not None else {}),
          **({"diar_head": dict(src=os.path.abspath(a.diar_head), weight=a.diar_weight, ramp=list(ramp), threshold=a.act_threshold)} if a.diar_head else {}),
          **({"act_shift": a.act_shift} if a.act_shift else {}),                                # 2026-10-27 E8: 끄면 키 없음
          **({"slot_blocks": dict(begin_weight=a.begin_weight, begin_lag=4, tag_act_lags=a.tag_act_lags)} if a.slot_blocks else {}),   # 2026-10-28 E9: 끄면 키 없음
          **({"declare_range": True} if a.declare_range else {}),                               # 2026-10-28: 끄면 키 없음
          **({"slot_shuffle": a.slot_shuffle} if a.slot_shuffle > 0 else {}),                   # 2026-10-30 활동 추종 학습: 끄면 키 없음
          **({"slot_remap": a.slot_remap} if a.slot_remap > 0 else {}),                         # 2026-10-30 활동 추종 학습: 끄면 키 없음
          **({"slot_gate": dict(lags=a.slot_gate, fa=a.gate_fa)} if a.slot_gate else {}),       # 2026-11-02 E13 활동 게이트: 끄면 키 없음
          **({"x_slot": dict(drop=a.miss_drop, short=a.miss_short, late=a.miss_late)} if a.x_slot else {}), **({"n3d_soft": True} if a.n3d_soft else {}),   # 2026-11-03 E14
          **({"shrink_speakers": True} if a.shrink_speakers and not a.window_draws else {}),    # 2026-10-28 E9b: 격자 창 줄이기(끄면 키 없음)
          **({"slot_audio": dict(d=a.slot_audio_dim, blocks=a.slot_audio_blocks, kernel=a.slot_audio_kernel, mix_drop=a.mix_drop)} if a.slot_audio else {}),   # 2026-10-29 E10
          **({"n3d": dict(path=os.path.abspath(a.n3d), train_chunk=a.n3d_train_chunk, chunk=a.n3d_chunk, threshold=a.n3d_threshold)} if a.n3d else {}))   # 2026-10-29 E11
if a.dry_run:
    names = {v: k for k, v in sp_ids.items()}
    for key, ds in train_sets.items():
        if not key.startswith(("dlg_", "sim_")): continue
        s = ds.cst_sequence(0, max(delays)); P = len(ds.cst_prefix(s["langs"], max(delays)))
        def show(p):
            t = s["ids"][p]; lab = s["labels"][p]
            return (("A" if s["chunk_of"][p] >= 0 else names.get(t) or repr(tok.decode([t]))) + ("" if lab != -100 or s["is_input"][p] else "[m]") + (f"[p={s['soft_w'][p]}]" if p in s["soft_pos"] else "")
                    + ("↵" if s["is_input"][p] and lab == sp_ids["<NEXT_AUDIO>"] else ""))      # E9: 입력 위치의 멈춤 label(앞 슬롯 끝)
        log(f"dry-run {key} {s['cid']}@{s['t0']} L {s['L']} δ{max(delays)} · 화자 {s['speakers']} · EOT {s['n_eot']} soft {s['n_soft']} · 머리 {tok.decode(s['ids'][:P])!r}")
        log(" ".join(show(p) for p in range(P, min(len(s["ids"]), P + 400))))
        if getattr(ds, "trans_on", False):                                                     # 번역 모드 예(첫 목표 언어)
            tg = ds.tgt_options(0)[0]; s = ds.cst_sequence(0, max(delays), tgt=tg); P = len(ds.cst_prefix(s["langs"], max(delays), list(s["tgt"])))
            log(f"dry-run 번역 {key} → {tg} · 번역 토큰 {s['n_tr']} · 입력만 {len(s['tr_input'])} · 창 {s['win']} · 머리 {tok.decode(s['ids'][:P])!r}")
            log(" ".join(show(p) for p in range(P, min(len(s["ids"]), P + 400))))
        if rc:                                                                                # 뒤집은 예: 회복 지점 앞뒤
            for j in range(min(len(ds), 50)):
                s = ds.cst_sequence(j, max(delays), rng=random.Random(a.seed + j), recover=rc._replace(p=1.0))
                if s["flip"]: break
            if s["flip"]:
                f = s["flip"]; lo, hi = max(P, f["p_start"] - 12), min(len(s["ids"]), f["p_end"] + 12)
                log(f"dry-run 뒤집기 {key} {s['cid']}@{s['t0']} {f['kind']} · 구간 위치 [{f['p_start']}, {f['p_end']}) · 회복 지점 정답 {names.get(s['ids'][f['p_end']]) or repr(tok.decode([s['ids'][f['p_end']]]))}")
                log(" ".join(("|" if p == f["p_end"] else "") + show(p) for p in range(lo, hi)))
    json.dump(dict(args=vars(a), sp_ids=sp_ids, stage3=fp, sets={k: dict(v.stats) for k, v in train_sets.items()}, sizes={k: len(v) for k, v in train_sets.items()}, check=check),
              open(os.path.join(out, "dry-run.json"), "w"), indent=1, ensure_ascii=False, default=str)
    log("dry-run 끝 →", os.path.join(out, "dry-run.json")); sys.exit(0)

# ── Trainer
from transformers import TrainingArguments
from vapasr.hf.trainer import VapAsrTrainer, PreemptCallback
from vapasr.hf.stage import last_complete_checkpoint                                         # 저장 완료 checkpoint-N(한 파일·1.7B 조각 저장 모두 — 선점이 끊은 것은 건너뛴다)
from vapasr.hf.packing import budget_sampler_factory

class Stage3Trainer(VapAsrTrainer):
    """데이터: 셋 가중 라운드로빈 + collate_stage3. 로그: 대화/replay 별 손실·턴 손실을 따로 평균(loss_dlg·loss_rep·loss_turn_dlg·soft_per_step)."""
    def bs_of(self, name): return self.bs_ko if self.train_sets[name].lang == "Korean" else self.bs_en
    def get_train_dataloader(self):
        return WeightedRoundRobinLoader(self.train_sets, self.bs_of, weights=weights, anchor=[k for k in self.train_sets if k.startswith("dlg_")], seed=self.args.seed,
                                        rank=self.args.process_index, world=self.args.world_size, num_workers=self.num_workers, collate_fn=collate_stage3,
                                        sampler_factory=budget_sampler_factory(self.pack_max_tokens, self.batch_max_tokens, self.pack_max_bs))
    def evaluate(self, *args, **kw): return {}
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, o = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        kind = "dlg" if "labels_alt" in inputs else "rep"; c = self.__dict__.setdefault("_kind", {})
        c.setdefault(f"loss_{kind}", []).append(float(loss.detach()))
        if kind == "dlg" and getattr(o, "loss_turn", None) is not None: c.setdefault("loss_turn_dlg", []).append(float(o.loss_turn))
        if kind == "dlg": c.setdefault("soft", []).append(int(inputs.get("n_soft", 0))); c.setdefault("flip", []).append(int(inputs.get("n_flip", 0))); c.setdefault("rev", []).append(int(inputs.get("n_rev", 0))); c.setdefault("tr", []).append(int(inputs.get("n_tr", 0)))
        if kind == "dlg" and getattr(o, "loss_act", None) is not None: c.setdefault("loss_spk", []).append(float(o.loss_act)); c.setdefault("spk_acc", []).append(float(o.act_acc))
        if kind == "dlg":                                                                      # V3-p1 교대·머무름 통계·EOT-문맥 dropout(켰을 때만 — 없으면 로그 키도 없다)
            for k in ("n_switch", "n_hold", "n_eot_drop", "n_remap"):
                if k in inputs: c.setdefault(k[2:], []).append(int(inputs[k]))
            for k in ("top1_switch", "switch_miss", "switch_wrong_slot", "top1_hold"):
                if getattr(o, k, None) is not None: c.setdefault(k, []).append(float(getattr(o, k)))
        return (loss, o) if return_outputs else loss
    def log(self, logs, start_time=None):
        c = self.__dict__.get("_kind")
        if c and "loss" in logs:
            for k, v in c.items():
                if v: logs[{"soft": "soft_per_dlg_step", "flip": "flip_per_dlg_step", "rev": "rev_per_dlg_step", "tr": "tr_per_dlg_step", "switch": "switch_per_dlg_step",
                            "hold": "hold_per_dlg_step", "eot_drop": "eot_drop_per_dlg_step", "remap": "remap_per_dlg_step"}.get(k, k)] = round(sum(v) / len(v), 4)
            self._kind = {}
        return super().log(logs, start_time)

targs = TrainingArguments(output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=1, num_train_epochs=a.epochs, max_steps=a.max_steps if a.max_steps > 0 else -1,
                          learning_rate=a.lr, weight_decay=a.wd, warmup_steps=a.warmup, lr_scheduler_type="cosine", max_grad_norm=1.0, **{"bf16": True, **training_args_kwargs(a)},
                          logging_strategy="steps", logging_steps=a.log_every, logging_first_step=True, eval_strategy="no", save_strategy="steps", save_steps=a.save_every,
                          save_total_limit=a.save_total_limit, save_safetensors=True, seed=a.seed, data_seed=a.seed, report_to=[x for x in a.report_to.split(",") if x and x != "none"],
                          logging_dir=os.path.join(out, "tb"), remove_unused_columns=False, disable_tqdm=True, ignore_data_skip=True, dataloader_num_workers=a.num_workers,
                          label_names=["labels"], log_level="warning" if main else "error", ddp_find_unused_parameters=False, ddp_broadcast_buffers=False, ddp_timeout=10800)
model.config.stage3 = fp
if a.xl_slot_order != "lang": model.config.xl_slot_order = a.xl_slot_order                     # V3 R4: 평가·디코드가 config 로 같은 교차 언어 prefix·슬롯 순서를 쓴다(끄면 키 없음)
last = last_complete_checkpoint(out) if a.resume == "auto" else (None if a.resume == "none" else a.resume); fdiff = []
if last:
    fdiff = fingerprint_diff(last, fp)
    if fdiff and not a.force_resume: sys.exit(f"재개 거부: {last} 의 config.stage3 가 지금 설정·데이터와 다르다 {fdiff[:8]} — 새 --out-dir 을 쓰거나 --force-resume")
    log(f"재개 ← {last}" + (f" · !! --force-resume: 지문 차이 {fdiff}" if fdiff else " · 지문 일치"))
preempt_cb = PreemptCallback(out, gloo_pg)
trainer = Stage3Trainer(model=model, args=targs, train_sets=train_sets, dev_sets={}, tokenizer=tok, bs_en=6, bs_ko=16, lr_adapter=a.lr_adapter, lr_encoder=a.lr_encoder,
                        max_per_chunk=0, num_workers=a.num_workers, language_schedule="balanced", gloo_pg=gloo_pg,
                        pack_max_tokens=a.pack_max_tokens, pack_max_bs=a.pack_max_bs, batch_max_tokens=a.batch_max_tokens, callbacks=[preempt_cb], processing_class=tok)
tdl = trainer.get_train_dataloader(); plan = tdl.describe()
empty = sorted(k for k, sp_ in tdl.samplers.items() if weights.get(k, 0) > 0 and len(sp_) == 0)     # rank 당 batch 0 이면 RoundRobinLoader._cycle 이 내지 않고 돈다(멈춤)
if empty: sys.exit(f"rank 당 batch 가 0 인 셋: {empty} — 셋이 너무 작거나(world {world}) 창 draw 하나가 비었다(--window-draws·--max-windows 확인)")
log(f"섞기: steps/epoch {plan['steps_per_epoch']} · 비중 {plan['share']} · 셋 배치/epoch {plan['batches_per_epoch']} · world {world}")
run_rec = dict(args=vars(a), init=a.init, sp_ids=sp_ids, stage3=fp, resume_from=last, resume_fingerprint_diff=fdiff, trainable_m=n_tr / 1e6, encoder_saved=bool(model.config.encoder_saved),
               sizes={k: len(v) for k, v in train_sets.items()}, sets={k: dict(v.stats) for k, v in train_sets.items()}, plan=plan, check=check, world=world, speed=speed,
               started=time.strftime("%F %T"))
run_path = os.path.join(out, time.strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid()}.json")
if main: json.dump(run_rec, open(run_path, "w"), indent=1, ensure_ascii=False, default=str); log(f"실행 기록 → {run_path}")
res = trainer.train(resume_from_checkpoint=last); st = trainer.state
done = st.global_step >= st.max_steps and not preempt_cb.fired
log(f"train 종료: step {st.global_step}/{st.max_steps} {'완료' if done else '(선점/중단 → 재시작 시 이어서)'} · {res.metrics}")
if done:
    fin = os.path.join(out, "final"); trainer.save_model(fin)
if done and main:
    tok.save_pretrained(fin)
    json.dump(run_rec, open(os.path.join(fin, "stage3_run.json"), "w"), indent=1, ensure_ascii=False, default=str)
    json.dump(dict(args=vars(a), log_history=st.log_history[-300:], steps=st.global_step), open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
    if not a.no_check_save:                                                                   # 재로드 parity: state_dict(인코더 포함)·대화 배치 손실·tokenizer·지문
        def batch_loss(m, x):
            m.eval(); xx = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in x.items()}
            nw = m.config.next_weight_ko if x["lang"][0] == "Korean" else m.config.next_weight
            kw = {k: xx[k] for k in IN_KEYS if k in xx}
            if "soft_w_full" in xx: kw["soft_w"] = xx["soft_w_full"]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): o = m(**kw, next_weight=nw)
            return {k: round(float(getattr(o, k)), 5) for k in ("loss",) + LOSS_KEYS if getattr(o, k, None) is not None}
        dk = next(k for k in train_sets if k.startswith("dlg_"))
        from vapasr.data.cst_dialogue import collate_cst
        xb = collate_cst([train_sets[dk][i] for i in range(min(4, len(train_sets[dk])))]); mem = getattr(trainer.model, "module", trainer.model)
        re_ = VapAsrForStreamingASR.from_pretrained(fin, encoder_path=a.nemotron_dir).cuda()
        sd1 = {k: v.detach().float().cpu() for k, v in mem.state_dict().items()}; sd2 = {k: v.detach().float().cpu() for k, v in re_.state_dict().items()}
        miss = sorted(set(sd1) - set(sd2)); extra = sorted(set(sd2) - set(sd1)); shape = [k for k in sd1 if k in sd2 and sd1[k].shape != sd2[k].shape]
        diff = sorted(((k, float((sd1[k] - sd2[k]).abs().max())) for k in sd1 if k in sd2 and sd1[k].shape == sd2[k].shape and float((sd1[k] - sd2[k]).abs().max()) > 1e-6), key=lambda x: -x[1])
        from safetensors import safe_open
        saved_enc = sum(1 for f in glob.glob(os.path.join(fin, "*.safetensors")) for k in safe_open(f, "pt").keys() if k.startswith("encoder."))
        l1, l2 = batch_loss(mem, xb), batch_loss(re_, xb)
        try:
            from vapasr.hf import load_tokenizer; tk = load_tokenizer(fin); tok_ok = all(tk.convert_tokens_to_ids(k) == v for k, v in sp_ids.items()); tok_err = None
        except Exception as e: tok_ok, tok_err = False, f"{type(e).__name__}: {str(e)[:200]}"
        rc = json.load(open(os.path.join(fin, "config.json"))); fp_ok = rc.get("stage3") == json.loads(json.dumps(fp))
        reg_ok = rc.get("sem_registry") == {k: sp_ids[k] for k in SEM_SPECIALS + [TURN_TOKEN]} and EOT not in rc.get("blocked_ids", [])
        parity = (tok_ok and fp_ok and reg_ok and not miss and not extra and not shape and not diff and saved_enc > 0 and abs(l1["loss"] - l2["loss"]) <= 5e-3 * max(1.0, abs(l1["loss"])))
        json.dump(dict(parity=parity, stage3_config_ok=fp_ok, eot_registry_ok=reg_ok, memory=l1, reloaded=l2, missing=miss[:50], extra=extra[:50], shape_mismatch=shape[:50],
                       value_diff=diff[:50], encoder_saved_keys=saved_enc, tokenizer_reload_ok=tok_ok, tokenizer_reload_error=tok_err, batch_n_soft=xb.get("n_soft")),
                  open(os.path.join(out, "parity.json"), "w"), indent=1)
        log(f"check-save: 누락 {miss[:5]} · 추가 {extra[:5]} · shape {shape[:5]} · 값 차이 {len(diff)} {diff[:5]} · 인코더 저장 키 {saved_enc} · 손실 메모리 {l1} / 재로드 {l2} · "
            f"tokenizer {'OK' if tok_ok else tok_err} · config.stage3 {'OK' if fp_ok else '불일치'} · <EOT> 등록 {'OK' if reg_ok else '불일치'}")
        log("PARITY OK" if parity else "PARITY FAIL — 저장된 final 이 학습 모델과 다르다(parity.json)")
        del re_; torch.cuda.empty_cache()
    open(os.path.join(out, "DONE"), "w").write(time.strftime("%F %T")); log("final →", fin)
if world > 1: dist.destroy_process_group()
