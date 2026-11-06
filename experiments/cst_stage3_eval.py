#!/usr/bin/env python
"""Stage 3 평가(P4) — held-out 2 인 대화 창을 두 슬롯 prefix 로 스트리밍 디코드해 화자 귀속·받아쓰기·턴 끝 <EOT> 를 잰다.

  python experiments/cst_stage3_eval.py run --checkpoint <final> --data <phase2 dir> --split <phase2>/splits/v2.json --out <dir> \\
      [--corpora aihub134-2,aihub134-1,aihub71631,otoSpeech,kec,ztc,nsc] [--deltas 4,6,10] [--limit-windows 400] [--rank r --world-size n] [--verify]
  python experiments/cst_stage3_eval.py summarize --out <dir>

창: 코퍼스마다 split held-out 대화의 20–40 s 무음 시작 격자 창(CstDialogueDataset, seed 0 — 학습 창 규칙과 같다). --limit-windows 는 길이 순 고른 간격 표본.
  71631 held-out 은 Stage 1·2 학습에 노출됐다(P0) — 'seen' 으로 표시한다. 134-1 held-out 은 화자 10 명이 71631 에 나온다(대화는 다름).
입력: mono 캐시 crop(학습과 같은 오디오), 창마다 인코더 한 번(singleton), 디코드는 언어·δ 별 배치(vapasr/hf/batch_decode.decode_batch, 학습과 같은 차단 목록).
  --verify: 언어·δ 마다 첫 배치의 처음·끝 창을 model.stream_decode 로 다시 풀어 토큰·시각이 같은지 확인한다.
  --oracle: 모델 없이 학습 타깃 시퀀스(정답)를 디코드 출력처럼 바꿔 채점한다 — 지표 sanity(slot·cp 0, 양성 EOT 재현율 1).
  --from-predictions DIR: 그 폴더 predictions-rank*.jsonl 의 원출력(emitted)을 다시 채점한다(디코드 없이 — 채점 규칙을 고친 뒤).
  --act-tag-guide(2026-10-28, E7 — --window full · 화자 분리 헤드 모델): 회의 통째 디코드에서 화자 태그를 활동으로 이끈다(long_session.make_retag — 태그를 내면
    그 뒤 말이 올 오디오 [k − δ − 2, k] 의 활동이 가장 많은 슬롯으로, 모델 슬롯도 말하고 있으면 그대로). 행에 retag{tags, changed}.
  --relabel-act DIR [--relabel-oracle](2026-10-28, 진단 — --from-predictions · --window full): 가설 단어를 활동으로 다시 귀속해 채점한다(cst_eval.relabel_by_activity).
    DIR = diar_eval run --session --save-probs 의 <out>/aosc/probs(회의마다 <cid>@full.npz — prob·ref), --relabel-oracle 이면 그 npz 의 정답 활동(ref)으로.
  E9 슬롯 블록 체크포인트(config.slot_blocks, 2026-10-28)는 자동으로 슬롯 블록 디코드(창: batch_decode.decode_blocks_batch, --window full: long_session.long_block_decode
    — 추측 prefill)·오라클(block_sequence)을 쓰고, 행에 llm_der(<BEGIN>·<EOT> 로 만든 슬롯 활동의 DER 셈 — diar.der_counts, diar_eval 과 같은 프레임 마스크),
    --window full 이면 forwards·chunk_ms(청크 디코드 ms 평균)·chunk_ms_p90. --verify 는 배치 대 추측 prefill 직렬을 대조한다. 참조·채점은 같다.
    --oracle --blocks: 학습 전에 E9 정답 시퀀스를 같은 규칙으로 채점한다(cp·agn 0, LLM DER 는 구간 끝 반올림만).
  E10 슬롯 추출기 체크포인트(2026-10-29)는 태그마다 슬롯 소리 토큰을 더해 디코드한다(model.slot_audio_of). --mix-off: 섞인 소리 토큰을 빈 임베딩으로 — 슬롯 소리만으로(config 에 mix_off).
  --n3d PATH(E11, 2026-10-29): 고정 화자 분리 전문가 Nemotron 3 Diarization 의 활동을 슬롯 활동으로 쓴다(model.add_n3d — 창은 창마다 새로, --window full 은 회의 전체
    스트리밍 한 번). E9·E10 체크포인트에 붙이면 학습 없는 진단, E11 체크포인트는 config 의 경로가 기본이다. --aosc 와 함께 쓰지 않는다(config 에 n3d).
  활동 추종 진단(2026-10-30, 사용자 "LLM 활동 추종 진단 구현해놓자" — wiki output-n3d-branch-20261029 6.5 절): 슬롯 블록 디코드에 슬롯 활동이 있으면(E6 헤드·E11 전문가 등)
    행에 act_runs(그 활동의 슬롯별 [시작, 끝) 프레임)·follow(cst_eval.follow_stats — 단어가 활동이 켜진 슬롯에 들어갔나, 다른 슬롯·무음·겹침, 정답으로 누가 맞았나 —
    창 --follow-window BACK,FWD 기본 3,1)·cp_follow(어긋난 단어를 활동 슬롯으로 옮긴 cp 오류 수)를 남기고, summarize 가 합친다.
    --act-gate BACK,FWD: 활동 게이트 디코드(batch_decode.ActGate) — 라운드 k 슬롯 s 의 첫 출력이 글자면 프레임 k − δ, <BEGIN> 이면 k − 4 근처 [−BACK, +FWD] 에
    그 슬롯 활동이 없을 때 그 라운드 그 슬롯을 버린다(<EOT> 는 그대로). 행에 gated{text, begin}, config 에 act_gate. --verify(배치 대 직렬)도 같은 게이트로 대조한다.
    --follow-n3d(--from-predictions 전용): 원출력 행에 act_runs 가 없으면(2026-10-30 전 E11·--n3d 평가) 원출력 config 의 전문가 설정(n3d)으로 활동을 다시 계산한다(GPU —
    전문가만 읽는다). 창은 원 평가처럼 배치로 돌리므로 --batch-size·--world-size 를 원 평가와 같게 준다(AMI 창은 16·8).
  --relabel-follow(2026-10-30, 사용자 "출력 쪽 재귀속을 먼저 실험하고 그 다음에 학습으로 고치는 것이 시도하는 게 나을거 같아"): 가설 단어 중 그 단어 슬롯의
    활동이 꺼지고 다른 슬롯이 켜진 단어(follow_stats 의 off_other — 창 --follow-window)만 활동 슬롯으로 옮겨 채점한다. 단어 오디오 프레임 + FWD 까지의 활동만 쓰고
    LLM 문맥은 바꾸지 않으므로 스트리밍 디코드에서 그대로 낼 수 있는 출력이다(원출력 재채점 = 온라인 결과). 행 follow 는 옮기기 전 셈, relabeled = 옮긴 단어 수.
  --retag B,F(2026-11-05, 사용자 "① 디코드 때 재귀속 구현 진행해줘" — wiki output-agn-reduction-20261104 7 절): 회의 통째(--window full) 슬롯 블록 디코드에서
    --relabel-follow 와 같은 규칙(창 [f − B, f + F], f = 라운드 − δ)을 디코드 안에서 건다 — 라운드를 낸 뒤 활동을 어긴 칸을 활동 슬롯으로 옮기고 그 라운드를 학습 꼴로
    고쳐 다시 넣는다(batch_decode.BlockRetag · realtime_block_decode(retag)). 다음 라운드부터 LLM 이 고친 문맥을 본다(사후 재귀속과 다른 점). 번역 칸·X 칸과는 함께 쓰지 않는다.
    행 dretag{boxes, moved, tokens, rounds}, config 에 retag. 행 follow·cp_follow 는 고친 출력의 셈이다.
  --follow-detail(2026-10-30, 활동 추종 학습 설계용): 행에 follow_detail(센 단어마다 cst_eval.follow_stats detail 꼴 — 범주·슬롯·말 차례 위치·직전 화자)을 남긴다.
  --encoder-state <graft.pt>(U3, 2026-10-30): 모델의 인코더를 그 graft.pt 의 enc(화자 분리와 함께 학습한 인코더 — n3d_graft_train --unfreeze-layers)로 바꿔
    디코드한다 — ASR 지킴 확인(E11 의 cp·agn 이 그대로인가). config 에 encoder_state.
  --encoder-rc N(지연 완화, 2026-11-01, 사용자 "둘 다 구현해놓자"): 인코더 우측 문맥(80 ms 프레임 — 모델 설정의 att_context_size 목록 0·3·6·13 = 청크 80·320·560·1120 ms)을
    평가에서 바꾼다(학습은 3). 회의 통째 조각은 청크(N + 1) 배수로 맞춘다. 화자 분리기 지연은 --n3d-chunk·--n3d-rc 로 따로 늘린다. config 에 encoder_rc.
  --graft-expert <graft.pt>(G3, 2026-11-01, 사용자 "G3 구현해놓자" — wiki output-diarizer-replacement-20261031): 고정 전문가 자리(model.n3d — E11 또는 --n3d)에
    이식 학생(n3d_graft.GraftExpert — 우리 인코더 탭 → 어댑터 → N3D 위층 + G2 LoRA)을 끼워 그 활동으로 디코드한다(창은 창마다, 회의 통째는 회의 전체 한 번 —
    전문가와 같은 경로). 같은 graft.pt 를 --encoder-state 로도 주면 받아쓰기 인코더도 학생 인코더라 인코더가 하나다. 청크·임계는 모델 n3d_cfg. config 에 graft_expert.
  --en-norm whisper(2026-10-27): 영어 채점을 Whisper EnglishTextNormalizer 로(채움말 제거 등 — vapasr.data.cst_eval.set_en_norm). 외부 AMI 수치와 견줄 때
    --from-predictions 와 함께 새 --out 에 다시 매긴다(raw/sources/experiments/2026-10-27-std-rescore/rescore_std.sh). config 에 en_norm 이 남는다(ours 면 키 없음).
  --spk-guide on,margin[,span] [--spk-guide-hold] [--spk-guide-no-insert]: 화자 헤드 안내 디코딩(batch_decode.SpkGuide — model.spk_head 가 있는 Stage 3 체크포인트).
      2026-10-18 p9 ck3000: 삽입은 귀속을 나쁘게 한다(δ10 cp +4.1). 삽입 + 막기는 δ10 중립·δ4 cp −1.3.
      직렬 stream_decode 에는 안내가 없어 --verify 와 함께 쓰지 않는다. 행에 guide_insert·guide_hold(바꾼 태그 수)를 남긴다.
  --backchannel S,N: 참조 EOT 를 맞장구 규칙으로 만든다(맞장구 episode 는 EOT 참조에서 빠지고 상대 판정에서도 빠진다). 기본은 Phase 2 규칙.
  --window full(2026-10-24, 사용자 "AMI 평가에 한번에 돌릴 수 있도록 창 길이를 늘리고 싶은데, 자르지 않고"): 대화(회의)마다 창 하나 [0, duration_s) —
      자르지 않고 한 스트림으로 디코드한다(vapasr/hf/long_session: 겹침 조각 인코딩 encode_long + 긴 문맥 eager 디코드 long_decode, 창 하나씩).
      채점 규칙은 같아서 cp·agn 은 세션 단위 cpWER 다. --verify 는 앞 300 s 인코딩(파일 전체 forward 대비)과 앞 60 s 디코드(decode_batch 대비)를 잰다.
      --spk-guide·--switch-bias 와 함께 쓰지 않는다. 행에 positions(넣은 위치 수)·enc_s·dec_s.
      --context-s S(--window full 전용, 0 = 문맥 전부): 오디오는 자르지 않고 계속 디코드하되 LLM 문맥은 prefix + 최근 S 초(long_session.LongThinker 문맥 창).
  --target-oracle(E2, 2026-10-25 — wiki output-diarization-integration-20261025): 대상 화자 모드 체크포인트(adapter_inject)를 정답 화자 활동으로 잰다. 창의 화자마다
      그 화자 80 ms 활동을 인코더에 주입해 한 슬롯 prefix(그 화자 언어)로 따로 디코드하고, 화자별 출력을 창 화자 순서의 슬롯으로 모아 같은 규칙으로 채점한다.
      행에 emitted_tgt(화자별 원출력)·spill(슬롯 A 밖으로 낸 단위 수 — 대상 화자 모드인데 다른 슬롯 태그를 낸 말). 배치 디코드만(--verify·--switch-bias·--spk-guide·--window full 없이).
  --act-oracle(E5, 2026-10-26): 슬롯 활동 입력 체크포인트(adapter_slot)에 창의 정답 슬롯 활동(slot_activity — 창에서 먼저 말한 순서, 학습과 같은 슬롯)을
      오디오 토큰 입력으로 주고 한 스트림으로 디코드한다(decode_batch acts). 형식 상한 — 화자 분리가 완벽할 때 한 스트림이 얼마까지 가나. --verify·--window full 없이.
  --target-act DIR(E3, 2026-10-25): --target-oracle 의 정답 활동 대신 화자 분리 가지(E1)의 예측 활동 — experiments/diar_eval.py --save-probs 의 DIR/<창 id>.npz.
      예측 슬롯마다(확률 > --target-act-threshold 인 프레임이 --target-act-min-frames 이상) 그 0/1 활동을 주입해 디코드하고, 가설은 예측 슬롯 번호 자리에 둔다
      (cp·agn 은 순열과 무관, slot 은 예측 도착 순서 그대로). 언어는 창 대화 언어. 예측 슬롯이 없는 창도 행을 쓴다(가설 빈 — 모두 탈락). 행에 hyp_slots.
      V2 final 은 문맥 전부로 14 분 회의의 앞 3.5 분만 받아썼다(학습 창 20–90 s) — 회의 통째 점수는 문맥 창으로 잰다.
참조 = 그 δ 의 학습 타깃 시퀀스(cst_sequence)에서 전사 없는 음성 청크(mask_chunks — 학습 label −100)에 놓인 원소를 뺀 것. 전사 토큰 → 단어(시각 = 마지막
  토큰의 원래 끝 시각), EOT = episode 끝(p_end). 가설도 같은 청크에서 나온 토큰·EOT 를 버린다 — 참조·가설이 같은 규칙(2026-10-15 오라클 점검: 창 전체 토큰을
  참조로 두면 마스크 청크 때문에 오라클도 KO CER 7–15 % 가 나왔다).
지표(vapasr/data/cst_eval): slot·cp(화자 귀속)·agn(화자 무시) 오류율 — 한국어 CER(공백 제외), 영어·독일어 WER. 슬롯 뒤집힘 비율,
  EOT 재현율(양성 = 교대·침묵)·정밀도·보류 오방출·지연. 출력: predictions-rank{r}.jsonl(창·δ 별 원출력과 점수), summary.json."""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEN = {"aihub71631": "seen(Stage 1·2 노출)", "aihub134-1": "partly(화자 10 명 71631 과 겹침)"}


def is_full(a) -> bool:
    return getattr(a, "window", None) == "full"


def win_kw(a) -> dict:
    """V2 평가 창 설정(인자가 없는 호출자 — spk_probe·cst_attr_errors 의 옛 설정 — 는 V1 기본: 화자 2·20–40 s 격자).
    --window full 이면 창 목록은 windows_of 가 대화 길이로 만든다(full_windows) — 여기는 화자 상한·선언만."""
    kw = {} if is_full(a) else dict(window_s=tuple(getattr(a, "window", None) or (20.0, 40.0)), hop_s=float(getattr(a, "hop", None) or 10.0))
    ms = int(getattr(a, "max_speakers", None) or 2)
    if ms != 2: kw["max_speakers"] = ms
    if getattr(a, "window_draws", 0) and not is_full(a): kw["window_draws"] = int(a.window_draws)
    if getattr(a, "declare", None): kw["declare"] = int(a.declare)
    return kw


def win_conf(a) -> dict:
    """재개 지문에 넣을 V2 창 설정 — V1 기본이면 비운다(예전 run 의 config 그대로)."""
    kw = win_kw(a); out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items() if k not in ("window_s", "hop_s")}
    if is_full(a): out["window"] = "full"; out.update({"context_s": a.context_s} if getattr(a, "context_s", 0) else {})
    elif kw["window_s"] != (20.0, 40.0) or kw["hop_s"] != 10.0: out.update(window=list(kw["window_s"]), hop=kw["hop_s"])
    return out


def full_windows(path: str) -> list:
    """--window full: 대화마다 창 하나 (conv_id, 0, duration_s) — 회의 통째."""
    out = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l); out.append((r["conv_id"], 0.0, float(r["duration_s"])))
    return out


def windows_of(a, tok):
    """코퍼스별 held-out 창 → [(코퍼스, 데이터셋, 창 index)]."""
    import numpy as np
    from vapasr.data.cst_dialogue import CstDialogueDataset
    split = json.load(open(a.split, encoding="utf-8")) if a.split else None; out = []; sets = {}                 # split 없음 = 창 전부(held-out 재료로 만든 합성 평가 셋)
    for c in a.corpora.split(","):
        held = set(split["corpora"].get(c, {}).get("heldout", [])) if split else None
        if split is not None and not held: print(f"  {c}: held-out 없음 — 건너뜀", flush=True); continue
        path = os.path.join(a.data, f"{c}.refined.dialogues.jsonl"); kw = win_kw(a)
        if is_full(a): kw["windows"] = full_windows(path)                                       # 회의 통째 — 화자 상한을 넘는 대화는 버려진다(stats cst_dropped_speakers)
        ds = CstDialogueDataset([path], tok, delays=tuple(a.deltas), mono_cache_dir=os.path.join(a.data, "mono"), seed=0,
                                backchannel=a.backchannel, **kw, **({"xl_order": a.xl_order} if getattr(a, "xl_order", "lang") != "lang" else {}))   # V3 R4
        if is_full(a) and ds.stats.get("cst_dropped_speakers"): print(f"  !! {c}: 화자 상한 {ds.n_slots} 을 넘는 대화 {ds.stats['cst_dropped_speakers']} 개 — 뺀다", flush=True)
        idx = [i for i, it in enumerate(ds.items) if held is None or it[0] in held]
        if getattr(a, "subset", None):                                                          # 2026-11-05 판단용 빠른 평가: AMI 공식 분할 하나만(manifest dev·eval)
            keep = set(json.load(open(os.path.join(a.data, "manifest.json"), encoding="utf-8"))[a.subset])
            idx = [i for i in idx if ds.items[i][0].split(":", 1)[-1] in keep]
        idx.sort(key=lambda i: (ds.items[i][2], ds.items[i][0], ds.items[i][1]))
        if a.limit_windows and len(idx) > a.limit_windows: idx = [idx[j] for j in np.linspace(0, len(idx) - 1, a.limit_windows, dtype=int)]
        sets[c] = ds; out += [(c, i) for i in idx]; print(f"  {c}: {'held-out 대화 ' + str(len(held)) if held is not None else '전부'} · 창 {len(idx)}", flush=True)
    return sets, out


def words_of(tok, toks):
    """[(시각, tid)] → [(단어, 그 단어 마지막 바이트 토큰의 시각)]."""
    import bisect, re
    from vapasr.data.ast_eval import token_bytes
    if not toks: return []
    bs = token_bytes(tok, [t for _, t in toks]); raw = b"".join(bs); cum, c = [], 0
    for b in bs: c += len(b); cum.append(c)
    return [(m.group().decode("utf-8", "replace"), toks[min(bisect.bisect_left(cum, m.end()), len(toks) - 1)][0]) for m in re.finditer(rb"\S+", raw)]


def shard(items, sets, rank: int, world_size: int, balance: bool = False):
    """몫 r/w 의 창. balance 면 창 길이(초)가 긴 것부터 지금까지 합이 가장 작은 몫에 넣는다(같으면 번호가 작은 몫 — 늘 같은 배정).
    끄면 예전 그대로 j % w(창은 windows_of 의 길이 순)."""
    if not balance: return [x for j, x in enumerate(items) if j % world_size == rank]
    load = [0.0] * world_size; mine = []
    for j in sorted(range(len(items)), key=lambda j: (-float(sets[items[j][0]].items[items[j][1]][2]), j)):
        r = min(range(world_size), key=lambda q: (load[q], q)); load[r] += float(sets[items[j][0]].items[items[j][1]][2])
        if r == rank: mine.append(j)
    return [items[j] for j in sorted(mine)]


def window_ref(ds, i, delta, eot_id, tok):
    """창 i·δ 의 참조 → (슬롯별 [(단어, 시각)], [(슬롯, 끝 시각, p_end)] EOT, 마스크 청크 집합, 화자 순서).
    참조 = 그 δ 의 타깃 시퀀스에서 마스크 청크 밖 원소(학습 label 과 같은 범위)."""
    from vapasr.data.cst_dialogue import window_speakers, window_streams
    cid, t0w, L = ds.items[i]; d = ds.dlgs[cid]; K = int(round(L / 0.08))
    eps = ds.eps[cid]; sp_order = window_speakers(eps, t0w, L)
    streams, eot_p, _ = window_streams(eps, sp_order, t0w, L, eot_id)
    q = ds.cst_sequence(i, delta); mask = ds.mask_chunks(d, t0w, L, K, delta); ref_tok = [[] for _ in range(max(2, len(sp_order)))]; ref_eot = []
    for pos, src in enumerate(q["src_at"]):
        if src is None or (0 <= q["chunk_at"][pos] < K and q["chunk_at"][pos] in mask): continue
        x = streams[src[0]][src[1]]
        if len(x) == 3: ref_eot.append((src[0], x[1], eot_p.get((src[0], src[1]))))
        else: ref_tok[src[0]].append((x[1], x[0]))
    return [words_of(tok, r) for r in ref_tok], ref_eot, mask, sp_order


def verify_long(model, wav, prefix, max_flush, D=None, gate=None) -> None:
    """--window full --verify: 겹침 조각 인코딩(encode_long)을 파일 전체 forward 와 앞 300 s 로(조각 1000 프레임 — 조각 넷), 긴 디코드(long_decode)를
    E6 이후(화자 분리 헤드 — 스트림 1 = 예측 활동)는 스트림 0(인코더 특징)만 대조한다(2026-10-28: 인과 헤드 예측은 조각 앞 맥락에 따라 몇 프레임 달라져
    96461 몫 0 이 max|Δ| 1.0 으로 멈췄다 — 그 차이는 몫으로만 적는다).
    배치 디코드(decode_batch)와 앞 60 s 특징으로(블록 256 위치 — 블록·나머지 경로 둘 다) 대조한다.
    D(E9 슬롯 블록 — 선언 슬롯 수): 긴 디코드는 long_block_decode(추측 prefill), 배치는 decode_blocks_batch 로 대조한다.
    gate(활동 → batch_decode.ActGate 를 만드는 함수, --act-gate): 두 디코드에 같은 규칙의 새 게이트를 준다."""
    import numpy as np, torch
    from vapasr.hf.batch_decode import decode_batch
    from vapasr.hf.long_session import encode_long, long_decode
    dev = next(model.parameters()).device; n = min(len(wav), 300 * 16000); w0 = np.asarray(wav[:n], dtype=np.float32); K0 = int(round(n / 16000 / 0.08))
    whole = model.encode(torch.from_numpy(w0).to(dev)[None], torch.tensor([n], device=dev), torch.tensor([K0], device=dev))[0]
    ch = int(getattr(model.config, "encoder_right_context", 0) or 0) + 1
    seg = encode_long(model, w0, K0, core=1000 // ch * ch, ctx=1760 // ch * ch)                # 조각은 청크 배수(우측 문맥 3 이면 1000·1760 그대로)
    d = (whole[:1] - seg[:1]).abs().max().item()                                               # 스트림 0(인코더 특징)만 — 2026-10-28
    act = f" · 예측 활동 스트림 다른 프레임 몫 {float((whole[1] != seg[1]).any(-1).float().mean()):.4f}(인과 헤드는 조각 앞 맥락에 따라 달라진다 — 대조 밖)" \
        if getattr(model, "diar_head", None) is not None and whole.shape[0] > 1 else ""
    print(f"ENC-PARITY 앞 {n / 16000:.0f} s(조각 1000 프레임·앞 겹침 1760): 스트림 0 max|Δ| {d:.2e}{act}", flush=True)
    assert d < 1e-3, f"겹침 조각 인코딩이 파일 전체 forward 와 다르다 {d}"
    f60 = whole[:, :750]
    if D:
        from vapasr.hf.batch_decode import decode_blocks_batch
        from vapasr.hf.long_session import long_block_decode
        A60 = model.block_act(f60)
        ref = decode_blocks_batch(model, [f60], [list(prefix)], D, max_flush_rounds=max_flush, gates=[gate(A60)] if gate else None)[0]
        out, forced, rounds, used, _, _ = long_block_decode(model, f60, prefix, D, max_flush_rounds=max_flush, blk=256, gate=gate(A60) if gate else None)
    else:
        ref = decode_batch(model, [f60], [list(prefix)], max_flush_rounds=max_flush)[0]
        out, forced, rounds, used = long_decode(model, f60, prefix, max_flush_rounds=max_flush, blk=256)
    assert out == ref.emitted and forced == ref.forced and rounds == ref.flush_rounds, f"긴 디코드 ≠ 배치 디코드: 토큰 {len(out)}/{len(ref.emitted)} forced {forced}/{ref.forced}"
    print(f"DEC-PARITY 앞 60 s: 토큰 {len(out)} 같음(위치 {used})", flush=True)


def target_act_of(d, wid, K, thr):
    """E3: diar_eval --save-probs 의 창 확률(d/<창 id>.npz, (K, 슬롯) float16) → 확률 > thr 인 프레임이 1 인 (K, 슬롯) float 텐서."""
    import numpy as np, torch
    p = np.load(os.path.join(d, f"{wid}.npz"))["prob"].astype(np.float32); assert p.shape[0] == K, (wid, p.shape, K)
    return torch.from_numpy((p > thr).astype(np.float32))


def target_oracle_loop(a, model, tok, sets, items, dest, done, spk, EOT, n_slots, dev) -> int:
    """--target-oracle(E2): 창의 화자마다 정답 활동 주입 → 한 슬롯 prefix 로 디코드 → 슬롯(창 화자 순서)으로 모아 채점. 쓴 행 수를 돌려준다.
    창 여러 개의 (창, 화자) 행을 모아 인코딩(활동만 다른 같은 오디오 — 패딩 배치)·디코드(prefix 가 같은 행끼리, 최대 --batch-size 행)한다."""
    import numpy as np, torch
    from vapasr.data.cst_dialogue import slot_activity, window_speakers
    from vapasr.data.cst_eval import parse_emitted, score_window, match_eot, eot_tol
    from vapasr.hf.batch_decode import decode_batch
    n, t0 = 0, time.time(); words = lambda toks: words_of(tok, toks); per = max(1, a.batch_size // 4)   # 창 묶음(화자 4 명이면 batch-size 행)
    pred = getattr(a, "target_act", None)                                                       # E3: 예측 활동 폴더(없으면 정답 활동 — E2)
    with open(dest, "a", buffering=1) as fo, torch.inference_mode():
        for c in dict.fromkeys(x[0] for x in items):
            ds = sets[c]; todo = [i for cc, i in items if cc == c and not all((f"{ds.items[i][0]}@{ds.items[i][1]}", dl) in done for dl in a.deltas)]
            for b0 in range(0, len(todo), per):
                rows = []                                                                       # (창 i, 화자 또는 예측 슬롯, 언어, wav, K, 활동 (K,))
                for i in todo[b0: b0 + per]:
                    cid, t0w, L = ds.items[i]; d = ds.dlgs[cid]; K = int(round(L / 0.08)); allspk, alllang = ds.slots(i)
                    wav = np.asarray(ds.mono(cid)[int(round(t0w * 16000)): int(round((t0w + L) * 16000))], dtype=np.float32)
                    if pred:
                        act = target_act_of(pred, f"{cid}@{t0w}", K, a.target_act_threshold)
                        rows += [(i, s_, d.lang, wav, K, act[:, s_]) for s_ in range(act.shape[1]) if act[:, s_].sum() >= a.target_act_min_frames]
                    else:
                        rows += [(i, x, alllang[allspk.index(x)] if x in allspk else d.lang, wav, K, torch.tensor(slot_activity(ds.eps[cid], [x], t0w, L, K, 1), dtype=torch.float32)[:, 0])
                                 for x in window_speakers(ds.eps[cid], t0w, L)[:n_slots]]
                feats = []
                for r0 in range(0, len(rows), a.batch_size):                                    # 인코딩: 패딩 배치(행마다 K 로 자른다)
                    rr = rows[r0: r0 + a.batch_size]; Nw = max(len(r[3]) for r in rr); Km = max(r[4] for r in rr)
                    W = torch.zeros(len(rr), Nw, device=dev); A = torch.zeros(len(rr), Km, device=dev)
                    for j, (i, x, _, wav, K, act) in enumerate(rr): W[j, : len(wav)] = torch.from_numpy(wav).to(dev); A[j, :K] = act.to(dev)
                    f = model.encode(W, torch.tensor([len(r[3]) for r in rr], device=dev), torch.tensor([r[4] for r in rr], device=dev), act=A)
                    feats += [f[j, :, : rr[j][4]] for j in range(len(rr))]
                for delta in a.deltas:
                    pres = [ds.cst_prefix([r[2]], delta) for r in rows]; states = [None] * len(rows)
                    for pre in dict.fromkeys(tuple(p) for p in pres):                          # 한 배치에 prefix 하나 — 언어가 다른 화자는 나눠 돈다
                        js = [j for j, p in enumerate(pres) if tuple(p) == pre]
                        for g0 in range(0, len(js), a.batch_size):
                            jj = js[g0: g0 + a.batch_size]
                            for j, st_ in zip(jj, decode_batch(model, [feats[j] for j in jj], [list(pre)] * len(jj), max_flush_rounds=a.max_flush)): states[j] = st_
                    for i in todo[b0: b0 + per]:                                                # 행이 없는 창(예측 슬롯 없음)도 채점한다
                        cid, t0w, L = ds.items[i]; d = ds.dlgs[cid]; K = int(round(L / 0.08))
                        if (f"{cid}@{t0w}", delta) in done: continue
                        js = [j for j, r in enumerate(rows) if r[0] == i]
                        ref_w, ref_eot, mask, sp_order = window_ref(ds, i, delta, EOT, tok)
                        if not pred: assert sp_order[: len(js)] == [rows[j][1] for j in js], (sp_order, js)
                        slots_ = [rows[j][1] for j in js] if pred else list(range(len(js)))     # 가설 슬롯: 예측 슬롯 번호(E3) 또는 창 화자 순서(E2)
                        hyp_w, he_all, spill = [[] for _ in range(max(slots_, default=-1) + 1)], [], 0
                        for s_, j in zip(slots_, js):
                            hs, he = parse_emitted(states[j].emitted, spk, EOT, drop_chunks=mask)
                            hyp_w[s_] = words(hs[0]); spill += sum(len(h) for h in hs[1:]); he_all += [(s_, t) for k_, t in he if k_ == 0]
                        sc = score_window(ref_w, hyp_w, d.lang); sc["eot"] = match_eot(ref_eot, he_all, sc["perm"], eot_tol(delta))
                        row = dict(id=f"{cid}@{t0w}", corpus=c, lang=d.lang, delta=delta, L=L, K=K, speakers=sp_order, n_spk=len(sp_order), masked_chunks=len(mask),
                                   forced=sum(states[j].forced for j in js), flush_rounds=max(states[j].flush_rounds for j in js), emitted=[], target_oracle=True,
                                   emitted_tgt=[[list(e) for e in states[j].emitted] for j in js], spill=spill, **({"hyp_slots": slots_} if pred else {}),
                                   ref_text=[" ".join(w_ for w_, _ in r) for r in ref_w], hyp_text=[" ".join(w_ for w_, _ in h) for h in hyp_w], **sc)
                        fo.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
                print(f"  {c}: {min(b0 + per, len(todo))}/{len(todo)} 창 · {n} 행 · {time.time() - t0:.0f}s", flush=True)
    return n


def run(a):
    import numpy as np, torch
    from vapasr.data.cst_eval import parse_emitted, score_window, match_eot, eot_tol, emitted_from_sequence, set_en_norm, emitted_from_blocks, activity_from_events, \
        follow_stats, act_runs, act_from_runs, attribute_x, merge_x
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    set_en_norm(getattr(a, "en_norm", "ours"))                                                # 영어 채점 정규화(ours 기본 · whisper 외부 비교용)
    gate_w = tuple(getattr(a, "act_gate", None) or ()); fwin = tuple(getattr(a, "follow_window", None) or (3, 1)); fn3 = bool(getattr(a, "follow_n3d", False))   # 활동 추종 진단(2026-10-30)
    assert len(gate_w) in (0, 2) and len(fwin) == 2 and min(gate_w + fwin) >= 0, "--act-gate·--follow-window 는 BACK,FWD(0 이상 정수 둘)"
    assert not gate_w or not (a.oracle or a.from_predictions), "--act-gate 는 모델 디코드 전용(--oracle·--from-predictions 없이)"
    assert not fn3 or a.from_predictions, "--follow-n3d 는 --from-predictions 와 함께(원출력에 활동이 없을 때 전문가 활동을 다시 계산한다)"
    assert not getattr(a, "encoder_state", None) or not (a.oracle or a.from_predictions), "--encoder-state 는 모델 디코드 전용"
    assert not getattr(a, "graft_expert", None) or not (a.oracle or a.from_predictions), "--graft-expert 는 모델 디코드 전용"
    assert getattr(a, "encoder_rc", None) is None or not (a.oracle or a.from_predictions), "--encoder-rc 는 모델 디코드 전용"
    gl_, gt_ = getattr(a, "gate_lags", None), getattr(a, "gate_thresh", None)                    # E13 게이트 진단(2026-11-03): 학습 없이 추론 게이트만 바꾼다
    assert (gl_ is None and gt_ is None) or not (a.oracle or a.from_predictions), "--gate-lags·--gate-thresh 는 모델 디코드 전용"
    assert gl_ is None or gl_ >= 0, f"--gate-lags 는 0 이상: {gl_}"
    assert gt_ is None or 0.0 < gt_ < 1.0, f"--gate-thresh 는 (0, 1): {gt_}"
    rlf = bool(getattr(a, "relabel_follow", False))
    assert not (rlf and getattr(a, "relabel_act", None)), "--relabel-follow 와 --relabel-act 는 함께 쓰지 않는다"
    rtw = tuple(getattr(a, "retag", None) or ())                                                 # 디코드 때 재귀속(2026-11-05)
    assert not rtw or (len(rtw) == 2 and min(rtw) >= 0 and is_full(a) and not (a.oracle or a.from_predictions or gate_w)), \
        "--retag 는 BACK,FWD(0 이상 둘), 회의 통째(--window full) 모델 디코드 전용(--oracle·--from-predictions·--act-gate 없이)"
    sb = None
    if getattr(a, "switch_bias", None):                                                        # V3 C3 상태 인지 교대 bias(교차 언어 창만) — 없으면 예전 디코드 그대로
        from vapasr.hf.batch_decode import SwitchBias
        sb = SwitchBias.parse(a.switch_bias)
        assert not (sb.on and a.verify), "--switch-bias 는 배치 디코드 전용이다(직렬 stream_decode 에 bias 가 없다) — --verify 없이(전부 0 은 같은 토큰열이라 허용)"
        assert not (a.oracle or a.from_predictions or a.spk_guide), "--switch-bias 는 모델 디코드 전용이고 --spk-guide 와 함께 쓰지 않는다"
    prev = {}
    if a.from_predictions:
        import glob
        for f in glob.glob(os.path.join(a.from_predictions, "predictions-rank*.jsonl")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l); prev[(r["id"], r["delta"])] = r
        print(f"  원출력 {len(prev)} 행 ← {a.from_predictions}", flush=True)
    if a.oracle or a.from_predictions:
        from transformers import AutoTokenizer
        model = None; tok = AutoTokenizer.from_pretrained(a.checkpoint); cfg = json.load(open(os.path.join(a.checkpoint, "config.json"))); sp = cfg["sp_ids"]; blocked = cfg["blocked_ids"]
    else:
        from vapasr.hf.infer import load_model
        from vapasr.hf.batch_decode import decode_batch, SpkGuide, word_init_mask, logits_size
        model, tok = load_model(a.checkpoint, encoder_path=a.encoder, dtype=getattr(torch, a.dtype)); sp = model.config.sp_ids; blocked = model.config.blocked_ids
        if getattr(a, "slot_enc_off", False):                                                   # 2026-11-06 E15b 진단: 슬롯 소리 토큰 0(섞인 소리·인코더 그대로)
            assert getattr(model, "adapter_slotenc", None) is not None, "--slot-enc-off 는 E15b(슬롯 조건 인코더) 체크포인트 전용"
            model.slot_enc_off = True
        if getattr(a, "diar_head_override", None):                                              # E7: AOSC 로 학습한 헤드로 바꿔 끼운다(같은 cfg·탭 — diar_train --aosc final)
            xo = torch.load(a.diar_head_override, map_location="cpu"); assert model.diar_head is not None and dict(xo["cfg"]) == dict(model.config.diar_head_cfg) \
                and list(xo["taps"]) == list(model.config.diar_taps), f"헤드 cfg·탭이 다르다: {a.diar_head_override}"
            assert int(xo.get("shift", 0) or 0) == int(getattr(model.config, "act_shift", 0) or 0), f"헤드 shift {xo.get('shift', 0)} ≠ 모델 act_shift — E8 규약이 다르다"
            model.diar_head.load_state_dict(xo["state"])
            if xo.get("aosc"): model.config.diar_aosc = dict(xo["aosc"])
            print(f"  화자 분리 헤드 ← {a.diar_head_override}(aosc {xo.get('aosc')})", flush=True)
        if getattr(a, "encoder_state", None):                                                    # U3: 화자 분리와 함께 학습한 인코더로 바꿔 끼운다(ASR 지킴 확인)
            xs = torch.load(a.encoder_state, map_location="cpu"); assert xs.get("enc") is not None, f"인코더 state(enc)가 없다: {a.encoder_state}"
            model.encoder.enc.load_state_dict(xs["enc"]); model.encoder.eval()
            print(f"  인코더 ← {a.encoder_state}(U3 — 아래 {xs.get('unfreeze')} 층 학습, step {xs.get('step')})", flush=True)
        if getattr(a, "n3d", None) or getattr(model, "n3d", None) is not None:                 # E11: 고정 화자 분리 전문가(평가에서 붙이거나 — E9·E10 진단 — 설정을 바꾼다)
            model.add_n3d(a.n3d or None, chunk=a.n3d_chunk, rc=a.n3d_rc, threshold=a.n3d_threshold)
            print(f"  고정 화자 분리 전문가 n3d ← {model.config.n3d_cfg}", flush=True)
        if getattr(a, "encoder_rc", None) is not None:                                           # 지연 완화: 인코더 우측 문맥(받아쓰기 특징만 — 학생 GraftExpert 는 제 설정)
            assert not getattr(a, "graft_expert", None), "--encoder-rc 는 --graft-expert 와 함께 쓰지 않는다(학생 인코더는 제 우측 문맥)"
            model.encoder.enc.set_default_att_context_size([int(model.config.encoder_left_context), int(a.encoder_rc)]); model.config.encoder_right_context = int(a.encoder_rc)
            print(f"  인코더 우측 문맥 ← {a.encoder_rc} 프레임(청크 {(a.encoder_rc + 1) * 80} ms, 학습은 3)", flush=True)
        if getattr(a, "graft_expert", None):                                                     # G3: 고정 전문가 자리에 이식 학생 — 활동만 바꾸고 디코드 경로는 그대로
            assert getattr(model, "n3d", None) is not None, "--graft-expert 는 고정 전문가 자리(E11 체크포인트 또는 --n3d)에 끼운다"
            from vapasr.hf.n3d_graft import GraftExpert
            c = model.config.n3d_cfg
            model.n3d = GraftExpert(a.graft_expert, nemotron_dir=a.encoder, chunk=int(c["chunk"]), rc=int(c["rc"]), threshold=float(c["threshold"]))
            print(f"  화자 활동 ← 이식 학생 {model.n3d.cfg}(G3 — 고정 전문가 대신)", flush=True)
        if gl_ is not None or gt_ is not None:                                                    # E13 게이트 진단: 게이트 창 · 게이트만 낮은 임계(태그 활동은 그대로)
            assert int(getattr(model.config, "slot_gate", 0) or 0) > 0, "--gate-lags·--gate-thresh 는 활동 게이트 모델(config.slot_gate — E13)"
            assert gt_ is None or getattr(model, "n3d", None) is not None, "--gate-thresh 는 전문가 활동 모델(E11·E13 · --graft-expert)"
            if gl_ is not None: model.config.slot_gate = int(gl_)
            if gt_ is not None: model.keep_n3d_probs = True                                        # n3d_act 가 같은 활동을 내며 확률을 last_n3d_probs 에 남긴다
            print(f"  E13 게이트 ← {'끔(선언 슬롯 전부)' if not model.config.slot_gate else f'창 {model.config.slot_gate} 청크'}(학습 12)"
                  f"{'' if gt_ is None else f' · 전문가 확률 > {gt_} 로 연다(태그 활동은 그대로)'}", flush=True)
        wi = word_init_mask(tok, logits_size(model)) if sb else None
    a.xl_order = (cfg.get("xl_slot_order") if model is None else getattr(model.config, "xl_slot_order", None)) or "lang"   # V3 R4: 교차 언어 슬롯 순서(창·prefix)
    from vapasr.data.cst_dialogue import SLOTS
    n_slots = max(2, int(getattr(a, "max_speakers", None) or 2))
    if model is not None and len(model.config.slot_tokens) < n_slots:                      # V1(2 슬롯) 기준선: 같은 다화자 창을 2 슬롯으로 잰다(prefix 선언도 2)
        print(f"  모델 슬롯 {model.config.slot_tokens} < 창 화자 상한 {n_slots} — 모델 슬롯 수로 디코드·채점(참조는 창 화자 전부)", flush=True); n_slots = len(model.config.slot_tokens)
    n_parse = len(model.config.slot_tokens) if model is not None else n_slots              # 출력 해석은 모델 슬롯 전부(V2 4) — 2 인 창에서 C·D 로 낸 말은 삽입으로 센다(텍스트로 읽지 않는다)
    spk = tuple(sp[t] for t in SLOTS[:n_parse]); EOT = sp["<EOT>"]
    XI = int(sp["<SPK_5>"]) if (cfg.get("x_slot") if model is None else getattr(model.config, "x_slot", False)) else -1   # E14 X 칸(2026-11-03): 해석은 슬롯 태그 + X
    spk_p = spk + ((XI,) if XI >= 0 else ())
    blocks = bool(cfg.get("slot_blocks") if model is None else getattr(model.config, "slot_blocks", False)) or bool(getattr(a, "blocks", False))   # E9 슬롯 블록(2026-10-28)
    assert not getattr(a, "blocks", False) or a.oracle, "--blocks 는 --oracle 과 함께(학습 전 E9 형식 점검)"
    BEGIN = sp.get("<ONSET>") if blocks else None
    mix_off = bool(getattr(a, "mix_off", False))
    assert not mix_off or (model is not None and getattr(model, "adapter_mixnull", None) is not None) or a.from_predictions, "--mix-off 는 E10(슬롯 추출기) 체크포인트 디코드"
    if blocks:
        assert not (sb or a.spk_guide or getattr(a, "target_oracle", False) or getattr(a, "act_tag_guide", False)), \
            "E9 슬롯 블록은 --switch-bias·--spk-guide·--target-oracle·--act-tag-guide 와 함께 쓰지 않는다(태그는 정해진 입력)"
        from vapasr.hf.batch_decode import decode_blocks_batch, ActGate, BlockRetag
        from vapasr.hf.diar import der_counts
        from vapasr.data.cst_dialogue import slot_activity as _slot_act
    assert not (gate_w or fn3) or blocks, "--act-gate·--follow-n3d 는 E9 슬롯 블록(태그 활동이 있는) 체크포인트·원출력"
    mk_gate = (lambda A_, dl: ActGate(A_, dl, BEGIN, EOT, *gate_w)) if gate_w else None             # 활동 게이트 디코드(2026-10-30 진단)
    fx = None
    if fn3:                                                                                     # 활동 추종: 원출력에 활동이 없으면 원 평가의 전문가 설정으로 다시 계산
        import glob
        pcs = sorted(glob.glob(os.path.join(a.from_predictions, "config-rank*.json"))); n3c = json.load(open(pcs[0])).get("n3d") if pcs else None
        assert n3c, f"--follow-n3d: 원출력 config 에 n3d 가 없다 — 고정 전문가(E11·--n3d) 평가의 원출력만: {a.from_predictions}"
        from vapasr.hf.n3d import N3DExpert
        fx = N3DExpert(n3c); fx_n = int(cfg.get("slot_input_slots") or 4); fdev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  활동 추종: 원출력 활동을 전문가로 다시 계산 ← {n3c}", flush=True)
    if EOT in blocked and not a.oracle: print("!! <EOT> 가 decode 차단 목록에 있다(Stage 3 학습 전 체크포인트) — EOT 는 나오지 않는다", flush=True)
    full = is_full(a); bs = 1 if full else a.batch_size                                         # 회의 통째: 창(회의) 하나씩 긴 직렬 디코드
    tor = bool(getattr(a, "target_oracle", False)); act_or = bool(getattr(a, "act_oracle", False))
    acfg = None
    assert not (getattr(a, "aosc", False) and model is not None and getattr(model, "n3d", None) is not None), "--aosc 는 고정 화자 분리 전문가(n3d) 모델과 함께 쓰지 않는다(활동은 전문가가)"
    if getattr(a, "aosc", False):                                                              # E7: 회의 통째의 예측 활동을 AOSC 스트리밍으로(회의 내내 같은 슬롯)
        from vapasr.hf.diar import AoscConfig
        assert full and model is not None and getattr(model, "diar_head", None) is not None, "--aosc 는 --window full 과 화자 분리 헤드(E6·E7) 모델"
        acfg = AoscConfig.from_dict(getattr(model.config, "diar_aosc", None))
        if getattr(a, "aosc_chunk", None): acfg.chunk = int(a.aosc_chunk)
        print(f"  AOSC {acfg.to_dict()}", flush=True)
    if act_or:
        assert model is not None and getattr(model, "adapter_slot", None) is not None and not (tor or a.verify or is_full(a) or a.oracle or a.from_predictions), \
            "--act-oracle: adapter_slot 이 있는 E5 체크포인트, 배치 디코드만(--target-oracle·--verify·--window full·--oracle 없이)"
        from vapasr.data.cst_dialogue import slot_activity, window_speakers
    if tor: assert model is not None and getattr(model, "adapter_inject", None) is not None and not (sb or a.spk_guide or a.verify or full), \
        "--target-oracle: adapter_inject 가 있는 E2 체크포인트, 배치 디코드만(--verify·--switch-bias·--spk-guide·--window full 없이)"
    assert full or not a.context_s, "--context-s 는 --window full 전용"
    assert not getattr(a, "act_tag_guide", False) or (full and model is not None and getattr(model, "diar_head", None) is not None), "--act-tag-guide 는 --window full 과 화자 분리 헤드 모델(E6·E7) 디코드"
    assert not getattr(a, "relabel_act", None) or (full and a.from_predictions), "--relabel-act 는 --from-predictions 와 --window full(진단 재채점)"
    assert not getattr(a, "relabel_oracle", False) or getattr(a, "relabel_act", None), "--relabel-oracle 은 --relabel-act 와 함께"
    assert tor or not getattr(a, "target_act", None), "--target-act 는 --target-oracle 과 함께(E3 — 정답 대신 예측 활동)"
    if full:
        assert not (sb or a.spk_guide), "--window full 은 긴 직렬 디코드(long_decode)라 --switch-bias·--spk-guide 와 함께 쓰지 않는다"
        if model is not None: from vapasr.hf.long_session import encode_long, long_decode, long_block_decode
    sets, items = windows_of(a, tok)
    items = shard(items, sets, a.rank, a.world_size, getattr(a, "balance", False))
    out = os.path.join(a.out); os.makedirs(out, exist_ok=True)
    conf = dict(checkpoint=os.path.abspath(a.checkpoint), config_sha256=hashlib.sha256(open(os.path.join(a.checkpoint, "config.json"), "rb").read()).hexdigest(),
                split_sha256=hashlib.sha256(open(a.split, "rb").read()).hexdigest() if a.split else None, corpora=a.corpora, deltas=a.deltas, limit_windows=a.limit_windows, dtype=a.dtype,
                world_size=a.world_size, eot_blocked=EOT in blocked, oracle=bool(a.oracle), from_predictions=a.from_predictions,
                **({"subset": a.subset} if getattr(a, "subset", None) else {}), **({"balance": True} if getattr(a, "balance", False) else {}),   # 2026-11-05: 끄면 키 없음
                **({"backchannel": list(a.backchannel)} if a.backchannel else {}), **win_conf(a),
                **({"spk_guide": list(a.spk_guide), "spk_guide_hold": bool(a.spk_guide_hold), **({"spk_guide_no_insert": True} if a.spk_guide_no_insert else {})} if a.spk_guide else {}),
                **({"switch_bias": sb.policy()} if sb else {}), **({"xl_slot_order": a.xl_order} if a.xl_order != "lang" else {}), **({"target_oracle": True} if tor else {}),
                **({"act_oracle": True} if act_or else {}), **({"aosc": acfg.to_dict()} if acfg is not None else {}),
                **({"diar_head_override": os.path.abspath(a.diar_head_override)} if getattr(a, "diar_head_override", None) else {}),
                **({"n3d": dict(model.config.n3d_cfg)} if model is not None and getattr(model, "n3d", None) is not None else {}),
                **({"en_norm": a.en_norm} if getattr(a, "en_norm", "ours") != "ours" else {}),
                **({"act_tag_guide": True} if getattr(a, "act_tag_guide", False) else {}), **({"slot_blocks": True} if blocks else {}), **({"mix_off": True} if mix_off else {}),
                **({"act_gate": list(gate_w)} if gate_w else {}), **({"follow_window": list(fwin)} if fwin != (3, 1) else {}), **({"follow_n3d": True} if fn3 else {}),
                **({"encoder_state": os.path.abspath(a.encoder_state)} if getattr(a, "encoder_state", None) else {}),
                **({"graft_expert": os.path.abspath(a.graft_expert)} if getattr(a, "graft_expert", None) else {}),
                **({"encoder_rc": int(a.encoder_rc)} if getattr(a, "encoder_rc", None) is not None else {}),
                **({"gate_lags": int(gl_)} if gl_ is not None else {}), **({"gate_thresh": float(gt_)} if gt_ is not None else {}),
                **({"relabel_follow": list(fwin)} if rlf else {}), **({"retag": list(rtw)} if rtw else {}), **({"slot_enc_off": True} if getattr(a, "slot_enc_off", False) else {}),
                **({"relabel_act": os.path.abspath(a.relabel_act), "relabel_oracle": bool(a.relabel_oracle)} if getattr(a, "relabel_act", None) else {}),
                **({"target_act": os.path.abspath(a.target_act), "target_act_threshold": a.target_act_threshold, "target_act_min_frames": a.target_act_min_frames}
                   if getattr(a, "target_act", None) else {}))
    if a.spk_guide:
        assert not a.verify, "--spk-guide 는 배치 디코드 전용이다(직렬 stream_decode 에 안내가 없다) — --verify 없이"
        assert model is not None and getattr(model, "spk_head", None) is not None, "--spk-guide: 화자 헤드가 있는 체크포인트가 필요하다"
        ws_cache = {}
        def word_start(tid):
            if tid not in ws_cache: ws_cache[tid] = tok.decode([tid]).startswith(" ")
            return ws_cache[tid]
    cp = os.path.join(out, f"config-rank{a.rank}.json")
    if os.path.exists(cp): assert json.load(open(cp)) == conf, "재개 설정 불일치 — 새 --out 을 쓴다"
    else: json.dump(conf, open(cp, "w"), indent=1)
    dest = os.path.join(out, f"predictions-rank{a.rank}.jsonl")
    done = {(r["id"], r["delta"]) for r in map(json.loads, open(dest))} if os.path.exists(dest) else set()
    dev = next(model.parameters()).device if model is not None else None; checked = set(); t0 = time.time(); n = 0
    words = lambda toks: words_of(tok, toks)

    if tor:
        n = target_oracle_loop(a, model, tok, sets, items, dest, done, spk, EOT, n_slots, dev)
        print(f"DONE rank={a.rank} rows={n}", flush=True); return
    n3 = model is not None and getattr(model, "n3d", None) is not None and getattr(model, "adapter_slotinj", None) is None   # E11: 창 평가는 고정 전문가를 배치로(창마다 새로) — E15b 는 model.encode 가 창마다 전문가 활동으로 슬롯 조건 인코딩
    with open(dest, "a", buffering=1) as fo, torch.inference_mode():
        for c in dict.fromkeys(x[0] for x in items):
            ds = sets[c]; todo = [i for cc, i in items if cc == c]
            for b0 in range(0, len(todo), bs):
                batch = todo[b0:b0 + bs]; feats = []; meta = []; enc_s = None; n3w = []; fxw = []; GA = []   # GA: --gate-thresh 게이트 활동(feats 와 같은 자리)
                for i in batch:
                    cid, t0w, L = ds.items[i]; d = ds.dlgs[cid]; K = int(round(L / 0.08)); meta.append((i, cid, t0w, L, K, d))
                    if a.oracle or a.from_predictions:
                        if fx is not None: fxw.append(np.asarray(ds.mono(cid)[int(round(t0w * 16000)): int(round((t0w + L) * 16000))], dtype=np.float32))
                        continue
                    wav = np.asarray(ds.mono(cid)[int(round(t0w * 16000)): int(round((t0w + L) * 16000))], dtype=np.float32)
                    if full:
                        if all((f"{cid}@{t0w}", dl) in done for dl in a.deltas): continue           # 끝난 회의는 인코딩도 건너뛴다(아래 δ 고리도 건너뜀)
                        if a.verify and "full" not in checked:
                            verify_long(model, wav, ds.cst_prefix(ds.slots(i)[1][:n_slots], a.deltas[0]), a.max_flush, D=len(ds.slots(i)[1][:n_slots]) if blocks else None,
                                        gate=(lambda A_: mk_gate(A_, a.deltas[0])) if mk_gate else None); checked.add("full")
                        te = time.time(); feats.append(encode_long(model, wav, K, aosc=acfg)); enc_s = round(time.time() - te, 1)
                        if gt_ is not None: GA.append((model.last_n3d_probs[0, :K, : int(model.config.slot_input_slots)] > gt_).cpu().numpy())   # encode_long 이 회의 전체로 한 번 돈 전문가
                        continue
                    w = torch.from_numpy(wav).to(dev)
                    if n3: n3w.append(w)                                                                # E11: 전문가는 아래에서 배치 창을 한 번에
                    feats.append(model.encode(w[None], torch.tensor([len(wav)], device=dev), torch.tensor([K], device=dev), **({"n3d": False} if n3 else {}))[0])
                if n3 and feats and not full:                                                           # E11: 고정 전문가를 배치로 한 번 돌려 스트림 1 에(창마다 새로 — model.encode 와 같은 값)
                    Kb = torch.tensor([m[4] for m in meta], device=dev)
                    A = model.n3d_act(torch.nn.utils.rnn.pad_sequence(n3w, batch_first=True), torch.tensor([len(x) for x in n3w], device=dev), Kb)
                    feats = [model.with_act_stream(f[None], A[j: j + 1, : int(Kb[j])])[0] for j, f in enumerate(feats)]
                    if gt_ is not None: GA = [(model.last_n3d_probs[j, : int(Kb[j]), : A.shape[-1]] > gt_).cpu().numpy() for j in range(len(feats))]
                A_np = [None] * len(meta); FA = [None] * len(meta); AS = [None] * len(meta)             # 활동 추종: 디코드가 받은 슬롯 활동(K, N) · --follow-n3d 로 다시 계산한 활동 · E14 X 귀속용 확률
                if fx is not None and fxw and not all((f"{m[1]}@{m[2]}", dl) in done for m in meta for dl in a.deltas):
                    Kb = torch.tensor([m[4] for m in meta], device=fdev)                                # 원 평가와 같은 꼴(창: 배치, 회의 통째: 회의 전체 한 번)
                    Ab = fx.act(torch.nn.utils.rnn.pad_sequence([torch.from_numpy(w) for w in fxw], batch_first=True).to(fdev), torch.tensor([len(w) for w in fxw], device=fdev), Kb, fx_n)
                    FA = [Ab[j, : int(Kb[j])].cpu().numpy() for j in range(len(meta))]
                for delta in a.deltas:
                    if all((f"{cid}@{t0w}", delta) in done for _, cid, t0w, *_ in meta): continue
                    prefixes = [ds.cst_prefix(ds.slots(m[0])[1][:n_slots], delta) for m in meta]        # 선언 슬롯 수 = max(2, 화자 수) 또는 --declare, 모델 슬롯 수까지(2 화자 셋은 예전과 같다)
                    if a.from_predictions:
                        from types import SimpleNamespace
                        if not all((f"{m[1]}@{m[2]}", delta) in prev for m in meta): print(f"  !! 원출력 없음 δ{delta} — 건너뜀", flush=True); continue
                        states = [SimpleNamespace(emitted=[tuple(e) for e in prev[(f"{m[1]}@{m[2]}", delta)]["emitted"]], forced=prev[(f"{m[1]}@{m[2]}", delta)]["forced"],
                                                  flush_rounds=prev[(f"{m[1]}@{m[2]}", delta)]["flush_rounds"],
                                                  **({"gated": prev[(f"{m[1]}@{m[2]}", delta)]["gated"]} if "gated" in prev[(f"{m[1]}@{m[2]}", delta)] else {})) for m in meta]
                    elif a.oracle:
                        from types import SimpleNamespace
                        states = []
                        for m in meta:
                            if blocks:                                                                          # E9: 슬롯 블록 정답 시퀀스
                                q = ds.block_sequence(m[0], delta); em = emitted_from_blocks(q["ids"], q["is_input"], q["tag_chunk"], q["slot_at"], spk_p,
                                                                                             x_id=XI if XI >= 0 else None, chunk_at=q["chunk_at"])
                            else:
                                q = ds.cst_sequence(m[0], delta); em = emitted_from_sequence(q["ids"], q["is_input"], q["chunk_of"], sp["<NEXT_AUDIO>"], sp["<EMPTY_AUDIO>"])
                            states.append(SimpleNamespace(emitted=em, forced=0, flush_rounds=0))
                    elif full:
                        from types import SimpleNamespace
                        states = []
                        for j, m in enumerate(meta):
                            rt, rs = None, None
                            if getattr(a, "act_tag_guide", False):                                   # E7: 화자 태그를 활동(스트림 1)으로 이끈다
                                from vapasr.hf.long_session import make_retag
                                Na = int(model.adapter_slot.in_features); rs = {}; rt = make_retag(feats[j][1, :, :Na], spk[:Na], delta, stats=rs)
                            td = time.time(); bx = {}
                            if blocks:                                                                          # E9: 추측 prefill 긴 디코드
                                Aj = model.block_act(feats[j]); A_np[j] = None if Aj is None else (Aj.float() > 0.5).cpu().numpy(); AS[j] = None if Aj is None else Aj.float().cpu().numpy()
                                assert not mk_gate or Aj is not None, "--act-gate: 이 모델은 슬롯 활동이 없다(model.block_act)"
                                g = mk_gate(Aj, delta) if mk_gate else None
                                assert not rtw or Aj is not None, "--retag: 이 모델은 슬롯 활동이 없다(model.block_act)"
                                rg = BlockRetag(Aj, delta, *rtw) if rtw else None                               # 디코드 때 재귀속(2026-11-05)
                                em, fc, rd, used, nf, tk = long_block_decode(model, feats[j], prefixes[j], len(ds.slots(m[0])[1][:n_slots]), max_flush_rounds=a.max_flush, context_s=a.context_s,
                                                                             mix_off=mix_off, gate=g, gate_act=GA[j] if GA else None, retag=rg)
                                bx = dict(forwards=nf, chunk_ms=round(float(np.mean(tk)), 2), chunk_ms_p90=round(float(np.percentile(tk, 90)), 2), **({"gated": dict(g.blocked)} if g else {}),
                                          **({"dretag": dict(rg.stats)} if rg else {}))
                            else: em, fc, rd, used = long_decode(model, feats[j], prefixes[j], max_flush_rounds=a.max_flush, context_s=a.context_s, retag=rt)
                            states.append(SimpleNamespace(emitted=em, forced=fc, flush_rounds=rd, positions=used, dec_s=round(time.time() - td, 1), **({"retag": rs} if rs is not None else {}),
                                                          **({"blk": bx} if bx else {})))
                            print(f"  {c}: {m[1]} {m[3] / 60:.1f} 분 δ{delta} · 위치 {used} · 토큰 {len(em)} · 인코딩 {enc_s} s · 디코드 {states[-1].dec_s} s", flush=True)
                    else:
                        acts = [torch.tensor(slot_activity(ds.eps[m[1]], window_speakers(ds.eps[m[1]], m[2], m[3])[:n_slots], m[2], m[3], m[4], int(model.config.slot_input_slots)),
                                             dtype=torch.float32, device=dev) for m in meta] if act_or else None   # E5 정답 슬롯 활동
                        guide = SpkGuide(delay=delta, on=a.spk_guide[0], margin=a.spk_guide[1], span=int(a.spk_guide[2]) if len(a.spk_guide) > 2 else 2,
                                         hold=a.spk_guide_hold, insert=not a.spk_guide_no_insert, word_start=word_start) if a.spk_guide else None
                        states = [None] * len(meta)
                        A_dec = [model.block_act(feats[j], acts[j] if acts is not None else None) for j in range(len(meta))] if blocks else None   # 태그 활동(디코드가 받는 것)
                        if blocks: A_np = [None if x is None else (x.float() > 0.5).cpu().numpy() for x in A_dec]; AS = [None if x is None else x.float().cpu().numpy() for x in A_dec]
                        assert not mk_gate or all(x is not None for x in A_dec), "--act-gate: 이 모델은 슬롯 활동이 없다(model.block_act)"
                        modes = ["xl" if len(set(ds.slots(m[0])[1][:n_slots])) > 1 else "same" for m in meta] if sb else None   # 교차 언어 = 선언 언어가 다른 창
                        for pre in dict.fromkeys(tuple(p) for p in prefixes):                          # decode_batch 는 한 배치에 prefix 하나 — 선언 슬롯 수가 다른 창은 나눠 돈다
                            js = [j for j, p in enumerate(prefixes) if tuple(p) == pre]
                            if blocks:                                                                      # E9: 슬롯 블록 배치 디코드(선언 슬롯 수 = prefix 의 언어 수)
                                Dj = len(ds.slots(meta[js[0]][0])[1][:n_slots]); gs = [mk_gate(A_dec[j], delta) for j in js] if mk_gate else None
                                for q_, (j, st_) in enumerate(zip(js, decode_blocks_batch(model, [feats[j] for j in js], [list(pre)] * len(js), Dj, max_flush_rounds=a.max_flush,
                                                                          acts=[acts[j] for j in js] if acts is not None else None, mix_off=mix_off, gates=gs,
                                                                          gate_acts=[GA[j] for j in js] if GA else None))):
                                    states[j] = st_
                                    if gs: st_.gated = dict(gs[q_].blocked)
                                continue
                            for j, st_ in zip(js, decode_batch(model, [feats[j] for j in js], [list(pre)] * len(js), max_flush_rounds=a.max_flush, spk_guide=guide,
                                                               **(dict(switch_bias=sb, row_modes=[modes[j] for j in js], word_init=wi) if sb else {}),
                                                               **(dict(acts=[acts[j] for j in js]) if acts is not None else {}))): states[j] = st_
                    key = (ds.dlgs[meta[0][1]].lang, delta)
                    if a.verify and model is not None and key not in checked and not full:
                        for j in sorted({0, len(meta) - 1}):
                            if blocks:                                                                      # E9: 추측 prefill 직렬(LongThinker) 대 배치
                                from vapasr.hf.long_session import long_block_decode as _lbd
                                serial, forced, _, _, _, _ = _lbd(model, feats[j], prefixes[j], len(ds.slots(meta[j][0])[1][:n_slots]), max_flush_rounds=a.max_flush, blk=256, mix_off=mix_off,
                                                                  gate=mk_gate(A_dec[j], delta) if mk_gate else None)
                            else: serial, forced, _, rounds = model.stream_decode(feats[j], prefixes[j], max_flush_rounds=a.max_flush)
                            assert serial == states[j].emitted and forced == states[j].forced, f"직렬/배치 불일치 {meta[j][1]}@{meta[j][2]} δ{delta}"
                        checked.add(key); print(f"PARITY {key} PASS", flush=True)
                    for j, ((i, cid, t0w, L, K, d), s) in enumerate(zip(meta, states)):
                        if (f"{cid}@{t0w}", delta) in done: continue
                        ref_w, ref_eot, mask, sp_order = window_ref(ds, i, delta, EOT, tok)
                        ev = [] if blocks else None
                        hs, he = parse_emitted(s.emitted, spk_p, EOT, drop_chunks=mask, begin_id=BEGIN, events=ev)
                        xw = words(hs[-1]) if XI >= 0 else []; hs = hs[: len(spk)]                      # E14: X 칸 단어(아래에서 전문가 활동으로 슬롯에 붙인다)
                        n_xe = sum(1 for s_, _ in he if s_ >= len(spk)); he = [(s_, t) for s_, t in he if s_ < len(spk)]   # X 칸 <EOT>(학습에 없다 — 모델 실수)는 버리고 센다
                        hw = hw0 = [words(h) for h in hs]
                        llm_der = None
                        if blocks:                                                                      # E9 LLM 쪽 화자 분리(<BEGIN>·<EOT> 구간) — diar_eval 과 같은 프레임 마스크
                            Nd = max(len(spk), len(sp_order)); ha = torch.from_numpy(activity_from_events(ev, K, len(spk), delta).astype(np.float32))
                            ra = torch.tensor(_slot_act(ds.eps[cid], sp_order, t0w, L, K, n_slots=Nd), dtype=torch.float32).reshape(K, Nd)
                            fm = torch.ones(K, dtype=torch.bool)
                            for s_, e_ in ds._bad_spans(d):
                                x0, x1 = s_ - t0w, e_ - t0w
                                if x1 > 0 and x0 < L: fm[max(0, int(x0 / 0.08)): min(K, int(np.ceil(x1 / 0.08)))] = False
                            llm_der = der_counts(torch.nn.functional.pad(ha, (0, Nd - ha.shape[1])), ra, fm)
                        pr_ = prev.get((f"{cid}@{t0w}", delta), {})                                     # 활동 추종: 디코드 활동 > 다시 계산한 전문가 활동 > 원출력 act_runs
                        fa = A_np[j] if A_np[j] is not None else FA[j] if FA[j] is not None else act_from_runs(pr_["act_runs"], K) if blocks and "act_runs" in pr_ else None
                        nrl = None
                        if rlf:                                                                         # 출력 쪽 재귀속: 활동을 어긴 단어만 활동 슬롯으로(스트리밍 그대로)
                            assert blocks and fa is not None, f"--relabel-follow: 활동이 없다(슬롯 블록 디코드 행·act_runs 또는 --follow-n3d) {cid}@{t0w}"
                            fs_, hw = follow_stats(hw0, fa, delta, frame_mask=fm.numpy(), back=fwin[0], fwd=fwin[1]); nrl = fs_["off_other"]
                        if getattr(a, "relabel_act", None):                                             # 진단: 가설 단어를 활동으로 다시 귀속
                            from vapasr.data.cst_eval import relabel_by_activity
                            z = np.load(os.path.join(a.relabel_act, f"{cid}@full.npz")); A_ = z["ref" if a.relabel_oracle else "prob"].astype(np.float32)
                            assert A_.shape[0] == K, f"활동 프레임 {A_.shape[0]} ≠ 회의 프레임 {K}: {cid}"
                            hw = relabel_by_activity(hw, A_, delta)
                        xs = None
                        if xw:                                                                          # E14: X 단어 → 전문가 확률(임계 아래 포함)이 가장 큰 슬롯(원출력에 남긴 것이 있으면 그것)
                            xs = pr_["x_slots"] if len(pr_.get("x_slots") or ()) == len(xw) else attribute_x(xw, AS[j] if AS[j] is not None else fa, delta)
                        addx = (lambda h: merge_x(h, xw, xs)) if xw else (lambda h: h)
                        sc = score_window(ref_w, addx(hw), d.lang)
                        sc["eot"] = match_eot(ref_eot, he, sc["perm"], eot_tol(delta))
                        fx_row = {}
                        if blocks and fa is not None:                                                   # 활동 추종(LLM 원래 슬롯 — relabel 전)
                            perm0 = sc["perm"] if hw is hw0 and not xw else score_window(ref_w, hw0, d.lang)["perm"]   # 추종은 LLM 이 슬롯 칸에 쓴 단어만(X 빼고)
                            fdl = [] if getattr(a, "follow_detail", False) else None
                            fst, moved = follow_stats(hw0, fa, delta, ref_act=ra.numpy(), perm=perm0, frame_mask=fm.numpy(), back=fwin[0], fwd=fwin[1], detail=fdl)
                            fx_row = dict(follow=fst, cp_follow=score_window(ref_w, addx(moved), d.lang)["cp"], act_runs=act_runs(fa), **({"follow_detail": fdl} if fdl is not None else {}))
                        row = dict(id=f"{cid}@{t0w}", corpus=c, lang=d.lang, delta=delta, L=L, K=K, speakers=sp_order, n_spk=len(sp_order), masked_chunks=len(mask),
                                   forced=s.forced, flush_rounds=s.flush_rounds, emitted=[list(e) for e in s.emitted],
                                   **({"guide_insert": s.guide_insert, "guide_hold": s.guide_hold} if hasattr(s, "guide_insert") else {}),
                                   **({"sb_flips": s.sb_flips, "sb_points": s.sb_points} if hasattr(s, "sb_flips") else {}),
                                   **({"positions": s.positions, "enc_s": enc_s, "dec_s": s.dec_s} if hasattr(s, "positions") else {}),
                                   **({"retag": s.retag} if hasattr(s, "retag") else {}), **({"llm_der": llm_der} if llm_der is not None else {}), **(s.blk if hasattr(s, "blk") else {}),
                                   **({"gated": s.gated} if hasattr(s, "gated") else {}), **fx_row, **({"relabeled": nrl} if nrl is not None else {}),
                                   **({"n_x": len(xw), "x_slots": xs} if xw else {}), **({"n_x_eot": n_xe} if XI >= 0 and n_xe else {}),
                                   ref_text=[" ".join(w for w, _ in r) for r in ref_w], hyp_text=[" ".join(w for w, _ in h) for h in addx(hw)], **sc)
                        fo.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
                print(f"  {c}: {min(b0 + bs, len(todo))}/{len(todo)} 창 · {n} 행 · {time.time() - t0:.0f}s", flush=True)
    print(f"DONE rank={a.rank} rows={n}", flush=True)


def summarize(a):
    import glob
    from vapasr.data.cst_eval import pool, follow_pool, follow_line
    rows = [json.loads(l) for p in sorted(glob.glob(os.path.join(a.out, "predictions-rank*.jsonl"))) for l in open(p, encoding="utf-8")]
    res = {}
    for c in sorted({r["corpus"] for r in rows}):
        for d in sorted({r["delta"] for r in rows if r["corpus"] == c}):
            rr = [r for r in rows if r["corpus"] == c and r["delta"] == d]
            res[f"{c}/d{d}"] = dict(pool(rr), exposure=SEEN.get(c, "unseen"), unit="CER" if rr[0]["lang"] == "Korean" else "WER",
                                    one_speaker_windows=sum(r["n_spk"] == 1 for r in rr), forced_tokens=sum(r["forced"] for r in rr))
            dz = [r["llm_der"] for r in rr if "llm_der" in r]                                    # E9 LLM 쪽 화자 분리(<BEGIN>·<EOT>) — 프레임 × 화자 합
            if dz:
                n_ = max(1.0, sum(x["ref"] for x in dz)); m_, f_, c_ = (sum(x[k] for x in dz) for k in ("miss", "fa", "conf_best"))
                res[f"{c}/d{d}"].update(llm_der=round(100 * (m_ + f_ + c_) / n_, 2), llm_miss=round(100 * m_ / n_, 2), llm_fa=round(100 * f_ / n_, 2), llm_conf=round(100 * c_ / n_, 2))
            fp = follow_pool(rr)                                                                  # 활동 추종 진단(2026-10-30)
            if fp: res[f"{c}/d{d}"]["follow"] = fp
            if any("chunk_ms" in r for r in rr): res[f"{c}/d{d}"]["chunk_ms_mean"] = round(sum(r["chunk_ms"] * r["K"] for r in rr if "chunk_ms" in r) / max(1, sum(r["K"] for r in rr if "chunk_ms" in r)), 2)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    for k, v in res.items():
        print(f"{k:22s} {v['unit']} slot {v['err_slot']:6.2f} cp {v['err_cp']:6.2f} agn {v['err_agn']:6.2f} swap {v['swap_share']:.3f} · "
              f"EOT R {v['eot_recall']:.3f} P {v['eot_precision']:.3f} hold {v['eot_hold_fire']:.3f} lat {v['eot_lat_median']} · 창 {v['windows']} ({v['exposure']})"
              + (f" · LLM DER {v['llm_der']:.2f}(놓침 {v['llm_miss']:.2f} 헛 {v['llm_fa']:.2f} 혼동 {v['llm_conf']:.2f})" if "llm_der" in v else "")
              + (f" · 청크 {v['chunk_ms_mean']} ms" if "chunk_ms_mean" in v else "") + (follow_line(v["follow"]) if "follow" in v else ""))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run"); q.add_argument("--checkpoint", required=True); q.add_argument("--data", required=True); q.add_argument("--split", default=None, help="held-out split — 없으면 창 전부(합성 평가 셋)")
    q.add_argument("--out", required=True); q.add_argument("--corpora", default="aihub134-2,aihub134-1,aihub71631,otoSpeech,kec,ztc,nsc")
    q.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[4, 6, 10]); q.add_argument("--limit-windows", type=int, default=400)
    q.add_argument("--rank", type=int, default=0); q.add_argument("--world-size", type=int, default=1); q.add_argument("--batch-size", type=int, default=32)
    q.add_argument("--subset", choices=("dev", "eval"), default=None, help="AMI(--data 의 manifest.json): 공식 분할 하나의 회의만 — 판단용 빠른 평가(2026-11-05)")
    q.add_argument("--balance", action="store_true", help="몫을 창 길이 합이 고르게 나눈다(긴 것부터 가장 가벼운 몫에 — 회의 통째 몫 여럿, 2026-11-05). 끄면 j %% world_size")
    q.add_argument("--max-flush", type=int, default=None); q.add_argument("--dtype", default="float32"); q.add_argument("--encoder", default=None); q.add_argument("--verify", action="store_true")
    q.add_argument("--oracle", action="store_true", help="모델 없이 정답 시퀀스를 채점(지표 sanity)")
    q.add_argument("--mix-off", action="store_true", help="E10 슬롯 추출기 체크포인트: 섞인 소리 토큰을 빈 임베딩으로(슬롯 소리 토큰만으로 받아쓰기)")
    q.add_argument("--blocks", action="store_true", help="--oracle 을 E9 슬롯 블록 정답(block_sequence)으로 — 체크포인트 config 에 slot_blocks 가 없어도(학습 전 점검)")
    q.add_argument("--spk-guide", default=None, type=lambda s: [float(x) for x in s.split(",")], help="화자 헤드 안내 디코딩 'on,margin[,span]'(예 0.5,0.3)")
    q.add_argument("--spk-guide-hold", action="store_true", help="안내: 헤드가 현재 화자를 가리키면 모델의 태그 전환을 막는다(EOT 용 태그도 막혀 EOT 재현율이 줄 수 있다)")
    q.add_argument("--spk-guide-no-insert", action="store_true", help="안내: 태그 삽입을 끈다(--spk-guide-hold 와 함께 = 막기만)")
    q.add_argument("--from-predictions", default=None, help="이 폴더의 원출력을 다시 채점(디코드 없음)")
    q.add_argument("--en-norm", choices=["ours", "whisper"], default="ours", help="영어 채점 정규화 — whisper = Whisper EnglishTextNormalizer(채움말 제거 등, 외부 비교용)")
    q.add_argument("--act-tag-guide", action="store_true", help="E7: --window full 에서 화자 태그를 활동으로 이끈다(long_session.make_retag)")
    q.add_argument("--relabel-act", default=None, help="진단: --from-predictions·--window full 에서 가설 단어를 활동(diar_eval --save-probs 의 probs 폴더)으로 다시 귀속")
    q.add_argument("--relabel-oracle", action="store_true", help="--relabel-act 의 npz 에서 예측 대신 정답 활동(ref)으로")
    q.add_argument("--backchannel", default=None, type=lambda s: (float(s.split(",")[0]), int(s.split(",")[1])),
                   help="참조 EOT 를 맞장구 규칙(최대초,최대토큰)으로 — 그 규칙으로 학습한 모델(--backchannel)을 같은 정의로 잰다. 창·전사 참조는 같다")
    q.add_argument("--max-speakers", type=int, default=2, help="V2: 창 화자 상한(최대 4 — 슬롯 C·D 를 켠 체크포인트)")
    q.add_argument("--window", type=lambda s: s if s == "full" else tuple(float(x) for x in s.split(",")), default=(20.0, 40.0),
                   help="창 길이 lo,hi(s) — V1 20,40. full = 대화(회의)마다 창 하나, 자르지 않고 한 스트림으로(긴 디코드)")
    q.add_argument("--hop", type=float, default=10.0); q.add_argument("--window-draws", type=int, default=0, help="1 = 격자 대신 이어 덮기 창(draw 0) — 긴 창 평가")
    q.add_argument("--context-s", type=float, default=0.0, help="--window full: LLM 문맥 창(초, 0 = 문맥 전부) — 오디오는 자르지 않고 문맥만 prefix + 최근 S 초")
    q.add_argument("--target-oracle", action="store_true", help="E2: 대상 화자 모드 체크포인트를 정답 화자 활동 주입·화자별 디코드로 잰다")
    q.add_argument("--target-act", default=None, help="E3: 정답 대신 예측 활동(diar_eval --save-probs 의 probs 폴더)")
    q.add_argument("--act-oracle", action="store_true", help="E5: 슬롯 활동 입력 체크포인트에 정답 슬롯 활동을 주고 한 스트림으로 디코드")
    q.add_argument("--aosc", action="store_true", help="E7: --window full 에서 화자 분리 헤드 예측을 회의 전체 AOSC 스트리밍으로(diar.aosc_stream — config.diar_aosc 또는 기본)")
    q.add_argument("--aosc-chunk", type=int, default=None, help="--aosc 추론 청크(프레임, 기본 6)")
    q.add_argument("--diar-head-override", default=None, help="E7: 모델의 화자 분리 헤드를 이 diar.pt(diar_train --aosc final)로 바꾼다 — aosc 설정도 따라온다")
    q.add_argument("--n3d", default=None, help="E11(2026-10-29): 고정 화자 분리 전문가 Nemotron 3 Diarization(.nemo)을 붙여 슬롯 활동을 그것으로(vapasr/hf/n3d — 창은 창마다, "
                                               "--window full 은 회의 전체 스트리밍). 슬롯 활동 입력(E6·E9·E10) 체크포인트에서 헤드 활동 대신 — E11 체크포인트는 config 의 경로가 기본")
    q.add_argument("--n3d-chunk", type=int, default=None, help="--n3d 추론 청크(80 ms 프레임, 기본 4 = 인코더 블록)"); q.add_argument("--n3d-rc", type=int, default=None, help="--n3d 오른쪽 문맥(기본 0)")
    q.add_argument("--n3d-threshold", type=float, default=None, help="--n3d 활동 임계(기본 0.5)")
    ints = lambda s: tuple(int(x) for x in s.split(","))
    q.add_argument("--act-gate", default=None, type=ints, help="활동 게이트 디코드 BACK,FWD(예 3,1 — batch_decode.ActGate): 슬롯 첫 출력의 오디오 근처에 그 슬롯 활동이 없으면 그 라운드 그 슬롯을 버린다")
    q.add_argument("--follow-window", default=(3, 1), type=ints, help="활동 추종 진단 창 BACK,FWD(단어 오디오 프레임 기준, 기본 3,1 — cst_eval.follow_stats)")
    q.add_argument("--follow-n3d", action="store_true", help="--from-predictions: 원출력에 act_runs 가 없으면 원 평가 config 의 전문가(n3d)로 활동을 다시 계산(GPU)")
    q.add_argument("--encoder-state", default=None, help="U3: 인코더를 이 graft.pt 의 enc(화자 분리와 함께 학습)로 바꿔 디코드 — ASR 지킴 확인")
    q.add_argument("--graft-expert", default=None, help="G3: 고정 전문가 자리에 이 graft.pt 의 이식 학생(GraftExpert)을 끼워 그 활동으로 디코드")
    q.add_argument("--encoder-rc", type=int, default=None, choices=(0, 3, 6, 13), help="지연 완화: 인코더 우측 문맥(80 ms 프레임 — 모델 설정의 att_context_size 목록, 학습은 3)")
    q.add_argument("--gate-lags", type=int, default=None, help="E13 게이트 진단(2026-11-03): 추론 활동 게이트 창(청크 — config.slot_gate 를 덮는다, 학습 12). 0 = 게이트 끔(선언 슬롯 전부)")
    q.add_argument("--gate-thresh", type=float, default=None, help="E13 게이트 진단: 게이트만 전문가 확률 > 이 값으로 연다(태그·오디오 활동은 전문가 임계 0.5 그대로)")
    q.add_argument("--slot-enc-off", action="store_true", help="E15b 진단(2026-11-06): 슬롯 소리 토큰을 0 으로(섞인 소리 토큰·인코더는 그대로) — 슬롯 경로를 썼는지")
    q.add_argument("--retag", default=None, type=ints, help="디코드 때 재귀속 BACK,FWD(예 3,1 — batch_decode.BlockRetag, 회의 통째 슬롯 블록 디코드): 라운드마다 활동을 어긴 칸을 옮기고 라운드를 고쳐 다시 넣는다")
    q.add_argument("--relabel-follow", action="store_true", help="출력 쪽 재귀속: 활동을 어긴 가설 단어만 활동 슬롯으로 옮겨 채점(창 --follow-window, 스트리밍 그대로 낼 수 있는 출력)")
    q.add_argument("--follow-detail", action="store_true", help="행에 follow_detail(단어마다 추종 범주·슬롯·말 차례 위치·직전 화자 — cst_eval.follow_stats detail)을 남긴다")
    q.add_argument("--target-act-threshold", type=float, default=0.5); q.add_argument("--target-act-min-frames", type=int, default=3, help="예측 슬롯을 쓰는 최소 활성 프레임(80 ms)")
    q.add_argument("--declare", type=int, default=None, help="prefix 선언 슬롯 수 하한(기본 max(2, 화자 수); 화자 수를 모르는 추론 흉내 = 4)")
    q.add_argument("--switch-bias", default=None, help="V3 C3 상태 인지 교대 bias 'eot=B,res=B,wi=B'(batch_decode.SwitchBias — 선언 언어가 다른 창만, 기본 끔). "
                                                       "직렬 stream_decode 에 bias 가 없어 0 이 아니면 --verify 와 함께 쓰지 않는다. config 에 정책, 행에 sb_flips·sb_points")
    q = sub.add_parser("summarize"); q.add_argument("--out", required=True)
    a = ap.parse_args(); {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
