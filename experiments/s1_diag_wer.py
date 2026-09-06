#!/usr/bin/env python
"""Stage 1 WER 진단 — 교사 강제 정확도 vs 자유 실행 오류(S/D/I), 예시. 원인 분해: 과소학습·용량 vs 노출 편향 vs 방출 결정.
python experiments/s1_diag_wer.py --ckpt <ckpt.pt> [--n 20] [--sets dev-clean,kspon-dev] [--gpu K]
"""
import os, sys, json, argparse, subprocess, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=20); ap.add_argument("--sets", default="dev-clean,dev-other,kspon-dev")
ap.add_argument("--delay", type=int, default=2); ap.add_argument("--gpu", default=None); ap.add_argument("--M", type=int, default=4); a = ap.parse_args()
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    if a.gpu is None:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        a.gpu = str(max([[int(v) for v in l.split(",")] for l in q.stdout.strip().splitlines()], key=lambda r: r[2] - r[1])[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import numpy as np, torch, torch.nn as nn, jiwer
from vapasr.data.textnorm import score_en, score_ko
from vapasr.uslm.mono_data import MonoStreamDataset, CHUNK_S
from vapasr.uslm.mono_model import MonoInterleavedASR
from vapasr.uslm.model import Adapter
from qwen_asr import Qwen3ASRModel
dev = "cuda"; QWEN = os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B")
qm = Qwen3ASRModel.from_pretrained(QWEN, dtype=torch.bfloat16, device_map=dev, max_new_tokens=8)
root = next(v for v in vars(qm).values() if isinstance(v, nn.Module)); thinker = root.thinker; tok = qm.processor.tokenizer; del root.thinker.audio_tower
SPEC = {"dev-clean": ("librispeech-dev", "dev-clean", "stream"), "dev-other": ("librispeech-dev", "dev-other", "stream"), "kspon-dev": ("kspon-dev", "dev", "utt")}
sets = {k: MonoStreamDataset([SPEC[k][0]], tok, mode=SPEC[k][2], subsets=[SPEC[k][1]], delays=(a.delay,), max_per_chunk=a.M, seed=1, max_items=a.n) for k in a.sets.split(",")}
sp_ids = next(iter(sets.values())).sp_ids
model = MonoInterleavedASR(thinker, tok, Adapter(), sp_ids, lora_r=16).to(dev); model.adapter.float()
model.load_trainable_state(torch.load(a.ckpt, map_location="cpu")); model.eval(); model.thinker.merge_adapter()
NEXT = sp_ids["<NEXT_AUDIO>"]; out = {}
for name, ds in sets.items():
    lang = ds.items[0]["lang"]; tf_hit = tf_tot = 0; next_hit = next_tot = 0; ranks = []; R, H, ex = [], [], []
    for i in range(len(ds)):
        f, ids, is_input, chunk_of, st, it, _, _ = ds.sequence(i, a.delay)
        feats = torch.from_numpy(f)[None].to(dev); ids_t = torch.tensor(ids)[None].to(dev); ia = torch.tensor([c >= 0 for c in chunk_of])[None].to(dev); co = torch.tensor(chunk_of)[None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            E = model.build(feats, ids_t, ia, co); logits = model.thinker(inputs_embeds=E).logits[0, :-1].float()
        tgt = torch.tensor(ids[1:], device=dev); inp = torch.tensor(is_input[1:], device=dev)
        text_pos = (~inp) & (tgt != NEXT); next_pos = (~inp) & (tgt == NEXT)
        pred = logits.argmax(-1); tf_hit += int((pred[text_pos] == tgt[text_pos]).sum()); tf_tot += int(text_pos.sum()); next_hit += int((pred[next_pos] == NEXT).sum()); next_tot += int(next_pos.sum())
        # 텍스트 위치에서 정답의 순위(top-5 안인가) 와 '정답 대신 NEXT 를 고른' 비율
        r = (logits[text_pos] > logits[text_pos].gather(1, tgt[text_pos][:, None])).sum(1); ranks += r.tolist()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): emitted, _, _, _ = model.stream_decode(feats[0], ds.prefix(lang, a.delay), a.M, next_bias=0.0)
        ref = it["tokens"]; R.append(tok.decode([t for t, _ in ref])); H.append(tok.decode([t for _, t in emitted]))
        if len(ex) < 3: ex.append((R[-1][:90], H[-1][:90]))
    norm = (lambda x: score_ko(x, False)) if lang == "Korean" else score_en
    Rn, Hn = [norm(x) for x in R], [norm(x) for x in H]
    if lang == "Korean": Rn, Hn = [" ".join(x) for x in Rn], [" ".join(x) for x in Hn]      # 문자 단위 S/D/I
    m = jiwer.process_words(Rn, Hn); ranks = np.array(ranks)
    out[name] = dict(n=len(ds), tf_text_top1=tf_hit / max(1, tf_tot), tf_text_top5=float((ranks < 5).mean()), tf_next_acc=next_hit / max(1, next_tot),
                     free_err=m.wer, sub=m.substitutions, dele=m.deletions, ins=m.insertions, hits=m.hits, ref_tokens=m.hits + m.substitutions + m.deletions, examples=ex)
    o = out[name]; tot = o["ref_tokens"]
    print(f"[{name}] n={o['n']} | 교사강제 텍스트 top-1 {o['tf_text_top1']:.3f} top-5 {o['tf_text_top5']:.3f} · NEXT 정확도 {o['tf_next_acc']:.3f} | 자유실행 오류 {o['free_err']:.3f} = S {o['sub']/tot:.3f} + D {o['dele']/tot:.3f} + I {o['ins']/tot:.3f}", flush=True)
    for r_, h_ in ex: print(f"    ref: {r_}\n    hyp: {h_}", flush=True)
json.dump(out, open(os.path.splitext(a.ckpt)[0] + "-diag.json", "w"), ensure_ascii=False, indent=1)
