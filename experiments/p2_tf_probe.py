#!/usr/bin/env python
"""Phase 2 teacher-forced 진단 — 평가 창(p2_eval_lanes 의 report.json 창 목록)에 참조 시퀀스를 그대로 넣고 (1) 종류별(텍스트·태그·ONSET·EOT·NEXT) top-1 정확도, (2) audio 위치에서의 태그 확률
p(첫 화자 태그 | 청크 k) 을 청크 1·참조 첫 ONSET 청크·그 사이 최대로 잰다. free-running 결과와 대조해 "인식이 약한가, 구조 결정이 약한가" 를 가른다.
  python experiments/p2_tf_probe.py --model <p2-D1/final> --data <phase2 dir> --from-eval <eval dir>[,<eval dir>] --out <json> [--mono-cache <dir>]"""
import os, sys, json, time, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--from-eval", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--delay", type=int, default=4); ap.add_argument("--delay-onset", type=int, default=0); ap.add_argument("--R", type=int, default=6); ap.add_argument("--mono-cache", default=None); ap.add_argument("--gpu", default=None); ap.add_argument("--window", type=float, nargs=2, default=(20.0, 40.0)); ap.add_argument("--hop", type=float, default=10.0)
a = ap.parse_args()
if a.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import numpy as np, torch
from vapasr.hf import VapAsrForStreamingASR, load_tokenizer
from vapasr.data.dialogue import Dialogue, CHUNK_S
from vapasr.data.dialogue_dataset import DialogueWindowDataset, collate_dialogue
def log(*s): print(*s, flush=True)
model = VapAsrForStreamingASR.from_pretrained(a.model); tok = load_tokenizer(a.model); model.cuda().eval(); reg = dict(model.config.phase2_registry)
lane_ids = {reg[n] for n in ("<SPK_A>", "<SPK_B>", "<SPK_3>", "<SPK_4>", "<SPK_5>", "<SPK_6>") if n in reg}; ONSET, EOT, NEXT = reg["<ONSET>"], reg["<EOT>"], model.next_audio; SPK_A = reg["<SPK_A>"]

def proxy_tokens(d):
    n = 0
    for u in d.utterances:
        if not u.text or u.tokens: continue
        enc = tok(" " + u.text, add_special_tokens=False)["input_ids"]
        if enc: u.tokens = [(t, u.start + (u.end - u.start) * (i + 1) / len(enc)) for i, t in enumerate(enc)]; n += 1
    return n
res = {}
for ev in a.from_eval.split(","):
    rep = json.load(open(os.path.join(ev, "report.json")))
    for corpus, s in rep["sets"].items():
        refined = os.path.join(a.data, f"{corpus}.refined.dialogues.jsonl"); plain = os.path.join(a.data, f"{corpus}.dialogues.jsonl"); path = refined if os.path.exists(refined) else plain
        dl = [Dialogue.from_json(l) for l in open(path, encoding="utf-8") if l.strip()]
        if sum(1 for d in dl for u in d.utterances if u.tokens) == 0:
            for d in dl: proxy_tokens(d)
            path = os.path.join(a.out + ".tmp", f"_{corpus}.proxy.jsonl"); os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for d in dl: f.write(d.to_json() + "\n")
        ds = DialogueWindowDataset([path], tok, R=a.R, window_s=tuple(a.window), hop_s=a.hop, delays=(a.delay,), delay_onset=a.delay_onset, seed=rep["args"]["seed"], mono_cache_dir=a.mono_cache)
        idx = {(it[0], round(it[1], 3), round(it[2], 3)): i for i, it in enumerate(ds.items)}
        rows = []; kinds_acc = collections.defaultdict(lambda: [0, 0])
        for w in s["windows"]:
            i = idx.get((w["conv"], round(w["t0"], 3), round(w["L"], 3)))
            if i is None: log(f"  ! {w['name']} 창 없음"); continue
            seq = ds.sequence(i, a.delay); x = collate_dialogue([ds[i]]); x = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in x.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                feats = model.encode(x["wav"], x["wav_len"], x["K"]); h = model.thinker.model(inputs_embeds=model.build(feats, x["ids"], x["is_audio"], x["chunk_of"]), attention_mask=x["mask"]).last_hidden_state[0]
                logits = model.thinker.lm_head(h[:-1]).float()                    # 위치 j 의 logits 가 ids[j+1] 을 예측
            ids = seq["ids"]; kinds = seq["kinds"]; labels = seq["labels"]; pred = logits.argmax(-1).tolist(); probs = torch.softmax(logits, -1)
            kacc = collections.defaultdict(lambda: [0, 0])
            for j in range(len(ids) - 1):
                if labels[j + 1] == -100: continue
                kd = kinds[j + 1]; kd = "tag" if kd == "tag" else kd; kacc[kd][0] += int(pred[j] == ids[j + 1]); kacc[kd][1] += 1
            for kd, (c, n) in kacc.items(): kinds_acc[kd][0] += c; kinds_acc[kd][1] += n
            # audio 위치별 태그 확률(모든 lane 태그 합)과 <SPK_A> 확률
            audio_pos = [j for j, k in enumerate(kinds) if k == "audio"]; chunk_of = seq["chunk_of"]
            p_tag = {chunk_of[j]: float(sum(probs[j, t] for t in lane_ids)) for j in audio_pos}; p_a = {chunk_of[j]: float(probs[j, SPK_A]) for j in audio_pos}
            eps, _ = ds.window_episodes(ds.dlgs[seq["cid"]], seq["t0"], seq["L"]); first_on = None
            for k, em in __import__("vapasr.data.dialogue_interleave", fromlist=["serialize"]).serialize([e for e in eps if e.lane], (seq["K"] - 0.5) * CHUNK_S, ds.sp, delay_text=a.delay, delay_onset=a.delay_onset)[0]:
                if any(e.kind == "onset" for e in em): first_on = k; break
            before = [p_tag[k] for k in range(0, first_on) if k in p_tag] if first_on else []
            rows.append(dict(name=w["name"], K=seq["K"], first_onset=first_on, p_tag_k1=round(p_tag.get(1, 0.0), 3), p_spkA_k1=round(p_a.get(1, 0.0), 3), p_tag_at_onset=(round(p_tag.get(first_on, 0.0), 3) if first_on is not None else None),
                             p_tag_max_before=(round(max(before), 3) if before else None), p_tag_mean_before=(round(float(np.mean(before)), 3) if before else None), acc={k: round(c / n, 3) for k, (c, n) in kacc.items() if n}))
            log(f"  {w['name']} K{seq['K']} first_onset {first_on} p_tag@1 {rows[-1]['p_tag_k1']} p_tag@onset {rows[-1]['p_tag_at_onset']} max_before {rows[-1]['p_tag_max_before']} acc {rows[-1]['acc']}")
        res[corpus] = dict(tag=s["tag"], acc={k: dict(acc=round(c / n, 4), n=n) for k, (c, n) in kinds_acc.items() if n}, windows=rows)
        log(f"[{corpus}] teacher-forced acc {res[corpus]['acc']}")
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1); log("→", a.out)
