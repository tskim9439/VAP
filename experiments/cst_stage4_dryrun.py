#!/usr/bin/env python
"""Stage 4 S2 dry-run — 같은 언어 대화 창을 번역 모드(네 흐름: 두 화자 전사 + 번역, vapasr/data/cst_dialogue.py build_cst4_sequence)로 직렬화해
검사·집계한다(decision-stage4-prep-20261019). GPU 불필요.

  python experiments/cst_stage4_dryrun.py --data <phase2 dir> --tokenizer <Stage 2 체크포인트(final)> --split <phase2>/splits/v2.json \\
      --trans <targets dir>[,<dir>] [--trans-input <dir>] [--revive-dir <dir>] --corpora ... --out <json> [--only-translated] [--show N]
--trans: ast_build_targets 산출(행 id '<conv>|<화자>|<utt>', stage4_dialogue_units 단위), --trans-input: 되살린 발화의 번역(입력만), --revive-dir: 되살리기 재료.
코퍼스마다(프로세스 하나): held-out 을 뺀 창(Stage 3 와 같은 창)마다 목표 언어(tgt_of — 한·독 → 영, 영 → 독·한)별로 δ 하나(무작위)를 골라
problems_of(네 흐름 다시 읽기·태그 규칙·입력만 label·soft 결정·EOT 수)를 검사하고, 같은 창·δ 의 Stage 3 시퀀스와 길이를 비교한다.
집계: 시퀀스 길이(Stage 4/3 비), est_lens 과소 추정, 창당 번역 토큰·태그, 번역 없는 행(miss: 판정 탈락, cut: 창 시작 전에 시작한 행), <EOT> 흐름(번역 꼬리 뒤/전사),
label 마스크 비율, flush 로 밀린 번역 토큰. --only-translated: 번역 행이 있는 대화의 창만(시험 생성처럼 일부 대화만 번역했을 때).
--show N: 코퍼스마다 N 창을 디코드해 표준 출력에만 쓴다(전사·번역 문장 — 파일로 남기지 않는다). 출력 --out 에는 집계만."""
import os, sys, json, time, random, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--tokenizer", required=True); ap.add_argument("--split")
ap.add_argument("--trans", required=True); ap.add_argument("--trans-input", default=""); ap.add_argument("--revive-dir", default="")
ap.add_argument("--corpora", default="aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc"); ap.add_argument("--delays", default="4,5,6,7,8,9,10")
ap.add_argument("--backchannel", default="1.0,3"); ap.add_argument("--out", required=True); ap.add_argument("--audio-sample", type=int, default=10)
ap.add_argument("--procs", type=int, default=7); ap.add_argument("--max-windows", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--only-translated", action="store_true"); ap.add_argument("--show", type=int, default=0)
a = ap.parse_args()
DELAYS = tuple(int(x) for x in a.delays.split(","))


def pct(v, ps=(50, 90, 99, 100)): return {f"p{p}": (round(float(np.percentile(v, p)), 3) if len(v) else None) for p in ps}


def show(ds, tok, s):
    """번역 모드 시퀀스 하나를 사람이 읽게: 청크 경계 '|', 특수 토큰 이름 그대로(표준 출력 전용)."""
    names = {ds.csp.next_audio: "|", ds.eot_id: "<EOT>", ds.trans_id: "<TRANS>", ds.spk_ids[0]: "<SPK_A>", ds.spk_ids[1]: "<SPK_B>", ds.csp.empty_audio: "<EMPTY>"}
    out, buf = [], []
    for t, inp, lab in zip(s["ids"], s["is_input"], s["labels"]):
        if inp: continue
        if t in names:
            if buf: out.append(tok.decode(buf)); buf = []
            if t != ds.csp.next_audio or (out and out[-1] != "|"): out.append(names[t] + ("" if lab != -100 or t == ds.csp.next_audio else "~"))
        else: buf.append(t)
    if buf: out.append(tok.decode(buf))
    return " ".join(out)


def run(corpus):
    from transformers import AutoTokenizer
    from vapasr.data.cst_dialogue import CstDialogueDataset, cst_head_text, load_revive
    from vapasr.data.ast_dataset import load_ast_targets
    t0 = time.time(); tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tr, tr_st, _ = load_ast_targets([p for p in a.trans.split(",") if p])
    tri, tri_st, _ = load_ast_targets([p for p in a.trans_input.split(",") if p]) if a.trans_input else ({}, collections.Counter(), "")
    rv_path = os.path.join(a.revive_dir, f"{corpus}.revive.jsonl") if a.revive_dir else ""
    rv = load_revive([rv_path]) if rv_path and os.path.exists(rv_path) else None
    held = set(json.load(open(a.split, encoding="utf-8"))["corpora"].get(corpus, {}).get("heldout", [])) if a.split else set()
    bc = tuple(float(x) if i == 0 else int(x) for i, x in enumerate(a.backchannel.split(","))) if a.backchannel else None
    ds = CstDialogueDataset([os.path.join(a.data, f"{corpus}.refined.dialogues.jsonl")], tok, delays=DELAYS, exclude=held, mono_cache_dir=os.path.join(a.data, "mono"),
                            seed=a.seed, backchannel=bc, revive=rv, trans=tr, trans_input=tri, ast_frac=1.0)
    convs = {r.split("|")[0] for r in tr} | {r.split("|")[0] for r in tri}
    if a.only_translated: ds.items = [it for it in ds.items if it[0] in convs]
    rep = dict(corpus=corpus, held_out_dialogues=len(held), stats=dict(ds.stats), trans_rows=sum(1 for r in tr if r.split("|")[0] in ds.dlgs),
               trans_input_rows=sum(1 for r in tri if r.split("|")[0] in ds.dlgs), trans_load=dict(tr_st), windows=len(ds), sec_init=round(time.time() - t0, 1))
    lang = next((ds.dlgs[c].lang for c, _, _ in ds.items), "Korean"); tgts = list(ds.tgt_of.get(lang, ()))
    rep["prefix"] = {}
    for tg in tgts:
        pre = ds.cst_prefix(lang, DELAYS[0], tg); body = pre[len(ds._pre):-1]; text = tok.decode(body)
        rep["prefix"][tg] = dict(text=text, n_tokens=len(pre), ok=text == cst_head_text([lang, lang], [tg, tg]) and body.count(ds.spk_ids[0]) == 2 and body.count(ds.spk_ids[1]) == 2)
    rng = random.Random(a.seed); idx = list(range(len(ds)))
    if a.max_windows and len(idx) > a.max_windows: idx = sorted(rng.sample(idx, a.max_windows))
    est = ds.est_lens(); agg = {tg: collections.defaultdict(list) for tg in tgts}; win = {tg: collections.Counter() for tg in tgts}
    probs = collections.Counter(); ex = []; nbad = 0; shown = 0; t1 = time.time()
    for n, i in enumerate(idx):
        for tg in tgts:
            d = rng.choice(DELAYS); s = ds.cst_sequence(i, d, tgt=tg); s3 = ds.cst_sequence(i, d); P = ds.problems_of(s); g = agg[tg]
            for p in P: probs[p.split(":")[0]] += 1
            nbad += bool(P)
            if P and len(ex) < 10: ex.append(dict(item=list(ds.items[i]), tgt=tg, delay=d, problems=P[:5]))
            out = [j for j, inp in enumerate(s["is_input"]) if not inp]; lab = s["labels"]
            g["len"].append(len(s["ids"])); g["ratio"].append(len(s["ids"]) / max(1, len(s3["ids"]))); g["est_ratio"].append(len(s["ids"]) / max(1, int(est[i])))
            g["n_tr"].append(s["n_tr"]); g["trans_tags"].append(sum(1 for j in out if s["ids"][j] == ds.trans_id))
            g["spk_tags"].append(sum(1 for j in out if s["ids"][j] in ds.spk_ids)); g["spk_tags3"].append(sum(1 for j, inp in enumerate(s3["is_input"]) if not inp and s3["ids"][j] in ds.spk_ids))
            g["mask"].append(sum(1 for j in out if lab[j] == -100) / max(1, len(out))); g["mask3"].append(sum(1 for j, inp in enumerate(s3["is_input"]) if not inp and s3["labels"][j] == -100) / max(1, sum(1 for x in s3["is_input"] if not x)))
            g["tr_flush"].append(sum(1 for j in out if s["chunk_at"][j] == s["K"] and s["src_at"][j] is not None and s["src_at"][j][0] >= 2 and len(s["streams"][s["src_at"][j][0]][s["src_at"][j][1]]) == 2))
            g["eot_in_tr"].append(sum(1 for p in s["eot_pos"] if s["src_at"][p][0] >= 2)); g["soft"].append(s["n_soft"]); g["soft3"].append(s3["n_soft"])
            win[tg].update(s["win"])
            if shown < a.show and s["n_tr"] and s["win"].get("miss_miss", 0) == 0:
                print(f"\n=== {corpus} {ds.items[i]} → {tg} δ{d} (~ = label −100)\n{show(ds, tok, s)}", flush=True); shown += 1
        if n and n % 5000 == 0: print(f"  {corpus}: {n}/{len(idx)} {time.time() - t1:.0f}s", flush=True)
    rep.update(windows_checked=len(idx), sequences_with_problems=nbad, problems=dict(probs), problem_examples=ex, sec_check=round(time.time() - t1, 1))
    rep["by_tgt"] = {}
    for tg in tgts:
        g = agg[tg]; w = win[tg]; rows_in = w.get("miss_miss", 0) + w.get("miss_cut", 0)
        rep["by_tgt"][tg] = dict(seq_len=pct(g["len"]), over_16384=int(sum(1 for x in g["len"] if x > 16384)), len_ratio_vs_stage3=pct(g["ratio"], (50, 90, 100)),
                                 est_under=int(sum(1 for x in g["est_ratio"] if x > 1.0)), est_ratio=pct(g["est_ratio"], (50, 90, 100)),
                                 tr_tokens=pct(g["n_tr"], (10, 50, 90)), trans_tags_per_window=round(float(np.mean(g["trans_tags"])), 2) if g["trans_tags"] else None,
                                 spk_tags_per_window=round(float(np.mean(g["spk_tags"])), 2) if g["spk_tags"] else None,
                                 spk_tags_per_window_stage3=round(float(np.mean(g["spk_tags3"])), 2) if g["spk_tags3"] else None,
                                 masked_label_frac=pct(g["mask"], (50, 90)), masked_label_frac_stage3=pct(g["mask3"], (50, 90)), tr_in_flush=int(sum(g["tr_flush"])),
                                 eot_total=int(w.get("eot", 0)), eot_after_translation=int(sum(g["eot_in_tr"])), soft_total=int(sum(g["soft"])), soft_total_stage3=int(sum(g["soft3"])),
                                 rows_without_translation=dict(miss=int(w.get("miss_miss", 0)), cut=int(w.get("miss_cut", 0)), total=rows_in),
                                 window_stats=dict(w))
    aud = []                                                                                    # 실제 학습 경로(__getitem__, ast_frac 1 → 번역 모드)
    for i in rng.sample(range(len(ds)), min(a.audio_sample, len(ds))):
        x = ds[i]; n = int(x["wav"].shape[0]); K = int(x["K"])
        aud.append(dict(id=x["id"], samples=n, K=K, len_ok=abs(n - K * 1280) <= 1280, n_tr=x["n_tr"]))
    rep["audio_sample"] = dict(n=len(aud), len_ok=all(z["len_ok"] for z in aud), n_tr=[z["n_tr"] for z in aud])
    rep["sec"] = round(time.time() - t0, 1)
    print(json.dumps({k: rep[k] for k in ("corpus", "windows_checked", "problems", "sec")}, ensure_ascii=False), flush=True)
    return rep


if __name__ == "__main__":
    import multiprocessing as mp
    corpora = a.corpora.split(",")
    with mp.get_context("spawn").Pool(min(a.procs, len(corpora))) as pool: reps = pool.map(run, corpora)
    res = dict(args=vars(a), corpora={r["corpus"]: r for r in reps},
               all_ok=all(not r["problems"] and all(p["ok"] for p in r["prefix"].values()) and r["audio_sample"]["len_ok"] for r in reps))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("all_ok", res["all_ok"], "→", a.out)
