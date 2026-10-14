#!/usr/bin/env python
"""Stage 3 P3 dry-run — 2 인 대화 코퍼스 전체 창을 Stage 3 형식(vapasr/data/cst_dialogue.py)으로 직렬화해 검사·집계한다. GPU 불필요.

  python experiments/cst_stage3_dryrun.py --data <phase2 dir> --tokenizer <Stage 2 체크포인트(final)> --split <phase2>/splits/v2.json \\
      --corpora aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc --delays 4,5,6,7,8,9,10 --out <json> [--audio-sample 20] [--procs 7]
코퍼스마다(프로세스 하나): split held-out 대화를 빼고 창(20–40 s, hop 10 s, 무음 시작)을 만든 뒤, 창마다 δ 하나(무작위)로 시퀀스를 만들어
problems_of(태그 규칙·슬롯별 토큰 순서·soft 결정 위치·EOT 수)를 검사한다. 집계: 창 수·3 화자 이상 제외·화자 수, 시퀀스 길이 분위,
창당 텍스트·EOT·soft·태그 결정, p_end 분포, 마스크 비율, flush 토큰, EOT 생략 사유. --audio-sample N: 창 N 개를 실제로 꺼내(mono 캐시·crop)
파형 길이·K·RMS 를 본다. prefix 는 tokenizer 로 되읽어 문자열과 특수 토큰 id 를 대조한다. 출력은 --out 하나(읽기 전용 입력)."""
import os, sys, json, time, random, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--tokenizer", required=True); ap.add_argument("--split")
ap.add_argument("--corpora", default="aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc"); ap.add_argument("--delays", default="4,5,6,7,8,9,10")
ap.add_argument("--out", required=True); ap.add_argument("--audio-sample", type=int, default=20); ap.add_argument("--procs", type=int, default=7)
ap.add_argument("--max-windows", type=int, default=0, help="코퍼스당 상한(0 = 전부, 스모크용)"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
DELAYS = tuple(int(x) for x in a.delays.split(","))


def pct(v, ps=(50, 90, 99, 100)): return {f"p{p}": (round(float(np.percentile(v, p)), 3) if len(v) else None) for p in ps}


def run(corpus):
    from transformers import AutoTokenizer
    from vapasr.data.cst_dialogue import CstDialogueDataset, cst_head_text, SLOTS
    t0 = time.time(); tok = AutoTokenizer.from_pretrained(a.tokenizer)
    held = set(json.load(open(a.split, encoding="utf-8"))["corpora"].get(corpus, {}).get("heldout", [])) if a.split else set()
    path = os.path.join(a.data, f"{corpus}.refined.dialogues.jsonl")
    ds = CstDialogueDataset([path], tok, delays=DELAYS, exclude=held, mono_cache_dir=os.path.join(a.data, "mono"), seed=a.seed)
    rep = dict(corpus=corpus, held_out_dialogues=len(held), stats=dict(ds.stats), sec_init=round(time.time() - t0, 1))
    # prefix: 문자열 되읽기 + 슬롯 태그가 특수 토큰 한 개씩인지
    lang = next(iter(ds.dlgs.values())).lang if ds.dlgs else "Korean"; pre = ds.cst_prefix(lang, DELAYS[0])
    body = pre[len(ds._pre):-1]; text = tok.decode(body)
    rep["prefix"] = dict(text=text, ok=text == cst_head_text([lang, lang]) and body.count(ds.spk_ids[0]) == 1 and body.count(ds.spk_ids[1]) == 1, n_tokens=len(pre),
                         slot_ids={SLOTS[0]: ds.spk_ids[0], SLOTS[1]: ds.spk_ids[1]}, eot_id=ds.eot_id)
    # 대화 단위 EOT 결과 분포
    out = collections.Counter(ep.outcome for eps in ds.eps.values() for ep in eps); rep["episode_outcomes"] = dict(out)
    rng = random.Random(a.seed); idx = list(range(len(ds)))
    if a.max_windows and len(idx) > a.max_windows: idx = sorted(rng.sample(idx, a.max_windows))
    L, ntext, neot, nsoft, ntagdec, nflush, mask_frac, nspk = [], [], [], [], [], [], [], collections.Counter()
    pend = collections.Counter(); win = collections.Counter(); probs = collections.Counter(); ex = []; nbad = 0; t1 = time.time()
    for n, i in enumerate(idx):
        d = rng.choice(DELAYS); s = ds.cst_sequence(i, d); P = ds.problems_of(s)
        for p in P: probs[p.split(":")[0]] += 1
        nbad += bool(P)
        if P and len(ex) < 10: ex.append(dict(item=list(ds.items[i]), delay=d, problems=P[:5]))
        L.append(len(s["ids"])); ntext.append(s["st"].tokens); neot.append(s["n_eot"]); nsoft.append(s["n_soft"]); nflush.append(s["n_flush"]); nspk[len(s["speakers"])] += 1
        ntagdec.append(sum(1 for q in s["soft_pos"] if s["ids"][q] != ds.eot_id))
        out_pos = [j for j, inp in enumerate(s["is_input"]) if not inp]; mask_frac.append(sum(1 for j in out_pos if s["labels"][j] == -100) / max(1, len(out_pos)))
        for q in s["soft_pos"]: pend[round(s["soft_w"][q], 2)] += 1
        win.update(s["win"])
        if n and n % 20000 == 0: print(f"  {corpus}: {n}/{len(idx)} {time.time() - t1:.0f}s", flush=True)
    rep.update(windows_checked=len(idx), windows_with_problems=nbad, problems=dict(probs), problem_examples=ex,
               seq_len=pct(L), over_16384=int(sum(1 for x in L if x > 16384)), text_tokens=pct(ntext, (10, 50, 90)), eot_per_window=round(float(np.mean(neot)), 2) if neot else None,
               soft_per_window=round(float(np.mean(nsoft)), 2) if nsoft else None, tag_decisions=int(sum(ntagdec)), soft_total=int(sum(nsoft)),
               soft_p_end=dict(sorted(pend.items())), flush_tokens=pct(nflush, (50, 90, 100)), masked_label_frac=pct(mask_frac, (50, 90)), speakers_per_window=dict(nspk),
               eot_window_stats=dict(win), sec_check=round(time.time() - t1, 1))
    # 오디오 표본: 실제 학습 경로(__getitem__)
    aud = []
    for i in rng.sample(range(len(ds)), min(a.audio_sample, len(ds))):
        x = ds[i]; n = int(x["wav"].shape[0]); K = int(x["K"]); r = 10 * np.log10(float((x["wav"].double() ** 2).mean()) + 1e-12)
        aud.append(dict(id=x["id"], samples=n, K=K, len_ok=abs(n - K * 1280) <= 1280, rms_db=round(r, 1), n_eot=x["n_eot"], n_soft=x["n_soft"]))
    rep["audio_sample"] = dict(n=len(aud), len_ok=all(z["len_ok"] for z in aud), rms_db=pct([z["rms_db"] for z in aud], (10, 50)), bad=[z for z in aud if not z["len_ok"] or z["rms_db"] < -50][:5])
    rep["sec"] = round(time.time() - t0, 1); print(json.dumps({k: rep[k] for k in ("corpus", "windows_checked", "problems", "seq_len", "eot_per_window", "sec")}, ensure_ascii=False), flush=True)
    return rep


if __name__ == "__main__":
    import multiprocessing as mp
    corpora = a.corpora.split(",")
    with mp.get_context("spawn").Pool(min(a.procs, len(corpora))) as pool: reps = pool.map(run, corpora)
    res = dict(args=vars(a), corpora={r["corpus"]: r for r in reps}, all_ok=all(not r["problems"] and r["prefix"]["ok"] and r["audio_sample"]["len_ok"] for r in reps))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("all_ok", res["all_ok"], "→", a.out)
