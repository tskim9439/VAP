#!/usr/bin/env python
"""대화 발화 단독 디코드 대조 — Stage 3 회귀 평가와 같은 held-out 창(cst_stage3_eval.windows_of)의 전사 발화를 하나씩 잘라 단일 화자 형식으로 디코드한다.
대화 받아쓰기 오류(agn·cp)가 단일 화자 평가 셋보다 큰 까닭을 나눈다: 대화 조건(두 화자 혼합·긴 문맥·귀속) 몫 대 도메인·참조 몫.

  python experiments/cst_isolated_asr.py prepare --checkpoint <ck(tokenizer)> --data <phase2> --split <v2.json> --out <dir> [--corpora …] [--limit-windows 200] [--pad 0.15]
  python experiments/eval_single_turn_asr.py run --manifest <dir>/manifest.jsonl --checkpoint <ck> --encoder <nemo> --out <dir>/<run> --deltas 4 10
  python experiments/cst_isolated_asr.py score --out <dir> --runs <run>[,…] [--dialogue <run>=<stage3-eval 결과 폴더>,…]

발화: 창 안에 통째로 들어가고 전사가 있는(DialogueDataset._untranscribed 가 아닌) 발화. 창이 겹치면 한 번만 쓴다.
조건: mix = 대화 평가와 같은 mono 혼합(캐시) 구간 [start − pad, end + pad] — 상대 화자 소리가 섞일 수 있다.
      chan = 그 화자 채널만 같은 구간(발화 구간 RMS 를 −23 dBFS 로 — mono 혼합의 화자 목표와 같다).
참조: 발화 정렬 토큰(u.tokens)을 학습 tokenizer 로 단어로 바꾼 것 — 대화 평가 참조(window_ref)와 같은 문자열. 채점은 cst_eval.units·ed(대화 평가와 같은 단위).
산출: <out>/audio/<조건>/<코퍼스>/*.wav, <out>/manifest.jsonl(eval_single_turn_asr 형식 + 대화·발화·화자·겹침), score-<run>.json. 아무것도 지우지 않는다."""
import argparse, collections, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def overlap_share(d, k):
    """발화 k 구간 중 다른 화자 발화(전사 여부 무관)와 겹치는 비율."""
    u = d.utterances[k]; iv = sorted((max(u.start, v.start), min(u.end, v.end)) for v in d.utterances if v.speaker != u.speaker and v.start < u.end and v.end > u.start)
    tot, cur = 0.0, None
    for s, e in iv:
        if cur is None or s > cur[1]: tot += 0 if cur is None else cur[1] - cur[0]; cur = [s, e]
        else: cur[1] = max(cur[1], e)
    if cur: tot += cur[1] - cur[0]
    return tot / max(1e-6, u.end - u.start)


def prepare(a):
    import numpy as np, soundfile as sf
    from transformers import AutoTokenizer
    from cst_stage3_eval import windows_of, words_of
    from vapasr.data.dialogue_mix import load_channel
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    sets, items = windows_of(a, tok)
    items.sort(key=lambda x: (x[0], sets[x[0]].items[x[1]][0], sets[x[0]].items[x[1]][1]))      # 대화끼리 모아 채널을 한 번씩만 읽는다
    seen, rows, chans, skipped = set(), [], {}, collections.Counter()
    man = os.path.join(a.out, "manifest.jsonl"); assert not os.path.exists(man), f"{man} 가 이미 있다 — 새 --out 을 쓴다"
    for j, (c, i) in enumerate(items):
        ds = sets[c]; cid, t0w, L = ds.items[i]; d = ds.dlgs[cid]; mono = None
        if j % 100 == 0: print(f"  창 {j}/{len(items)} · {c} · 행 {len(rows)}", flush=True)
        for k, u in enumerate(d.utterances):
            if not (u.start >= t0w and u.end <= t0w + L): continue
            if (cid, k) in seen: continue
            seen.add((cid, k))
            if ds._untranscribed(d, u) or not u.tokens: skipped["untranscribed"] += 1; continue
            ref = " ".join(w for w, _ in words_of(tok, [(t, tid) for tid, t in u.tokens]))
            if not ref.strip(): skipped["empty_ref"] += 1; continue
            s0, s1 = max(0.0, u.start - a.pad), min(d.duration_s, u.end + a.pad)
            if mono is None: mono = ds.mono(cid)
            ov = round(overlap_share(d, k), 4); stem = f"{cid.split(':', 1)[1].replace('/', '_')}_{k:04d}"
            if not any(k0 == cid for k0, _ in chans): chans.clear()                         # 대화가 바뀌면 이전 대화 채널을 놓는다
            if (cid, u.speaker) not in chans:
                ref_ch = d.channels.get(u.speaker); chans[(cid, u.speaker)] = load_channel(ref_ch, d.duration_s) if ref_ch is not None else None
            ch = chans[(cid, u.speaker)]
            for cond in ("mix", "chan"):
                if cond == "mix": x = np.asarray(mono[int(round(s0 * 16000)): int(round(s1 * 16000))], dtype=np.float32)
                else:
                    if ch is None: skipped["no_channel"] += 1; continue
                    x = ch[int(round(s0 * 16000)): int(round(s1 * 16000))].astype(np.float32)
                    core = ch[int(round(u.start * 16000)): int(round(u.end * 16000))]
                    rms = float(np.sqrt(np.mean(core.astype(np.float64) ** 2) + 1e-12)); x = x * (10 ** (-23 / 20) / max(rms, 1e-6))
                    pk = float(np.max(np.abs(x))) if len(x) else 0.0
                    if pk > 0.98: x = x * (0.98 / pk)
                if len(x) < 1600: skipped["too_short"] += 1; continue
                p = os.path.join(a.out, "audio", cond, c, stem + ".wav"); os.makedirs(os.path.dirname(p), exist_ok=True)
                sf.write(p, x, 16000, subtype="PCM_16")
                rows.append(dict(id=f"{c}/{cond}/{stem}", dataset=f"{c}-{cond}", lang=d.lang, path=p, raw_reference=ref, reference=ref, sample_rate=16000,
                                 num_samples=len(x), audio_s=round(len(x) / 16000, 4), audio_format="wav", corpus=c, cond=cond, conv=cid, utt=k,
                                 speaker=u.speaker, start=u.start, end=u.end, overlap=ov, window=f"{cid}@{t0w}"))
    with open(man, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = collections.Counter((r["corpus"], r["cond"]) for r in rows)
    print(json.dumps(dict(rows=len(rows), per=dict((f"{k[0]}/{k[1]}", v) for k, v in sorted(n.items())), skipped=dict(skipped)), ensure_ascii=False, indent=1), flush=True)


def score(a):
    from vapasr.data.cst_eval import units, ed
    man = {r["id"]: r for r in map(json.loads, open(os.path.join(a.out, "manifest.jsonl"), encoding="utf-8"))}
    dlg = {}
    for x in (a.dialogue or "").split(","):
        if x: name, p = x.split("=", 1); dlg[name] = json.load(open(os.path.join(p, "summary.json")))
    res = {}
    for run in a.runs.split(","):
        acc = collections.defaultdict(collections.Counter)
        for p in sorted(glob.glob(os.path.join(a.out, run, "predictions-rank*.jsonl"))):
            for l in open(p, encoding="utf-8"):
                x = json.loads(l); m = man[x["id"]]
                ru, hu = units(m["reference"], m["lang"]), units(x["hypothesis"], m["lang"]); e = ed(ru, hu)
                for g in ("all", "no_overlap" if m["overlap"] == 0 else "overlap", "short" if m["end"] - m["start"] < 1.5 else "long"):
                    c = acc[(m["corpus"], m["cond"], x["delta"], g)]; c["err"] += e; c["ref"] += len(ru); c["hyp"] += len(hu); c["utts"] += 1
        res[run] = {f"{k[0]}/{k[1]}/d{k[2]}/{k[3]}": dict(err=round(100 * v["err"] / max(1, v["ref"]), 2), hyp_ref=round(v["hyp"] / max(1, v["ref"]), 3), utts=v["utts"], ref_units=v["ref"])
                    for k, v in sorted(acc.items())}
        json.dump(res[run], open(os.path.join(a.out, f"score-{run}.json"), "w"), ensure_ascii=False, indent=1)
        print(f"== {run}: 발화 단독 오류 %(CER/WER, hyp/ref) — mix 전체 · chan 전체 · chan 겹침 없음 · chan 1.5 s 미만" + (f" | 대화 평가 agn · cp({a.dialogue_name or ''})" if dlg else ""))
        corpora = sorted({k[0] for k in acc}, key=lambda c: ["aihub134-1", "aihub134-2", "aihub71631", "otoSpeech", "kec", "nsc", "ztc"].index(c) if c in
                         ["aihub134-1", "aihub134-2", "aihub71631", "otoSpeech", "kec", "nsc", "ztc"] else 99)
        for dlt in sorted({k[2] for k in acc}):
            for c in corpora:
                f = lambda cond, g: (lambda v: f"{v['err']:5.1f}({v['hyp_ref']:.2f}, {v['utts']})" if v else "    –")(res[run].get(f"{c}/{cond}/d{dlt}/{g}"))
                s = f"  δ{dlt:<2d} {c:11s} mix {f('mix', 'all')} · chan {f('chan', 'all')} · 겹침 없음 {f('chan', 'no_overlap')} · 짧은 {f('chan', 'short')}"
                for name, sm in dlg.items():
                    v = sm.get(f"{c}/d{dlt}")
                    if v: s += f" | {name} agn {v['err_agn']:5.1f} cp {v['err_cp']:5.1f}"
                print(s)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--checkpoint", required=True, help="참조 단어를 만들 tokenizer(학습 체크포인트)")
    q.add_argument("--data", required=True); q.add_argument("--split", required=True); q.add_argument("--out", required=True)
    q.add_argument("--corpora", default="aihub134-2,aihub134-1,aihub71631,otoSpeech,kec,ztc,nsc"); q.add_argument("--limit-windows", type=int, default=200)
    q.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[4, 10]); q.add_argument("--pad", type=float, default=0.15)
    q.add_argument("--backchannel", default=(1.0, 3), type=lambda s: (float(s.split(",")[0]), int(s.split(",")[1])), help="창 규칙을 Stage 3 회귀 평가와 맞춘다")
    q = sub.add_parser("score"); q.add_argument("--out", required=True); q.add_argument("--runs", required=True)
    q.add_argument("--dialogue", default=None, help="이름=stage3-eval 결과 폴더[,…] — 같은 창의 대화 평가 agn·cp 를 옆에 적는다")
    q.add_argument("--dialogue-name", default=None)
    a = ap.parse_args(); {"prepare": prepare, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
