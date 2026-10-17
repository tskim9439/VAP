#!/usr/bin/env python
"""합성 2 인 대화 셋 만들기(Stage 3 화자 귀속 보강) — vapasr/data/sim_dialogue 로 세션을 합성하고 mono 혼합(+잡음·잔향)을 캐시한다.

  python experiments/sim_build_dialogues.py --lang English --out <dir> --sessions 2000 \\
      --phase2 <phase2 dir> --split <v2.json> --phase2-corpora otoSpeech --words @<list> --words @<list> [--words-frac 0.15] \\
      --pool-weights otoSpeech:0.3,swbd:0.3,cv:0.15,vp:0.15,ls:0.1 [--noise-bank <dir>] [--workers 16]

재료 풀: Phase 2 정제 대화(split 학습 분할만 — held-out 대화는 쓰지 않는다)의 코퍼스별 화자 발화 + Stage 1·2 words(화자 키를 아는 세트: ls·swbd·vp·cv).
  풀 이름 = Phase 2 코퍼스 이름 | words 키 접두어. 화자 키마다 발화 --min-units 개 이상만 쓴다.
시간 구조: --stats-corpora(기본 = --phase2-corpora)의 학습 분할 실제 대화에서 잰다(sim_dialogue.timing_stats).
세션 i(난수 = seed:i): 풀을 가중치로 고르고, 그 풀에서 다른 사람인 화자 키 둘을 고른다(Phase 2 는 다른 대화이면서 person_ids 가 겹치지 않게).
  각자 발화를 원래 순서로 임의 위치부터 꺼내 --dur 범위 길이로 합성한다(synth_session). 발화 오디오를 읽어 말소리 구간(speech_span)만 놓는다.
  --split-s 보다 긴 발화는 안쪽 쉼에서 구 단위로 나눠(split_points) 앞 조각부터 그 화자의 턴들에 쓴다. 재개 확률 --p-hold, 맞장구 배율 --bc-scale.
mono: 화자별 채널(놓인 말소리)을 dialogue_mix.mix_dialogue 와 같은 규칙(화자 발화 구간 RMS → −23 dBFS, ± --jitter-db, 피크 0.98)으로 맞춰 더하고,
  --noise-bank 가 있으면 섞은 뒤 잡음·잔향을 둘에 같이 입힌다(녹음 조건 차이가 화자 단서가 되지 않게).
  저장: <out>/mono/sim_<xx>/<conv>.npy(float16) — CstDialogueDataset(mono_cache_dir=<out>/mono) 의 캐시 규약이라 학습이 그대로 읽는다.
출력: <out>/sim_<xx>.refined.dialogues.jsonl, <out>/sim_<xx>.stats.json(세션·시간·풀별 수·겹침·교대·맞장구·시간 구조 요약·인자). 아무것도 지우지 않는다."""
import argparse, collections, hashlib, json, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHORT = {"Korean": "ko", "English": "en", "German": "de"}
G = {}


def _paths(vals):
    return [q for x in vals for q in ([l.strip() for l in open(x[1:]) if l.strip()] if x.startswith("@") else x.split(",")) if q]


def _keep(p, frac):
    return frac >= 1.0 or int(hashlib.sha1(p.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def _person_of_words_key(k):
    fam, rest = k.split(":", 1)
    return {f"ls:{rest.split('-')[0]}"} if fam == "ls" else {k}


def load_pools(a):
    from vapasr.data.dialogue import Dialogue
    from vapasr.data.sim_dialogue import units_from_dialogue, units_from_words_row, person_ids, timing_stats
    split = json.load(open(a.split, encoding="utf-8")); pools = collections.defaultdict(lambda: collections.defaultdict(dict)); persons = {}; stats_d = []; info = {}
    for c in a.phase2_corpora:
        held = set(split["corpora"].get(c, {}).get("heldout", [])); n_d = n_u = 0
        for line in open(os.path.join(a.phase2, f"{c}.refined.dialogues.jsonl"), encoding="utf-8"):
            d = Dialogue.from_json(line)
            if d.conv_id in held or d.lang != a.lang: continue
            n_d += 1
            if c in a.stats_corpora: stats_d.append(d)
            for u in units_from_dialogue(d): pools[c][u.key][u.uid] = u; persons[u.key] = person_ids(d, u.key.split("|")[2]); n_u += 1
        info[c] = dict(dialogues=n_d, units=n_u)
    files = [p for p in _paths(a.words) if _keep(p, a.words_frac)]; n_rows = 0
    for p in files:
        for line in open(p, encoding="utf-8"):
            if f'"lang": "{a.lang}"' not in line: continue
            r = json.loads(line); n_rows += 1
            for u in units_from_words_row(r):
                fam = u.key.split(":", 1)[0]; pools[fam][u.key][u.uid] = u; persons[u.key] = _person_of_words_key(u.key)
    info["words"] = dict(files=len(files), rows=n_rows)
    out = {}
    for name, keys in pools.items():
        kk = {k: sorted(v.values(), key=lambda u: u.order) for k, v in keys.items() if len(v) >= a.min_units}
        if len(kk) >= 2: out[name] = kk
        info.setdefault(name, {}).update(keys_all=len(keys), keys_used=len(kk), units_used=sum(len(v) for v in kk.values()))
    return out, persons, timing_stats(stats_d), info


def pick_keys(rng, keys, persons, tries=60):
    ks = list(keys)
    for _ in range(tries):
        a, b = rng.sample(ks, 2)
        if persons[a] & persons[b]: continue
        if "|" in a and a.split("|")[1] == b.split("|")[1]: continue
        return a, b
    return None


def mix_arrays(d, audio, spans, rng, jitter_db, target_dbfs=-23.0, peak=0.98):
    """dialogue_mix.mix_dialogue 와 같은 규칙을 이미 읽은 발화 오디오로: 화자별 채널에 말소리 구간을 놓고, 화자 발화 구간 RMS 로 target_dbfs(± jitter) 에 맞춰 더한다."""
    import numpy as np
    from vapasr.data.sim_dialogue import SR
    T = int(round(d.duration_s * SR)); mono = np.zeros(T, np.float32); gains = {}
    for spk in d.speakers:
        ch = np.zeros(T, np.float32); acc = 0.0; n = 0
        for u in d.utterances:
            if u.speaker != spk: continue
            uid = u.utt_id.split(":", 1)[1]; a, b = spans[uid]; x = audio[uid]; i0 = int(round(u.start * SR)); m = int(round((u.end - u.start) * SR))
            seg = x[int(round(a * SR)): int(round(a * SR)) + m]; seg = seg[: max(0, T - i0)]; ch[i0: i0 + len(seg)] += seg; acc += float(np.sum(seg.astype(np.float64) ** 2)); n += len(seg)
        rms = math.sqrt(acc / n) if n else 1e-6
        g = (10 ** (target_dbfs / 20)) / max(rms, 1e-6) * (10 ** (rng.uniform(-jitter_db, jitter_db) / 20) if jitter_db else 1.0); gains[spk] = round(g, 4); mono += ch * g
    pk = float(np.max(np.abs(mono))) if T else 0.0
    return (mono * (peak / pk) if pk > peak else mono), gains


def make_session(i):
    import numpy as np
    from vapasr.data.streams import load_utt_audio
    from vapasr.data.sim_dialogue import synth_session, speech_span, overlap_stats, split_points, split_unit, SimConfig, SR
    from vapasr.data.dialogue_dataset import mono_cache_path
    a = G["args"]; xx = SHORT[a.lang]; errs = []
    for attempt in range(5):
        rng = random.Random(f"{a.seed}:{i}:{attempt}"); names = [n for n in G["weights"] if n in G["pools"]]
        pool = rng.choices(names, weights=[G["weights"][n] for n in names])[0]; ks = pick_keys(rng, G["pools"][pool], G["persons"])
        if ks is None: continue
        us = []
        for k in ks:
            v = G["pools"][pool][k]; i0 = rng.randrange(len(v)); us.append((v[i0:] + v[:i0])[: a.max_units])
        audio, spans = {}, {}
        def load(u):
            if u.uid not in audio:
                audio[u.uid] = np.asarray(load_utt_audio(u.path, u.src_offset, u.dur) if u.src_offset is not None else load_utt_audio(u.path), dtype=np.float32)
            return audio[u.uid]
        def span(u):
            if u.uid not in spans: spans[u.uid] = speech_span(load(u), u.tokens)
            return spans[u.uid]
        def expand(u):                                                             # 긴 발화 → 쉼에서 구 단위 조각(조각 오디오는 부모를 잘라 캐시)
            x = load(u); parts = split_points(x, u.tokens, max_s=a.split_s); subs = split_unit(u, parts)
            for sub in subs[1:] if len(subs) == 1 else subs:
                k = int(sub.uid.rsplit("~", 1)[1]); p0, p1 = parts[k]; audio[sub.uid] = x[int(round(p0 * SR)): int(round(p1 * SR))]
            return subs
        try:
            d = synth_session(us, G["stats"], rng, rng.uniform(*a.dur), span, f"sim_{xx}_{i:06d}", f"sim_{xx}", a.lang, cfg=SimConfig(p_hold=a.p_hold, bc_scale=a.bc_scale),
                              meta=dict(pool=pool, keys=list(ks), attempt=attempt), expand=expand)
            if d.meta["turns"] < 4 or d.duration_s < 25.0: errs.append("too_short"); continue
            y, gains = mix_arrays(d, audio, spans, rng, a.jitter_db); d.meta["gains"] = gains
            if G.get("noise") is not None: y = G["noise"](y, random.Random(f"{a.seed}:{i}:aug"))
            f = mono_cache_path(os.path.join(a.out, "mono"), d); os.makedirs(os.path.dirname(f), exist_ok=True)
            tmp = f[:-4] + f".{os.getpid()}.tmp.npy"; np.save(tmp, np.asarray(y, dtype=np.float16)); os.replace(tmp, f)
            return i, d.to_json(), dict(overlap_stats(d), pool=pool, bc=d.meta["bc"], turns=d.meta["turns"], units=d.meta["units"], attempt=attempt, errors=errs)
        except Exception as e:
            errs.append(f"{type(e).__name__}: {str(e)[:80]}")
    return i, None, dict(failed=True, errors=errs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lang", required=True, choices=sorted(SHORT)); ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", type=int, default=2000); ap.add_argument("--dur", default="60,120"); ap.add_argument("--phase2", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--phase2-corpora", default=""); ap.add_argument("--stats-corpora", default=None); ap.add_argument("--words", action="append", default=[])
    ap.add_argument("--words-frac", type=float, default=0.15); ap.add_argument("--pool-weights", required=True); ap.add_argument("--min-units", type=int, default=10)
    ap.add_argument("--max-units", type=int, default=80); ap.add_argument("--jitter-db", type=float, default=3.0); ap.add_argument("--noise-bank", default=None)
    ap.add_argument("--noise-p", type=float, default=0.5); ap.add_argument("--noise-snr", default="10,30"); ap.add_argument("--rir-p", type=float, default=0.3)
    ap.add_argument("--p-hold", type=float, default=0.3, help="같은 화자 재개 확률(음수 = 실제 대화 값)"); ap.add_argument("--bc-scale", type=float, default=1.5)
    ap.add_argument("--split-s", type=float, default=4.0, help="이보다 긴 발화는 안쪽 쉼에서 구 단위로 나눈다")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    if a.p_hold < 0: a.p_hold = None
    a.dur = tuple(float(x) for x in a.dur.split(",")); a.phase2_corpora = [c for c in a.phase2_corpora.split(",") if c]
    a.stats_corpora = [c for c in (a.stats_corpora.split(",") if a.stats_corpora else a.phase2_corpora) if c]
    weights = {k: float(v) for k, v in (x.split(":") for x in a.pool_weights.split(",") if x)}
    os.makedirs(a.out, exist_ok=True); xx = SHORT[a.lang]; t0 = time.time()
    pools, persons, stats, info = load_pools(a)
    print(f"풀 {({k: (len(v), sum(len(u) for u in v.values())) for k, v in pools.items()})} · 시간 구조 대화 {stats.n_dialogues} · 교대 쌍 {stats.n_pairs} · 재개 {stats.p_hold:.2f} · "
          f"맞장구 {stats.bc_rate:.3f}/s · {time.time() - t0:.0f}s", flush=True)
    missing = [k for k, w in weights.items() if w > 0 and k not in pools]
    if missing: sys.exit(f"가중치 > 0 인데 쓸 수 있는 풀이 없다: {missing} (있는 풀 {sorted(pools)})")
    G.update(args=a, pools=pools, persons=persons, stats=stats, weights={k: w for k, w in weights.items() if w > 0})
    if a.noise_bank:
        from vapasr.data.noise_aug import from_args
        G["noise"] = from_args(a.noise_bank, a.noise_p, a.noise_snr, a.rir_p)
    import multiprocessing as mp
    lines, rows = {}, []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (i, line, st) in enumerate(pool.imap_unordered(make_session, range(a.sessions), chunksize=2)):
            if line: lines[i] = line
            rows.append(st)
            if (k + 1) % 100 == 0: print(f"  {k + 1}/{a.sessions} · {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(a.out, f"sim_{xx}.refined.dialogues.jsonl")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for i in sorted(lines): f.write(lines[i] + "\n")
    os.replace(path + ".tmp", path)
    ok = [r for r in rows if not r.get("failed")]; q = lambda xs, p: round(float(sorted(xs)[int(p * (len(xs) - 1))]), 3) if xs else None
    summ = dict(lang=a.lang, sessions=len(ok), failed=len(rows) - len(ok), hours=round(sum(r["duration_s"] for r in ok) / 3600, 2),
                speech_hours=round(sum(r["speech_s"] for r in ok) / 3600, 2), overlap_ratio_mean=round(sum(r["overlap_ratio"] for r in ok) / max(1, len(ok)), 4),
                changes_per_min_mean=round(sum(r["changes_per_min"] for r in ok) / max(1, len(ok)), 2), bc_per_min=round(60 * sum(r["bc"] for r in ok) / max(1e-6, sum(r["duration_s"] for r in ok)), 2),
                self_overlap=sum(r["self_overlap"] for r in ok), pools=dict(collections.Counter(r["pool"] for r in ok)),
                timing=dict(dialogues=stats.n_dialogues, pairs=stats.n_pairs, p_hold=round(stats.p_hold, 3), bc_rate_per_s=round(stats.bc_rate, 4),
                            fto_q10_50_90=[q(stats.fto, p) for p in (0.1, 0.5, 0.9)], hold_q50=q(stats.hold, 0.5), turn_q10_50_90=[q(stats.turn, p) for p in (0.1, 0.5, 0.9)],
                            fto_negative_share=round(sum(x < 0 for x in stats.fto) / max(1, len(stats.fto)), 3)),
                pool_info=info, errors=dict(collections.Counter(e for r in rows for e in r.get("errors", [])).most_common(20)), args={k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(a).items()}, sec=round(time.time() - t0, 1))
    json.dump(summ, open(os.path.join(a.out, f"sim_{xx}.stats.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("args", "pool_info")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
