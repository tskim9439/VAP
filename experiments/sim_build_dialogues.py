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
출력: <out>/sim_<xx>.refined.dialogues.jsonl, <out>/sim_<xx>.stats.json(세션·시간·풀별 수·겹침·교대·맞장구·시간 구조 요약·인자). 아무것도 지우지 않는다.

교차 언어(Stage 4, decision-stage4-prep-20261019 — 이어 붙인 합성 대화): --lang2 와 그 재료(--phase2-corpora2·--words2·--pool-weights2)를 주면
  화자 a 는 --lang 풀, 화자 b 는 --lang2 풀에서 고른다(코퍼스가 달라 사람 겹침 검사는 없다). 시간 구조는 두 언어의 실제 대화에서 함께 잰다.
  이름 xx = '<l1>-<l2>'(예 sim_ko-en), Dialogue.lang = '<L1>+<L2>', meta.spk_lang = {a: L1, b: L2} — CstDialogueDataset 이 슬롯을 언어로 고정한다.
  번역(--trans, 반복·쉼표·@목록 — ast_build_targets 산출: Stage 2 단일 화자 타깃(words 행 id)·S1 대화 타깃(conv|화자|utt)): 발화 Unit 에 붙여 놓고
  (sim_dialogue 모듈 설명), <out>/targets-sim_<xx>/<xx>/u-*.jsonl.gz(load_ast_targets 형식, 행 id = <conv>|<화자>|<utt_id 의 # 앞>)로 쓴다.
  교차 언어는 상대 언어 번역이 있는 발화만 재료로 쓴다(--allow-untranslated 로 끔).
--heldout: 평가 셋 — split held-out 대화만 재료로 쓰고 words 는 쓰지 않는다(학습 합성 셋은 held-out 을 쓰지 않으므로 겹치지 않는다)."""
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


def load_pools(a, lang=None, corpora=None, words=None, stats_corpora=None, TR=None, need=None, min_keys=2):
    """→ (풀 {이름: {화자 키: [Unit…]}}, 사람 id, 시간 구조용 실제 대화, 정보). 기본 인자는 --lang 쪽. TR: 번역 타깃({행 id: {목표 언어: (ids, 시각)}}),
    need: 그 목표 언어 번역이 있는 발화만 남긴다(교차 언어). min_keys: 풀에 남길 화자 키 수(같은 언어는 한 풀에서 둘을 고르므로 2)."""
    from vapasr.data.dialogue import Dialogue
    from vapasr.data.sim_dialogue import units_from_dialogue, units_from_words_row, person_ids
    lang = lang or a.lang; corpora = a.phase2_corpora if corpora is None else corpora; words = a.words if words is None else words
    stats_corpora = a.stats_corpora if stats_corpora is None else stats_corpora
    split = json.load(open(a.split, encoding="utf-8")); pools = collections.defaultdict(lambda: collections.defaultdict(dict)); persons = {}; stats_d = []
    info = {}; n_drop = collections.Counter()
    def keep(u, pool):
        if need is None or need in u.trans: return True
        n_drop[pool] += 1; return False
    for c in corpora:
        held = set(split["corpora"].get(c, {}).get("heldout", [])); n_d = n_u = 0
        for line in open(os.path.join(a.phase2, f"{c}.refined.dialogues.jsonl"), encoding="utf-8"):
            d = Dialogue.from_json(line)
            if (d.conv_id not in held if a.heldout else d.conv_id in held) or d.lang != lang: continue
            n_d += 1
            if c in stats_corpora: stats_d.append(d)
            for u in units_from_dialogue(d, trans=TR):
                if keep(u, c): pools[c][u.key][u.uid] = u; persons[u.key] = person_ids(d, u.key.split("|")[2]); n_u += 1
        info[c] = dict(dialogues=n_d, units=n_u)
    files = [] if a.heldout else [p for p in _paths(words) if _keep(p, a.words_frac)]; n_rows = 0              # 평가 셋(--heldout)은 학습 words 를 쓰지 않는다
    for p in files:
        for line in open(p, encoding="utf-8"):
            if f'"lang": "{lang}"' not in line: continue
            r = json.loads(line); n_rows += 1
            for u in units_from_words_row(r, trans=TR.get(r["id"]) if TR else None):
                fam = u.key.split(":", 1)[0]
                if keep(u, fam): pools[fam][u.key][u.uid] = u; persons[u.key] = _person_of_words_key(u.key)
    info["words"] = dict(files=len(files), rows=n_rows)
    out = {}
    for name, keys in pools.items():
        kk = {k: sorted(v.values(), key=lambda u: u.order) for k, v in keys.items() if len(v) >= a.min_units}
        if len(kk) >= min_keys: out[name] = kk
        info.setdefault(name, {}).update(keys_all=len(keys), keys_used=len(kk), units_used=sum(len(v) for v in kk.values()), units_untranslated=n_drop[name])
    return out, persons, stats_d, info


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
    a = G["args"]; xx = G["xx"]; errs = []
    for attempt in range(5):
        rng = random.Random(f"{a.seed}:{i}:{attempt}")
        if G.get("pools2") is None:
            names = [n for n in G["weights"] if n in G["pools"]]
            pool = rng.choices(names, weights=[G["weights"][n] for n in names])[0]; ks = pick_keys(rng, G["pools"][pool], G["persons"])
            if ks is None: continue
            src = [G["pools"][pool][k] for k in ks]
        else:                                                                      # 교차 언어: 화자 a = --lang 풀, b = --lang2 풀(코퍼스가 달라 사람이 겹치지 않는다)
            picks = []
            for P, W in ((G["pools"], G["weights"]), (G["pools2"], G["weights2"])):
                names = [n for n in W if n in P]; pn = rng.choices(names, weights=[W[n] for n in names])[0]; k = rng.choice(list(P[pn])); picks.append((pn, k, P[pn][k]))
            pool = "+".join(x[0] for x in picks); ks = [x[1] for x in picks]; src = [x[2] for x in picks]
        us = []
        for v in src:                                                              # 행 첫 조각에서 시작한다(이어지는 조각 Unit.cont 를 건너뜀 — sim_dialogue 행 무결성)
            i0 = rng.randrange(len(v)); j = 0
            while v[i0].cont and j < len(v): i0 = (i0 + 1) % len(v); j += 1
            us.append((v[i0:] + v[:i0])[: a.max_units])
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
            d = synth_session(us, G["stats"], rng, rng.uniform(*a.dur), span, f"sim_{xx}_{i:06d}", f"sim_{xx}", G["lang_name"], cfg=SimConfig(p_hold=a.p_hold, bc_scale=a.bc_scale),
                              meta=dict(pool=pool, keys=list(ks), attempt=attempt, **({"spk_lang": {"a": a.lang, "b": a.lang2}} if a.lang2 else {})), expand=expand)
            if d.meta["turns"] < 4 or d.duration_s < 25.0: errs.append("too_short"); continue
            tr = d.meta.pop("trans", None)
            y, gains = mix_arrays(d, audio, spans, rng, a.jitter_db); d.meta["gains"] = gains
            if G.get("noise") is not None: y = G["noise"](y, random.Random(f"{a.seed}:{i}:aug"))
            f = mono_cache_path(os.path.join(a.out, "mono"), d); os.makedirs(os.path.dirname(f), exist_ok=True)
            tmp = f[:-4] + f".{os.getpid()}.tmp.npy"; np.save(tmp, np.asarray(y, dtype=np.float16)); os.replace(tmp, f)
            return i, d.to_json(), dict(overlap_stats(d), pool=pool, bc=d.meta["bc"], turns=d.meta["turns"], units=d.meta["units"], attempt=attempt, errors=errs,
                                        utts=len(d.utterances), utts_translated=len(tr or {})), (d.conv_id, tr)
        except Exception as e:
            errs.append(f"{type(e).__name__}: {str(e)[:80]}")
    return i, None, dict(failed=True, errors=errs), None


def write_targets(out_dir, xx, trans_all):
    """세션 번역 → <out>/targets-sim_<xx>/<xx>/u-NNNNN.jsonl.gz(load_ast_targets 형식). 행 id = <conv>|<화자>|<utt_id 의 # 앞>(CstDialogueDataset 의 행 —
    정제 조각 #n 이 한 세션에 같이 놓이면 한 행으로 합친다). → (행 수, 토큰 수)."""
    import gzip
    td = os.path.join(out_dir, f"targets-sim_{xx}", xx); os.makedirs(td, exist_ok=True); buf = []; k = [0]; n = [0, 0]
    def flush():
        if not buf: return
        f = os.path.join(td, f"u-{k[0]:05d}.jsonl.gz"); tmp = f + ".partial"
        with gzip.open(tmp, "wt", encoding="utf-8") as g: g.write("\n".join(buf) + "\n")
        os.replace(tmp, f); k[0] += 1; buf.clear()
    for i in sorted(trans_all):
        cid, tr = trans_all[i]; rows = {}
        for uid, v in tr.items():
            for g, toks in v.items(): rows.setdefault((f"{cid}|{uid.split(':', 1)[0]}|{uid.split('#')[0]}", g), []).extend(toks)
        for (rid, g), toks in rows.items():
            toks.sort(key=lambda z: z[1]); buf.append(json.dumps(dict(id=rid, tgt=g, tokens=toks), ensure_ascii=False)); n[0] += 1; n[1] += len(toks)
            if len(buf) >= 50000: flush()
    flush()
    return n[0], n[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lang", required=True, choices=sorted(SHORT)); ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", type=int, default=2000); ap.add_argument("--dur", default="60,120"); ap.add_argument("--phase2", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--phase2-corpora", default=""); ap.add_argument("--stats-corpora", default=None); ap.add_argument("--words", action="append", default=[])
    ap.add_argument("--words-frac", type=float, default=0.15); ap.add_argument("--pool-weights", required=True); ap.add_argument("--min-units", type=int, default=10)
    ap.add_argument("--max-units", type=int, default=80); ap.add_argument("--jitter-db", type=float, default=3.0); ap.add_argument("--noise-bank", default=None)
    ap.add_argument("--noise-p", type=float, default=0.5); ap.add_argument("--noise-snr", default="10,30"); ap.add_argument("--rir-p", type=float, default=0.3)
    ap.add_argument("--p-hold", type=float, default=0.3, help="같은 화자 재개 확률(음수 = 실제 대화 값)"); ap.add_argument("--bc-scale", type=float, default=1.5)
    ap.add_argument("--split-s", type=float, default=4.0, help="이보다 긴 발화는 안쪽 쉼에서 구 단위로 나눈다")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("교차 언어·번역(Stage 4)")
    g.add_argument("--lang2", default=None, choices=sorted(SHORT), help="화자 b 의 언어(주면 교차 언어 세션)"); g.add_argument("--phase2-corpora2", default="")
    g.add_argument("--stats-corpora2", default=None); g.add_argument("--words2", action="append", default=[]); g.add_argument("--pool-weights2", default="")
    g.add_argument("--trans", action="append", default=[], help="번역 타깃(ast_build_targets 산출, 반복·쉼표·@목록) — Stage 2 단일 화자·S1 대화")
    g.add_argument("--allow-untranslated", action="store_true", help="교차 언어에서 상대 언어 번역이 없는 발화도 재료로 쓴다")
    g.add_argument("--heldout", action="store_true", help="평가 셋: split held-out 대화만 재료로 쓴다(words 없음) — 학습 셋과 겹치지 않는다")
    a = ap.parse_args()
    if a.p_hold < 0: a.p_hold = None
    a.dur = tuple(float(x) for x in a.dur.split(",")); a.phase2_corpora = [c for c in a.phase2_corpora.split(",") if c]
    a.stats_corpora = [c for c in (a.stats_corpora.split(",") if a.stats_corpora else a.phase2_corpora) if c]
    a.phase2_corpora2 = [c for c in a.phase2_corpora2.split(",") if c]
    a.stats_corpora2 = [c for c in (a.stats_corpora2.split(",") if a.stats_corpora2 else a.phase2_corpora2) if c]
    if a.lang2 and (a.lang2 == a.lang or not a.pool_weights2): sys.exit("--lang2 는 --lang 과 다른 언어이고 --pool-weights2 가 필요하다")
    weights = {k: float(v) for k, v in (x.split(":") for x in a.pool_weights.split(",") if x)}
    weights2 = {k: float(v) for k, v in (x.split(":") for x in a.pool_weights2.split(",") if x)}
    xx = SHORT[a.lang] + (f"-{SHORT[a.lang2]}" if a.lang2 else ""); lang_name = a.lang + (f"+{a.lang2}" if a.lang2 else "")
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    from vapasr.data.sim_dialogue import timing_stats
    TR = None
    if a.trans:
        from vapasr.data.ast_dataset import load_ast_targets
        TR, tr_st, _ = load_ast_targets(_paths(a.trans)); print(f"번역 타깃 {dict(tr_st)} · {time.time() - t0:.0f}s", flush=True)
    need = (lambda lg: lg) if a.lang2 and not a.allow_untranslated else (lambda lg: None)
    pools, persons, stats_d, info = load_pools(a, TR=TR, need=need(a.lang2), min_keys=1 if a.lang2 else 2); pools2 = None
    if a.lang2:
        pools2, persons2, stats_d2, info2 = load_pools(a, lang=a.lang2, corpora=a.phase2_corpora2, words=a.words2, stats_corpora=a.stats_corpora2, TR=TR, need=need(a.lang), min_keys=1)
        persons.update(persons2); stats_d += stats_d2; info = {"a": info, "b": info2}
    stats = timing_stats(stats_d)
    show = lambda P: {k: (len(v), sum(len(u) for u in v.values())) for k, v in P.items()}
    print(f"풀 {show(pools)}" + (f" · 풀2 {show(pools2)}" if pools2 is not None else "") + f" · 시간 구조 대화 {stats.n_dialogues} · 교대 쌍 {stats.n_pairs} · 재개 {stats.p_hold:.2f} · "
          f"맞장구 {stats.bc_rate:.3f}/s · {time.time() - t0:.0f}s", flush=True)
    missing = [k for k, w in weights.items() if w > 0 and k not in pools] + [f"2:{k}" for k, w in weights2.items() if a.lang2 and w > 0 and k not in pools2]
    if missing: sys.exit(f"가중치 > 0 인데 쓸 수 있는 풀이 없다: {missing} (있는 풀 {sorted(pools)}{' / ' + str(sorted(pools2)) if pools2 is not None else ''})")
    G.update(args=a, pools=pools, persons=persons, stats=stats, weights={k: w for k, w in weights.items() if w > 0}, xx=xx, lang_name=lang_name,
             pools2=pools2, weights2={k: w for k, w in weights2.items() if w > 0})
    if a.noise_bank:
        from vapasr.data.noise_aug import from_args
        G["noise"] = from_args(a.noise_bank, a.noise_p, a.noise_snr, a.rir_p)
    import multiprocessing as mp
    lines, rows, trans_all = {}, [], {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (i, line, st, tr) in enumerate(pool.imap_unordered(make_session, range(a.sessions), chunksize=2)):
            if line: lines[i] = line
            if line and tr and tr[1]: trans_all[i] = tr
            rows.append(st)
            if (k + 1) % 100 == 0: print(f"  {k + 1}/{a.sessions} · {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(a.out, f"sim_{xx}.refined.dialogues.jsonl")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for i in sorted(lines): f.write(lines[i] + "\n")
    os.replace(path + ".tmp", path)
    n_tr_rows, n_tr_tok = write_targets(a.out, xx, trans_all) if trans_all else (0, 0)
    ok = [r for r in rows if not r.get("failed")]; q = lambda xs, p: round(float(sorted(xs)[int(p * (len(xs) - 1))]), 3) if xs else None
    summ = dict(lang=a.lang, lang2=a.lang2, sessions=len(ok), trans=dict(rows=n_tr_rows, tokens=n_tr_tok, utts=sum(r["utts"] for r in ok),
                utts_translated=sum(r["utts_translated"] for r in ok)) if TR else None, failed=len(rows) - len(ok), hours=round(sum(r["duration_s"] for r in ok) / 3600, 2),
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
