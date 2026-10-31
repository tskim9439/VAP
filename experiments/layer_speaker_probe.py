#!/usr/bin/env python
"""인코더 층별 화자 구분력(2026-10-31 — 사용자 "고정 화자 분리기를 대신할 수 있는 방법을 한번 모색해봐", wiki output-diarizer-replacement-20261031).
회의 통째 화자 혼동은 대부분 긴 거리의 정체 어긋남이다(identity_drift.py — E7a 회의 전체 짝 혼동 13.44·19.52 대 30 s 구간 짝 3.97·5.69). 어느 표현에
회의 단위 화자 정체가 남아 있는지 층마다 잰다.
  구간: AMI 회의 통째에서 한 사람만 말하는 프레임(참조 전사 구간, 전사 없는 음성 뺌)이 이어지는 곳을 양끝 2 프레임 깎고 30 프레임(2.4 s)씩 자른다
        (20 프레임 미만은 버리고, N3D 40 s 창 경계를 넘지 않게). 화자마다 시간 순 앞 3 개(등록) + 나머지에서 고르게 --per-spk 개.
  표현(층마다 구간 프레임 평균):
    ours — 고정 Nemotron ASR 인코더(--ckpt 의 encoder.enc): 서브샘플링 출력(층 -1) + 0–23 층 출력. 회의 통째를 겹침 조각(segment_plan)으로 — 스트리밍과 같은 값.
    u3   — --graft(U3 graft.pt 의 enc): 같은 층들. u3-cache = 어댑터 출력 ĥ_8(학생 화자 캐시가 담는 값).
    e7a  — --diar(E7a diar.pt): 가지 입력 임베딩(AOSC 캐시가 담는 값).
    n3d  — Nemotron 3 Diarization 의 층 입력 h_0–h_30 과 마지막 출력(40 s 창 오프라인 — 창 안 전체 어텐션). h_8 이 교사 캐시 자리(U2 이식 층)와 같은 꼴이다.
  잼(eval 회의, 같은 회의 안 구간끼리):
    EER — 구간 짝 코사인으로 같은/다른 화자 판별, 시간 거리(≤ 60 s · 60–600 s · > 600 s)별.
    등록 맞히기 — 화자마다 앞 3 구간 평균(등록)과 가장 가까운 화자를 고른 정확도, 등록 끝에서 지난 시간(≤ 5 분 · 5–15 분 · > 15 분)별.
      AOSC 캐시가 앞에서 고른 대표 프레임으로 뒤의 화자를 잇는 것과 같은 물음이다.
    raw = 회의 평균을 빼고 정규화한 코사인. wccn = 같은 코퍼스 dev 회의의 화자 안 공분산으로 희게 한 뒤 코사인(표현·층마다 dev 에서 맞춤 — 같은 기회).
  python experiments/layer_speaker_probe.py run --ckpt <E11 final> --n3d <.nemo> [--graft <graft.pt>] [--diar <E7a final>] --out <dir> [--rank r --world-size n]
  python experiments/layer_speaker_probe.py report --out <dir>
산출: <out>/emb/<코퍼스>__<회의>.npz(구간 메타 + 표현별 (구간, 층, D) fp16 — report 는 모델 없이 이것만 읽는다), <out>/probe.json, <out>/probe.log. 아무것도 지우지 않는다."""
import argparse, glob, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNK_S, SR, HOP = 0.08, 16000, 1280
NW, PIECE, MIN_PIECE, TRIM, N_ENROLL = 500, 30, 20, 2, 3                                      # N3D 창(프레임) · 구간 길이 · 최소 · 양끝 깎기 · 등록 구간 수
DIST, AGE = (60.0, 600.0), (300.0, 900.0)                                                      # EER 시간 거리 · 등록 뒤 지난 시간 경계(초)


def pieces(act, mask):
    """(K, n) 0/1 활동, (K,) 셈 마스크 → [(화자, 시작, 끝)] — 한 사람만 말하는 연속 프레임을 깎고 PIECE 씩 자른 구간(N3D 창 경계 안)."""
    K = len(act); single = (act.sum(1) == 1) & mask; spk = act.argmax(1); out = []; k = 0
    while k < K:
        if not single[k]: k += 1; continue
        s, e = spk[k], k
        while e < K and single[e] and spk[e] == s: e += 1
        p, b = k + TRIM, e - TRIM
        while p + MIN_PIECE <= b:
            q = min(p + PIECE, b, (p // NW + 1) * NW)
            if q - p >= MIN_PIECE: out.append((int(s), p, q))
            p = q
        k = e
    return out


def choose(segs, n_spk, per_spk):
    """화자마다 시간 순 앞 N_ENROLL 개(등록) + 나머지에서 고르게 per_spk 개 → [(화자, 시작, 끝, 등록)] 시간 순. 구간이 N_ENROLL + 1 개 미만인 화자는 뺀다."""
    out = []
    for s in range(n_spk):
        g = sorted((x for x in segs if x[0] == s), key=lambda x: x[1])
        if len(g) <= N_ENROLL: continue
        rest = g[N_ENROLL:]; idx = np.unique(np.linspace(0, len(rest) - 1, min(per_spk, len(rest))).round().astype(int))
        out += [(*x, 1) for x in g[:N_ENROLL]] + [(*rest[i], 0) for i in idx]
    return sorted(out, key=lambda x: x[1])


class Acc:
    """구간별 프레임 합(표현마다 (구간, 층, D))."""
    def __init__(self, seg_id, n, dev): self.seg_id, self.n, self.dev, self.s, self.c = seg_id, n, dev, {}, {}

    def add(self, name, g0, X):
        """X (층, T, D) = 회의 프레임 [g0, g0 + T) 의 표현 → 구간 합에 더한다."""
        ids = self.seg_id[g0: g0 + X.shape[1]]; v = ids >= 0
        if not v.any(): return
        idx = torch.from_numpy(ids[v]).to(self.dev).long(); loc = torch.from_numpy(np.nonzero(v)[0]).to(self.dev).long()
        if name not in self.s: self.s[name] = torch.zeros(self.n, X.shape[0], X.shape[2], device=self.dev); self.c[name] = torch.zeros(self.n, device=self.dev)
        self.s[name].index_add_(0, idx, X[:, loc].float().permute(1, 0, 2)); self.c[name].index_add_(0, idx, torch.ones(len(idx), device=self.dev))

    def mean(self, name): return (self.s[name] / self.c[name].clamp(min=1)[:, None, None]).half().cpu().numpy()


def enc_pass(enc, wav, K, ch, acc, name, dev, extras=()):
    """회의 통째 → acc[name] = 서브샘플링 출력 + 0 … 마지막 층(겹침 조각 — 스트리밍과 같은 값). extras = [(이름, fn(f (1, 1 + 층 - 1, Ks, D) — 0 = 마지막 층, j + 1 = j 층) → (Ks, d))]."""
    from vapasr.hf.long_session import segment_plan
    nl = len(enc.enc.layers); taps = tuple(range(nl - 1))
    for seg, Ks, a0, c0, n in segment_plan(wav, K, ch):
        got = {}
        hk = enc.enc.pre_encode.register_forward_hook(lambda m_, i_, o: got.__setitem__("pre", o[0] if isinstance(o, tuple) else o))
        try: f = enc(torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32)).to(dev)[None], torch.tensor([len(seg)], device=dev), torch.tensor([Ks], device=dev), taps=taps)
        finally: hk.remove()
        pre = got["pre"][0].float()[:Ks]; pre = torch.nn.functional.pad(pre, (0, 0, 0, Ks - pre.shape[0]))
        X = torch.cat([pre[None], f[0, 1:], f[0, :1]], 0)                                          # (1 + nl, Ks, D): 서브샘플링, 0 … nl - 1 층
        o = c0 - a0; acc.add(name, c0, X[:, o: o + n])
        for nm, fn in extras: acc.add(nm, c0, fn(f)[None, o: o + n])
        del f, X


def n3d_pass(m, wav, K, acc, dev, batch: int = 8):
    """회의 통째 → acc['n3d'] = N3D 층 입력 h_0 … h_30 + 마지막 출력(40 s 창 오프라인)."""
    from experiments.n3d_graft_probe import n3d_hidden
    starts = list(range(0, K, NW))
    for b0 in range(0, len(starts), batch):
        ws = [wav[s * HOP: min(K, s + NW) * HOP] for s in starts[b0: b0 + batch]]
        W = torch.zeros(len(ws), max(len(w) for w in ws))
        for j, w in enumerate(ws): W[j, : len(w)] = torch.from_numpy(np.asarray(w, dtype=np.float32))
        hs, _, elen = n3d_hidden(m, W.to(dev), torch.tensor([len(w) for w in ws], device=dev))
        H = torch.stack(hs, 1)                                                                      # (B, L + 1, T, 512)
        for j, s in enumerate(starts[b0: b0 + batch]):
            T = min(int(elen[j]), H.shape[2], K - s); acc.add("n3d", s, H[j, :, :T])


def run(a):
    from experiments.cst_stage3_eval import windows_of
    from vapasr.data.cst_dialogue import slot_activity, window_speakers
    from vapasr.hf import load_tokenizer
    from vapasr.hf.diar import DiarBranch, load_encoder
    from vapasr.hf.n3d import load_n3d
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"; ed = os.path.join(a.out, "emb"); os.makedirs(ed, exist_ok=True)
    log = open(os.path.join(a.out, f"probe-rank{a.rank}.log"), "a", buffering=1); say = lambda s: (print(s, flush=True), log.write(s + "\n"))
    enc, _ = load_encoder(a.ckpt, a.nemotron_dir, dev); ch = int(json.load(open(os.path.join(a.ckpt, "config.json"))).get("encoder_right_context", 0) or 0) + 1
    m, _ = load_n3d(a.n3d, dev); tok = load_tokenizer(a.ckpt)
    u3 = ad = br = None
    if a.graft:                                                                                     # U3: 아래 층을 학습한 인코더 + 어댑터(학생 캐시 값 ĥ_8)
        from vapasr.hf.n3d_graft import GraftAdapter
        x = torch.load(a.graft, map_location="cpu"); assert x.get("enc") is not None, "--graft 는 인코더를 함께 학습한 graft.pt(U3)"
        u3, _ = load_encoder(x["init"], a.nemotron_dir, dev); u3.enc.load_state_dict(x["enc"]); u3.eval()
        ad = GraftAdapter(**x["cfg"]); ad.load_state_dict(x["state"]); ad.to(dev).eval(); gt = tuple(x["taps"])
    if a.diar:                                                                                      # E7a: 가지 입력 임베딩(AOSC 캐시 값) — 인코더가 --ckpt 와 같아야 한다
        y = torch.load(os.path.join(a.diar, "diar.pt"), map_location="cpu"); br = DiarBranch(**y["cfg"]); br.load_state_dict(y["state"]); br.to(dev).eval(); dt = tuple(y["taps"])
        e2, _ = load_encoder(y["init"], a.nemotron_dir, dev)
        assert all(torch.equal(p, q) for p, q in zip(enc.enc.state_dict().values(), e2.enc.state_dict().values())), "--diar 의 인코더가 --ckpt 와 다르다"
        del e2
    man = json.load(open(os.path.join(a.data, "manifest.json"))); sub = {**{k: "dev" for k in man["dev"]}, **{k: "eval" for k in man["eval"]}}
    ns = argparse.Namespace(corpora=a.corpora, data=a.data, split=None, deltas=[10], limit_windows=0, backchannel=(1.0, 3), max_speakers=4,
                            window="full", hop=10.0, window_draws=0, declare=4)                       # 화자 4 명 넘는 회의는 빠진다(회의 통째 평가와 같은 회의)
    sets, items = windows_of(ns, tok); items = [it for j, it in enumerate(items) if j % a.world_size == a.rank and sub.get(sets[it[0]].items[it[1]][0].split(":", 1)[1])]
    if a.limit: items = items[: a.limit]
    say(f"[{time.strftime('%F %T')}] rank {a.rank}/{a.world_size}: 회의 {len(items)} · ours{' + u3' if u3 else ''}{' + e7a' if br else ''} + n3d · 화자당 {a.per_spk} + 등록 {N_ENROLL}")
    with torch.inference_mode():
        for c, i in items:
            ds = sets[c]; cid, _, L = ds.items[i]; d = ds.dlgs[cid]; dest = os.path.join(ed, f"{c}__{cid.split(':', 1)[1]}.npz")
            if os.path.exists(dest): continue
            t0 = time.time(); K = int(round(L / CHUNK_S)); wav = np.asarray(ds.mono(cid)[: int(round(L * SR))], dtype=np.float32)
            sp = window_speakers(ds.eps[cid], 0.0, L); act = np.asarray(slot_activity(ds.eps[cid], sp, 0.0, L, K, n_slots=len(sp)), dtype=np.int8).reshape(K, len(sp))
            mk = np.ones(K, bool)
            for s_, e_ in ds._bad_spans(d): mk[max(0, int(s_ / CHUNK_S)): min(K, int(math.ceil(e_ / CHUNK_S)))] = False
            sel = choose(pieces(act, mk), len(sp), a.per_spk)
            if not sel: say(f"  {c} {cid}: 구간 없음 — 건너뜀"); continue
            seg_id = np.full(K, -1, np.int64)
            for j, (_, s0, e0, _) in enumerate(sel): seg_id[s0: e0] = j
            acc = Acc(seg_id, len(sel), dev)
            ex = [("e7a", lambda f: br.embed(f[:, [0] + [1 + t for t in dt]])[0])] if br else []
            enc_pass(enc, wav, K, ch, acc, "ours", dev, ex)
            if u3 is not None:
                def _cache(f):
                    with torch.autocast("cuda", dtype=torch.bfloat16): return ad(f[:, [0] + [1 + t for t in gt]])[0].float()
                enc_pass(u3, wav, K, ch, acc, "u3", dev, [("u3-cache", _cache)])
            n3d_pass(m, wav, K, acc, dev)
            out = dict(spk=np.array([x[0] for x in sel]), start=np.array([x[1] for x in sel]), end=np.array([x[2] for x in sel]), enroll=np.array([x[3] for x in sel]),
                       speakers=np.array(sp), sub=np.array(sub[cid.split(":", 1)[1]]), K=np.array(K))
            out.update({f"emb_{nm}": acc.mean(nm) for nm in acc.s})
            np.savez(dest + ".tmp.npz", **out); os.replace(dest + ".tmp.npz", dest)
            say(f"  {c} {cid}({out['sub']}): {K * CHUNK_S / 60:.1f} 분 · 화자 {len(sp)} · 구간 {len(sel)} · " + " ".join(f"{nm} {tuple(acc.s[nm].shape[1:])}" for nm in acc.s) + f" · {time.time() - t0:.0f}s")
    say(f"DONE rank {a.rank} {time.strftime('%F %T')}")


def eer(sc, lab):
    """점수·라벨(같은 화자 = 1) → EER %(짝이 한쪽뿐이면 None)."""
    o = np.argsort(-sc, kind="stable"); l = lab[o].astype(np.float64); P = l.sum(); N = len(l) - P
    if P == 0 or N == 0: return None
    frr = 1 - np.cumsum(l) / P; far = np.cumsum(1 - l) / N; i = int(np.argmin(np.abs(frr - far)))
    return round(float(100 * (frr[i] + far[i]) / 2), 2)


def unit(X): return X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)


def wccn(dev_rows, shrink: float):
    """dev 회의들 [(X (n, D) 회의 평균 뺀 값, spk (n,))] → 화자 안 공분산 희게 하는 W (D, D)(수축 shrink·평균 고윳값)."""
    R = np.concatenate([X[spk == s] - X[spk == s].mean(0) for X, spk in dev_rows for s in np.unique(spk)], 0).astype(np.float64)
    C = R.T @ R / max(1, len(R)); e, V = np.linalg.eigh(C); lam = shrink * max(float(e.mean()), 1e-12)
    return (V * (1.0 / np.sqrt(np.clip(e, 0, None) + lam))) @ V.T


def score(rows, W=None):
    """eval 회의들 [(X, spk, start, end, enroll)] → (EER 시간 거리 3 칸, 등록 맞히기 % 지난 시간 3 칸, 셈)."""
    sc = [[] for _ in range(3)]; lb = [[] for _ in range(3)]; hit = np.zeros(3); tot = np.zeros(3)
    for X, spk, st, en, er in rows:
        Z = unit(X @ W if W is not None else X); S = Z @ Z.T; iu = np.triu_indices(len(Z), 1)
        dt = np.abs(st[:, None] - st[None, :])[iu] * CHUNK_S; same = (spk[:, None] == spk[None, :])[iu]; b = np.digitize(dt, DIST)
        for k in range(3): sc[k].append(S[iu][b == k]); lb[k].append(same[b == k])
        ss = [s for s in np.unique(spk) if (er[spk == s] == 1).any()]
        cen = unit(np.stack([Z[(spk == s) & (er == 1)].mean(0) for s in ss])); end = {s: en[(spk == s) & (er == 1)].max() for s in ss}
        for j in np.nonzero(er == 0)[0]:
            if spk[j] not in end or st[j] < end[spk[j]]: continue
            k = int(np.digitize((st[j] - end[spk[j]]) * CHUNK_S, AGE)); tot[k] += 1; hit[k] += ss[int(np.argmax(cen @ Z[j]))] == spk[j]
    e = [eer(np.concatenate(sc[k]), np.concatenate(lb[k])) if sc[k] else None for k in range(3)]
    return e, [round(float(100 * hit[k] / tot[k]), 1) if tot[k] else None for k in range(3)], dict(pairs=[int(sum(len(x) for x in lb[k])) for k in range(3)], tests=tot.astype(int).tolist())


def report(a):
    log = open(os.path.join(a.out, "probe.log"), "a", buffering=1); say = lambda s: (print(s, flush=True), log.write(s + "\n"))
    files = sorted(glob.glob(os.path.join(a.out, "emb", "*.npz"))); by = {}
    for p in files:                                                                                 # 한 번에 메모리로(npz 는 열쇠마다 다시 읽는다)
        z = np.load(p); by.setdefault(os.path.basename(p).split("__")[0], []).append({k: z[k] for k in z.files})
    res = {}
    say(f"[{time.strftime('%F %T')}] {a.out} — 회의 {len(files)} · 수축 {a.shrink} · EER 거리 ≤{DIST[0]:g}/{DIST[0]:g}–{DIST[1]:g}/>{DIST[1]:g} s · 등록 뒤 ≤{AGE[0]:g}/{AGE[0]:g}–{AGE[1]:g}/>{AGE[1]:g} s")
    for c, zs in by.items():
        dv = [z for z in zs if str(z["sub"]) == "dev"]; ev = [z for z in zs if str(z["sub"]) == "eval"]
        names = [k[4:] for k in ev[0] if k.startswith("emb_")] if ev else []
        say(f"== {c}: dev {len(dv)} · eval {len(ev)} 회의 · eval 구간 {sum(len(z['spk']) for z in ev)}")
        for nm in names:
            nL = ev[0][f"emb_{nm}"].shape[1]; off = -1 if nm in ("ours", "u3") else 0
            for l in range(nL):
                get = lambda z: z[f"emb_{nm}"][:, l].astype(np.float32); cen = lambda X: X - X.mean(0)
                rows = [(cen(get(z)), z["spk"], z["start"], z["end"], z["enroll"]) for z in ev]
                W = wccn([(cen(get(z)), z["spk"]) for z in dv], a.shrink) if dv else None
                er_, id_, n_ = score(rows); ew, iw, _ = score(rows, W) if W is not None else ([None] * 3, [None] * 3, None)
                lay = "cache" if nL == 1 else l + off
                res.setdefault(c, {}).setdefault(nm, []).append(dict(layer=lay, eer_raw=er_, eer_wccn=ew, id_raw=id_, id_wccn=iw, **n_))
                f = lambda v: " · ".join("–" if x is None else f"{x:g}" for x in v)
                say(f"  {nm:>8} {str(lay):>5}: EER raw {f(er_)} | wccn {f(ew)} || 등록 맞히기 raw {f(id_)} | wccn {f(iw)}")
    json.dump(dict(shrink=a.shrink, dist=DIST, age=AGE, res=res), open(os.path.join(a.out, "probe.json"), "w"), indent=1)
    say("DONE report")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--ckpt", required=True); r.add_argument("--n3d", required=True); r.add_argument("--out", required=True)
    r.add_argument("--graft", default=None); r.add_argument("--diar", default=None); r.add_argument("--nemotron-dir", default=None)
    r.add_argument("--data", default="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1"); r.add_argument("--corpora", default="ami-ihmmix,ami-sdm")
    r.add_argument("--per-spk", type=int, default=30); r.add_argument("--limit", type=int, default=0); r.add_argument("--rank", type=int, default=0); r.add_argument("--world-size", type=int, default=1)
    q = sp.add_parser("report"); q.add_argument("--out", required=True); q.add_argument("--shrink", type=float, default=0.05)
    a = ap.parse_args(); run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
