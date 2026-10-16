"""MLX 엔진 vs 골든(torch fp32 CPU stream_decode) 검증.

(1) thinker 만: <GOLD>/<case>/feats.npy 로
    - teacher-forced: 참조 입력열을 그대로 다시 먹이며 결정 행마다 argmax 일치 (불일치 때 참조·MLX 마진)
    - free-running: SegmentDecoder(순차/투기/fast = 투기 + q4 초안 + 라운드 끝 투기)로 emitted·forced·flush 동일성, 첫 분기,
      토큰 편집거리, final 모드 (q=1 걸음과 lookahead 걸음 둘 다)
    정밀도: fp32 / fp16(잔차 fp32 + 재채점) / fp16-plain / bf16 (+ q8 보고용)
(2) 전체 엔진: wav.npy → 인코더(W2) → 토큰 (패킷 크기 --packets p20|rand|oneshot), 특징 차이, 라운드 불변식,
    final_redecode (MLX 인코더 특징으로) vs 골든 final_ids

무거운 작업: lockf 로 감싸 실행 (app-design §5).
  python tools/verify_engine.py --part 1 [--precisions fp32,fp16,fp16-plain,bf16,q8] [--cases a,b]
  python tools/verify_engine.py --part 2 --precisions fp32,fp16 [--encoder-precision fp16] [--packets rand] [--no-final]"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLD = "/Volumes/Samsung_T5/vapkt-models/app-work/golden-s4"
OUT_DIR = "/private/tmp/claude-501/-Users-taesookim-Desktop-VAPKT/e6f0e8db-a78d-4321-bce1-f9fa25adbcf1/scratchpad/work-w3"


def load_case(d: str) -> dict:
    with open(os.path.join(d, "meta.json")) as f:
        m = json.load(f)
    m["_dir"], m["_name"] = d, os.path.basename(d)
    m["_delay"] = m.get("delta", m.get("delay"))
    m["_final"] = m.get("final_ids", m.get("offline_ids"))
    m["_n_total"] = m.get("n_total", m.get("n_samples_with_tail"))
    return m


def ref_decisions(m: dict) -> dict | None:
    """골든 결정 기록: make_reference.py 의 decisions = {pos, k, id, margin, id2} (prefix 결정 없음, 강제 결정 포함)."""
    d = m.get("decisions")
    if isinstance(d, dict) and "id" in d and "margin" in d:
        return d
    if isinstance(d, list) and d and isinstance(d[0], (list, tuple)):           # golden-tiny 형식 [[tag, id, margin], ...]
        d = [x for x in d if x[0] != "prefix"]
        return dict(id=[int(x[1]) for x in d], margin=[float(x[2]) for x in d])
    return None


def margin(row: np.ndarray) -> float:
    t = np.partition(row, -2)[-2:]
    return float(t[1] - t[0])


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def split_runs(ids: list, spk: tuple, trans: int) -> tuple[list, list]:
    run, src, tr = 0, [], []
    for t in ids:
        if t in spk:
            run = 0
        elif t == trans:
            run = 1
        else:
            (src if run == 0 else tr).append(t)
    return src, tr


def teacher_forced(eng, m: dict, feats: np.ndarray) -> dict:
    """참조 궤적을 그대로 먹이며 매 결정의 MLX argmax 를 본다 (순차, 병합 NEXT)."""
    import mlx.core as mx
    th, sp, cfg = eng.thinker, eng.sp, eng.cfg
    cap, K = cfg["runaway_cap"], int(m["K"])
    max_total = int(6.0 * K) + 64
    by = defaultdict(list)
    for k, t in m["emitted"]:
        by[int(k)].append(int(t))
    ce = eng.adapter(mx.array(feats)).astype(th.in_dtype)
    eN, eE = th.embed([sp.next])[0], th.embed([sp.empty])[0]
    prefix = list(m["prefix_ids"])
    th.forward(th.embed(prefix), 0, sel=[])
    pos, pending, n_out = len(prefix), False, 0
    rows = []
    for k in list(range(K)) + [K + r for r in range(int(m["flush_rounds"]))]:
        toks = by.get(k, [])
        x = ce[k] if k < K else eE
        xs = [eN, x] if pending else [x]
        if pos + len(xs) + len(toks) + 1 > th.max_len:
            return dict(error=f"KV 용량 초과 (pos {pos})")
        ids, L = th.forward_ids_logits(mx.stack(xs), pos)
        pos += len(xs)
        for i in range(len(toks) + 1):
            if i < len(toks):
                exp = toks[i]
            else:
                exp = None if (i >= cap or n_out + i >= max_total) else sp.next   # cap 에서 끊긴 결정은 참조 argmax 를 모른다
            rows.append((k, i, exp, ids[0], margin(L[0]), pos - 1))
            if i < len(toks):
                ids, L = th.forward_ids_logits(th.embed([toks[i]]), pos)
                pos += 1
        pending = True
        n_out += len(toks)
    ref = ref_decisions(m)
    full = ref is not None and len(ref["id"]) == len(rows)
    if full:                                                       # 강제 결정의 참조 argmax 도 안다
        bad_pos = sum(1 for j, r in enumerate(rows) if "pos" in ref and ref["pos"][j] != r[5])
        rows = [(r[0], r[1], int(ref["id"][j]), r[3], r[4], r[5], float(ref["margin"][j])) for j, r in enumerate(rows)]
    else:
        bad_pos = None
        rows = [r + (None,) for r in rows]
    cmp = [(j, r) for j, r in enumerate(rows) if r[2] is not None]
    bad = [(j, r) for j, r in cmp if r[2] != r[3]]
    mism = [dict(idx=j, k=r[0], i=r[1], pos=r[5], ref=r[2], mlx=r[3], mlx_margin=round(r[4], 5),
                 ref_margin=(round(r[6], 6) if r[6] is not None else None)) for j, r in bad]
    rm = [r[6] for r in rows if r[6] is not None]
    return dict(decisions=len(rows), compared=len(cmp), disagree=len(bad), agree_rate=1 - len(bad) / max(1, len(cmp)),
                min_mlx_margin=round(min(r[4] for r in rows), 6) if rows else None, mismatches=mism[:20],
                ref_aligned=full, pos_mismatch=bad_pos,
                max_ref_margin_at_disagree=max((x["ref_margin"] for x in mism if x["ref_margin"] is not None), default=None),
                n_ref_margin_lt_1e3=sum(v < 1e-3 for v in rm) if rm else None)


def free_running(eng, m: dict, feats: np.ndarray, speculative: bool, final: bool, draft: bool = False) -> dict:
    import mlx.core as mx
    K = int(m["K"])
    eng.speculative = speculative
    on0 = getattr(eng, "draft_on", False)
    eng.draft_on = bool(draft and getattr(eng, "draft", None) is not None)
    try:
        seg = eng.new_segment(m["src"], m.get("tgt"), int(m["_delay"]), known_K=K)
    finally:
        eng.draft_on = on0
    t0 = time.perf_counter()
    f = mx.array(feats)
    for i in range(0, K, 4):                                       # 인코더 청크 단위로 넣는다
        seg.feed_frames(f[i:i + 4])
    seg.finish_frames(K, n_total=m.get("_n_total"))
    ms = (time.perf_counter() - t0) * 1000
    ref = [tuple(map(int, x)) for x in m["emitted"]]
    got = list(seg.out)
    first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
    if first is None and len(ref) != len(got):
        first = min(len(ref), len(got))
    src, tr = split_runs([t for _, t in got], eng.spk_ids, eng.trans_id)
    res = dict(identical=(got == ref and seg.forced == m["forced"] and seg.flush_rounds == m["flush_rounds"]),
               tokens_ref=len(ref), tokens=len(got), forced=(seg.forced, m["forced"]), flush=(seg.flush_rounds, m["flush_rounds"]),
               first_diff=first, edit=edit_distance([t for _, t in ref], [t for _, t in got]),
               k_shift=sum(1 for a, b in zip(ref, got) if a[1] == b[1] and a[0] != b[0]),
               src_text=eng.tokens.decode(src), tr_text=eng.tokens.decode(tr), ms=round(ms, 1),
               forwards=seg.stats["forwards"], nonfinite=eng.thinker.nonfinite, kv_full=seg.kv_full,
               drafted=seg.stats.get("drafted"), accepted=seg.stats.get("accepted"), res_used=seg.stats.get("res_used"))
    for key in ("src_text", "transcript_text"):
        if key in m:
            res["src_text_equal"] = res["src_text"] == m[key]
            break
    for key in ("tr_text", "translation_text"):
        if key in m:
            res["tr_text_equal"] = res["tr_text"] == m[key]
            break
    if final and m.get("_final") is not None:
        ids, fms = seg.final_job(lookahead=0).run()
        ref_f = list(m["_final"])
        la = (getattr(eng, "final_lookahead", 0) or 11) if hasattr(eng, "final_lookahead") else 0
        if la:                                                      # lookahead 걸음도 같은 id 인지 (엔진 기본이 0 인 fp16 도 11 로 잰다)
            job = seg.final_job(lookahead=la)
            ids_la, fms_la = job.run()
            res["final_lookahead"] = dict(identical=ids_la == ref_f, lookahead=la, ms=round(fms_la, 1), steps=job.steps)
        fd = next((i for i, (a, b) in enumerate(zip(ref_f, ids)) if a != b), None)
        if fd is None and len(ref_f) != len(ids):
            fd = min(len(ref_f), len(ids))
        fmg = m.get("final_margins")
        res["final"] = dict(identical=ids == ref_f, n=len(ids), n_ref=len(ref_f), edit=edit_distance(ids, ref_f), ms=round(fms, 1),
                            first_diff=fd, ref_margin_at_diff=(fmg[fd] if fd is not None and fmg and fd < len(fmg) else None))
    return res


def part1(args, cases) -> dict:
    from engine import Engine
    out = {}
    for prec in args.precisions:
        t0 = time.perf_counter()
        eng = Engine(args.model, kv_capacity=args.kv, precision=prec, encoder=False, strict=not args.loose, draft="q4")
        print(f"[{prec}] 적재 {time.perf_counter() - t0:.1f}s", flush=True)
        for m in cases:
            feats = np.load(os.path.join(m["_dir"], "feats.npy")).astype(np.float32)
            r = dict(prefix_ok=list(m["prefix_ids"]) == eng.tokens.prefix_ids(m["src"], m.get("tgt"), int(m["_delay"])),
                     total_cap_hit=bool(m.get("total_cap_hit")))
            r["teacher"] = teacher_forced(eng, m, feats)
            r["seq"] = free_running(eng, m, feats, False, final=not args.no_final)
            if not args.no_spec:
                r["spec"] = free_running(eng, m, feats, True, final=False)
            if not args.no_spec and getattr(eng, "draft", None) is not None:
                r["fast"] = free_running(eng, m, feats, True, final=False, draft=True)
            out.setdefault(m["_name"], {})[prec] = r
            tf, sq, spc = r["teacher"], r["seq"], r.get("spec", {})
            print(f"  {m['_name']:28s} {prec:10s} TF {tf.get('compared', 0) - tf.get('disagree', 0)}/{tf.get('compared', 0)}"
                  f" | seq {'=' if sq['identical'] else '≠'} (첫 분기 {sq['first_diff']}, 편집 {sq['edit']}, {sq['ms']} ms)"
                  f" | spec {'=' if spc.get('identical') else '≠'} (편집 {spc.get('edit')}, {spc.get('ms')} ms)"
                  + (f" | fast {'=' if r['fast']['identical'] else '≠'} (편집 {r['fast']['edit']}, {r['fast']['ms']} ms)" if "fast" in r else "")
                  + (f" | final {'=' if sq['final']['identical'] else '≠'} (편집 {sq['final']['edit']})" if "final" in sq else "")
                  + (f" la {'=' if sq['final_lookahead']['identical'] else '≠'}" if "final_lookahead" in sq else "")
                  + (" | 참조 total_cap" if r["total_cap_hit"] else ""), flush=True)
        eng.close()
        del eng
        gc.collect()
    return out


def packet_sizes(n: int, mode: str, seed: int) -> list[int]:
    """p20 = 20 ms 고정, rand = 1–24000 샘플 들쭉날쭉 (케이스마다 범위를 바꿈), oneshot = 한 번에."""
    if mode == "oneshot":
        return [n]
    if mode == "p20":
        return [320] * (n // 320) + ([n % 320] if n % 320 else [])
    rng = np.random.default_rng(seed)
    lo, hi = [(1, 400), (64, 2048), (2048, 24000)][seed % 3]
    out, s = [], 0
    while s < n:
        x = int(min(rng.integers(lo, hi + 1), n - s))
        out.append(x)
        s += x
    return out


def part2(args, cases) -> dict:
    import mlx.core as mx
    from engine import Engine
    try:
        import engine.encoder  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"인코더 모듈을 불러올 수 없어 (2) 를 건너뜁니다: {e}")
        return {"skipped": str(e)}
    out = {}
    for prec in args.precisions:
        eng = Engine(args.model, kv_capacity=args.kv, precision=prec, encoder_precision=args.encoder_precision,
                     draft="off" if args.no_draft else "q4")
        for ci, m in enumerate(cases):
            wp = os.path.join(m["_dir"], "wav.npy")
            if not os.path.exists(wp):
                continue
            wav = np.load(wp).astype(np.float32)
            n_total = int(m.get("_n_total") or len(wav))
            n_real = int(m.get("n_real") or (n_total - 16000))
            seg = eng.new_segment(m["src"], m.get("tgt"), int(m["_delay"]))
            rounds, i, late = [], 0, 0
            t0 = time.perf_counter()
            for p in packet_sizes(n_real, args.packets, ci):        # 20 ms / 무작위(마이크처럼) / 한 번에
                rs = seg.feed(wav[i:i + p])
                i += p
                late += sum(r.need_samples > seg.samples_fed for r in rs)
                rounds += rs
            rounds += seg.finish(n_total - n_real)
            ms = (time.perf_counter() - t0) * 1000
            ref = [tuple(map(int, x)) for x in m["emitted"]]
            gf = np.array(seg.features())
            rf = np.load(os.path.join(m["_dir"], "feats.npy"))
            n = min(len(gf), len(rf))
            K = seg.K
            r = dict(K=(K, m["K"]), identical=(list(seg.out) == ref and seg.forced == m["forced"] and seg.flush_rounds == m["flush_rounds"]),
                     edit=edit_distance([t for _, t in ref], [t for _, t in seg.out]), tokens=(len(seg.out), len(ref)),
                     feat_max_abs=float(np.abs(gf[:n] - rf[:n]).max()) if n else None, ms=round(ms, 1),
                     rtf=round(ms / 1000 / (n_total / 16000), 3), packets=args.packets,
                     # 라운드 불변식: 프레임마다 한 라운드(0..K-1 순서), flush 는 k ≥ K·is_tail, 필요 샘플은 이미 도착
                     rounds_ok=([x.k for x in rounds if not x.is_flush] == list(range(K))
                                and all(x.k >= K and x.is_tail for x in rounds if x.is_flush) and late == 0),
                     tail_raised=seg.stats["tail_raised"], overrun=seg.stats["overrun"], k_mismatch=seg.stats["k_mismatch"])
            if not args.no_final and m.get("_final") is not None:
                ids, fms = seg.final_redecode()                     # 엔진 기본 lookahead
                r["final"] = dict(identical=ids == list(m["_final"]), edit=edit_distance(ids, list(m["_final"])), ms=round(fms, 1),
                                  lookahead=getattr(eng, "final_lookahead", 0))
            out.setdefault(m["_name"], {})[prec] = r
            print(f"  {m['_name']:28s} {prec:10s} K {r['K']} tokens {'=' if r['identical'] else '≠'} (편집 {r['edit']})"
                  f" feat max|Δ| {r['feat_max_abs']:.2e} RTF {r['rtf']} rounds_ok {r['rounds_ok']}"
                  + (f" | final {'=' if r['final']['identical'] else '≠'} (편집 {r['final']['edit']}, {r['final']['ms']} ms)" if "final" in r else ""),
                  flush=True)
        eng.close()
        del eng
        gc.collect()
        mx.clear_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=GOLD)
    ap.add_argument("--model", default=None)
    ap.add_argument("--cases", default="")
    ap.add_argument("--part", default="1", choices=["1", "2", "all"])
    ap.add_argument("--precisions", default="fp32,fp16,fp16-plain,bf16,q8")
    ap.add_argument("--encoder-precision", default="fp32")
    ap.add_argument("--packets", default="p20", choices=["p20", "rand", "oneshot"], help="(2) 의 feed 패킷 크기")
    ap.add_argument("--kv", type=int, default=2048)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--no-final", action="store_true")
    ap.add_argument("--no-draft", action="store_true", help="(2) 에서 q4 초안 끄기 (기본: 켬)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--loose", action="store_true", help="s4 모양 검사 끄기 (작은 가짜 체크포인트 스모크 테스트용)")
    args = ap.parse_args()
    args.precisions = [p for p in args.precisions.split(",") if p]
    names = [c for c in args.cases.split(",") if c]
    if not names:
        idx = os.path.join(args.gold, "index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                ix = json.load(f)
            names = [c if isinstance(c, str) else c.get("name", c.get("case")) for c in (ix.get("cases", ix) if isinstance(ix, dict) else ix)]
        else:
            names = sorted(n for n in os.listdir(args.gold) if os.path.exists(os.path.join(args.gold, n, "meta.json")))
    cases = [load_case(os.path.join(args.gold, n)) for n in names]
    print(f"골든 {args.gold}: {len(cases)} 케이스, 정밀도 {args.precisions}", flush=True)
    res = dict(gold=args.gold, cases=names, precisions=args.precisions, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    if args.part in ("1", "all"):
        res["part1"] = part1(args, cases)
    if args.part in ("2", "all"):
        res["part2"] = part2(args, cases)
    res["summary"] = summarize(res)
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    out = args.out or os.path.join(OUT_DIR, f"verify-part{args.part}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print("JSON:", out)


def summarize(res: dict) -> dict:
    s = {}
    for prec in res["precisions"]:
        rows = [v[prec] for v in res.get("part1", {}).values() if prec in v]
        if rows:
            tf = [r["teacher"] for r in rows if "compared" in r["teacher"]]
            s[f"part1/{prec}"] = dict(
                cases=len(rows),
                teacher_agree=f"{sum(t['compared'] - t['disagree'] for t in tf)}/{sum(t['compared'] for t in tf)}",
                seq_identical=f"{sum(r['seq']['identical'] for r in rows)}/{len(rows)}",
                spec_identical=f"{sum(r.get('spec', {}).get('identical', False) for r in rows)}/{len(rows)}",
                fast_identical=f"{sum(r.get('fast', {}).get('identical', False) for r in rows)}/{sum('fast' in r for r in rows)}",
                final_lookahead_identical=f"{sum(r['seq'].get('final_lookahead', {}).get('identical', False) for r in rows)}/"
                                          f"{sum('final_lookahead' in r['seq'] for r in rows)}",
                final_identical=f"{sum(r['seq'].get('final', {}).get('identical', False) for r in rows)}/{sum('final' in r['seq'] for r in rows)}",
                nonfinite=max(r["seq"]["nonfinite"] for r in rows))
        rows2 = [v[prec] for k, v in res.get("part2", {}).items() if isinstance(v, dict) and prec in v]
        if rows2:
            s[f"part2/{prec}"] = dict(cases=len(rows2), identical=f"{sum(r['identical'] for r in rows2)}/{len(rows2)}",
                                      feat_max_abs=max(r["feat_max_abs"] or 0 for r in rows2),
                                      rounds_ok=f"{sum(r.get('rounds_ok', False) for r in rows2)}/{len(rows2)}",
                                      final_identical=f"{sum(r['final']['identical'] for r in rows2 if 'final' in r)}/"
                                                      f"{sum('final' in r for r in rows2)}")
    return s


if __name__ == "__main__":
    main()
