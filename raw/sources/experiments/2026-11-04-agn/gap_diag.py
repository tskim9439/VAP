#!/usr/bin/env python
"""외부 기준과의 격차가 어디서 오는가(2026-11-05, 사용자 "외부 기준과의 격차가 어디서 오는지 진단을 먼전 수행해보자").
우리 시스템과 외부 기준(Multitalker Parakeet + N3D 등 baseline_ami_ext 행)의 AMI 회의 통째 원출력을 같은 참조·같은 셈으로 나눠 본다. 디코드 없음, 읽기만.

  python gap_diag.py --data <ami-bench-v1> --tok <토크나이저> --sys 이름=ours:<결과 폴더> --sys 이름=ext:<결과 폴더> … [--subset eval] [--json out]

참조: window_ref(δ10, 전사 없는 구간 마스크 뒤) 단어를 agn_diag.build_ref 그대로 — 채움말(hmm·mm·mhm·mmm·uh·um) 빼고 소문자·구두점 제거, 겹침 = 그 프레임에
  말하는 참조 화자 수(episode 구간, 자기 포함) 1 · 2 · 3+.
가설: ours = 행 emitted 를 cst_stage3_eval 과 같은 규칙(parse_emitted·마스크·words_of)으로, 시각 = 방출 청크 끝 − δ·80 ms − 40 ms.
  ext = 행 hyp_words(화자, 단어, 회의 시각 — 외부 시스템 구간 안 고른 근사)에서 baseline_ami_ext 와 같은 마스크(int(t/80 ms) + δ 가 마스크 청크면 뺀다).
두 셈(단어 시각은 짝짓기에 쓰지 않는다 — 외부 기준의 시각 근사에 흔들리지 않게)
  cp  : 참조 화자 ↔ 가설 화자 짝(편집 거리 합 최소, 헝가리안) 뒤 화자마다 편집 거리 정렬 → 참조 단어마다 맞음 · 대치 · 삭제. 짝 없는 가설 화자의 말은 삽입.
  orc : meeteval tcORC-WER(collar 5 s)의 배정(참조 발화 → 가설 화자, 화자 무시)을 받아, 가설 화자마다 배정된 참조 발화 단어를 시각순으로 이어 편집 거리 정렬.
        참조 발화 = 그 화자의 AMI 발화 구간으로 묶은 참조 단어, 가설 구간 = 같은 화자에서 2 s 넘게 끊기지 않는 단어 묶음(qlt_tc 와 같다).
  놓침 = 대치 + 삭제(참조 단어 몫). cp 놓침 − orc 놓침 ≈ 화자 귀속 탓(맞게 받아썼지만 다른 화자 칸에 쓴 것), orc 놓침 = 받아쓰기 탓.
--relabel B,F(ours): 행 act_runs(디코드가 받은 활동)로 출력 재귀속(cst_stage3_eval --relabel-follow 와 같은 follow_stats)을 얹어 잰다.
"""
import argparse, bisect, collections, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, "experiments")]
CHUNK = 0.08


def align_status(ref, hyp):
    """ref·hyp 단어 목록 → (참조 단어마다 'ok'|'sub'|'del', 삽입 수). 편집 거리 정렬(대치·삭제·삽입 비용 1)."""
    from vapasr.data.cst_eval import align
    st = [None] * len(ref); ins = 0
    for i, j in align(ref, hyp):
        if i is None: ins += 1
        elif j is None: st[i] = "del"
        else: st[i] = "ok" if ref[i] == hyp[j] else "sub"
    return st, ins


def runs(words, gap=2.0):
    """[(단어, 시각)] → [(시작, 끝, [단어])] — gap 넘게 끊기지 않는 묶음."""
    out = []
    for w, t in sorted(words, key=lambda x: x[1]):
        if out and t - out[-1][1] <= gap: out[-1][1] = t; out[-1][2].append(w)
        else: out.append([t, t, [w]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--tok", required=True); ap.add_argument("--sys", action="append", required=True)
    ap.add_argument("--subset", default="eval"); ap.add_argument("--corpora", default="ami-ihmmix,ami-sdm"); ap.add_argument("--delta", type=int, default=10)
    ap.add_argument("--collar", type=float, default=5.0); ap.add_argument("--json", default=None)
    ap.add_argument("--relabel", type=lambda x: tuple(int(v) for v in x.split(",")), default=None, help="ours: 출력 재귀속 창 B,F(예 3,1 — 행 act_runs 활동)")
    a = ap.parse_args()
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from transformers import AutoTokenizer
    from meeteval.io import SegLST
    from meeteval.wer.api import tcorcwer
    import cst_stage3_eval as E
    from cst_stage3_eval import windows_of, words_of
    from vapasr.data.cst_eval import parse_emitted, ed, follow_stats, act_from_runs
    from agn_diag import build_ref, keep
    tok = AutoTokenizer.from_pretrained(a.tok); EOT = tok.convert_tokens_to_ids("<EOT>")
    cfg = json.load(open(os.path.join(a.tok, "config.json")))
    spk_ids = tuple(tok.convert_tokens_to_ids(t) for t in cfg["slot_tokens"]); BEGIN = tok.convert_tokens_to_ids("<ONSET>") if cfg.get("slot_blocks") else None
    ns = argparse.Namespace(data=a.data, split=None, corpora=a.corpora, deltas=[a.delta], limit_windows=0, backchannel=(1.0, 3), window="full", hop=10.0, max_speakers=4,
                            window_draws=0, declare=4, xl_order="lang", subset=a.subset)
    sets, items = windows_of(ns, tok); where = {sets[c].items[i][0]: (c, i) for c, i in items}
    refs = {}
    for cid, (c, i) in where.items():
        ds = sets[c]; d = ds.dlgs[cid]; R, _, _, mask, _ = build_ref(E, ds, i, a.delta, EOT, tok)
        _, _, _, sp_order = E.window_ref(ds, i, a.delta, EOT, tok)
        utt = {}
        for k, sp in enumerate(sp_order): utt[k] = sorted((u.start, u.end) for u in d.utterances if u.speaker == sp)
        for q in R:                                                                              # 참조 단어 → 그 화자의 AMI 발화(orc 참조 구간)
            ut = utt.get(q["s"]) or [(q["t"], q["t"])]; st = [x[0] for x in ut]; j = max(0, bisect.bisect_right(st, q["t"] + 0.2) - 1)
            if q["t"] > ut[j][1] + 0.3 and j + 1 < len(ut) and abs(ut[j + 1][0] - q["t"]) < q["t"] - ut[j][1]: j += 1
            q["utt"] = (q["s"], j, ut[j][0], ut[j][1])
        refs[cid] = (R, mask)
    out = {}
    for spec in a.sys:
        name, _, rest = spec.partition("="); kind, _, path = rest.partition(":")
        rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(path, "predictions-rank*.jsonl"))) for l in open(f, encoding="utf-8")]
        rows = [r for r in rows if r["delta"] == a.delta and r["id"].split("@")[0] in where]
        print(f"== {name}({kind}) {path} · 행 {len(rows)}", flush=True)
        agg = collections.defaultdict(lambda: collections.Counter())
        for r in rows:
            cid = r["id"].split("@")[0]; c = r["corpus"]; R, mask = refs[cid]
            if kind == "ours":
                slots, _ = parse_emitted(r["emitted"], spk_ids, EOT, drop_chunks=mask, begin_id=BEGIN)
                ws_ = [words_of(tok, toks) for toks in slots]
                if a.relabel:                                                                   # 출력 재귀속(cst_stage3_eval --relabel-follow 와 같은 follow_stats — 활동을 어긴 단어만 활동 슬롯으로)
                    c_, i_ = where[cid]; ds_ = sets[c_]; K_ = int(r["K"]); fm_ = np.ones(K_, bool)
                    for s_, e_ in ds_._bad_spans(ds_.dlgs[cid]):
                        if e_ > 0 and s_ < r["L"]: fm_[max(0, int(s_ / CHUNK)): min(K_, int(np.ceil(e_ / CHUNK)))] = False
                    _, ws_ = follow_stats(ws_, act_from_runs(r["act_runs"], K_), a.delta, frame_mask=fm_, back=a.relabel[0], fwd=a.relabel[1])
                H = {f"h{k}": [(w, te - a.delta * CHUNK - 0.04) for w, te in ws] for k, ws in enumerate(ws_)}
            else:
                H = collections.defaultdict(list)
                for sp, w, t in r["hyp_words"]:
                    if int(t / CHUNK) + a.delta not in mask: H[sp].append((w, t))
            H = {k: [(keep(w), t) for w, t in v if keep(w)] for k, v in H.items()}; H = {k: v for k, v in H.items() if v}
            hk = sorted(H); n_s = max(R, key=lambda q: q["s"])["s"] + 1 if R else 0
            Rs = [[q for q in sorted(R, key=lambda q: q["t"]) if q["s"] == s] for s in range(n_s)]
            # cp: 화자 짝 → 정렬
            n = max(len(Rs), len(hk)); C = np.zeros((n, n))
            hw = [[w for w, _ in sorted(H[k], key=lambda x: x[1])] for k in hk]
            for s in range(n):
                for h in range(n): C[s, h] = ed([q["w"] for q in Rs[s]] if s < len(Rs) else [], hw[h] if h < len(hw) else [])
            ri, hi = linear_sum_assignment(C); ins_cp = 0
            for s, h in zip(ri, hi):
                rr = Rs[s] if s < len(Rs) else []; st, ins = align_status([q["w"] for q in rr], hw[h] if h < len(hw) else []); ins_cp += ins
                for q, x in zip(rr, st): q["cp"] = x
            # orc: meeteval tcORC 배정(참조 발화 → 가설 화자)
            segs = collections.defaultdict(list)
            for q in R: segs[q["utt"]].append(q)
            rk = sorted(segs, key=lambda u: (u[2], u[0], u[1]))
            ref_l = SegLST([dict(session_id="x", speaker=f"r{u[0]}", start_time=float(u[2]), end_time=float(max(u[3], u[2] + 0.01)),
                                 words=" ".join(q["w"] for q in sorted(segs[u], key=lambda q: q["t"]))) for u in rk])
            hyp_l = SegLST([dict(session_id="x", speaker=k, start_time=float(s0), end_time=float(max(s1, s0 + 0.01)), words=" ".join(ws))
                            for k in hk for s0, s1, ws in runs(H[k])])
            res = tcorcwer(ref_l, hyp_l, collar=a.collar)["x"]; asg = list(res.assignment)
            assert len(asg) == len(rk), (len(asg), len(rk))
            by = collections.defaultdict(list)
            for u, hspk in zip(rk, asg): by[hspk] += sorted(segs[u], key=lambda q: q["t"])
            ins_orc = 0; used = set()
            for hspk, qs in by.items():
                st, ins = align_status([q["w"] for q in qs], [w for w, _ in sorted(H.get(hspk, []), key=lambda x: x[1])]); ins_orc += ins; used.add(hspk)
                for q, x in zip(qs, st): q["orc"] = x
            ins_orc += sum(len(H[k]) for k in hk if k not in used)
            g = agg[c]; g["ins_cp"] += ins_cp; g["ins_orc"] += ins_orc; g["meeteval_orc_err"] += res.errors; g["meeteval_orc_len"] += res.length
            for q in R:
                ov = "1" if q["ov"] <= 1 else "2" if q["ov"] == 2 else "3+"
                for key in (ov, "all"):
                    g[f"n|{key}"] += 1; g[f"cp_{q['cp']}|{key}"] += 1; g[f"orc_{q['orc']}|{key}"] += 1
        res_s = {}
        for c in sorted(agg):
            g = agg[c]; N = g["n|all"]; d_ = {}
            for key in ("1", "2", "3+", "all"):
                n_ = max(1, g[f"n|{key}"]); cpm = 100 * (g[f"cp_sub|{key}"] + g[f"cp_del|{key}"]) / n_; om = 100 * (g[f"orc_sub|{key}"] + g[f"orc_del|{key}"]) / n_
                d_[key] = dict(n=g[f"n|{key}"], share=round(100 * g[f"n|{key}"] / max(1, N), 1), cp_miss=round(cpm, 2), orc_miss=round(om, 2), attr=round(cpm - om, 2),
                               orc_del=round(100 * g[f"orc_del|{key}"] / n_, 2), orc_sub=round(100 * g[f"orc_sub|{key}"] / n_, 2))
            d_["cp_wer"] = round(100 * (g["cp_sub|all"] + g["cp_del|all"] + g["ins_cp"]) / max(1, N), 2)
            d_["orc_wer"] = round(100 * (g["orc_sub|all"] + g["orc_del|all"] + g["ins_orc"]) / max(1, N), 2)
            d_["ins_cp"] = round(100 * g["ins_cp"] / max(1, N), 2); d_["ins_orc"] = round(100 * g["ins_orc"] / max(1, N), 2)
            d_["meeteval_tcorc"] = round(100 * g["meeteval_orc_err"] / max(1, g["meeteval_orc_len"]), 2)
            res_s[c] = d_
            print(f"  {c:11s} cp {d_['cp_wer']:6.2f} · orc {d_['orc_wer']:6.2f}(meeteval tcORC {d_['meeteval_tcorc']:.2f}) · 삽입 cp {d_['ins_cp']:.2f} orc {d_['ins_orc']:.2f}", flush=True)
            for key in ("1", "2", "3+", "all"):
                x = d_[key]
                print(f"    겹침 {key:3s} 몫 {x['share']:5.1f} % · 놓침 cp {x['cp_miss']:6.2f} = 받아쓰기(orc) {x['orc_miss']:6.2f}(삭제 {x['orc_del']:.2f} 대치 {x['orc_sub']:.2f}) + 귀속 {x['attr']:5.2f}", flush=True)
        out[name] = res_s
    if a.json: json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)
    # 격차 분해: 첫 시스템 − 둘째 시스템, 참조 단어 몫 가중(놓침 차이 × 몫 = 전체 놓침 차이에 대한 기여)
    names = list(out)
    if len(names) >= 2:
        A_, B_ = out[names[0]], out[names[1]]
        print(f"== 격차 {names[0]} − {names[1]}(참조 단어 100 개당 — 겹침 몫 × 놓침 차이)", flush=True)
        for c in A_:
            if c not in B_: continue
            parts = []
            for key in ("1", "2", "3+"):
                sh = A_[c][key]["share"] / 100
                parts.append(f"겹침 {key}: 받아쓰기 {sh * (A_[c][key]['orc_miss'] - B_[c][key]['orc_miss']):+.2f} · 귀속 {sh * (A_[c][key]['attr'] - B_[c][key]['attr']):+.2f}")
            print(f"  {c:11s} cp {A_[c]['cp_wer'] - B_[c]['cp_wer']:+.2f} = 놓침 {A_[c]['all']['cp_miss'] - B_[c]['all']['cp_miss']:+.2f} + 삽입 {A_[c]['ins_cp'] - B_[c]['ins_cp']:+.2f} | " + " | ".join(parts), flush=True)


if __name__ == "__main__":
    main()
