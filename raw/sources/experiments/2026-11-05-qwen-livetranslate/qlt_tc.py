#!/usr/bin/env python
"""시각 제약 화자 귀속·화자 무시 오류율(meeteval tcpWER · tcORC-WER, CHiME-8 방식)을 Qwen 과 우리 모델에 똑같이 잰다(맥 로컬, 2026-11-05).
왜: 우리 agn(화자 무시)은 모든 화자 단어를 시각순으로 이어 비교해 단어 시각에 민감하다. Qwen API 는 단어 시각을 주지 않아(원문 item 구간에 고르게 놓음)
  정답 전사로 만든 로그도 agn 9.5 % 가 나왔다(qlt_selftest oracle-ami, cpWER 0.40 · DER 0.61). tcORC-WER 은 참조 발화를 가설 화자에 통째로 최적 배정하고
  발화 안 순서·시각(collar 안)만 보므로 구간 시각만 있는 시스템에도 공정하다 — 우리 모델에도 같은 셈을 적용해 견준다.

  python qlt_tc.py --kind qwen|ours --pred <결과 폴더(predictions-rank*.jsonl)> --data <ami-bench-v1> --tok <토크나이저> [--subset eval] [--collar 5] [--json <out>]
참조 구간: window_ref(δ — 전사 없는 구간 마스크 뒤 단어, 시각 = 토큰 끝)를 그 화자의 AMI 발화 구간(utterances)으로 묶는다(행 ref_text 와 같은 단어인지 대조).
가설 구간: qwen = 행 hyp_segs(원문 item, 마스크 뒤) · ours = 행 emitted 를 cst_stage3_eval 과 같은 규칙(parse_emitted·마스크·words_of)으로 풀어
  단어 시각 = 방출 청크 끝 − δ·80 ms − 40 ms(오디오 시각 — baseline_ami_ext vapasr-spk 와 같은 환산; 행 hyp_text 와 같은 단어인지 대조),
  같은 슬롯에서 --ours-gap(2 s) 넘게 끊기지 않는 단어를 한 구간으로 묶는다 — 단어마다 정규화하면 숫자 등이 깨져 cpWER 이 0.5 pt 높게 나왔다(24.96 대 24.48).
정규화: 구간마다 Whisper EnglishTextNormalizer(표준). 읽기만 한다."""
import argparse, bisect, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, "experiments")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["qwen", "ours"], required=True); ap.add_argument("--pred", required=True); ap.add_argument("--data", required=True); ap.add_argument("--tok", required=True)
    ap.add_argument("--subset", default="eval"); ap.add_argument("--corpora", default="ami-ihmmix,ami-sdm"); ap.add_argument("--delta", type=int, default=10)
    ap.add_argument("--collar", type=float, default=5.0); ap.add_argument("--json", default=None)
    ap.add_argument("--ours-gap", type=float, default=2.0, help="ours: 같은 슬롯 단어를 한 구간으로 묶는 최대 간격(s)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from whisper_normalizer.english import EnglishTextNormalizer
    from meeteval.io import SegLST
    from meeteval.wer import combine_error_rates
    from meeteval.wer.api import tcpwer, tcorcwer, cpwer
    from cst_stage3_eval import windows_of, window_ref, words_of
    from vapasr.data.cst_eval import parse_emitted
    from vapasr.data.cst_dialogue import SLOTS
    norm = EnglishTextNormalizer(); tok = AutoTokenizer.from_pretrained(a.tok); EOT = tok.convert_tokens_to_ids("<EOT>")
    cfg = json.load(open(os.path.join(a.tok, "config.json"))); slot_toks = cfg.get("slot_tokens") or list(SLOTS[:2])
    spk_ids = tuple(tok.convert_tokens_to_ids(t) for t in slot_toks); BEGIN = tok.convert_tokens_to_ids("<ONSET>") if cfg.get("slot_blocks") else None
    assert not cfg.get("x_slot"), "X 칸(E14) 체크포인트는 아직 — 해석 규칙을 더해야 한다"
    ns = argparse.Namespace(data=a.data, split=None, corpora=a.corpora, deltas=[a.delta], limit_windows=0, backchannel=(1.0, 3), window="full", hop=10.0, max_speakers=4,
                            window_draws=0, declare=4, xl_order="lang", subset=a.subset)
    sets, items = windows_of(ns, tok); where = {sets[c].items[i][0]: (c, i) for c, i in items}
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.pred, "predictions-rank*.jsonl"))) for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if r["delta"] == a.delta and r["id"].split("@")[0] in where]
    R, H, chk = {}, {}, dict(ref_mismatch=0, hyp_mismatch=0, rows=len(rows))
    seg = lambda sid, spk, s, e, txt: dict(session_id=sid, speaker=spk, start_time=float(s), end_time=float(max(e, s + 0.01)), words=norm(txt))
    for r in rows:
        cid = r["id"].split("@")[0]; c, i = where[cid]; ds = sets[c]; d = ds.dlgs[cid]; sid = r["id"]
        ref_w, _, mask, sp_order = window_ref(ds, i, a.delta, EOT, tok)
        chk["ref_mismatch"] += sum(" ".join(w for w, _ in ref_w[k]) != r["ref_text"][k] for k in range(len(ref_w)))
        rs = R.setdefault(r["corpus"], [])
        for k, words in enumerate(ref_w):
            if not words: continue
            sp = sp_order[k] if k < len(sp_order) else None
            ut = sorted((u.start, u.end) for u in d.utterances if u.speaker == sp) or [(words[0][1], words[-1][1])]
            st = [x[0] for x in ut]; groups = {}
            for w, t in words:
                j = max(0, bisect.bisect_right(st, t + 0.2) - 1)                                # t 앞에서 시작한 마지막 발화, 끝을 많이 넘으면 다음 발화와 가까운 쪽
                if t > ut[j][1] + 0.3 and j + 1 < len(ut) and abs(ut[j + 1][0] - t) < t - ut[j][1]: j += 1
                groups.setdefault(j, []).append(w)
            for j, ws in groups.items(): rs.append(seg(sid, f"r{k}", ut[j][0], ut[j][1], " ".join(ws)))
        hs = H.setdefault(r["corpus"], [])
        if a.kind == "qwen":
            for spk, s, e, txt in r["hyp_segs"]: hs.append(seg(sid, spk, s, e, txt))
        else:
            slots, _ = parse_emitted(r["emitted"], spk_ids, EOT, drop_chunks=mask, begin_id=BEGIN)
            for k, toks in enumerate(slots):
                ws = words_of(tok, toks)
                if k < len(r["hyp_text"]): chk["hyp_mismatch"] += " ".join(w for w, _ in ws) != r["hyp_text"][k]
                run = []                                                                        # 같은 슬롯에서 --ours-gap 넘게 끊기지 않는 단어 묶음 = 한 구간(정규화·시각 처리를 Qwen item 과 같게)
                for w, te in ws + [(None, None)]:
                    t = None if te is None else te - a.delta * 0.08 - 0.04
                    if run and (w is None or t - run[-1][1] > a.ours_gap):
                        hs.append(seg(sid, f"h{k}", run[0][1], run[-1][1], " ".join(x for x, _ in run))); run = []
                    if w is not None: run.append((w, t))
    out = dict(check=chk, collar=a.collar, kind=a.kind, pred=os.path.abspath(a.pred))
    print(f"== {a.pred} ({a.kind}) · {a.subset} δ{a.delta} · collar {a.collar} s · 대조 {chk}", flush=True)
    for c in sorted(R):
        ref = SegLST([x for x in R[c] if x["words"].strip()]); hyp = SegLST([x for x in H.get(c, []) if x["words"].strip()])
        res = {}
        for name, fn in (("cpWER", lambda: cpwer(ref, hyp)), ("tcpWER", lambda: tcpwer(ref, hyp, collar=a.collar)), ("tcORC-WER", lambda: tcorcwer(ref, hyp, collar=a.collar))):
            e = combine_error_rates(*fn().values()); res[name] = dict(wer=round(100 * e.error_rate, 2), errors=e.errors, length=e.length)
        out[c] = res
        print(f"  {c:11s} " + " · ".join(f"{k} {v['wer']:6.2f}({v['errors']}/{v['length']})" for k, v in res.items()), flush=True)
    if a.json: json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
