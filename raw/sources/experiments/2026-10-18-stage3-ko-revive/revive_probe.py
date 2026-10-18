# 격리 발화 되살리기 로더 경로 점검(2026-10-18) — stage3_train 과 같은 창 규칙(20–40 s, hop 10, grid, 맞장구 1.0·3, held-out 제외)으로
# CstDialogueDataset 을 끈 것·켠 것(revive)으로 만들어, 같은 창(코퍼스마다 300 개, δ10)의 출력 label 마스크 비율·텍스트에 나타난 교대 태그와
# 그중 학습되는 것·<EOT> 결정·전사 토큰 수를 비교한다. 결과 파일에는 숫자만 쓴다.
#   python revive_probe.py <out.json> <corpus,…> [--show]   (--show: 되살린 구간 하나를 [m] 표시와 함께 표준오류로 — 확인용, 저장하지 않는다)
import json, random, sys, time
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.hf import load_tokenizer
from vapasr.data.cst_dialogue import CstDialogueDataset, load_revive

P = "/soundai/users/tskim/VAPKT-data/data/phase2"; RV = P + "/revive-asr-v1"; N = 300; DL = 10
tok = load_tokenizer("/soundai/Model/VAPASR/stage3-st3-p10-sim/checkpoint-3000"); split = json.load(open(P + "/splits/v2.json")); rep = {}
for c in sys.argv[2].split(","):
    t0 = time.time(); held = set(split["corpora"].get(c, {}).get("heldout", []))
    kw = dict(delays=(DL,), exclude=held, window_s=(20.0, 40.0), hop_s=10.0, seed=0, backchannel=(1.0, 3)); f = [f"{P}/{c}.refined.dialogues.jsonl"]
    off = CstDialogueDataset(f, tok, **kw); on = CstDialogueDataset(f, tok, revive=load_revive([f"{RV}/{c}.revive.jsonl"]), **kw)
    assert off.items == on.items, c
    idx = random.Random(0).sample(range(len(on)), min(N, len(on))); bad = lex = 0; shown = "--show" not in sys.argv
    agg = {k: dict(out=0, masked=0, tags=0, tags_sup=0, eot_sup=0, text_sup=0) for k in ("off", "on")}
    for i in idx:
        for k, ds in (("off", off), ("on", on)):
            s = ds.cst_sequence(i, DL); a = agg[k]; outs = [p for p in range(len(s["ids"])) if not s["is_input"][p]]; tags = [p for p in outs if s["ids"][p] in ds.spk_ids]
            a["out"] += len(outs); a["masked"] += sum(s["labels"][p] == -100 for p in outs)
            a["tags"] += max(0, len(tags) - 1); a["tags_sup"] += sum(s["labels"][p] != -100 for p in tags[1:])          # 첫 태그 뒤 = 교대
            a["eot_sup"] += sum(s["labels"][p] != -100 for p in s["eot_pos"])
            a["text_sup"] += sum(1 for p in outs if s["src_at"][p] is not None and s["ids"][p] != ds.eot_id and s["labels"][p] != -100)
            if k == "on":
                bad += bool(ds.problems_of(s)); lex += len(s["lex"])
                if not shown and s["lex"]:
                    p0 = min(p for p, q in enumerate(s["src_at"]) if q is not None and (q[0], q[1]) in s["lex"]); shown = True
                    show = lambda p: (tok.convert_ids_to_tokens(s["ids"][p]) if s["ids"][p] >= 151643 else tok.decode([s["ids"][p]])) + ("[m]" if s["labels"][p] == -100 else "")
                    print(f"[예시 {c} {s['cid']}@{s['t0']}] " + " ".join(show(p) for p in range(max(0, p0 - 40), min(len(s["ids"]), p0 + 80)) if not s["is_input"][p]), file=sys.stderr)
    n = len(idx); r = dict(windows=n, revived_utts=on.stats.get("cst_revived"), revive_missing=on.stats.get("cst_revive_missing"), problems=bad, lex_per_win=round(lex / n, 1))
    for k, a in agg.items():
        r.update({f"{k}_masked_pct": round(100 * a["masked"] / a["out"], 1), f"{k}_tag_changes": round(a["tags"] / n, 2), f"{k}_tag_changes_sup": round(a["tags_sup"] / n, 2),
                  f"{k}_eot_sup": round(a["eot_sup"] / n, 2), f"{k}_text_sup": round(a["text_sup"] / n, 1)})
    rep[c] = dict(r, sec=round(time.time() - t0)); print(json.dumps({c: rep[c]}, ensure_ascii=False), flush=True)
json.dump(rep, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
