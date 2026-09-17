#!/usr/bin/env python
"""세션 단위 held-out split v1 (D1c 부터 학습 제외; [[decision-phase2-eval-plan-v2]]).
  python experiments/p2_make_split.py --data <phase2 dir> --out <phase2 dir>/splits/v1.json
규칙: AMI 공식 ASR test 회의(EN2002·ES2004·ES2014·IS1009·TS3003 a–d), ICSI 표준 eval(Bed004·Bed009·Bed016·Bmr005·Bmr019·Bro018), NOTSOFAR dev 서브셋,
      71631·134-1·134-2 는 conv_id 에서 화자 id 를 뽑아 화자 5 % 를 고르고 그 화자가 낀 대화 전부(화자 누설 방지), otoSpeech 는 meta 의 배우 id 5 %(없으면 conv 5 %)."""
import os, sys, json, re, hashlib, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--frac", type=float, default=0.05); a = ap.parse_args()
AMI_TEST = {f"{p}{s}" for p in ("EN2002", "ES2004", "ES2014", "IS1009", "TS3003") for s in "abcd"}; ICSI_EVAL = {"Bed004", "Bed009", "Bed016", "Bmr005", "Bmr019", "Bro018"}
def h01(s): return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
def rows(corpus):
    p = os.path.join(a.data, f"{corpus}.dialogues.jsonl")
    for l in open(p, encoding="utf-8"):
        if l.strip(): d = json.loads(l); yield d["conv_id"], d
out = {}
for corpus in ("aihub71631", "aihub134-1", "aihub134-2", "otoSpeech", "ami", "notsofar", "icsi"):
    p = os.path.join(a.data, f"{corpus}.dialogues.jsonl")
    if not os.path.exists(p): print("skip", corpus); continue
    held = []; rule = ""; total = 0; hours_h = 0.0
    if corpus == "ami": rule = "AMI 공식 ASR test 20 회의"
    elif corpus == "icsi": rule = "ICSI 표준 eval 6 회의"
    elif corpus == "notsofar": rule = "NOTSOFAR-1 dev 서브셋"
    spk_of = {}
    for cid, d in rows(corpus):
        total += 1; key = cid.split(":", 1)[-1]
        if corpus == "ami": hit = key in AMI_TEST
        elif corpus == "icsi": hit = key in ICSI_EVAL
        elif corpus == "notsofar": hit = d.get("split") == "dev"
        elif corpus.startswith("aihub"):
            ids = re.findall(r"(?:^|_)(\d{3,6})G\d", key) or re.findall(r"\d{4,}", key); spk_of[cid] = ids; hit = None
        else:
            actors = [str(x) for x in (d.get("meta", {}).get("actors") or d.get("meta", {}).get("actor_ids") or [])]; hit = any(h01("oto:" + x) < a.frac for x in actors) if actors else h01(cid) < a.frac
        if hit: held.append(cid); hours_h += d["duration_s"] / 3600
    if corpus.startswith("aihub"):
        speakers = sorted({s for v in spk_of.values() for s in v}); held_spk = {s for s in speakers if h01(f"{corpus}:{s}") < a.frac}
        for cid, d in rows(corpus):
            if any(s in held_spk for s in spk_of.get(cid, [])) or (not spk_of.get(cid) and h01(cid) < a.frac): held.append(cid); hours_h += d["duration_s"] / 3600
        rule = f"화자 id {len(held_spk)}/{len(speakers)} ({a.frac:.0%}) 가 낀 대화 전부"
    elif corpus == "otoSpeech": rule = f"배우 id {a.frac:.0%} (없으면 conv {a.frac:.0%})"
    out[corpus] = dict(rule=rule, n_heldout=len(held), n_total=total, hours_heldout=round(hours_h, 1), heldout=sorted(held)); print(corpus, rule, f"{len(held)}/{total}", f"{hours_h:.1f} h")
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(dict(version="v1", frac=a.frac, corpora=out), open(a.out, "w"), ensure_ascii=False, indent=1); print("→", a.out)
