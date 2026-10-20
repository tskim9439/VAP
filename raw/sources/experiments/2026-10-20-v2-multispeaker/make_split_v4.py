# split v4(V2 학습용): v3 에서 회의 코퍼스의 공식 평가 분할을 모두 held-out 으로 옮긴다 — 학습이 공식 dev·eval 을 보지 않게(2026-10-20 사용자 질문:
# "AMI-IHM 과 AMI-SDM 으로도 평가 진행하려는 데, 학습 셋과 겹치지는 않지?").
#   AMI: v2 held-out 20(공식 eval 16 + ES2014a–d) + 공식 dev 18(full-corpus ASR 분할: ES2011a–d · IB4001–4004·4010·4011 · IS1008a–d · TS3004a–d)
#   NOTSOFAR: 원본 경로가 dev_set(36, v2 held-out 그대로)·eval_set(129, 지금까지 학습 쪽)인 회의 전부 → 학습은 train_set(72)만
#   나머지 코퍼스는 v3 그대로(V1 평가 창과 같다). 출력 파일이 이미 있으면 덮지 않는다.
#   python3 make_split_v4.py <phase2 폴더>
import json, os, sys
P2 = sys.argv[1]; src = os.path.join(P2, "splits", "v3.json"); dst = os.path.join(P2, "splits", "v4.json")
assert not os.path.exists(dst), f"{dst} 가 이미 있다 — 덮지 않는다"
s = json.load(open(src, encoding="utf-8"))
AMI_DEV = ["ES2011a", "ES2011b", "ES2011c", "ES2011d", "IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011",
           "IS1008a", "IS1008b", "IS1008c", "IS1008d", "TS3004a", "TS3004b", "TS3004c", "TS3004d"]
AMI_EVAL = ["EN2002a", "EN2002b", "EN2002c", "EN2002d", "ES2004a", "ES2004b", "ES2004c", "ES2004d",
            "IS1009a", "IS1009b", "IS1009c", "IS1009d", "TS3003a", "TS3003b", "TS3003c", "TS3003d"]
dur, nsf_split = {}, {}
for c in ("ami", "notsofar"):
    for l in open(os.path.join(P2, f"{c}.refined.dialogues.jsonl"), encoding="utf-8"):
        d = json.loads(l); dur[d["conv_id"]] = d["duration_s"]
        if c == "notsofar":
            ch = next(iter(d["channels"].values())); p = ch.get("path") or ch["pieces"][0][0]
            nsf_split[d["conv_id"]] = "dev" if "/dev_set/" in p else "eval" if "/eval_set/" in p else "train" if "/train_set/" in p else "?"
assert "?" not in nsf_split.values(), "NOTSOFAR 원본 분할을 모르는 회의가 있다"
ami = s["corpora"]["ami"]; held = set(ami["heldout"]) | {f"ami:{m}" for m in AMI_DEV}
assert {f"ami:{m}" for m in AMI_EVAL} <= held and all(f"ami:{m}" in dur for m in AMI_DEV)
nsf = s["corpora"]["notsofar"]; nheld = set(nsf["heldout"]); assert nheld == {k for k, v in nsf_split.items() if v == "dev"}, "v3 NOTSOFAR held-out 이 dev_set 과 다르다"
nheld |= {k for k, v in nsf_split.items() if v == "eval"}
h = lambda ids: round(sum(dur[i] for i in ids) / 3600, 2)
s["corpora"]["ami"] = dict(ami, heldout=sorted(held), n_heldout=len(held), hours_heldout=h(held),
                           rule=ami.get("rule", "") + " | v4: + 공식 dev 18(full-corpus ASR) — 공식 eval 16 은 v2 부터 held-out",
                           official={"eval": [f"ami:{m}" for m in AMI_EVAL], "dev": [f"ami:{m}" for m in AMI_DEV]})
s["corpora"]["notsofar"] = dict(nsf, heldout=sorted(nheld), n_heldout=len(nheld), hours_heldout=h(nheld),
                                rule=nsf.get("rule", "") + " | v4: 원본 dev_set·eval_set 회의 전부 held-out, 학습은 train_set 만",
                                official={k: sorted(i for i, v in nsf_split.items() if v == k) for k in ("dev", "eval", "train")})
s["version"] = "v4"; s["derived_from"] = "v3"
json.dump(s, open(dst + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1); os.replace(dst + ".tmp", dst)
for c in ("ami", "notsofar"):
    x = s["corpora"][c]; tot = [k for k in dur if k.startswith(c + ":")]
    print(c, "held-out", x["n_heldout"], f"({x['hours_heldout']} h)", "· 학습", len(tot) - x["n_heldout"], f"({h(set(tot) - set(x['heldout']))} h)")
print("→", dst)
