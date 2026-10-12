"""scores-all.json(simulst_common.py score 산출) → tables.txt: 세트·방향마다 시스템/설정별 BLEU·chrF·LAAL·AL·빈 번역.
usage: python make_tables.py scores-all.json > tables.txt"""
import collections
import json
import sys

d = json.load(open(sys.argv[1]))
rows = collections.defaultdict(dict)
for k, v in d.items():
    sysn, setting, ds = k.split("/"); rows[ds][f"{sysn}/{setting}"] = v
order = ["fleurs-de-en", "covost2-de-en", "fleurs-en-de", "fleurs-en-de-peak05", "covost2-en-de", "fleurs-en-ko", "fleurs-en-ko-peak05", "fleurs-ko-en"]
for ds in order:
    n = {v["n"] for v in rows[ds].values()}
    print(f"== {ds} (n={'/'.join(map(str, sorted(n)))})")
    for name, v in sorted(rows[ds].items()):
        lat = (f"LAAL {v['laal_s']:5.2f} (중앙 {v['laal_median_s']:5.2f}) AL {v['al_s']:5.2f}" if v.get("laal_s") is not None else "(오프라인)")
        bad = "  ← 무효(VAC 없이 실행, 재실행 중)" if name.startswith("simulstreaming-cascade/c1000-ft25") and not name.endswith("-vac") else ""
        print(f"   {name:38s} BLEU {v['bleu']:6.2f} chrF {v['chrf']:6.2f} {lat} 빈 번역 {v['empty_translations']}{bad}")
