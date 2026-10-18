# Stage 2 단일 화자 퇴행 축약 평가 비교: python3 cmp_regress.py s4 p1 …  (results/stage3-regress-<태그>/<세트>/summary.json)
import json, sys, os
R = "/soundai/users/tskim/VAPKT-data/results"; tags = sys.argv[1:]
def load(t, s):
    p = f"{R}/stage3-regress-{t}/{s}/summary.json"
    return json.load(open(p)) if os.path.exists(p) else None
print("세트/δ".ljust(34), " | ".join(f"{t:>14s}" for t in tags))
for s in ("fleurs", "vpaa-kspon", "yodas-de", "libri-other"):
    S = {t: load(t, s) for t in tags}
    if not all(S.values()): print(s, "결과 없음", [t for t in tags if not S[t]]); continue
    for g in sorted(S[tags[0]]["groups"]):
        m = S[tags[0]]["groups"][g]["metrics"]; key = "cer_nospace" if "cer_nospace" in m else "wer"
        print(f"{g} {key}".ljust(34), " | ".join(f"{100 * S[t]['groups'][g]['metrics'][key]['rate']:14.2f}" for t in tags))
    for g in sorted(S[tags[0]].get("translation") or {}):
        print(f"{g} BLEU·LAAL".ljust(34), " | ".join(f"{S[t]['translation'][g]['bleu']:7.2f}·{S[t]['translation'][g]['laal_s']:5.2f}" for t in tags))
