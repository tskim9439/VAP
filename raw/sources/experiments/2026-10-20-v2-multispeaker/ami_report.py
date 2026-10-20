# AMI 공식 분할별 다화자 점수 — cst_stage3_eval 원출력(ami-bench-v1: ami-ihmmix·ami-sdm)을 공식 dev·eval 로 나눠 합친다(창 cpWER·agn·EOT, cst_eval.pool). 읽기만 한다.
#   python3 ami_report.py <stage3-eval 결과 폴더> [<폴더> …]   (vapasr 가 import 되는 python — 예 PYTHONPATH=/soundai/users/tskim/VAPKT)
import glob, json, os, sys
from vapasr.data.cst_eval import pool
MAN = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json"))
SUB = {**{m: "dev" for m in MAN["dev"]}, **{m: "eval" for m in MAN["eval"]}}
for d in sys.argv[1:]:
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "predictions-rank*.jsonl"))) for l in open(f, encoding="utf-8")]
    print(f"== {d} ({len(rows)} 행)")
    for c in sorted({r["corpus"] for r in rows}):
        for sub in ("eval", "dev"):
            for dl in sorted({r["delta"] for r in rows}):
                rr = [r for r in rows if r["corpus"] == c and r["delta"] == dl and SUB[r["id"].split(":", 1)[1].split("@")[0]] == sub]
                if not rr: continue
                p = pool(rr)
                print(f"  {c:11s} {sub:4s} δ{dl:<2d} cpWER {p['err_cp']:6.2f} · agn {p['err_agn']:6.2f} · 귀속 몫 {p['err_cp'] - p['err_agn']:5.2f} · slot {p['err_slot']:6.2f} · "
                      f"EOT R {p['eot_recall']:.3f} P {p['eot_precision']:.3f} · 창 {p['windows']} · 회의 {len({r['id'].split('@')[0] for r in rr})} · 화자 수 {sorted({r['n_spk'] for r in rr})}")
