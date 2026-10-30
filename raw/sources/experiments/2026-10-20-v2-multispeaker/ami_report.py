# AMI 공식 분할별 다화자 점수 — cst_stage3_eval 원출력(ami-bench-v1: ami-ihmmix·ami-sdm)을 공식 dev·eval 로 나눠 합친다(창 cpWER·agn·EOT, cst_eval.pool). 읽기만 한다.
#   python3 ami_report.py <stage3-eval 결과 폴더> [<폴더> …]   (vapasr 가 import 되는 python — 예 PYTHONPATH=/soundai/users/tskim/VAPKT)
import glob, json, os, sys
from vapasr.data.cst_eval import pool, follow_pool, follow_line
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
                dz = [r["llm_der"] for r in rr if "llm_der" in r]                                   # E9(2026-10-28): <BEGIN>·<EOT> 구간의 LLM 쪽 DER·청크 디코드 ms
                ld = ""
                if dz:
                    n_ = max(1.0, sum(x["ref"] for x in dz)); ld = (f" · LLM DER {100 * sum(x['miss'] + x['fa'] + x['conf_best'] for x in dz) / n_:.2f}"
                                                                    f"(놓침 {100 * sum(x['miss'] for x in dz) / n_:.2f} 헛 {100 * sum(x['fa'] for x in dz) / n_:.2f} 혼동 {100 * sum(x['conf_best'] for x in dz) / n_:.2f})")
                ck = [r for r in rr if "chunk_ms" in r]
                if ck: ld += f" · 청크 {sum(r['chunk_ms'] * r['K'] for r in ck) / sum(r['K'] for r in ck):.1f} ms"
                fp = follow_pool(rr)                                                                  # 2026-10-30 활동 추종 진단(행에 follow 가 있을 때)
                if fp: ld += follow_line(fp)
                print(f"  {c:11s} {sub:4s} δ{dl:<2d} cpWER {p['err_cp']:6.2f} · agn {p['err_agn']:6.2f} · 귀속 몫 {p['err_cp'] - p['err_agn']:5.2f} · slot {p['err_slot']:6.2f} · "
                      f"EOT R {p['eot_recall']:.3f} P {p['eot_precision']:.3f} · 창 {p['windows']} · 회의 {len({r['id'].split('@')[0] for r in rr})} · 화자 수 {sorted({r['n_spk'] for r in rr})}{ld}")
