# 모델 출력(원출력 emitted)에서 슬롯 교대가 <EOT> 뒤에만 일어나는가. 교대 = 지금 슬롯과 다른 태그. "<EOT> 뒤" = 지금 슬롯의 마지막 글자 이후 교대 전까지 <EOT> 가 있었음.
# <EOT> 뒤 행방: 다음 내용이 다른 슬롯(교대)인지 같은 슬롯(머무름)인지. 읽기만.
import json, glob, collections
A, B, EOT, TR = 151707, 151708, 151722, 151726
def stats(rows):
    c = collections.Counter()
    for r in rows:
        cur = None; eot_pending = False
        for k, t in r["emitted"]:
            if t in (A, B):
                if cur is not None and t != cur:
                    c["switch"] += 1; c["switch_after_eot" if eot_pending else "switch_no_eot"] += 1
                    if eot_pending: c["eot_then_switch"] += 1
                    eot_pending = False
                cur = t
            elif t == EOT: c["eot"] += 1; eot_pending = True
            elif t == TR: continue
            else:
                if eot_pending: c["eot_then_same"] += 1                    # <EOT> 뒤 같은 슬롯이 다시 말함
                eot_pending = False
    return c
R = "/soundai/users/tskim/VAPKT-data/results"
for name, d in (("V1 ck2000 · 같은 언어 dev", R + "/stage3-eval-st4-m2a-ck2000"), ("R1x final · 같은 언어 dev", R + "/stage3-eval-v3-p1-tel-xlw-final"),
                ("V1 ck2000 · AMI", R + "/stage3-eval-ami-st4-m2a-ck2000"), ("R1x final · AMI(진행 중 몫)", R + "/stage3-eval-ami-v3-p1-tel-xlw-final")):
    rows = [json.loads(l) for f in sorted(glob.glob(d + "/predictions-rank*.jsonl")) for l in open(f)]
    rows = [r for r in rows if int(r["delta"]) == 10 and r.get("n_spk", 0) >= 2]
    if not rows: print(name, "없음"); continue
    c = stats(rows); s = max(1, c["switch"]); e = max(1, c["eot"])
    print(f"{name}: 창 {len(rows)} · 모델 교대 {c['switch']}: <EOT> 뒤 {100 * c['switch_after_eot'] / s:.1f} % · <EOT> 없이 {100 * c['switch_no_eot'] / s:.1f} %"
          f" │ <EOT> {c['eot']}: 뒤에 교대 {100 * c['eot_then_switch'] / e:.1f} % · 같은 화자 계속 {100 * c['eot_then_same'] / e:.1f} %")
