#!/usr/bin/env python
"""오류 예산: 우리 cp 오류를 유형별로 나누고, 각 유형을 얼마나 줄이면 외부 기준(Multitalker Parakeet + N3D)을 넘는지 계산한다(2026-11-06,
사용자 "우리 모델의 오류 유형을 분석하고 … 각 유형의 오류율을 어느 정도까지 내릴 수 있다면 Multitalker parakeet 보다 앞설 수 있는지 분석해봐").
입력은 gap_diag.py --json 결과(첫 시스템 = 우리, 둘째 = 외부 기준). 계산만 한다.

  python err_budget.py <gap_diag.json> [--md]

유형(참조 단어 100 개당 cp 오류 몫 — 합 = cp):
  혼자 말 받아쓰기 = 혼자 말 몫 × orc 놓침률, 겹친 말 받아쓰기 = 둘·셋+ 몫 × orc 놓침률,
  화자 귀속 = (cp 놓침 − orc 놓침) + (cp 삽입 − orc 삽입) — 맞게 받아썼지만 다른 화자 칸에 쓴 단어와 그 때문에 생긴 삽입,
  삽입 = orc 삽입(어느 화자에도 맞지 않는 헛말).
넘는 조건: 우리 합 < 외부 기준 합. 겹친 말 놓침률 m_o 손익분기 = (외부 합 − 혼자 몫·m1 − 귀속 − 삽입) / 겹침 몫.
"""
import argparse, json


def parts(x):
    N = x["all"]["n"]; sh = {k: x[k]["n"] / N for k in ("1", "2", "3+")}
    t1 = sh["1"] * x["1"]["orc_miss"]; t2 = sh["2"] * x["2"]["orc_miss"]; t3 = sh["3+"] * x["3+"]["orc_miss"]
    att = x["all"]["attr"] + (x["ins_cp"] - x["ins_orc"]); ins = x["ins_orc"]
    so = sh["2"] + sh["3+"]
    return dict(sh=sh, so=so, t1=t1, t2=t2, t3=t3, att=att, ins=ins, tot=t1 + t2 + t3 + att + ins, cp=x["cp_wer"],
                m1=x["1"]["orc_miss"], m2=x["2"]["orc_miss"], m3=x["3+"]["orc_miss"], mo=100 * (t2 + t3) / (100 * so) if so else 0.0,
                d2=x["2"]["orc_del"], s2=x["2"]["orc_sub"], d1=x["1"]["orc_del"], s1=x["1"]["orc_sub"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("json"); a = ap.parse_args()
    d = json.load(open(a.json)); (na, A), (nb, B) = list(d.items())[:2]
    for c in A:
        if c not in B: continue
        u, p = parts(A[c]), parts(B[c])
        print(f"== {c} · {na} cp {u['cp']:.2f}(유형 합 {u['tot']:.2f}) · {nb} cp {p['cp']:.2f}(유형 합 {p['tot']:.2f}) · 격차 {u['cp'] - p['cp']:+.2f}")
        print(f"   몫: 혼자 {100 * u['sh']['1']:.1f} % · 둘 {100 * u['sh']['2']:.1f} % · 셋+ {100 * u['sh']['3+']:.1f} %")
        rows = [("혼자 말 받아쓰기", "t1", f"놓침률 {u['m1']:.1f} 대 {p['m1']:.1f} %(삭제 {u['d1']:.1f}·대치 {u['s1']:.1f} 대 {p['d1']:.1f}·{p['s1']:.1f})"),
                ("둘 겹침 받아쓰기", "t2", f"놓침률 {u['m2']:.1f} 대 {p['m2']:.1f} %(삭제 {u['d2']:.1f}·대치 {u['s2']:.1f} 대 {p['d2']:.1f}·{p['s2']:.1f})"),
                ("셋+ 겹침 받아쓰기", "t3", f"놓침률 {u['m3']:.1f} 대 {p['m3']:.1f} %"),
                ("화자 귀속", "att", ""), ("삽입(헛말)", "ins", "")]
        print(f"   {'유형':14s} {'우리':>6s} {'(몫)':>7s} {'외부':>6s} {'차':>6s}  비고")
        for name, k, note in rows:
            print(f"   {name:14s} {u[k]:6.2f} ({100 * u[k] / u['tot']:4.1f} %) {p[k]:6.2f} {u[k] - p[k]:+6.2f}  {note}")
        print(f"   겹친 말(둘+셋+) 놓침률 {u['mo']:.1f} 대 {p['mo']:.1f} % · 1 %p 당 cp {u['so']:.3f} · 혼자 말 1 %p 당 cp {u['sh']['1']:.3f}")
        # 손익분기: 겹친 말 놓침률 m_o* (다른 유형 시나리오별)
        P = p["tot"]
        scen = [("나머지 그대로", u["m1"], u["att"], u["ins"]),
                ("귀속 → 외부 수준", u["m1"], p["att"], u["ins"]),
                ("귀속·혼자 말 → 외부 수준", p["m1"], p["att"], u["ins"]),
                ("귀속 0 · 혼자 말 → 외부 수준", p["m1"], 0.0, u["ins"])]
        for name, m1, att, ins in scen:
            mo = (P - u["sh"]["1"] * m1 - att - ins) / u["so"]
            print(f"   손익분기 [{name}] 겹친 말 놓침률 < {mo:5.1f} %" + ("  (불가능 — 겹침을 다 받아써도 못 넘는다)" if mo < 0 else ""))
        # 겹친 말을 외부 수준까지 내렸을 때 혼자 말 놓침률 손익분기
        for name, mo, att in [("겹침 → 외부 수준, 귀속 그대로", p["mo"], u["att"]), ("겹침·귀속 → 외부 수준", p["mo"], p["att"])]:
            m1 = (P - u["so"] * mo - att - u["ins"]) / u["sh"]["1"]
            print(f"   손익분기 [{name}] 혼자 말 놓침률 < {m1:5.1f} %(지금 {u['m1']:.1f}, 외부 {p['m1']:.1f})")


if __name__ == "__main__":
    main()
