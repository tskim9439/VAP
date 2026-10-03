"""번역 단위 정책별 오라클 지연(문맥 정렬 기반). 목표 단어 j 는 '정렬 누적 최대 cummax_{k≤j} a_k 이상인 첫 경계'에서 나온다(순서 유지·확정).
정책: MU(유효 절단 중 새 단어 생기는 첫 위치) · <SEM_END>(A, 스트림 끝 포함) · 발화 끝 · wait-k(k=3,5; 단어 단위, 정렬 무시 — 번역 불변 보장 없음).
지표(초, 음성 기준): 첫 토큰 지연 · LAAL(Papi 2022 식: τ = 원천 끝에 처음 도달한 목표 위치, 이상 지연 (j−1)·|X|/max(|Y|,|Y*|), |Y*|=|Y|) · 평균 원천 대기."""
import json, sys, statistics as st, importlib.util
spec = importlib.util.spec_from_file_location("p", "/soundai/users/tskim/VAPKT/experiments/cst_translation_unit_probe.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
streams_f, runs = sys.argv[1], sys.argv[2:]
S = {json.loads(l)["id"]: json.loads(l) for l in open(streams_f)}
def emit(bounds, cm, n):
    b = sorted(set(bounds) | {n}); return [next(x for x in b if x >= c) for c in cm]
def laal(d, X, T):
    tau = next((j for j, v in enumerate(d) if v >= X - 1e-6), len(d) - 1)
    return sum(d[j] - j * X / T for j in range(tau + 1)) / (tau + 1)
for path in runs:
    rows = [json.loads(l) for l in open(path) if '"L"' in l]
    R = {k: {"first": [], "laal": [], "units": []} for k in ("MU", "SEM_END", "utt_end", "wait3", "wait5")}
    for r in rows:
        s = S[r["id"]]; n = s["n"]; ends = s["ends"]; X = ends[-1]
        a, C, M = P.units(r); T = len(a)
        cm, m = [], 0
        for v in a: m = max(m, v); cm.append(m)
        A = [i for i in s["A"] if 1 <= i <= n]
        pol = {"MU": emit(M, cm, n), "SEM_END": emit(A, cm, n), "utt_end": [n] * T,
               "wait3": [min(n, 3 + int(j * n / T)) for j in range(T)], "wait5": [min(n, 5 + int(j * n / T)) for j in range(T)]}
        for k, b in pol.items():
            d = [ends[x - 1] for x in b]
            R[k]["first"].append(d[0]); R[k]["laal"].append(laal(d, X, T))
    lang = rows[0]["lang"]
    print(f"\n{lang}→{'English' if lang == 'Korean' else 'Korean'} ({len(rows)} 스트림, 평균 발화 {st.mean(S[r['id']]['ends'][-1] for r in rows):.1f} s)")
    for k, v in R.items():
        print(f"  {k:8s} 첫 토큰 {st.mean(v['first']):.2f} s (중앙 {st.median(v['first']):.2f}) · LAAL {st.mean(v['laal']):.2f} s (중앙 {st.median(v['laal']):.2f})")
