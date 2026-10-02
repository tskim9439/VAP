#!/usr/bin/env python3
"""<SEM_END> 대 번역 단위 겹침 측정 (wiki/outputs/output-simulst-translation-unit-survey-20261002.md §5).

semcommit 라벨 스트림을 교사 LLM(Qwen3.8-27B 또는 EXAONE-4.0, 판정 LLM 규칙)으로 KO→EN·EN→KO 번역한 뒤 다음을 잰다.

- 문맥 정렬(Hibiki, arXiv 2502.03382): 목표 단어 j 의 정렬 위치 a_j = argmax_i [L_j(i) − L_j(i−1)].
  L_j(i) = 원천 앞 i 단어만 보고 강제 디코드한 전체 번역에서 단어 j 토큰들의 로그확률 합(i = 0…n).
- 번역 단위 경계:
  - 유효 절단 C = {i : {j : a_j ≤ i} 가 목표 앞부분 {1…k}}. 그 앞까지의 번역이 이후 바뀌지 않는 원천 위치다.
  - MU 경계 M = C 중 새 목표 단어가 생기는 첫 위치. 번역 접두 일관성 MU(Zhang et al. EMNLP 2020)를 정렬로 근사한 것이다.
- 비교:
  - <SEM_END> A 자리가 C 에 드는 비율
  - 단위 길이(단어·초)
  - 한국어 연결어미 규칙 음성(rule_conn_*) 자리가 C·M 에 드는 비율
  - 정책별 원천 단어의 평균 대기 시간(다음 경계까지 초): MU · <SEM_END> · 스트림 끝

  sample:  python experiments/cst_translation_unit_probe.py sample --per-lang 300 --out <dir>/streams.jsonl
  run:     CUDA_VISIBLE_DEVICES=0 python experiments/cst_translation_unit_probe.py run --streams <dir>/streams.jsonl --lang Korean --out <dir>/ko2en.jsonl
  analyze: python experiments/cst_translation_unit_probe.py analyze --runs <dir>/ko2en.jsonl <dir>/en2ko.jsonl
run 은 이어서 할 수 있다(끝난 id 는 건너뛴다)."""
import argparse
import json
import os
import random
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SNAP = "/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637"
TGT = {"Korean": "English", "English": "Korean"}


def word_ends(w, n):
    """단어별 끝 시각(초). words 필드에 끝 시각이 있으면 쓰고, 없으면 길이를 균등 분할."""
    ws = w.get("words")
    if isinstance(ws, list) and len(ws) == n and all(isinstance(x, dict) for x in ws):
        for key in ("end_time", "end", "end_s", "t1"):
            if all(key in x for x in ws):
                return [float(x[key]) for x in ws], "words"
    d = float(w["duration_s"])
    return [d * (k + 1) / n for k in range(n)], "uniform"


def sample(a):
    rng = random.Random(a.seed)
    W = [l.strip() for l in open(f"{a.snap}/main-words.list")]
    L = [l.strip() for l in open(f"{a.snap}/main-labels.list")]
    pairs = list(zip(W, L)); rng.shuffle(pairs)
    out, per_part, n = [], Counter(), Counter()
    for wf, lf in pairs:
        lang = "English" if "/en-" in wf else "Korean"
        if n[lang] >= a.per_lang:
            continue
        words = {}
        for l in open(wf):
            r = json.loads(l); words[r["id"]] = r
        rows = [json.loads(l) for l in open(lf)]; rng.shuffle(rows)
        for r in rows:
            w = words.get(r["id"])
            if not w or n[lang] >= a.per_lang or per_part[wf] >= a.per_part:
                continue
            toks = w["text"].split(); nw = len(toks)
            if not (a.min_words <= nw <= a.max_words) or float(w["duration_s"]) > a.max_s:
                continue
            disp = (w.get("display_text") or "").split()
            src = disp if len(disp) == nw else toks
            ends, timing = word_ends(w, nw)
            cands = [dict(i=c["after_word"] + 1, grade=c["grade"], why=c.get("why", ""), punct=bool(c.get("punct")))
                     for c in r["candidates"]]
            out.append(dict(id=r["id"], lang=lang, set=w.get("set"), part=Path(wf).parent.parent.name, n=nw, src=src,
                            src_is_display=src is disp, duration_s=float(w["duration_s"]), ends=ends, timing=timing,
                            A=sorted(c["i"] for c in cands if c["grade"] == "A"),
                            conn=sorted(c["i"] for c in cands if c["why"].startswith("rule_conn")),
                            cands=cands))
            n[lang] += 1; per_part[wf] += 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"streams": dict(n), "timing": dict(Counter(r["timing"] for r in out)),
                      "display_text": sum(r["src_is_display"] for r in out)}, ensure_ascii=False))


def messages(src_lang, text):
    tgt = TGT[src_lang]
    return [{"role": "user", "content": f"Translate the following {src_lang} speech transcript into natural {tgt}. "
                                        f"Output only the {tgt} translation, nothing else.\n\n{text}"}]


def run(a):
    import torch
    from vapasr.data.semcommit_llm import LLM
    streams = [json.loads(l) for l in open(a.streams) if json.loads(l)["lang"] == a.lang]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = [s for s in streams if s["id"] not in done]
    llm = LLM(a.model, a.kind, device="cuda", dtype="bfloat16")
    llm.load()
    tok = llm.tok
    print(f"model {a.model} · {a.lang}→{TGT[a.lang]} · todo {len(todo)}/{len(streams)}", flush=True)
    t0 = time.time()
    with open(a.out, "a") as f:
        for k, s in enumerate(todo):
            full = llm.generate_json([llm.build_prompt(messages(s["lang"], " ".join(s["src"])))],
                                     max_new_tokens=a.max_new_tokens, batch_size=1)[0]
            y = full["text"].split("</think>")[-1].strip().split("\n")[0].strip()
            tw = y.split()
            if not tw:
                f.write(json.dumps(dict(id=s["id"], lang=s["lang"], error="empty translation", raw=full["text"]), ensure_ascii=False) + "\n"); continue
            enc = tok(y, add_special_tokens=False, return_offsets_mapping=True)
            yids, offs = enc["input_ids"], enc["offset_mapping"]
            starts, pos = [], 0                                    # 목표 단어 문자 구간 → 토큰별 단어 번호
            for w_ in tw:
                p = y.index(w_, pos); starts.append((p, p + len(w_))); pos = p + len(w_)
            tword = []
            for (c0, c1) in offs:
                j = next((jj for jj, (s0, s1) in enumerate(starts) if c0 < s1 and c1 > s0), None)
                tword.append(j if j is not None else (tword[-1] if tword else 0))
            ctx = [llm.encode(llm.build_prompt(messages(s["lang"], " ".join(s["src"][:i])))) for i in range(s["n"] + 1)]
            seqs = [c + yids for c in ctx]
            tok_lp = []
            for b0 in range(0, len(seqs), a.batch_size):
                ids, mask, posi = llm._left_pad(seqs[b0:b0 + a.batch_size])
                keep = len(yids) + 1
                with torch.inference_mode():
                    try:
                        out = llm.model(input_ids=ids, attention_mask=mask, position_ids=posi, use_cache=False, logits_to_keep=keep)
                    except TypeError:
                        out = llm.model(input_ids=ids, attention_mask=mask, position_ids=posi, use_cache=False)
                lg = out.logits[:, -keep:].float().log_softmax(-1)
                tgt = ids[:, -keep:]
                lp = lg[:, :-1].gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)   # (B, len(yids))
                tok_lp += lp.cpu().tolist()
                del out, lg
            Lw = [[0.0] * len(tw) for _ in range(len(seqs))]       # L_j(i) = 단어 j 토큰 로그확률 합
            for i, row in enumerate(tok_lp):
                for t, v in enumerate(row):
                    Lw[i][tword[t]] += v
            rec = dict(id=s["id"], lang=s["lang"], model=a.model, translation=y, tgt_words=len(tw), n=s["n"],
                       L=[[round(v, 4) for v in r] for r in Lw])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if (k + 1) % 10 == 0:
                print(f"{k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


def units(r):
    """정렬·유효 절단·MU 경계. i 는 읽은 원천 단어 수(1…n)."""
    n, L = r["n"], r["L"]; T = len(L[0])
    a = [max(range(1, n + 1), key=lambda i: (L[i][j] - L[i - 1][j], -i)) for j in range(T)]
    C, M, last_k = [], [], 0
    for i in range(1, n + 1):
        S = {j for j in range(T) if a[j] <= i}
        k = len(S)
        if S == set(range(k)):                                     # 목표 앞부분만 정렬됨 → 유효 절단
            C.append(i)
            if k > last_k:                                         # 새 목표 단어가 생기는 첫 유효 절단 = MU 경계
                M.append(i); last_k = k
    if n not in C:
        C.append(n)
    if not M or M[-1] != n:
        M.append(n)
    return a, C, M


def wait_s(bounds, ends, n):
    """원천 단어마다 다음 경계(포함)까지의 대기 시간 평균(초). 경계 = 읽은 단어 수."""
    b = sorted(set(bounds) | {n}); out = []
    for p in range(1, n + 1):
        nb = next(x for x in b if x >= p)
        out.append(ends[nb - 1] - ends[p - 1])
    return out


def analyze(a):
    S = {}
    for l in open(a.streams):
        r = json.loads(l); S[r["id"]] = r
    for path in a.runs:
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if "L" in r and r["id"] in S]
        if not rows:
            continue
        lang = rows[0]["lang"]; agg = defaultdict(list); cnt = Counter()
        for r in rows:
            s = S[r["id"]]; n = s["n"]; ends = s["ends"]
            _, C, M = units(r)
            Cs, Ms = set(C), set(M)
            A = [i for i in s["A"] if 1 <= i <= n]
            for i in A:
                cnt["A"] += 1; cnt["A_in_C"] += i in Cs; cnt["A_in_M"] += i in Ms
                cnt["A_end"] += i == n
            for i in s["conn"]:
                if 1 <= i < n:
                    cnt["conn"] += 1; cnt["conn_in_C"] += i in Cs; cnt["conn_in_M"] += i in Ms
            mid_M = [i for i in M if i < n]
            cnt["M_mid"] += len(mid_M); cnt["M_mid_at_A"] += sum(i in set(A) for i in mid_M)
            agg["units_M"].append(len(M)); agg["units_A"].append(len(set(A) | {n}))
            agg["words_per_M"].append(n / len(M)); agg["words_per_A"].append(n / len(set(A) | {n}))
            agg["sec_per_M"].append(s["duration_s"] / len(M))
            agg["wait_M"] += wait_s(M, ends, n); agg["wait_A"] += wait_s(A, ends, n); agg["wait_end"] += wait_s([], ends, n)
            cnt["streams"] += 1; cnt["timing_" + s["timing"]] += 1
        pct = lambda x, y: f"{100 * cnt[x] / max(cnt[y], 1):.1f}% ({cnt[x]}/{cnt[y]})"
        print(f"\n## {lang} → {TGT[lang]}  ({cnt['streams']} 스트림, 시각 {dict((k[7:], v) for k, v in cnt.items() if k.startswith('timing_'))})")
        print(f"- <SEM_END> A 가 유효 절단(번역 불변)인 비율: {pct('A_in_C', 'A')} · MU 경계와 정확히 일치: {pct('A_in_M', 'A')} · A 중 스트림 끝: {pct('A_end', 'A')}")
        print(f"- 스트림 중간 MU 경계 중 A 자리: {pct('M_mid_at_A', 'M_mid')}  (A 의 MU 재현율)")
        if cnt["conn"]:
            print(f"- 한국어 연결어미 규칙 음성 자리가 유효 절단: {pct('conn_in_C', 'conn')} · MU 경계: {pct('conn_in_M', 'conn')}")
        m = lambda k: f"{st.mean(agg[k]):.2f}"; md = lambda k: f"{st.median(agg[k]):.2f}"
        print(f"- 스트림당 단위 수 MU {m('units_M')} vs A {m('units_A')} · 단위당 단어 MU {m('words_per_M')} vs A {m('words_per_A')} · MU 단위당 초 {m('sec_per_M')}")
        print(f"- 원천 단어의 평균 대기(다음 경계까지, 초): MU {m('wait_M')} (중앙 {md('wait_M')}) · <SEM_END> {m('wait_A')} (중앙 {md('wait_A')}) · 스트림 끝 {m('wait_end')}")


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("sample")
    q.add_argument("--snap", default=SNAP); q.add_argument("--out", required=True); q.add_argument("--per-lang", type=int, default=300)
    q.add_argument("--per-part", type=int, default=12); q.add_argument("--min-words", type=int, default=6); q.add_argument("--max-words", type=int, default=60)
    q.add_argument("--max-s", type=float, default=30.0); q.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("run")
    q.add_argument("--streams", required=True); q.add_argument("--lang", choices=["Korean", "English"], required=True); q.add_argument("--out", required=True)
    q.add_argument("--model", default="/soundai/Model/Qwen3.8-27B"); q.add_argument("--kind", default="qwen38", choices=["qwen38", "exaone4"])
    q.add_argument("--batch-size", type=int, default=16); q.add_argument("--max-new-tokens", type=int, default=256)
    q = sub.add_parser("analyze")
    q.add_argument("--streams", required=True); q.add_argument("--runs", nargs="+", required=True)
    a = p.parse_args()
    {"sample": sample, "run": run, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
