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


PARTIAL_NOTE = ("The transcript may be incomplete because the speaker is still talking: translate only what has been said so far, "
                "do not guess or complete the rest.")


def messages_partial(src_lang, text, final):
    tgt = TGT[src_lang]
    note = "" if final else " " + PARTIAL_NOTE
    return [{"role": "user", "content": f"Translate the following {src_lang} speech transcript into natural {tgt}.{note} "
                                        f"Output only the {tgt} translation, nothing else.\n\n{text}"}]


def cut_loop(text, n=3, reps=3):
    """같은 n-gram 이 연달아 reps 번 이상 반복되면 첫 반복 뒤를 잘라낸다(탐욕 디코드 반복 붕괴)."""
    w = text.split()
    for i in range(len(w)):
        for m in range(1, n + 1):
            if i + m * reps <= len(w) and all(w[i:i + m] == w[i + m * r:i + m * (r + 1)] for r in range(1, reps)):
                return " ".join(w[:i + m])
    return text


def simul(a):
    """확정형(revision-free) 동시 번역 흉내: 정책 경계 b 마다 원천 앞 b 단어를 번역하되 지금까지 확정한 번역을 어시스턴트 앞부분으로
    강제하고, 새로 생성한 부분을 확정한다(Zhang 2020 의 강제 디코딩). 정책:
    - MU: 문맥 정렬 MU 경계(--align 런)
    - SEM_END: 라벨 A
    - waitK: K 단어 뒤 원천 단어마다 목표 1 단어
    - utt_end: 발화 끝 한 번 = 오프라인
    출력은 정책별 최종 번역."""
    from vapasr.data.semcommit_llm import LLM
    S = {json.loads(l)["id"]: json.loads(l) for l in open(a.streams)}
    al = {json.loads(l)["id"]: json.loads(l) for l in open(a.align) if '"L"' in l}
    ids = [i for i in al if S[i]["lang"] == a.lang][: a.max_streams]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    llm = LLM(a.model, a.kind, device="cuda", dtype="bfloat16"); llm.load()
    sep = " "
    def step(s, b, committed, max_new):
        final = b >= s["n"]
        prompt = llm.build_prompt(messages_partial(s["lang"], " ".join(s["src"][:b]), final)) + (committed + sep if committed else "")
        r = llm.generate_json([prompt], max_new_tokens=max_new, batch_size=1)[0]
        return cut_loop(r["text"].split("</think>")[-1].split("\n")[0].strip())
    t0 = time.time()
    with open(a.out, "a") as f:
        for k, sid in enumerate(i for i in ids if i not in done):
            s = S[sid]; n = s["n"]; _, C, M = units(al[sid]); rec = dict(id=sid, lang=s["lang"], n=n, out={}, commits={})
            off = step(s, n, "", a.max_new_tokens); rec["out"]["utt_end"] = off           # 오프라인 = 발화 끝 한 번
            cap = min(a.max_new_tokens, 2 * len(llm.encode(off)) + 8)                     # 단계별 생성 상한(반복 붕괴 방지)
            ratio = max(len(off.split()), 1) / n                                         # 목표/원천 단어 비(wait-k 보폭)
            pols = {"MU": M, "SEM_END": sorted(set(i for i in s["A"] if 1 <= i <= n) | {n})}
            for name, bounds in pols.items():
                committed, log = "", []
                for b in sorted(set(bounds) | {n}):
                    new = step(s, b, committed, cap)
                    committed = (committed + sep + new).strip() if new else committed; log.append([b, new])
                rec["out"][name] = committed; rec["commits"][name] = log
            for K in a.wait_k:                                                           # 비율 보정 wait-k: b 단어를 읽으면 목표 floor((b−K+1)·ratio) 단어까지
                committed, w = "", 0
                for b in range(min(K, n), n + 1):
                    if b < n:
                        allow = int((b - K + 1) * ratio)
                        if allow > w:
                            new = step(s, b, committed, cap).split()[: allow - w]
                            if new:
                                committed = (committed + sep + " ".join(new)).strip(); w += len(new)
                    else:
                        new = step(s, b, committed, cap)
                        committed = (committed + sep + new).strip() if new else committed
                rec["out"][f"wait{K}"] = committed
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if (k + 1) % 10 == 0:
                print(f"{k + 1} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


HAN_RE = __import__("re").compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
HANGUL_RE = __import__("re").compile(r"[\uac00-\ud7a3]")


def lang_ok(text, tgt):
    """목표 언어 출력인지: 한자·가나가 없고, 영어 목표면 한글이 없고, 한국어 목표면 한글이 글자의 30 % 이상."""
    import re
    if HAN_RE.search(text):
        return False
    if tgt == "English":
        return not HANGUL_RE.search(text)
    return len(HANGUL_RE.findall(text)) >= 0.3 * max(len(re.findall(r"\w", text)), 1)


def messages_incremental(src_lang, heard, committed, final):
    tgt = TGT[src_lang]
    note = "" if final else " The speaker is still talking: translate only what has been said so far and do not guess the rest."
    return [{"role": "user", "content":
             f"You are a simultaneous interpreter translating {src_lang} speech into {tgt}.\n"
             f"Source heard so far: {heard}\n"
             f"Translation already given (cannot be changed): {committed or '(none)'}\n"
             f"Translate only the part of the source that the translation already given does not cover yet, continuing it naturally in {tgt}.{note} "
             f"Output only the new {tgt} text, nothing else. If nothing new can be translated yet, output nothing."}]


def simul2(a):
    """보강 MU(MU2): 문맥 정렬의 유효 절단 중 최소 단위(단어·초)를 넘는 후보에서 접두 번역을 실제로 생성하고,
    그 결과가 전체 번역의 해당 구간(정렬상 새로 덮이는 목표 단어)과 chrF ≥ τ 로 일치하며 목표 언어일 때만 확정한다(Zhang 2020 의 생성 검증).
    확정한 번역은 강제 접두가 아니라 프롬프트로 알려 준다. 같은 프롬프트 방식의 SEM_END2·발화 끝(utt_end)도 함께 낸다.
    commits 에 (경계 단어 수, 확정 텍스트)를 남겨 score 가 지연을 계산한다."""
    import sacrebleu
    from vapasr.data.semcommit_llm import LLM
    S = {json.loads(l)["id"]: json.loads(l) for l in open(a.streams)}
    al = {json.loads(l)["id"]: json.loads(l) for l in open(a.align) if '"L"' in l}
    ids = [i for i in al if S[i]["lang"] == a.lang][: a.max_streams]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    llm = LLM(a.model, a.kind, device="cuda", dtype="bfloat16"); llm.load()
    def gen(s, b, committed, cap, final):
        tgt = TGT[s["lang"]]
        for tries in range(2):
            m = messages_incremental(s["lang"], " ".join(s["src"][:b]), committed, final)
            if tries:
                m[0]["content"] += f" Write the answer in {tgt} only."
            r = llm.generate_json([llm.build_prompt(m)], max_new_tokens=cap, batch_size=1)[0]
            out = cut_loop(r["text"].split("</think>")[-1].split("\n")[0].strip())
            if not out or lang_ok(out, tgt):
                return out
        return None                                                                   # 두 번 모두 다른 언어 → 확정하지 않음
    t0 = time.time(); stat = Counter()
    with open(a.out, "a") as f:
        for k, sid in enumerate(i for i in ids if i not in done):
            s = S[sid]; n = s["n"]; ends = s["ends"]; r_al = al[sid]
            a_al, C, M = units(r_al); full = r_al["translation"].split(); T = len(a_al)
            rec = dict(id=sid, lang=s["lang"], n=n, out={}, commits={})
            off = gen(s, n, "", a.max_new_tokens, True) or ""; rec["out"]["utt_end"] = off; rec["commits"]["utt_end"] = [[n, off]]
            cap = min(a.max_new_tokens, 2 * len(llm.encode(off or r_al["translation"])) + 8)
            min_w = a.min_words_ko if s["lang"] == "Korean" else a.min_words_en
            # MU2
            committed, log, k_prev, last_b = "", [], 0, 0
            for b in C:
                if b >= n:
                    break
                S_b = {j for j in range(T) if a_al[j] <= b}; k_b = len(S_b)
                if k_b <= k_prev or b - last_b < min_w or ends[b - 1] - (ends[last_b - 1] if last_b else 0.0) < a.min_sec:
                    continue
                stat["cand"] += 1
                new = gen(s, b, committed, cap, False)
                expect = " ".join(full[k_prev:k_b])
                if new and sacrebleu.sentence_chrf(new, [expect]).score >= a.tau:
                    committed = (committed + " " + new).strip(); log.append([b, new]); k_prev, last_b = k_b, b; stat["ok"] += 1
                else:
                    stat["reject_lang" if new is None else ("reject_empty" if not new else "reject_chrf")] += 1
            new = gen(s, n, committed, cap, True) or ""
            committed = (committed + " " + new).strip() if new else committed; log.append([n, new])
            rec["out"]["MU2"] = committed; rec["commits"]["MU2"] = log
            # SEM_END2 (같은 프롬프트 방식)
            committed, log = "", []
            for b in sorted(set(i for i in s["A"] if 1 <= i <= n) | {n}):
                new = gen(s, b, committed, cap, b >= n) or ""
                committed = (committed + " " + new).strip() if new else committed; log.append([b, new])
            rec["out"]["SEM_END2"] = committed; rec["commits"]["SEM_END2"] = log
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if (k + 1) % 10 == 0:
                print(f"{k + 1} {time.time() - t0:.0f}s {dict(stat)}", flush=True)
    print("DONE", dict(stat), flush=True)


def commit_laal(log, ends, out_words):
    """확정 기록 [(경계 단어 수, 텍스트)] → 목표 단어별 확정 시각으로 LAAL(초)·첫 토큰(초)."""
    d = []
    for b, txt in log:
        d += [ends[b - 1]] * len(txt.split())
    if not d:
        return None, None
    X = ends[-1]; T = max(len(d), out_words, 1)
    tau = next((j for j, v in enumerate(d) if v >= X - 1e-6), len(d) - 1)
    return sum(d[j] - j * X / T for j in range(tau + 1)) / (tau + 1), d[0]


def score(a):
    """정책별 최종 번역 vs 참조(다른 교사의 오프라인 번역) — sacrebleu BLEU·chrF(한국어 목표는 chrF 주 지표, BLEU tokenize intl).
    --streams 를 주면 원문과 정책별 번역·MU 확정 과정 예시를 --examples 개 출력한다(정책 간 차이가 큰 것부터 절반, 나머지는 무작위)."""
    import sacrebleu
    ref = {json.loads(l)["id"]: json.loads(l).get("translation") for l in open(a.ref)}
    rows = [json.loads(l) for l in open(a.simul)]
    rows = [r for r in rows if ref.get(r["id"])]
    tgt = TGT[rows[0]["lang"]]
    tokz = "intl" if tgt == "Korean" else "13a"
    print(f"{rows[0]['lang']}→{tgt}: {len(rows)} 스트림, 참조 = {a.ref}")
    for pol in rows[0]["out"]:
        hyp = [r["out"][pol] for r in rows]; R = [[ref[r["id"]] for r in rows]]
        b = sacrebleu.corpus_bleu(hyp, R, tokenize=tokz); c = sacrebleu.corpus_chrf(hyp, R)
        same = sum(r["out"][pol] == r["out"].get("utt_end") for r in rows)
        broken = sum(not r["out"][pol].strip() or not lang_ok(r["out"][pol], tgt) or cut_loop(r["out"][pol]) != r["out"][pol] for r in rows)
        lat = ""
        if a.streams and all(pol in r.get("commits", {}) for r in rows):
            SS = {json.loads(l)["id"]: json.loads(l) for l in open(a.streams)}
            vals = [commit_laal(r["commits"][pol], SS[r["id"]]["ends"], len(r["out"][pol].split())) for r in rows]
            vals = [v for v in vals if v[0] is not None]
            if vals:
                lat = f"  LAAL {sum(v[0] for v in vals) / len(vals):.2f}s  첫 확정 {sum(v[1] for v in vals) / len(vals):.2f}s"
        print(f"  {pol:8s} BLEU {b.score:5.1f}  chrF {c.score:5.1f}  길이비 {b.sys_len / max(b.ref_len, 1):.2f}  오프라인과 동일 {same}/{len(rows)}  붕괴 {broken}{lat}")
    if a.streams and a.examples:
        S = {json.loads(l)["id"]: json.loads(l) for l in open(a.streams)}
        def gap(r):
            o = r["out"].get("utt_end", "")
            return 100 - sacrebleu.sentence_chrf(r["out"].get("MU2", r["out"].get("MU", "")), [o]).score
        ordered = sorted(rows, key=gap, reverse=True)
        k = a.examples // 2
        rng = random.Random(0); rest = ordered[k:]; rng.shuffle(rest)
        for r in ordered[:k] + rest[: a.examples - k]:
            s = S[r["id"]]; ends = s["ends"]
            print(f"\n--- {r['id'][-16:]} ({s['set']}, {s['n']} 단어, {ends[-1]:.1f} s, MU 와 오프라인 chrF 차 {gap(r):.1f})")
            print(f"  원문     : {' '.join(s['src'])}")
            print(f"  참조(EXAONE): {ref[r['id']]}")
            for pol in ("utt_end", "SEM_END", "SEM_END2", "MU", "MU2"):
                if pol in r["out"]:
                    print(f"  {pol:8s}: {r['out'][pol]}")
            for b, new in r.get("commits", {}).get("MU2", r.get("commits", {}).get("MU", [])):
                print(f"    MU 확정 @{ends[b - 1]:5.2f}s [{' '.join(s['src'][:b])[-40:]}] → {new}")


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
    q = sub.add_parser("simul")
    q.add_argument("--streams", required=True); q.add_argument("--align", required=True); q.add_argument("--lang", choices=["Korean", "English"], required=True)
    q.add_argument("--out", required=True); q.add_argument("--model", default="/soundai/Model/Qwen3.8-27B"); q.add_argument("--kind", default="qwen38", choices=["qwen38", "exaone4"])
    q.add_argument("--max-streams", type=int, default=150); q.add_argument("--max-new-tokens", type=int, default=160); q.add_argument("--wait-k", type=int, nargs="*", default=[3, 5])
    q = sub.add_parser("simul2")
    q.add_argument("--streams", required=True); q.add_argument("--align", required=True); q.add_argument("--lang", choices=["Korean", "English"], required=True)
    q.add_argument("--out", required=True); q.add_argument("--model", default="/soundai/Model/Qwen3.8-27B"); q.add_argument("--kind", default="qwen38", choices=["qwen38", "exaone4"])
    q.add_argument("--max-streams", type=int, default=150); q.add_argument("--max-new-tokens", type=int, default=160)
    q.add_argument("--tau", type=float, default=50.0, help="접두 번역 vs 전체 번역 해당 구간 chrF 문턱")
    q.add_argument("--min-words-ko", type=int, default=2); q.add_argument("--min-words-en", type=int, default=3); q.add_argument("--min-sec", type=float, default=1.0)
    q = sub.add_parser("score")
    q.add_argument("--simul", required=True); q.add_argument("--ref", required=True)
    q.add_argument("--streams", default=None); q.add_argument("--examples", type=int, default=0)
    a = p.parse_args()
    {"sample": sample, "run": run, "analyze": analyze, "simul": simul, "simul2": simul2, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
