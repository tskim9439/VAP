import json, re, sys, sacrebleu, statistics as st
D = "/soundai/users/tskim/VAPKT-data/results/cst-unit-probe-v1"
HAN = re.compile(r"[一-鿿぀-ヿ]"); HANGUL = re.compile(r"[가-힣]")
def fail(text, tgt):
    if not text.strip(): return "empty"
    if HAN.search(text): return "중국어·일본어 문자"
    if tgt == "English" and HANGUL.search(text): return "원문(한국어) 그대로"
    if tgt == "Korean" and len(HANGUL.findall(text)) < 0.3 * len(re.findall(r"\w", text)): return "한국어 아님"
    w = text.split()
    for n in (1, 2, 3):
        for i in range(len(w) - 3 * n + 1):
            if w[i:i+n] == w[i+n:i+2*n] == w[i+2*n:i+3*n]: return "반복"
    return None
for name, refp, tgt in (("ko2en", "ko2en-exaone4", "English"), ("en2ko", "en2ko-exaone4", "Korean")):
    ref = {json.loads(l)["id"]: json.loads(l)["translation"] for l in open(f"{D}/{refp}.jsonl")}
    rows = [json.loads(l) for l in open(f"{D}/simul3-{name}-qwen38.jsonl")]
    print(f"\n## {name} ({len(rows)})")
    for pol in ("utt_end", "SEM_END", "MU"):
        f = [fail(r["out"][pol], tgt) for r in rows]
        from collections import Counter
        c = Counter(x for x in f if x)
        print(f"  {pol:8s} 붕괴 {sum(1 for x in f if x)}/{len(rows)} {dict(c)}")
    ok = [r for r in rows if not any(fail(r["out"][p], tgt) for p in ("utt_end", "SEM_END", "MU"))]
    for pol in ("utt_end", "SEM_END", "MU"):
        hyp = [r["out"][pol] for r in ok]; R = [[ref[r["id"]] for r in ok]]
        b = sacrebleu.corpus_bleu(hyp, R, tokenize="intl" if tgt == "Korean" else "13a"); c = sacrebleu.corpus_chrf(hyp, R)
        so = st.median(sacrebleu.sentence_chrf(r["out"][pol], [r["out"]["utt_end"]]).score for r in ok)
        print(f"  [붕괴 없는 {len(ok)}개] {pol:8s} BLEU {b.score:5.1f} chrF {c.score:5.1f} · 오프라인 대비 문장 chrF 중앙값 {so:5.1f}")
