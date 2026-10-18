#!/usr/bin/env python3
"""Stage 4 대화 번역 타깃의 원문 단위(decision-stage4-prep-20261019) — Phase 2 정제 대화의 발화를 ast_mono_generate.py 의 단위 형식으로 쓴다.
그 뒤는 Stage 2 와 같다: ast_mono_generate.py run --root DIR --prompt v3 --gen-dir gen-v3-bf16(Qwen3.8-27B, 단조 번역·판정·stable prefix),
fix(판정 v3 + 한 번 수리). 방향도 같다(DIRECTIONS: en→de·ko, ko→en, de→en).

행 = 원 발화. 정제(p2_refine)가 VAD 로 나눈 조각(<utt_id>#n)은 같은 화자의 원 발화로 다시 묶는다.
  격리 발화(flags — Stage 3 에서 되살린 것 포함)는 원문이 의심스러워 넣지 않는다. held-out 대화도 넣는다(평가 참조용, 학습은 split 으로 뺀다).
단어 시각 = 발화의 정렬 토큰. " " + 정규화 text 를 Qwen3-ASR 토크나이저로 다시 토큰화해(p2_align 과 같은 토큰화) id 가 같은지 확인하고,
  offset 으로 토큰을 단어에 묶어 단어의 마지막 토큰 시각을 쓴다. 나뉜 조각은 text 가 비어 있어 토큰을 decode 한 text 를 쓴다.
원문 = raw(대소문자·문장부호가 있으면 번역에 낫다 — otoSpeech·NSC·KEC 철자; 한국어·ZTC 는 text 와 같다)를 clean_text·glue 로 정리해 단어 시각에
  문자 정렬한다(align_times, 일치 비율이 낮으면 정규화 text 단어). 단위 나누기(split_units)·shard·manifest 는 ast_mono_generate prepare 와 같다.
  python experiments/stage4_dialogue_units.py --data <phase2> --corpora aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc --root DIR
--revive-dir DIR(Stage 3 되살리기 재료 <corpus>.revive.jsonl): 그 되살린 격리 발화만 단위로 쓴다(set = <corpus>#revived, 다른 --root 로).
  원문(라벨)이 의심스러워 그 번역은 Stage 4 학습에서 입력 문맥으로만 쓴다(label −100) — 번역 모드에서도 태그·<EOT> 구조가 끊기지 않게.
"""
import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ast_mono_generate import CODE, DIRECTIONS, align_times, clean_text, code_sha, glue, split_units  # noqa: E402
from vapasr.data.dialogue import Dialogue  # noqa: E402
from vapasr.data.cst_dialogue import REVIVABLE, load_revive  # noqa: E402


def word_times(tok, ids, times, text):
    """정렬 토큰(ids, times)을 " " + text 의 단어에 묶는다 → [(단어, 끝 시각)] | None(다시 토큰화한 id 가 다르면)."""
    s = " " + text; enc = tok(s, return_offsets_mapping=True, add_special_tokens=False)
    if list(enc["input_ids"]) != list(ids): return None
    spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", s)]; out = [None] * len(spans); j = 0
    for (a, b), t in zip(enc["offset_mapping"], times):
        if b <= a: continue
        while j < len(spans) - 1 and b - 1 >= spans[j][1]: j += 1
        out[j] = max(out[j] or 0.0, float(t))
    if any(x is None for x in out): return None
    return [(w, x) for (_, _, w), x in zip(spans, out)]


def dialogue_rows(d, tok, max_words, st, revive=None):
    """revive({utt_id: tokens} | None): 주면 그 되살린 격리 발화만(정렬 토큰 = 되살리기 재료), 아니면 격리되지 않은 발화만."""
    groups = defaultdict(list); miss = set(d.meta.get("missing", []))
    for i, u in enumerate(d.utterances, 1):
        if revive is not None:
            if u.tokens or not u.flags or not set(u.flags) <= REVIVABLE or i in miss or u.utt_id not in revive: continue
            u.tokens = list(revive[u.utt_id]); u.text = ""                      # 정규화 text 는 비어 있다 → 토큰을 decode 한다
        elif u.flags or not u.tokens: continue
        groups[(u.speaker, u.utt_id.split("#")[0])].append(u)
    for (spk, base), us in sorted(groups.items(), key=lambda g: min(u.start for u in g[1])):
        us.sort(key=lambda u: u.start); pairs = sorted((t for u in us for t in u.tokens), key=lambda x: x[1])
        ids, times = [int(t) for t, _ in pairs], [float(e) for _, e in pairs]
        text = us[0].text if len(us) == 1 and (us[0].text or "").strip() else tok.decode(ids).strip()
        words = word_times(tok, ids, times, text)
        if words is None: st["skip_retokenize"] += 1; continue
        src, toks, tt = "text", [w for w, _ in words], [x for _, x in words]
        raw = glue(clean_text(us[0].raw or "").split())
        if raw:
            rt, a_share, b_share = align_times(raw, words)
            if rt and a_share >= 0.85 and b_share >= 0.6: src, toks, tt = "raw", raw, rt
        st[f"src_{src}"] += 1
        units = [dict(seg=0, tokens=toks[lo:hi], times=[round(x, 3) for x in tt[lo:hi]], src=src) for lo, hi in split_units(toks, tt, max_words)]
        yield dict(id=f"{d.conv_id}|{spk}|{base}", set=d.corpus + ("#revived" if revive is not None else ""), lang=d.lang, duration_s=round(us[-1].end - us[0].start, 3), units=units)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--corpora", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--max-words", type=int, default=40); ap.add_argument("--shard-words", type=int, default=25000)
    ap.add_argument("--tokenizer", default=os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B")); ap.add_argument("--limit", type=int, default=0, help="코퍼스마다 대화 상한(파일럿)")
    ap.add_argument("--revive-dir", default=None, help="되살리기 재료 폴더 — 주면 되살린 격리 발화만(모듈 설명)")
    a = ap.parse_args(); root = Path(a.root); man_path = root / "units" / "manifest.json"
    if man_path.exists(): sys.exit(f"already prepared: {man_path}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    shards = {c: [] for c in DIRECTIONS}; writers = {}; stats = {}; t0 = time.monotonic()

    def flush(code, w):
        if not w["units"]: return
        dd = root / "units" / code; dd.mkdir(parents=True, exist_ok=True); name = f"u-{len(shards[code]):05d}"; tmp = dd / f"{name}.jsonl.gz.partial"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for u in w["units"]: f.write(json.dumps(u, ensure_ascii=False) + "\n")
        os.replace(tmp, dd / f"{name}.jsonl.gz"); shards[code].append(dict(name=name, units=len(w["units"]), words=w["words"], rows=w["rows"])); w.update(units=[], words=0, rows=0)

    for c in [x for x in a.corpora.split(",") if x]:
        st = stats.setdefault(c, Counter()); rv = None
        if a.revive_dir:
            f = os.path.join(a.revive_dir, f"{c}.revive.jsonl")
            if not os.path.exists(f): print(f"{c}: 되살리기 재료 없음 — 건너뜀", flush=True); continue
            rv = load_revive([f])
        for n, line in enumerate(open(os.path.join(a.data, f"{c}.refined.dialogues.jsonl"), encoding="utf-8")):
            if a.limit and n >= a.limit: break
            d = Dialogue.from_json(line); code = CODE[d.lang]; st["dialogues"] += 1
            for r in dialogue_rows(d, tok, a.max_words, st, None if rv is None else rv.get(d.conv_id, {})):
                st["rows"] += 1; st["hours"] += r["duration_s"] / 3600; st["units"] += len(r["units"]); st["words"] += sum(len(u["tokens"]) for u in r["units"])
                w = writers.setdefault(code, dict(units=[], words=0, rows=0))
                for k, u in enumerate(r["units"]):
                    w["units"].append(dict(uid=f"{r['id']}#{k}", row=r["id"], set=r["set"], k=k, n=len(r["units"]), **u)); w["words"] += len(u["tokens"])
                w["rows"] += 1
                if w["words"] >= a.shard_words: flush(code, w)
        print(f"[{time.monotonic() - t0:.0f} s] {c}: {dict(st)}", flush=True)
    for code, w in writers.items(): flush(code, w)
    stats = {k: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in stats.items()}
    man = dict(created=time.strftime("%F %T"), source="phase2 refined dialogues (stage4_dialogue_units)", corpora=a.corpora.split(","), data=a.data, revive_dir=a.revive_dir,
               max_words=a.max_words, shard_words=a.shard_words, tokenizer=a.tokenizer, code_sha256=code_sha(), shards=shards, stats=stats,
               totals={c: dict(shards=len(v), units=sum(x["units"] for x in v), words=sum(x["words"] for x in v), rows=sum(x["rows"] for x in v)) for c, v in shards.items()})
    man_path.parent.mkdir(parents=True, exist_ok=True); man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(json.dumps(dict(totals=man["totals"]), indent=1))


if __name__ == "__main__":
    main()
