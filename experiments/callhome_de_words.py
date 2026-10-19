#!/usr/bin/env python
"""CallHome 독일어 감독 발화 → 단일 화자 학습 항목 words.jsonl (사용자 결정 2026-10-20: "감독 발화만 단일 화자 학습 항목으로 쓰기").

대화 창으로는 마스크가 79 % 라 쓰지 않고, 두 교사(Qwen3-ASR-1.7B·Whisper large-v3) WER < 3 % 로 감독이 붙은 발화(callhome_de_pseudo_label build)를
YODAS 독일어 replay 와 같은 words.jsonl 행으로 만든다(semcommit_dataset 계약: tokens 스트림 시각·시간순, words = split_words, segments 의 src_offset_s 로 긴 wav 의 구간).
  입력: <phase2>/callhome-de.refined.dialogues.jsonl(정렬 토큰 내장), split(v3 — held-out 통화는 eval 파일로 따로)
  발화 구간: [start − pad, end + pad](pad 0.15 s, 통화 안으로 자름) — mono 라 상대 소리가 섞일 수 있지만 두 교사가 맞은 발화만이다.

  python experiments/callhome_de_words.py --data <phase2> --split <phase2>/splits/v3.json --tokenizer <학습 체크포인트> --out <dir>
산출: <dir>/callhome-de-words-{train,heldout}.jsonl, <dir>/lists/callhome-de-words-train.list(학습 --replay-set 용), <dir>/stats.json. 아무것도 지우지 않는다."""
import argparse, json, math, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--split", required=True); ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--pad", type=float, default=0.15); ap.add_argument("--min-words", type=int, default=1)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vapasr.data.dialogue import Dialogue
    from vapasr.data.semcommit_words import split_words
    from vapasr.data.semcommit_dataset import check_words
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    held = set(json.load(open(a.split, encoding="utf-8"))["corpora"]["callhome-de"]["heldout"])
    os.makedirs(os.path.join(a.out, "lists"), exist_ok=True)
    paths = {k: os.path.join(a.out, f"callhome-de-words-{k}.jsonl") for k in ("train", "heldout")}
    for p in paths.values(): assert not os.path.exists(p), f"{p} 가 이미 있다 — 새 --out 을 쓴다"
    st = Counter(); hours = Counter(); fo = {k: open(p + ".tmp", "w", encoding="utf-8") for k, p in paths.items()}
    for line in open(os.path.join(a.data, "callhome-de.refined.dialogues.jsonl"), encoding="utf-8"):
        d = Dialogue.from_json(line); part = "heldout" if d.conv_id in held else "train"; path = d.channels[d.speakers[0]].path
        for u in d.utterances:
            st["utts"] += 1
            if u.flags or not u.tokens: st["skip_unsupervised"] += 1; continue
            s0 = max(0.0, u.start - a.pad); s1 = min(d.duration_s, u.end + a.pad); dur = round(s1 - s0, 3)
            toks = [(int(t), round(min(max(float(x) - s0, 0.0), dur), 3)) for t, x in u.tokens]
            if any(toks[i][1] < toks[i - 1][1] for i in range(1, len(toks))): st["skip_time_reversal"] += 1; continue
            ws = split_words([t for t, _ in toks], [x for _, x in toks], tok)
            if not ws or len(ws) < a.min_words: st["skip_words"] += 1; continue
            words = [dict(i=w["i"], text=w["text"], a=w["a"], b=w["b"], end_time=w["end_time"], seg=0, tags=[]) for w in ws]
            try: check_words(words, [list(t) for t in toks])
            except ValueError as e: st[f"skip_{str(e).split(':')[0]}"] += 1; continue
            sess = d.conv_id.split(":", 1)[1]
            row = dict(id=f"chde-{sess}-{u.utt_id}", set="callhome-de", lang="German", K=math.ceil(dur / 0.08), duration_s=dur, text=" ".join(w["text"] for w in words),
                       tokens=[list(t) for t in toks], words=words,
                       segments=[dict(path=path, offset_s=0.0, silence_before_s=0.0, dur_s=dur, src_offset_s=round(s0, 3), utt_id=u.utt_id, raw_text=u.raw)],
                       source_corpus="callhome-de", conv_id=d.conv_id, speaker=u.speaker, split=part, training_eligible=True,
                       selection="qwen3asr_vs_whisper_wer_lt_0.03")
            fo[part].write(json.dumps(row, ensure_ascii=False) + "\n"); st[f"rows_{part}"] += 1; hours[part] += dur / 3600
    for k, f in fo.items(): f.close(); os.replace(paths[k] + ".tmp", paths[k])
    lst = os.path.join(a.out, "lists", "callhome-de-words-train.list")
    if not os.path.exists(lst): open(lst, "w").write(paths["train"] + "\n")
    stats = dict(st, hours={k: round(v, 2) for k, v in hours.items()}, pad_s=a.pad, tokenizer=a.tokenizer, split=a.split)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1); print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
