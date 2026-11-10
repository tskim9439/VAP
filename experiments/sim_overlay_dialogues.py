#!/usr/bin/env python
"""두 합성 2 인 대화를 통째로 겹친 4 인 대화를 만든다(2026-11-10, 사용자 "P2 와 같은 레시피로 진행했으면 좋겠는데" — 본 모델 LLM 학습용 v8 겹침 셋).
P2(화자 조건 인코더 검증)의 "다른 대화 창을 간섭원으로 덧섞기"를 본 모델 학습 경로에 맞춘 것이다. LLM 은 모든 화자를 받아쓰므로 간섭원 화자의 말도 정답에 들어가야 한다.
그래서 대화(JSON)를 합쳐 새 셋으로 미리 만든다. 학습 경로(mono 섞기·N3D 활동·슬롯 정답)는 그대로 쓴다.
  A(2 인) + B(2 인, 시각을 U(0, --max-offset) s 밀고 화자 a·b → c·d) → 화자 4 인, 발화·토큰·채널 조각을 합친다.
  섞기(dialogue_mix)는 화자마다 말소리 음량을 같은 dBFS 로 맞추므로, 두 대화가 비슷한 크기로 겹친다.
  python experiments/sim_overlay_dialogues.py --src <a.jsonl> [--src <b.jsonl> …] --out <폴더> [--key sim_en-ovl3] [--rounds 2] [--max-offset 5] [--seed 0]
  짝: --src 의 2 인 대화마다 다른 2 인 대화 하나(같은 언어)를 무작위로 고른다. 라운드마다 짝을 새로 고른다. 산출 <폴더>/<key>.refined.dialogues.jsonl·<key>.stats.json(새 폴더). 읽기만 하고 새 파일에만 쓴다.
"""
import argparse, json, os, random


def shift_utt(u, spk, off):
    u = dict(u); u["speaker"] = spk; u["start"] = round(u["start"] + off, 3); u["end"] = round(u["end"] + off, 3)
    u["tokens"] = [[tid, round(t + off, 3)] for tid, t in (u.get("tokens") or [])]
    u["utt_id"] = f"{spk}:{u.get('utt_id', '')}"
    return u


def overlay(a, b, off, key, i):
    ren = dict(zip(b["speakers"], ["c", "d"]))
    utts = [dict(u) for u in a["utterances"]] + [shift_utt(u, ren[u["speaker"]], off) for u in b["utterances"]]
    utts.sort(key=lambda u: (u["start"], u["speaker"]))
    ch = {k: v for k, v in a["channels"].items()}
    for s, v in b["channels"].items():
        ch[ren[s]] = dict(v, pieces=[[p[0], round(p[1] + off, 3), p[2], p[3]] for p in (v.get("pieces") or [])])
    meta = dict(synthetic="sim", mask_eot=True, overlay=[a["conv_id"], b["conv_id"]], offset=round(off, 3))
    return dict(conv_id=f"{key}_{i:06d}", corpus=key, lang=a["lang"], split="train", duration_s=round(max(a["duration_s"], b["duration_s"] + off), 3),
                speakers=list(a["speakers"]) + ["c", "d"], utterances=utts, channels=ch, observed_until_s=None, meta=meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True); ap.add_argument("--out", required=True); ap.add_argument("--key", default="sim_en-ovl3")
    ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--max-offset", type=float, default=5.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dst = os.path.join(a.out, f"{a.key}.refined.dialogues.jsonl")
    if os.path.exists(dst): raise SystemExit(f"이미 있다: {dst}")
    pool = []
    for f in a.src:
        for l in open(f, encoding="utf-8"):
            d = json.loads(l)
            if len(d["speakers"]) == 2 and d.get("split", "train") == "train" and not d["meta"].get("trans"): pool.append(d)
    langs = {d["lang"] for d in pool}
    if len(langs) != 1: raise SystemExit(f"언어가 하나여야 한다: {langs}")
    rng = random.Random(a.seed); os.makedirs(a.out, exist_ok=True); n = 0; dur = 0.0
    with open(dst + ".tmp", "w", encoding="utf-8") as w:
        for r in range(a.rounds):
            for x in pool:
                y = rng.choice(pool)
                while y is x: y = rng.choice(pool)
                d = overlay(x, y, rng.uniform(0.0, a.max_offset), a.key, n); n += 1; dur += d["duration_s"]
                w.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(dst + ".tmp", dst)
    json.dump(dict(sessions=n, hours=round(dur / 3600, 2), pool=len(pool), src=a.src, rounds=a.rounds, max_offset=a.max_offset, seed=a.seed, key=a.key),
              open(os.path.join(a.out, f"{a.key}.stats.json"), "w"), ensure_ascii=False, indent=1)
    print(f"대화 {n} · {dur / 3600:.1f} h · 2 인 재료 {len(pool)} → {dst}", flush=True)


if __name__ == "__main__":
    main()
