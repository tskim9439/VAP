#!/usr/bin/env python
"""자기 채널에서 들리지 않는 발화 표시(2026-10-26, CHiME-6 train — 착용 마이크가 죽은 구간): 정제된 대화의 화자 채널마다 20 ms 프레임 dB 를 재고,
발화 구간 dB 중앙값이 채널 바닥(10 백분위) + --min-snr-db 보다 낮은 발화에 flags "own_ch_inaudible" 를 붙인다. 그런 발화의 정렬 토큰은 죽은 채널 위에서 나온 것이라
믿을 수 없다 — 학습은 그 구간을 전사 없는 음성처럼 다룬다(dialogue_dataset._bad_spans: 손실 마스크, 활동 타깃은 유지). 전사 시각(발화 구간)은 그대로다.
--drop-conv 의 대화는 통째로 뺀다(예: 한 화자 채널이 세션 내내 죽어 mono 혼합 게인이 +65 dB 로 잡음을 키우는 chime6:S05).
  python experiments/p2_flag_channel_level.py --src <in>.refined.dialogues.jsonl --out <out>.refined.dialogues.jsonl [--min-snr-db 6] [--drop-conv a,b] [--procs 16]
출력이 이미 있으면 멈춘다(덮어쓰지 않는다). 통계 <out>.stats.json(화자 채널마다 바닥·발화 중앙 dB·표시 비율). 읽기 전용 입력."""
import argparse, json, multiprocessing as mp, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLAG, HOP_MS = "own_ch_inaudible", 20.0

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("--min-snr-db", type=float, default=6.0)
ap.add_argument("--drop-conv", default="", help="통째로 뺄 conv_id(쉼표)"); ap.add_argument("--procs", type=int, default=16)
a = ap.parse_args() if __name__ == "__main__" else None


def flag_dialogue(d, min_snr_db: float, sr: int = 16000):
    """Dialogue → 화자 채널 통계 목록(d.utterances 의 flags 를 제자리에서 고친다)."""
    from vapasr.data.dialogue_mix import load_channel
    from vapasr.data.vad import frame_db
    fps = 1000.0 / HOP_MS; out = []
    for spk in d.speakers:
        ref = d.channels.get(spk); us = [u for u in d.utterances if u.speaker == spk]
        if ref is None or not us: continue
        db = frame_db(load_channel(ref, d.duration_s, sr), sr, HOP_MS); floor = float(np.percentile(db, 10)); meds = []
        for u in us:
            i0 = min(int(u.start * fps), len(db) - 1); med = float(np.median(db[i0: max(int(u.end * fps), i0 + 1)])); meds.append(med)
            if med < floor + min_snr_db and FLAG not in (u.flags or []): u.flags = list(u.flags or []) + [FLAG]
        meds = np.array(meds)
        out.append(dict(conv=d.conv_id, speaker=spk, utts=len(us), floor_db=round(floor, 1), utt_median_db=round(float(np.median(meds)), 1),
                        flagged=int((meds < floor + min_snr_db).sum()), flagged_frac=round(float((meds < floor + min_snr_db).mean()), 3)))
    return out


def one(line):
    from vapasr.data.dialogue import Dialogue
    d = Dialogue.from_json(line); st = flag_dialogue(d, a.min_snr_db); return d.to_json(), st


def main():
    if os.path.exists(a.out): sys.exit(f"이미 있다(덮어쓰지 않는다): {a.out}")
    drop = {x for x in a.drop_conv.split(",") if x}; lines = [l for l in open(a.src, encoding="utf-8") if l.strip()]
    keep = [l for l in lines if json.loads(l)["conv_id"] not in drop]; t0 = time.time()
    with mp.get_context("fork").Pool(max(1, min(a.procs, len(keep)))) as pool: res = pool.map(one, keep, chunksize=1)
    tmp = a.out + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fo:
        for js, _ in res: fo.write(js + "\n")
    os.replace(tmp, a.out)
    chans = [s for _, st in res for s in st]; n = sum(c["utts"] for c in chans); f = sum(c["flagged"] for c in chans)
    summary = dict(src=a.src, dialogues=len(keep), dropped=sorted(drop & {json.loads(l)["conv_id"] for l in lines}), min_snr_db=a.min_snr_db, utts=n, flagged=f,
                   flagged_frac=round(f / max(n, 1), 4), sec=round(time.time() - t0, 1))
    json.dump(dict(summary, channels=chans), open(a.out.replace(".jsonl", "") + ".stats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    for c in sorted(chans, key=lambda c: -c["flagged_frac"])[:8]: print("  ", c, flush=True)
    print(f"FLAG_OK {a.out}", flush=True)


if __name__ == "__main__":
    main()
