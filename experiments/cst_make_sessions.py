#!/usr/bin/env python3
"""CST 매니페스트(vapasr/cst/schema.py JSONL) → 세션 연속 음성 + 타임라인 라벨 (겹침 없는 Natural/L0 조건).

  python experiments/cst_make_sessions.py --manifest /Volumes/Samsung_T5/VAPKT-DB/TAXI/cst/taxi-sessions.jsonl \\
      --audio-root /Volumes/Samsung_T5/VAPKT-DB/TAXI/cst/wav16k --sr 16000 --gap natural \\
      --out /Volumes/Samsung_T5/VAPKT-DB/TAXI/cst/sessions-natural

턴 파일 앞뒤 무음은 기본으로 잘라 낸다(trim_silence; 잘라 낸 길이는 turns[].trimmed_s). usable 턴만 대화 순서대로 이어 붙인다(garbage 턴은 빼고 그 자리는 meta.skipped 에 남긴다). 세션마다
  <out>/<세션>/mix.wav (mono)  <out>/<세션>/2ch.wav (speakers 순서대로 채널)  <out>/<세션>/timeline.json
과 전체 요약 <out>/sessions.jsonl(세션 메타 + 턴별 start_s·end_s·전사·참조)을 쓴다. seed = 세션 이름 해시 + --seed."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vapasr.cst.schema import Session
from vapasr.cst.timeline import overlap_ratio, place_turns, render, trim_silence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--audio-root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--sr", type=int, default=16000); p.add_argument("--gap", default="natural", choices=["natural", "mediated"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-trim", action="store_true", help="턴 파일 앞뒤 무음을 자르지 않음(기본은 자름 — trim_silence)")
    a = p.parse_args()
    import numpy as np
    import soundfile as sf
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summ = []; tot = 0.0
    for line in open(a.manifest):
        s = Session.from_json(line); turns = s.usable_turns()
        if not turns:
            continue
        audio, trims = {}, {}
        for t in turns:
            x, sr = sf.read(str(Path(a.audio_root) / t.audio), dtype="float32")
            assert sr == a.sr and x.ndim == 1, (t.audio, sr)
            if not a.no_trim:
                i, j = trim_silence(x, a.sr); trims[t.turn_id] = (round(i / a.sr, 4), round((len(x) - j) / a.sr, 4)); x = x[i:j]
            audio[t.turn_id] = x
        for t in turns:                                                    # 자른·리샘플한 길이로 갱신
            t.duration_s = round(len(audio[t.turn_id]) / a.sr, 4)
        seed = int(hashlib.sha1(s.session_id.encode()).hexdigest()[:8], 16) + a.seed
        placed = place_turns(turns, gap=a.gap, seed=seed)
        speakers = list(s.speakers)
        mono, st = render(placed, audio, speakers, a.sr)
        d = out / s.session_id.split("/")[-1]; d.mkdir(exist_ok=True)
        sf.write(str(d / "mix.wav"), mono, a.sr, subtype="PCM_16"); sf.write(str(d / "2ch.wav"), st, a.sr, subtype="PCM_16")
        byid = {t.turn_id: t for t in turns}
        tl = [dict(turn_id=q.turn_id, speaker=q.speaker, lang=byid[q.turn_id].lang, start_s=q.start_s, end_s=q.end_s, gap_before_s=q.gap_before_s,
                   transcript=byid[q.turn_id].transcript, translation=byid[q.turn_id].translation.text if byid[q.turn_id].translation else None,
                   translation_lang=byid[q.turn_id].translation.lang if byid[q.turn_id].translation else None,
                   trimmed_s=trims.get(q.turn_id)) for q in placed]
        rec = dict(session_id=s.session_id, corpus=s.corpus, redistributable=s.redistributable, speakers=s.speakers, channels=speakers,
                   sample_rate=a.sr, gap=a.gap, seed=seed, trim=not a.no_trim, duration_s=round(len(mono) / a.sr, 3), overlap_ratio=round(overlap_ratio(placed), 4),
                   skipped=[t.turn_id for t in s.turns if not t.usable], mix=str((d / "mix.wav").relative_to(out)), two_channel=str((d / "2ch.wav").relative_to(out)),
                   turns=tl)
        (d / "timeline.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1)); summ.append(rec); tot += rec["duration_s"]
    with (out / "sessions.jsonl").open("w") as f:
        for r in summ:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(dict(sessions=len(summ), hours=round(tot / 3600, 3), gap=a.gap,
                          speech_ratio=round(sum(t["end_s"] - t["start_s"] for r in summ for t in r["turns"]) / max(tot, 1e-9), 3)), ensure_ascii=False))


if __name__ == "__main__":
    main()
