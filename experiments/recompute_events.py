#!/usr/bin/env python
"""manifest 디렉토리의 npz 에서 (오디오 없이) 이벤트·τ 를 현재 규칙으로 다시 계산해 덮어쓰고, 채널별 발화 비율·품질 플래그를 stats 에 기록.
python experiments/recompute_events.py <manifest dir>   → 손상 npz 는 건너뛰고 stats.json 의 bad_files 에 기록
"""
import os, sys, json, glob, time, zipfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data.targets import derive_events, time_to_next_onset, EVENT_TYPES
d = sys.argv[1]; t0 = time.time(); ev_tot = {k: 0 for k in EVENT_TYPES}; n = 0; hours = 0.0; bad = []; flags = {}
for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
    try:
        z = np.load(p); va = z["vad"].astype(np.float32); fh = float(z["frame_hz"]); dur = float(z["duration"])
    except Exception as e:
        bad.append(os.path.basename(p)); continue
    ev = derive_events(va, fh); ttn = time_to_next_onset(va, fh)
    np.savez_compressed(p, vad=z["vad"], tau_bin=ttn["bin"].astype(np.int8), censored=ttn["censored"],
                        events=np.array([(t, s, EVENT_TYPES.index(k)) for t, s, k in ev], dtype=np.float32).reshape(-1, 3), frame_hz=fh, duration=dur)
    sr = va.mean(0).tolist(); fid = os.path.basename(p)[:-4]
    fl = []
    if min(sr) < 0.05: fl.append("one_channel_silent")
    if dur < 30: fl.append("too_short")
    if len(ev) == 0: fl.append("no_events")
    if fl: flags[fid] = dict(speech_ratio=[round(x, 3) for x in sr], duration=round(dur, 1), flags=fl)
    for _, _, k in ev: ev_tot[k] += 1
    n += 1; hours += dur / 3600
sp = os.path.join(d, "stats.json"); st = json.load(open(sp)) if os.path.exists(sp) else {}
st.update(n=n, hours=hours, events=ev_tot, events_recomputed=time.strftime("%Y-%m-%d %H:%M"), bad_files=bad, flagged=flags, n_flagged=len(flags))
json.dump(st, open(sp, "w"), indent=1, ensure_ascii=False)
print(json.dumps(dict(n=n, hours=round(hours, 1), events=ev_tot, bad=bad, n_flagged=len(flags), flag_kinds={k: sum(k in v["flags"] for v in flags.values()) for k in ("one_channel_silent", "too_short", "no_events")}, sec=round(time.time() - t0)), ensure_ascii=False))
