#!/usr/bin/env python
"""AI Hub 감정 태깅 자유대화 — 라벨(JSON)만으로 코퍼스 통계. 오디오 불필요.
스키마: Wav{SamplingRate,NumberOfChannel}, File{FileName,FileLength}, Speaker1/2{ID,...},
        Conversation[{Text,StartTime,EndTime,SpeakerNo,...emotion}], ConversationInfo{Domain,...}
실행: python experiments/aihub_label_stats.py <라벨 루트> [--out json]
"""
import os, sys, json, glob, argparse, time, collections
import numpy as np
def _f(x): return float(str(x).replace(',', ''))   # AI Hub 숫자는 천 단위 구분자 포함 ('1,003.24')
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", default=None); a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.root, "**", "*.json"), recursive=True))
print(f"json {len(files)}개 under {a.root}")

flen, utt_dur, gaps, overlaps, n_utt, n_switch, sr_set, ch_set, dec = [], [], [], [], [], [], collections.Counter(), collections.Counter(), collections.Counter()
domains = collections.Counter(); pairs = collections.Counter(); per_split = collections.defaultdict(lambda: [0, 0.0])
bad = 0
for f in files:
    try: d = json.load(open(f, encoding="utf-8"))
    except Exception: bad += 1; continue
    split = "Training" if "/Training/" in f else "Validation"
    L = _f(d["File"]["FileLength"]); flen.append(L); per_split[split][0] += 1; per_split[split][1] += L
    sr_set[d["Wav"].get("SamplingRate")] += 1; ch_set[d["Wav"].get("NumberOfChannel")] += 1
    domains[d.get("ConversationInfo", {}).get("Domain", "?")] += 1
    pairs[(d["Speaker1"]["ID"], d["Speaker2"]["ID"])] += 1
    conv = sorted(d.get("Conversation", []), key=lambda u: _f(u["StartTime"]))
    n_utt.append(len(conv)); sw = 0
    for i, u in enumerate(conv):
        s, e = _f(u["StartTime"]), _f(u["EndTime"]); utt_dur.append(e - s)
        st = u["StartTime"]; dec[len(st.split(".")[1]) if "." in st else 0] += 1
        if i:
            p = conv[i - 1]; ps, pe = _f(p["StartTime"]), _f(p["EndTime"])
            if u["SpeakerNo"] != p["SpeakerNo"]:
                sw += 1; g = s - pe
                (gaps if g >= 0 else overlaps).append(g if g >= 0 else -g)
    n_switch.append(sw)

def q(v, unit=""):
    v = np.array(v, dtype=float)
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)), max=float(v.max()), unit=unit) if len(v) else None
tot_h = sum(flen) / 3600
S = dict(files=len(files), bad=bad, total_hours=tot_h, per_split={k: dict(files=v[0], hours=v[1] / 3600) for k, v in per_split.items()},
         sampling_rate=dict(sr_set), channels=dict(ch_set), timestamp_decimals=dict(dec),
         file_length_s=q(flen, "s"), utterances_per_file=q(n_utt), speaker_switches_per_file=q(n_switch),
         utterance_dur_s=q(utt_dur, "s"), gap_s=q(gaps, "s"), overlap_s=q(overlaps, "s"),
         switch_total=int(sum(n_switch)), overlap_ratio_of_switches=(len(overlaps) / max(1, len(gaps) + len(overlaps))),
         unique_speaker_pairs=len(pairs), domains=dict(domains.most_common(12)))
# gap 히스토그램 (turn-taking 핵심): 0-0.2 / 0.2-0.5 / 0.5-1 / 1-2 / >2 s
edges = [0, 0.2, 0.5, 1.0, 2.0, 1e9]; h = np.histogram(gaps, bins=edges)[0] if gaps else []
S["gap_hist"] = {f"{edges[i]}-{edges[i+1] if edges[i+1] < 1e8 else 'inf'}s": int(c) for i, c in enumerate(h)}
print(json.dumps(S, indent=1, ensure_ascii=False))
if a.out: json.dump(S, open(a.out, "w"), indent=1, ensure_ascii=False); print("saved", a.out)
