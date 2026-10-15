import glob, collections, sys
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.corpora import parse_srt, SPEECH_LABELS, NONSPEECH_LABELS
same = collections.Counter(); alone = collections.Counter(); hrs_alone = collections.Counter(); ex = collections.defaultdict(list)
for f in sorted(glob.glob("/soundai/DB/raw/otoSpeech16k/*/speaker_*_annotation_a.srt")):
    e_ = parse_srt(f)
    for i, (s, e, lab, t) in enumerate(e_):
        if lab in SPEECH_LABELS or lab in NONSPEECH_LABELS or not t.strip(): continue
        nb = [e_[j] for j in (i - 1, i + 1) if 0 <= j < len(e_)]
        if any(x[3] == t and (abs(x[0] - e) < 0.05 or abs(s - x[1]) < 0.05) for x in nb): same[lab] += 1
        else:
            alone[lab] += 1; hrs_alone[lab] += e - s
            if len(ex[lab]) < 2: ex[lab].append((f.split("/")[-2], round(s, 1), round(e, 1), t[:60], [(x[2], x[3][:30]) for x in nb]))
for lab in sorted(set(same) | set(alone), key=lambda l: -(same[l] + alone[l])):
    print(f"{lab[:40]:40s} 이웃과 같은 전사 {same[lab]:6d} · 단독 {alone[lab]:6d} ({hrs_alone[lab] / 3600:.2f} h)  예 {ex[lab][:1]}")
