import glob, collections, sys
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.corpora import parse_srt, SPEECH_LABELS, NONSPEECH_LABELS
n = collections.Counter(); dur = collections.Counter(); ex = {}
for f in glob.glob("/soundai/DB/raw/otoSpeech16k/*/speaker_*_annotation_a.srt"):
    for s, e, lab, t in parse_srt(f):
        n[lab] += 1; dur[lab] += e - s
        if lab not in ex and t.strip(): ex[lab] = t[:70]
tot = sum(dur.values())
for lab, c in n.most_common():
    tag = "SPEECH" if lab in SPEECH_LABELS else ("NONSPEECH" if lab in NONSPEECH_LABELS else "!! 미분류")
    print(f"{lab[:40]:40s} {c:7d} {dur[lab] / 3600:7.2f} h ({dur[lab] / tot:.1%}) {tag:10s} 예: {ex.get(lab, '')!r}")
