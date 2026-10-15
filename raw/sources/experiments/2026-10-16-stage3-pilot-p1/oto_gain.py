# otoSpeech mono 혼합 이득: 옛 발화 구간(v1-dupsrt) vs 새 구간(재빌드 3)으로 화자별 RMS 이득 차이(dB). 표본 40 대화. 읽기만 한다.
import json, random, math, sys
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.dialogue import Dialogue
from vapasr.data.dialogue_mix import load_channel, _rms_over
D = "/soundai/users/tskim/VAPKT-data/data/phase2"
old = {json.loads(l)["conv_id"]: l for l in open(f"{D}/otoSpeech.v1-dupsrt.refined.dialogues.jsonl", encoding="utf-8")}
new = {json.loads(l)["conv_id"]: l for l in open(f"{D}/otoSpeech.refined.dialogues.jsonl", encoding="utf-8")}
ids = sorted(old); random.Random(0).shuffle(ids); diffs = []
for cid in ids[:40]:
    a, b = Dialogue.from_json(old[cid]), Dialogue.from_json(new[cid])
    for spk in a.speakers:
        x = load_channel(a.channels[spk], a.duration_s, 16000)
        ra = _rms_over(x, [(u.start, u.end) for u in a.utterances if u.speaker == spk], 16000)
        rb = _rms_over(x, [(u.start, u.end) for u in b.utterances if u.speaker == spk], 16000)
        diffs.append(20 * math.log10(max(ra, 1e-9) / max(rb, 1e-9)))
diffs.sort(); n = len(diffs)
print(f"이득 차이(dB, 새 − 옛 = 옛 RMS / 새 RMS): n {n} · 중앙값 {diffs[n // 2]:.2f} · p10 {diffs[n // 10]:.2f} · p90 {diffs[9 * n // 10]:.2f} · 최대 |{max(abs(d) for d in diffs):.2f}|")
