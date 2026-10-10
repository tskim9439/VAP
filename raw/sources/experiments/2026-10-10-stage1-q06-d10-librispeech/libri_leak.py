import json, re, collections
from pathlib import Path
SNAP = "/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637"
ASR = "/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list"
M = "/soundai/users/tskim/VAPKT-data/data/evaluation/single-turn-librispeech-v1"
test_spk, test_chap = set(), set()
for s in ("test-clean", "test-other"):
    for l in open(f"{M}/manifest-{s}.jsonl"):
        u = json.loads(l)["id"].rsplit("/", 1)[1]; sp, ch, _ = u.split("-"); test_spk.add(sp); test_chap.add((sp, ch))
pat = re.compile(r'"id": "ls-([a-z]+-[a-z]+(?:-\d+)?)-(\d+)-(\d+)-')
for name, lists in (("v035 snapshot (main+short)", [f"{SNAP}/main-words.list", f"{SNAP}/short-words.list"]), ("Stage1 ASR words", [ASR])):
    files = [p.strip() for L in lists for p in open(L) if "librispeech" in p]
    split, spk, rows = collections.Counter(), set(), 0
    for f in files:
        for line in open(f):
            m = pat.search(line)
            if not m: continue
            rows += 1; split[m.group(1)] += 1; spk.add(m.group(2))
    print(f"{name}: librispeech files={len(files)} rows={rows} splits={dict(split)} speakers={len(spk)} overlap_with_test_speakers={len(spk & test_spk)}")
print("test speakers", len(test_spk))
