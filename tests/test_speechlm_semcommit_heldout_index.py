import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import speechlm_semcommit_heldout_index as mod


def test_heldout_index_dedups_segments_and_marks_path_only(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(wav, np.full(16000, 0.1, dtype=np.float32), 16000)
    manifests = tmp_path / "manifests"
    for name in (*mod.DEFAULT_HASH_SETS, *mod.DEFAULT_PATH_SETS):
        root = manifests / name
        root.mkdir(parents=True)
        seg = dict(path=str(wav), dur_s=1.0)
        (root / "streams.jsonl").write_text(json.dumps(dict(segments=[seg, seg])) + "\n")
    gold = tmp_path / "gold"
    (gold / "wav").mkdir(parents=True)
    gold_wav = gold / "wav" / "g.wav"
    sf.write(gold_wav, np.full(16000, 0.2, dtype=np.float32), 16000)
    (gold / "words-gs-test.jsonl").write_text(json.dumps(dict(segments=[dict(
        path="/old/rack4/g.wav", dur_s=1.0)])) + "\n")
    entries, sources = mod.collect(manifests, gold, mod.DEFAULT_HASH_SETS, mod.DEFAULT_PATH_SETS)
    assert len(entries) == 2
    assert len(sources) == 6
    output = tmp_path / "out"
    result = mod.build(manifests, gold, output, workers=2)
    assert result["complete"] is True
    assert result["counts"]["ok"] == 2
    assert result["counts"]["set:aihub71631-dev"] == 1
    indexed = list(mod.rows(output / "identities.jsonl"))
    assert all(len(r["waveform_sha256"]) == 64 for r in indexed)
    assert mod.file_sha256(output / "identities.jsonl") == result["identities_sha256"]
