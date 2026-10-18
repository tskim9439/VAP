import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import speechlm_semcommit_review_bundle as mod


def test_review_sample_per_cell_and_html_escaping(tmp_path):
    sample = tmp_path / "review-sample.jsonl"
    sample.write_text("".join(json.dumps(dict(key=str(i), review_cell=["src", "short"],
                                              source_shard="part-0")) + "\n" for i in range(4)))
    assert [r["key"] for r in mod.select(sample, per_cell=2)] == ["0", "1"]
    card = dict(source="<unsafe>", review_cell=["src", "short"], key="k", duration_s=1.0,
                rms=0.1, clipping_ratio=0.0, acoustic_flags=[], audio_file="audio/1.wav",
                target_text="<script>", original_transcript="<x>", qc_state="QC_PASS_CANDIDATE",
                zero_duration_words=0)
    rendered = mod._html_card(card)
    assert "&lt;script&gt;" in rendered and "<script>" not in rendered
