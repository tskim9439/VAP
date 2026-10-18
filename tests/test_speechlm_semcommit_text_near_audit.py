from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import speechlm_semcommit_text_near_audit as mod


def test_near_text_screen_flags_similar_long_text_not_short_common_phrase():
    value = mod.normalized("The speaker's long, distinctive sentence!")
    index = {("English", anchor): [(value, 4.0, "librispeech-test")]
             for anchor in mod.anchors(value)}
    hit = mod.match("the speakers long distinctive sentence", "English", 4.1, index)
    assert hit and hit["heldout_set"] == "librispeech-test"
    assert mod.match("yes", "English", 1.0, index) is None
    assert mod.match("the speakers long distinctive sentence", "English", 20, index) is None
