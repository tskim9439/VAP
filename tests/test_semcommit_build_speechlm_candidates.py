import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "semcommit_build_speechlm_candidates",
    Path(__file__).resolve().parents[1] / "experiments/semcommit_build_speechlm_candidates.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def sample():
    aligned = dict(key="abc", source="SLM-SPEECH-000004", lang="Korean", duration_s=10.0,
                   target_text="운동을 해야지.", audio=dict(path="/x.tar::a.wav"),
                   alignment_ok=True, training_eligible=False, dedup_status="unverified",
                   source_shard="part-000000", timing_quality="forced_aligned_unverified",
                   words=[dict(char_start=0, char_end=2, end_s=0.5),
                          dict(char_start=2, char_end=3, end_s=0.6),
                          dict(char_start=4, char_end=7, end_s=1.0)],
                   tokens=[dict(id=1, char_start=0, char_end=2, end_s=0.5),
                           dict(id=2, char_start=2, char_end=3, end_s=0.6),
                           dict(id=3, char_start=3, char_end=7, end_s=1.0),
                           dict(id=4, char_start=7, char_end=8, end_s=1.0)])
    record = dict(key="abc", split="train", selection_state="KEEP_ASR_SILVER",
                  source_manifest=dict(split="train", num_channels=1,
                                       corpus="AIHub general free conversation",
                                       original_transcript="운동을 해야지"))
    return aligned, record


def test_candidate_preserves_unapproved_provenance_and_punctuation():
    aligned, record = sample()
    row, reason = mod.build_candidate(aligned, record)
    assert reason is None
    assert row["training_eligible"] is False and row["candidate_only"] is True
    assert row["duration_within_recipe"] is True
    assert row["text"] == "운동을 해야지"
    assert row["words"][0]["a"] == 0 and row["words"][0]["b"] == 2
    assert row["words"][1]["a"] == 2 and row["words"][1]["b"] == 4
    assert row["words"][1]["tags"] == ["punct_final"]


def test_candidate_rejects_eval_and_keeps_short_as_candidate():
    aligned, record = sample()
    record["split"] = "test"
    assert mod.build_candidate(aligned, record)[1] == "not_train_silver"
    record["split"] = "train"
    aligned["duration_s"] = 2.0
    row, reason = mod.build_candidate(aligned, record)
    assert reason is None and row["duration_within_recipe"] is False


def test_candidate_rejects_bad_mapping_and_unapproved_elevation():
    aligned, record = sample()
    aligned["training_eligible"] = True
    assert mod.build_candidate(aligned, record)[1] == "alignment_or_approval"
    aligned["training_eligible"] = False
    aligned["tokens"][2]["char_start"] = 1  # crosses prior lexical word
    assert mod.build_candidate(aligned, record)[1] == "token_mapping"


def test_tokenizer_path_strips_punctuation_into_tags(monkeypatch):
    """Tokenizer words keep punctuation inside ('해야지.'); semcommit words must be lexical with punct_final as a tag."""
    aligned, record = sample()
    canonical = [dict(i=0, text="운동을", a=0, b=2, end_time=0.6), dict(i=1, text="해야지.", a=2, b=4, end_time=1.0)]
    monkeypatch.setattr(mod, "split_words", lambda ids, times, tok, text=None: [dict(w) for w in canonical])
    row, reason = mod.build_candidate(aligned, record, tokenizer=object())
    assert reason is None and row["text"] == "운동을 해야지"
    assert [(w["text"], w["a"], w["b"], w["tags"]) for w in row["words"]] == [("운동을", 0, 2, []), ("해야지", 2, 4, ["punct_final"])]


def test_lexical_words_merge_and_ellipsis():
    W = lambda *ts: [dict(i=i, text=t, a=i, b=i + 1, end_time=0.1 * (i + 1)) for i, t in enumerate(ts)]
    out, why = mod.lexical_words(W("그래서,", "갔어요", "?"))            # 단독 구두점은 앞 단어로 합친다(토큰 범위 유지)
    assert why is None and [(w["text"], w["a"], w["b"], w["tags"]) for w in out] == [("그래서", 0, 1, ["punct_comma"]), ("갔어요", 1, 3, ["punct_final"])]
    assert out[1]["end_time"] == 0.2                                   # 합쳐도 단어 시각은 그대로
    out, _ = mod.lexical_words(W("\"", "네", "그런데..."))              # 머리 구두점은 다음 단어로, 말줄임은 문장 끝 아님
    assert [(w["text"], w["a"], w["b"], w["tags"]) for w in out] == [("네", 0, 2, []), ("그런데", 2, 3, [])]
    assert mod.lexical_words(W(".", "!"))[1] == "punct_only"
    assert mod.punct_tags("…") == [] and mod.punct_tags("?!") == ["punct_final"] and mod.punct_tags(".") == ["punct_final"]
