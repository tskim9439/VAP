from vapasr.data.approved_training import alignment_density_problem, affinity, join_supervision, pack_streams, stable_silence
from vapasr.data.interleave import build_interleaved, Specials


def rows(key="a", duration=3.0):
    selected = dict(key=key, source="kspon-full", corpus="kspon", split="train", subset="train-01",
                    lang="Korean", parent_id="p", utt_id=key, speaker=None, speaker_scope="source_or_unknown",
                    group_id=None, timeline="synthetic_stream", start_s=0, end_s=duration,
                    audio=dict(path=f"/x/d/{key}.pcm", offset_s=None, duration_s=None),
                    recommended_training_target=dict(text="안녕"),
                    auto_selection=dict(expected_waveform_sha256="wave"))
    aligned = dict(key=key, target_text="안녕", alignment_ok=True, training_eligible=False,
                   waveform_sha256="wave", duration_s=duration, target_sha256="text",
                   words=[dict(text="안녕")], tokens=[dict(id=7, end_s=2.0)],
                   timing_quality="forced_aligned_unverified")
    return selected, aligned


def test_join_is_asr_only_and_verbatim():
    s, a = rows(); r = join_supervision(s, a, "approval")
    assert r["target_text"] == "안녕" and r["target_normalization"] == "none"
    assert r["asr_training_eligible"] is True
    assert r["speaker_training_eligible"] is r["turn_training_eligible"] is False


def test_join_rejects_changed_audio_or_text():
    s, a = rows(); a["target_text"] = "다름"
    try: join_supervision(s, a, "approval")
    except ValueError as e: assert "target mismatch" in str(e)
    else: assert False


def test_pack_shifts_token_and_audio_offsets_deterministically():
    joined = []
    for key in ("a", "b"):
        s, a = rows(key); joined.append(join_supervision(s, a, "approval"))
    packed = pack_streams(joined, "approval")
    assert len(packed) == 1
    manifest, align = packed[0]
    gap = stable_silence("a", "b")
    assert manifest["segments"][1]["offset_s"] == 3.0 + gap
    assert align["utts"][1]["tokens"][0]["end_time"] == 5.0 + gap
    assert manifest["meta"]["speaker_supervision"] is False


def test_original_dialogue_never_crosses_parent_or_speaker():
    joined = []
    for key, parent, speaker in (("a", "p1", "0"), ("b", "p1", "1"), ("c", "p2", "0")):
        s, a = rows(key); s.update(source="ami", timeline="original_dialogue", parent_id=parent,
                                  group_id=parent, speaker=speaker)
        joined.append(join_supervision(s, a, "approval"))
    assert len(pack_streams(joined, "approval")) == 3


def test_flush_preserves_equal_timestamp_token_order():
    sp = Specials(next_audio=90, empty_audio=91, spk=(92, 93))
    chunks, _ = build_interleaved([[(3200, 1.0), (13, 1.0)]], .95, sp,
                                  delay_frames=2, add_spk_tags=False)
    assert chunks[-1][1] == [3200, 13, 91]


def test_alignment_density_guard_is_explicit():
    s, a = rows(); a["tokens"] = [dict(id=i, end_s=2.0) for i in range(9)]
    assert alignment_density_problem(join_supervision(s, a, "approval")) == "same_end_time_token_burst"
