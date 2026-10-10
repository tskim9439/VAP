import pytest

from vapasr.data.ast_eval import al, laal, split_runs, word_emission_times

SPK_A, SPK_B, TRANS = 151707, 151708, 151726


class _Tok:
    """byte-level pieces like Qwen: 'Ġ' = leading space."""
    def __init__(self, pieces):
        self.pieces = pieces

    def convert_ids_to_tokens(self, ids):
        return [self.pieces[i] for i in ids]


def test_split_runs_streaming_and_final():
    emitted = [(77, SPK_A), (77, 1), (77, TRANS), (77, 11), (78, SPK_A), (78, 2), (81, 3), (81, TRANS), (81, 12), (81, 13)]
    src, tr = split_runs(emitted, (SPK_A, SPK_B), TRANS)
    assert src == [(77, 1), (78, 2), (81, 3)] and tr == [(77, 11), (81, 12), (81, 13)]
    src, tr = split_runs([SPK_B, 5, 6, TRANS, 7], (SPK_A, SPK_B), TRANS)          # final mode: plain ids
    assert src == [(None, 5), (None, 6)] and tr == [(None, 7)]
    src, tr = split_runs([(0, 9), (1, TRANS), (1, 10)], (SPK_A, SPK_B), TRANS)     # untagged start counts as transcript
    assert src == [(0, 9)] and tr == [(1, 10)]


def test_word_emission_times_multi_token_word():
    tok = _Tok(["Es", "Ġge", "ht", "Ġum"])
    text, times = word_emission_times(tok, [(77, 0), (80, 1), (81, 2), (81, 3)])
    assert text == "Es geht um"
    assert times == pytest.approx([78 * 0.08, 82 * 0.08, 82 * 0.08])            # 'geht' completes with token 'ht' at chunk 81


def test_laal_and_al():
    times = [1.0, 2.0, 3.0, 5.0]                                                  # source ends at 4 s
    # tau = 4 (first word at/after 4 s); |Y| = 4, |Y*| = 8 -> LAAL uses 8
    assert al(times, 4.0) == pytest.approx(((1 - 0) + (2 - 1) + (3 - 2) + (5 - 3)) / 4)
    assert laal(times, 4.0, 8) == pytest.approx(((1 - 0) + (2 - 0.5) + (3 - 1) + (5 - 1.5)) / 4)
    assert laal([], 4.0, 3) is None
