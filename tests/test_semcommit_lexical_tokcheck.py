"""SemCommitDataset tokenizer 관문 × speechlm lexical words — 빌더가 가장자리 구두점을 태그로 뗀 단어('world' + punct_final)는
split_words 의 'world.' 와 달라도 같은 lexical_words 규칙으로 맞으면 통과해야 한다(2026-09-27 v0.3.5 부분 DB 학습 로드 실패)."""
import json
import pytest
from vapasr.data.semcommit_dataset import SemCommitDataset
from vapasr.data.semcommit_words import lexical_words


class PieceTok:
    def __init__(self, pieces): self.v = {"<|audio_pad|>": 149999}; self.pieces = dict(pieces)
    def add_tokens(self, toks, special_tokens=True):
        for t in toks: self.v.setdefault(t, 150000 + len(self.v) - 1)
    def convert_tokens_to_ids(self, t): return self.v[t]
    def __call__(self, text, add_special_tokens=False, **kw): return {"input_ids": [ord(c) % 1000 + 1 for c in text]}
    def convert_ids_to_tokens(self, ids): return [self.pieces.get(int(i)) for i in ids]


PIECES = {1001: "hello", 1002: "Ġworld", 1003: ".", 1004: "Ġso", 1005: "Ġdone", 1006: "?"}
TOKS = [[1001, 0.6], [1002, 0.9], [1003, 0.9], [1004, 1.3], [1005, 1.7], [1006, 1.7]]


def _row(words):
    return dict(id="slm-1", set="librispeech-test", lang="English", K=25, duration_s=2.0, text="hello world so done", tokens=TOKS,
                segments=[dict(path="/x/a.pcm", offset_s=0.1, silence_before_s=0.1, dur_s=1.8, utt_id="u1", raw_text="x")], words=words)


def _write(tmp_path, words):
    wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"
    wp.write_text(json.dumps(_row(words)) + "\n")
    lp.write_text(json.dumps(dict(id="slm-1", lang="English", candidates=[dict(after_word=1, grade="A", why="t", stageA=True)], turn_end=True)) + "\n")
    return str(wp), str(lp)


LEX = [dict(i=0, text="hello", a=0, b=1, end_time=0.6, seg=0, tags=[]), dict(i=1, text="world", a=1, b=3, end_time=0.9, seg=0, tags=["punct_final"]),
       dict(i=2, text="so", a=3, b=4, end_time=1.3, seg=0, tags=[]), dict(i=3, text="done", a=4, b=6, end_time=1.7, seg=0, tags=["punct_final"])]


def test_lexical_words_accepted(tmp_path):
    ds = SemCommitDataset(*_write(tmp_path, LEX), PieceTok(PIECES), online=False)
    assert ds.stats["tok_check"] == 1 and len(ds) == 1


def test_raw_split_words_still_accepted(tmp_path):
    raw = [dict(LEX[0]), dict(LEX[1], text="world."), dict(LEX[2]), dict(LEX[3], text="done?")]
    assert len(SemCommitDataset(*_write(tmp_path, raw), PieceTok(PIECES), online=False)) == 1


@pytest.mark.parametrize("pieces", [{**PIECES, 1004: "Ġdone", 1005: "Ġso"},        # different word order
                                    {**PIECES, 1003: "Ġx"}])                       # '.' becomes a word → different boundaries
def test_mismatch_still_rejected(tmp_path, pieces):
    with pytest.raises(ValueError, match="tokenizer_mismatch"):
        SemCommitDataset(*_write(tmp_path, LEX), PieceTok(pieces), online=False)


def test_lexical_words_shared_with_builder():
    import importlib.util, pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "semcommit_build_speechlm_candidates.py"
    spec = importlib.util.spec_from_file_location("bld", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.lexical_words is lexical_words
