"""final(오프라인) 모드 — vapasr/data/offline_seq 형식, 작은 모델로 손실·디코드(막는 토큰·멈춤). CPU."""
import importlib.util, pathlib
import pytest, torch
from vapasr.data.offline_seq import OfflineFormat, OFFLINE_PRE, OFFLINE_MID

_HERE = pathlib.Path(__file__).parent
def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", _HERE / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


class Tok:
    """형식 토큰은 고정 id, 텍스트는 문자 코드(작은 id)."""
    SPECIAL = {"<|audio_start|>": 151669, "<|audio_end|>": 151670, "<|audio_pad|>": 151676, "<|im_end|>": 151645, "<|im_start|>": 151644, "<asr_text>": 151704}
    def __init__(self, special=None): self.sp = dict(special or self.SPECIAL)
    def convert_tokens_to_ids(self, t): return self.sp.get(t, -1)
    def __call__(self, text, add_special_tokens=False, **kw): return {"input_ids": [ord(c) % 500 + 10 for c in text]}


def test_offline_sequence_layout():
    tok = Tok(); F = OfflineFormat(tok); s = F.sequence([501, 502, 503], K=5, lang="Korean")
    pre, mid, lang = tok(OFFLINE_PRE)["input_ids"], tok(OFFLINE_MID)["input_ids"], tok("language Korean<asr_text>")["input_ids"]
    assert s["ids"] == pre + [151669] + [151676] * 5 + [151670] + mid + lang + [501, 502, 503, 151645]
    a0 = len(pre) + 1
    assert [s["chunk_of"][a0 + k] for k in range(5)] == list(range(5)) and sum(c >= 0 for c in s["chunk_of"]) == 5
    assert [l for l in s["labels"] if l != -100] == [501, 502, 503, 151645] and s["prompt_len"] == len(s["ids"]) - 4
    assert s["is_input"] == [True] * s["prompt_len"] + [False] * 4 and s["n_text"] == 3
    with pytest.raises(ValueError, match="형식 토큰"): OfflineFormat(Tok({"<|im_end|>": 1}))


def test_offline_loss_and_decode_on_tiny_model():
    tp = _load("test_packing"); m = tp._tiny("sdpa"); c = m.thinker.config
    tok = Tok({"<|audio_start|>": getattr(c, "audio_start_token_id", 151669), "<|audio_end|>": getattr(c, "audio_end_token_id", 151670), "<|audio_pad|>": c.audio_token_id,
               "<|im_end|>": 151645, "<|im_start|>": 151644, "<asr_text>": 151704})
    F = OfflineFormat(tok); K = 6; s = F.sequence([11, 12, 13, 14], K, "English")
    feats = torch.randn(1, 1, K, 4, generator=torch.Generator().manual_seed(0))
    x = dict(ids=torch.tensor([s["ids"]]), is_audio=torch.tensor([[q >= 0 for q in s["chunk_of"]]]), chunk_of=torch.tensor([s["chunk_of"]]),
             labels=torch.tensor([s["labels"]]), mask=torch.ones(1, len(s["ids"]), dtype=torch.long), feats=feats)
    out = m(**x); assert torch.isfinite(out.loss) and int(out.n_labels) == 5                   # 텍스트 4 + <|im_end|>
    from vapasr.hf.offline_decode import offline_decode, offline_blocked_ids
    blocked = offline_blocked_ids(m, F)
    assert F.im_end not in blocked and c.audio_token_id in blocked and all(v in blocked for v in m.config.sp_ids.values())
    hyp = offline_decode(m, tok, feats[0, 0], "English", max_new_tokens=7, fmt=F)
    assert len(hyp) <= 7 and not set(hyp) & set(blocked) and F.im_end not in hyp
