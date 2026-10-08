import json
import os
import types

import pytest
import torch

from vapasr.data.semcommit_dataset import add_semcommit_specials, check_sem_ids
from vapasr.data.semcommit_tokens import DELAY_EXT, DELAY_EXT_REGISTRY, add_delay_ext_specials, add_delay_tokens

QWEN = "/soundai/Model/Qwen3-ASR-1.7B"
needs_qwen = pytest.mark.skipif(not os.path.isdir(QWEN), reason="needs the Qwen3-ASR tokenizer")


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN)


@needs_qwen
def test_delay_ext_ids_follow_sem():
    tok = _tok()
    ids = add_semcommit_specials(tok)
    ext = add_delay_ext_specials(tok)
    assert ext == DELAY_EXT_REGISTRY and ext["<DELAY_9>"] == ids["<SEM_END>"] + 1
    check_sem_ids({**ids, **ext})                                   # the dataset check accepts the extension after SEM
    assert add_delay_ext_specials(tok) == ext                       # idempotent
    with pytest.raises(AssertionError):
        check_sem_ids({**ids, "<DELAY_9>": ids["<SEM_END>"] - 1})


def test_delay_ext_needs_sem_first():
    class Tok:
        def get_vocab(self): return {}
    with pytest.raises(AssertionError):
        add_delay_ext_specials(Tok())


@needs_qwen
def test_add_delay_tokens_copies_delay8_and_blocks():
    tok = _tok()
    sp = add_semcommit_specials(tok)
    emb = torch.nn.Embedding(151936, 8)
    cfg = types.SimpleNamespace(sp_ids=dict(sp), special_tokens=["<SEM_END>"], blocked_ids=[1, 2], sem_registry={"<SEM_END>": sp["<SEM_END>"]})
    model = types.SimpleNamespace(config=cfg, get_input_embeddings=lambda: emb)
    before = emb.weight[sp["<DELAY_8>"]].clone()
    ids = add_delay_tokens(model, tok)
    for n in DELAY_EXT:
        assert torch.equal(emb.weight[ids[n]], before) and cfg.sp_ids[n] == ids[n] and ids[n] in cfg.blocked_ids
    assert model.blocked.tolist() == cfg.blocked_ids and cfg.special_tokens[-2:] == DELAY_EXT
    with torch.no_grad():
        emb.weight[ids["<DELAY_9>"]] += 1                                # trained row
    trained = emb.weight[ids["<DELAY_9>"]].clone()
    add_delay_tokens(model, tok)                                         # a second call keeps existing rows
    assert torch.equal(emb.weight[ids["<DELAY_9>"]], trained)


@needs_qwen
def test_add_delay_tokens_refuses_v02_checkpoint():
    tok = _tok()
    sp = add_semcommit_specials(tok)
    cfg = types.SimpleNamespace(sp_ids=dict(sp), special_tokens=[], blocked_ids=[], sem_registry={"<SEM_END>": 151723, "<TURN_END>": 151724})
    model = types.SimpleNamespace(config=cfg, get_input_embeddings=lambda: torch.nn.Embedding(151936, 4))
    with pytest.raises(AssertionError):
        add_delay_tokens(model, tok)


@needs_qwen
@pytest.mark.parametrize("with_tokenizer_files", [True, False])
def test_load_tokenizer_restores_delay_ext(tmp_path, with_tokenizer_files):
    from vapasr.hf import load_tokenizer
    tok = _tok()
    sp = {**add_semcommit_specials(tok), **add_delay_ext_specials(tok)}
    if with_tokenizer_files:
        tok.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(dict(model_type="vapasr", thinker_name_or_path=QWEN, sp_ids=sp,
                                                          sem_registry={"<SEM_END>": sp["<SEM_END>"]})))
    tok2 = load_tokenizer(str(tmp_path))
    assert {n: tok2.convert_tokens_to_ids(n) for n in sp} == sp
