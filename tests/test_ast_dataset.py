import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from vapasr.data.ast_dataset import AstStreamDataset, ast_head_text, build_ast_sequence
from vapasr.data.interleave import Specials

QWEN = "/soundai/Model/Qwen3-ASR-1.7B"
needs_qwen = pytest.mark.skipif(not os.path.isdir(QWEN), reason="needs the Qwen3-ASR tokenizer")
NEXT, EMPTY, SPK, TRANS, PAD = 100, 101, 102, 103, 99
SP = Specials(next_audio=NEXT, empty_audio=EMPTY, spk=(0, 0))


def _chunks(s):
    return {k: em for k, em in s["chunks"] if len(em) > 1}


def test_confirmed_example_layout():
    """wiki decision-ast-monotonic-data 9 example (δ4): transcript run <SPK_A>, translation run <TRANS>, tags only on a change."""
    src = [(1, 5.87), (2, 5.95), (3, 6.19), (4, 6.59), (5, 7.07)]          # it is about shared values
    tr = [(11, 5.87), (12, 6.19), (13, 6.19), (14, 7.07), (15, 7.07)]      # Es geht um gemeinsame Werte
    s = build_ast_sequence(src, tr, 100, [7, 8], PAD, SP, SPK, TRANS, 4)
    # chunk 81/92: the run before is already the transcript (78/86), so no <SPK_A> — tags only where the run changes
    assert _chunks(s) == {77: [SPK, 1, TRANS, 11, NEXT], 78: [SPK, 2, NEXT], 81: [3, TRANS, 12, 13, NEXT],
                          86: [SPK, 4, NEXT], 92: [5, TRANS, 14, 15, NEXT]}
    assert s["tr_in_flush"] == 0 and s["n_tr"] == 5 and s["n_tags"] == 6
    assert s["ids"][:2] == [7, 8] and all(l == -100 for l, inp in zip(s["labels"], s["is_input"]) if inp)


def test_asr_only_has_one_leading_tag():
    src = [(1, 0.40), (2, 0.41), (3, 1.00)]
    s = build_ast_sequence(src, [], 30, [7], PAD, SP, SPK, TRANS, 2)
    out = [t for t, inp in zip(s["ids"], s["is_input"]) if not inp and t != NEXT]
    assert out == [SPK, 1, 2, 3] and s["n_tags"] == 1


def test_head_text():
    assert ast_head_text("English", "<SPK_A>") == "language <SPK_A>English<asr_text>"
    assert ast_head_text("Korean", "<SPK_B>", "English") == "language <SPK_B>Korean\ntranslate <SPK_B>English<asr_text>"


# ── real tokenizer (mxc)
def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN)


@needs_qwen
def test_trans_token_id_and_model_setup():
    from vapasr.data.semcommit_dataset import add_semcommit_specials
    from vapasr.data.semcommit_tokens import add_delay_ext_specials, add_trans_special, add_ast_tokens, TRANS_REGISTRY
    tok = _tok()
    sp = {**add_semcommit_specials(tok), **add_delay_ext_specials(tok)}
    assert add_trans_special(tok) == TRANS_REGISTRY and add_trans_special(tok) == TRANS_REGISTRY          # idempotent
    emb = torch.nn.Embedding(151936, 8)
    cfg = types.SimpleNamespace(sp_ids=dict(sp), special_tokens=["<SEM_END>"], runaway_cap=8,
                                blocked_ids=sorted([1, sp["<SPK_A>"], sp["<SPK_B>"], sp["<DELAY_4>"]]))
    model = types.SimpleNamespace(config=cfg, get_input_embeddings=lambda: emb)
    ids = add_ast_tokens(model, tok)
    t = ids["<TRANS>"]
    assert t == 151726 and cfg.sp_ids["<TRANS>"] == t and cfg.special_tokens[-1] == "<TRANS>"
    assert cfg.blocked_ids == sorted([1, sp["<DELAY_4>"]]) and model.blocked.tolist() == cfg.blocked_ids
    assert cfg.runaway_cap == 32 and cfg.ast_format == "ast-v1"
    with torch.no_grad():
        emb.weight[t] += 1
    row = emb.weight[t].clone()
    add_ast_tokens(model, tok)                                               # second call keeps the trained row
    assert torch.equal(emb.weight[t], row)


def _row(tok, rid, text, t0=0.5, step=0.3):
    from vapasr.data.semcommit_words import split_words
    ids = tok(text, add_special_tokens=False)["input_ids"]
    times = [round(t0 + step * j, 3) for j in range(len(ids))]
    words = split_words(ids, times, tok)
    K = int((times[-1] + 0.6) / 0.08) + 1
    return dict(id=rid, set="test-set", lang="English", K=K, duration_s=K * 0.08, text=text,
                tokens=[[i, t] for i, t in zip(ids, times)],
                segments=[dict(path="/dev/null", offset_s=0.0, silence_before_s=0.0, dur_s=K * 0.08, utt_id=rid, raw_text=text)],
                words=[dict(w, tags=[], seg=0) for w in words])


@needs_qwen
def test_dataset_sequences_and_items(tmp_path):
    from vapasr.data.semcommit_dataset import add_semcommit_specials
    from vapasr.data.semcommit_tokens import add_delay_ext_specials, add_trans_special
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from ast_build_targets import token_times
    tok = _tok()
    sp = {**add_semcommit_specials(tok), **add_delay_ext_specials(tok)}
    trans = add_trans_special(tok)["<TRANS>"]
    rows = [_row(tok, "r1", "it is about shared values"), _row(tok, "r2", "we should wait")]
    p = tmp_path / "words.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    w_end = {w["text"]: w["end_time"] for w in rows[0]["words"]}
    text, toks = token_times(tok, ["Es", "geht", "um", "gemeinsame", "Werte."],
                             [w_end["it"], w_end["about"], w_end["about"], w_end["values"], w_end["values"]])
    assert text == "Es geht um gemeinsame Werte." and toks[-1][1] == w_end["values"]
    targets = {"r1": {"German": ([t for t, _ in toks], [x for _, x in toks])}}
    ds = AstStreamDataset(str(p), [], tok, sp_ids=sp, delays=(4, 10), online=False, ignore_labels=True, sem_free_unlabeled=True,
                          source_pool="main", ast_targets=targets, ast_frac=1.0, trans_id=trans, offline_frac=0.0, ko_textnorm=False)
    assert ds.stats["ast_items"] == 1 and len(ds) == 2
    for i in range(len(ds)):
        for d in ds.delays:
            assert ds.target_problems(i, d) == []
    i1 = next(i for i, it in enumerate(ds.items) if it["id"] == "r1")
    s, K, _ = ds.ast_sequence(i1, 4, 1, "German")
    head = tok.decode(s["ids"][: s["is_input"].index(False)])
    assert "language <SPK_B>English\ntranslate <SPK_B>German<asr_text><DELAY_4>" in head
    out = tok.decode([t for t, inp in zip(s["ids"], s["is_input"]) if not inp and t != sp["<NEXT_AUDIO>"]])
    assert out.startswith("<SPK_B>it<TRANS>Es") and "Werte." in out and out.count("<SPK_A>") == 0
    item = ds[i1]
    assert item["ast"] == 1 and item["tgt"] == "German" and trans in item["ids"].tolist()
    off = ds.ast_offline_sequence(i1, ds.items[i1]["K"], 0, "German")
    body = tok.decode([t for t, inp in zip(off["ids"], off["is_input"]) if not inp])
    assert body.startswith("<SPK_A>it is about shared values<TRANS>Es geht um gemeinsame Werte.")
