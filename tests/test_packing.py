"""Sequence packing (vapasr/hf/packing.py) — sampler budget, pack_batch indices, and packed == padded loss/gradients on a tiny real
Qwen3-ASR thinker (block-diagonal causal mask from position_ids, mRoPE position restart per sample)."""
import pytest, torch
from vapasr.hf.packing import PackedBudgetSampler, pack_batch, packed, packing_sampler_factory, est_lens
from vapasr.uslm.interleave_data import SPECIAL_TOKENS as PHASE1_SPECIALS, add_specials

BASE = 151705


class _DS:
    def __init__(self, Ks): self.items = [dict(K=k, tokens=[(1, 0)] * (k // 2)) for k in Ks]
    def __len__(self): return len(self.items)


def test_sampler_budget_and_rank_split():
    ds = _DS([3, 50, 7, 20, 20, 9, 100, 4, 60, 33] * 5); est = est_lens(ds)
    sp = PackedBudgetSampler(ds, max_tokens=400, max_bs=8, seed=1)
    seen = sorted(i for b in sp.batches for i in b); assert seen == list(range(len(ds)))          # every item exactly once
    for b in sp.batches:
        assert len(b) <= 8 and (len(b) == 1 or sum(int(est[i]) for i in b) <= 400)
    d = sp.describe(); assert d["batches"] == len(sp.batches) and 0 < d["fill"] <= 1
    r0, r1 = (PackedBudgetSampler(ds, 400, max_bs=8, seed=1, rank=r, world=2) for r in (0, 1))
    b0, b1 = list(r0), list(r1); assert len(b0) == len(b1) == len(sp.batches) // 2 and not {tuple(x) for x in b0} & {tuple(x) for x in b1}
    r0.set_epoch(1); assert list(r0) != b0 or len(b0) < 2                                        # per-epoch reshuffle
    big = PackedBudgetSampler(_DS([500, 2]), max_tokens=100); assert sorted(map(len, big.batches)) == [1, 1]   # over-budget item alone
    assert packing_sampler_factory(0) is None and isinstance(packing_sampler_factory(300)(ds, seed=0, rank=0, world=1), PackedBudgetSampler)


def _batch(lens, Ks, Din=4, seed=0):
    """Padded collate-like batch: 3 prefix tokens (label -100), then per chunk one audio slot + one text token (labelled)."""
    g = torch.Generator().manual_seed(seed); B, L, Kmax = len(lens), max(lens), max(Ks)
    ids = torch.zeros(B, L, dtype=torch.long); lab = torch.full((B, L), -100); ia = torch.zeros(B, L, dtype=torch.bool)
    ch = torch.full((B, L), -1); mask = torch.zeros(B, L, dtype=torch.long); pw = torch.zeros(B, L)
    for i, (n, k) in enumerate(zip(lens, Ks)):
        ids[i, :n] = torch.randint(10, 1000, (n,), generator=g); mask[i, :n] = 1
        for p in range(3, n):
            c = (p - 3) // 2
            if (p - 3) % 2 == 0 and c < k: ia[i, p] = True; ch[i, p] = c
            else: lab[i, p] = ids[i, p]
        pw[i, 4] = 2.0
    feats = torch.randn(B, 1, Kmax, Din, generator=g)
    return dict(ids=ids, is_audio=ia, chunk_of=ch, labels=lab, mask=mask, pos_weight=pw, feats=feats, lang=["English"] * B, n_sem=0)


def test_pack_batch_indices():
    b = _batch([9, 15, 6], [3, 6, 2]); p = pack_batch(b); n = [9, 15, 6]; N = sum(n)
    for k in ("ids", "is_audio", "labels", "pos_weight", "chunk_of", "position_ids"): assert p[k].shape == (1, N), k
    assert "mask" not in p and p["pack_n"] == 3 and p["pack_kmax"] == 6 and p["feats"] is b["feats"] and p["lang"] == b["lang"]
    assert p["position_ids"][0].tolist() == list(range(9)) + list(range(15)) + list(range(6))
    off = 0
    for i, ni in enumerate(n):
        seg = p["chunk_of"][0, off: off + ni]; src = b["chunk_of"][i, :ni]
        assert torch.equal(seg, torch.where(src >= 0, src + i * 6, src)); assert torch.equal(p["ids"][0, off: off + ni], b["ids"][i, :ni]); off += ni
    b2 = dict(b, soft_b=torch.tensor([0, 1, 2]), soft_pos=torch.tensor([4, 5, 1]), activity=torch.ones(3, 4, 2), activity_mask=torch.ones(3, 4))
    p2 = pack_batch(b2); assert p2["soft_pos"].tolist() == [4, 9 + 5, 24 + 1] and p2["soft_b"].tolist() == [0, 0, 0]
    assert p2["activity"].shape == (3, 6, 2) and p2["activity_mask"][:, 4:].sum() == 0              # padded to Kmax so the model can flatten it with the encoder output
    assert packed(lambda x: b)([None])["pack_n"] == 3


def _tiny(attn="sdpa", **cfg_kw):
    pytest.importorskip("qwen_asr")
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRThinkerConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRThinkerForConditionalGeneration
    from vapasr.hf import VapAsrConfig, VapAsrForStreamingASR

    class _Tok:
        def __init__(s): s.vocab = {}
        def add_tokens(s, toks, special_tokens=False):
            for t in toks: s.vocab.setdefault(t, BASE + len(s.vocab))
        def convert_tokens_to_ids(s, t): return s.vocab.get(t, 3)
        def __len__(s): return BASE + len(s.vocab)
    torch.manual_seed(0); sp = add_specials(_Tok())
    tcfg = Qwen3ASRThinkerConfig(audio_config=dict(encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=16, d_model=8, output_dim=32, downsample_hidden_size=8),
                                 text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                                  rope_scaling={"mrope_section": [4, 2, 2], "interleaved": True, "rope_type": "default"}))
    th = Qwen3ASRThinkerForConditionalGeneration(tcfg)
    cfg = VapAsrConfig(adapter_d_in=4, adapter_d_out=32, adapter_hidden=16, sp_ids=sp, special_tokens=list(PHASE1_SPECIALS), blocked_ids=[], act_hidden=4, **cfg_kw)
    m = VapAsrForStreamingASR(cfg, thinker=th).float().eval()
    from vapasr.speedup import set_attn_impl
    assert set_attn_impl(m, attn, log=lambda *_: None) == attn
    return m


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_packed_equals_padded_loss_and_grads(attn):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _tiny(attn).to(dev); b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in _batch([19, 31, 11], [6, 12, 4], seed=3).items()}
    x = {k: v for k, v in b.items() if k not in ("lang", "n_sem")}
    o1 = m(**x); g1 = torch.autograd.grad(o1.loss, [m.adapter.net[0].weight if hasattr(m.adapter, "net") else next(m.adapter.parameters()), m.get_input_embeddings().weight])
    p = pack_batch(b); xp = {k: v for k, v in p.items() if k not in ("lang", "n_sem", "pack_n", "pack_kmax")}
    o2 = m(**xp); g2 = torch.autograd.grad(o2.loss, [m.adapter.net[0].weight if hasattr(m.adapter, "net") else next(m.adapter.parameters()), m.get_input_embeddings().weight])
    assert int(o1.n_labels) == int(o2.n_labels) > 0
    assert torch.allclose(o1.loss, o2.loss, atol=1e-5, rtol=1e-5), (float(o1.loss), float(o2.loss))
    for a_, b_ in zip(g1, g2): assert torch.allclose(a_, b_, atol=1e-5, rtol=1e-4)
    # a packed row without position_ids restart (plain causal over the whole row) must differ — proves the block mask is doing work
    xw = dict(xp, position_ids=torch.arange(xp["ids"].shape[1], device=dev)[None])
    assert not torch.allclose(m(**xw).loss, o1.loss, atol=1e-4)


def test_packed_equals_padded_phase2_soft_and_activity():
    """Phase 2 inputs: soft EOT targets (labels_alt/soft_w) and the lane activity head over audio positions (activity (B,K,R))."""
    m = _tiny("sdpa", lanes=3); Ks = [6, 12, 4]; b = _batch([19, 31, 11], Ks, seed=5); B, L = b["ids"].shape; g = torch.Generator().manual_seed(1)
    la = torch.full((B, L), -100); sw = torch.ones(B, L)
    for i in range(B):
        p = int((b["labels"][i] != -100).nonzero()[1]); la[i, p] = 42; sw[i, p] = 0.7
    act = torch.zeros(B, max(Ks), 3); am = torch.zeros(B, max(Ks))
    for i, k in enumerate(Ks): act[i, :k] = (torch.rand(k, 3, generator=g) > 0.5).float(); am[i, :k] = 1
    b.update(labels_alt=la, soft_w_full=sw, activity=act, activity_mask=am)
    def run(x):
        x = {k: v for k, v in x.items() if k not in ("lang", "n_sem", "pack_n", "pack_kmax")}; x["soft_w"] = x.pop("soft_w_full"); return m(**x)
    o1, o2 = run(b), run(pack_batch(b))
    assert o1.loss_act is not None and abs(o1.loss_act - o2.loss_act) < 1e-5 and o1.n_soft == o2.n_soft > 0
    assert torch.allclose(o1.loss, o2.loss, atol=1e-5, rtol=1e-5), (float(o1.loss), float(o2.loss))


def test_flex_with_packing_refused():
    """flex_attention + packed BlockMask: loss matches sdpa but gradients do not (torch 2.9 / transformers 4.57, H200) — apply_speedups refuses it."""
    import types
    from vapasr.speedup import apply_speedups
    a = types.SimpleNamespace(attn_impl="flex_attention", pack_max_tokens=4096, pack_max_bs=64, optim="adamw_torch_fused", deepspeed=None)
    with pytest.raises(SystemExit): apply_speedups(object(), a, log=lambda *_: None, liger=False)


def test_padded_budget_sampler_and_loader_collate():
    """--batch-max-tokens: n × longest ≤ budget (padded collate, no packing) — short items get big batches, long ones small."""
    from vapasr.hf.packing import budget_sampler_factory
    ds = _DS([3, 50, 7, 20, 20, 9, 100, 4, 60, 33] * 5); est = est_lens(ds)
    sp = PackedBudgetSampler(ds, max_tokens=1200, max_bs=32, padded=True)
    assert sorted(i for b in sp.batches for i in b) == list(range(len(ds)))
    for b in sp.batches: assert len(b) == 1 or len(b) * max(int(est[i]) for i in b) <= 1200
    short = [b for b in sp.batches if max(int(est[i]) for i in b) < 60]; long_ = [b for b in sp.batches if min(int(est[i]) for i in b) > 200]
    assert min(map(len, short)) > max(map(len, long_)) and sp.describe()["mode"] == "padded"
    f = budget_sampler_factory(0, 1200, 32); assert f.pack is False and f(ds).padded
    assert budget_sampler_factory(1200, 0, 32).pack is True and budget_sampler_factory(0, 0) is None
    with pytest.raises(ValueError): budget_sampler_factory(100, 100)
    from vapasr.hf.data import RoundRobinLoader
    from vapasr.hf.packing import _Packed
    ident = lambda b: b
    rr = RoundRobinLoader({"a": ds}, lambda m: 4, num_workers=0, sampler_factory=f, collate_fn=ident); assert rr.collate_fn is ident
    rr = RoundRobinLoader({"a": ds}, lambda m: 4, num_workers=0, sampler_factory=budget_sampler_factory(1200, 0, 32), collate_fn=ident); assert isinstance(rr.collate_fn, _Packed)
