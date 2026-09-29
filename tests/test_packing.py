"""Sequence packing (vapasr/hf/packing.py) — sampler budget, pack_batch indices, and packed == padded loss/gradients on a tiny real
Qwen3-ASR thinker (block-diagonal causal mask from position_ids, mRoPE position restart per sample).
varlen attention (vapasr/hf/varlen_attention.py, --attn-impl varlen): cu_seqlens derivation and the CPU reference path equal sdpa
(padded auto-unpad, packed cu_seqlens, position_ids fallback, KV-cache decode), the cu kwargs survive gradient checkpointing, rows
of a batch get their own segments, and ambiguous inputs are refused; CUDA-only tests check torch's FA2 varlen kernel against the fp32
reference and bf16 varlen training against padded sdpa."""
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
    from vapasr.hf.varlen_attention import IMPL
    assert set_attn_impl(m, attn, log=lambda *_: None) == {"varlen": IMPL}.get(attn, attn)
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


# ── varlen attention (vapasr/hf/varlen_attention.py)
_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only: torch's FlashAttention-2 varlen kernel")


def _rel(a, b): return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def _xs(b): return {k: v for k, v in b.items() if k not in ("lang", "n_sem", "pack_n", "pack_kmax")}


def _params(m): return [m.adapter.net[0].weight, m.get_input_embeddings().weight, m.thinker.model.layers[0].self_attn.k_proj.weight]


def _loss_grads(m, x, autocast=False):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast): o = m(**x)
    return o, torch.autograd.grad(o.loss, _params(m))


def _count_kernel_calls(monkeypatch):
    """Record the token count (rows of q) of every varlen kernel call — proves the varlen path ran and what it saw."""
    from vapasr.hf import varlen_attention as VA
    calls, real = [], VA._flash_varlen
    def spy(q, *a, **k): calls.append(int(q.shape[0])); return real(q, *a, **k)
    monkeypatch.setattr(VA, "_flash_varlen", spy); return calls


def _spy_segments(monkeypatch):
    """Record every varlen kernel call as (q rows, cu_q tensor) and every cu tensor the model builds (modeling_vapasr.packed_kwargs):
    `cu is built[0]` proves a call used the caller's cu kwargs (branch 1), not a segmentation recomputed inside the layer."""
    from vapasr.hf import varlen_attention as VA
    import vapasr.hf.modeling_vapasr as MV
    rec = dict(kernel=[], built=[]); real_k, real_pk = VA._flash_varlen, MV.packed_kwargs
    def spy_k(q, k, v, cu_q, *a, **kw): rec["kernel"].append((int(q.shape[0]), cu_q)); return real_k(q, k, v, cu_q, *a, **kw)
    def spy_pk(*a, **kw): out = real_pk(*a, **kw); rec["built"].append(out["cu_seq_lens_q"]); return out
    monkeypatch.setattr(VA, "_flash_varlen", spy_k); monkeypatch.setattr(MV, "packed_kwargs", spy_pk)
    return rec


def _redraw_layers(m, std=0.2, seed=1):
    """Layer weights with std 0.2 so attention dominates the residual stream (with the 0.02 init a wrong attention barely shows)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p_ in m.thinker.model.layers.parameters():
            if p_.dim() == 2: p_.copy_(torch.randn(p_.shape, generator=g) * std)
    return m


def test_packed_kwargs_cu_seqlens():
    """cu_seqlens from packed position_ids (restart = new sample, transformers' find_packed_sequence_indices rule) and from lengths."""
    pytest.importorskip("transformers")
    from transformers.masking_utils import find_packed_sequence_indices
    from vapasr.hf.varlen_attention import packed_kwargs, _unpad
    pos = pack_batch(_batch([9, 15, 6], [3, 6, 2]))["position_ids"]
    kw = packed_kwargs(position_ids=pos)
    assert kw["cu_seq_lens_q"].tolist() == [0, 9, 24, 30] and kw["cu_seq_lens_q"].dtype == torch.int32 and kw["max_length_q"] == kw["max_length_k"] == 15
    assert kw["cu_seq_lens_k"] is kw["cu_seq_lens_q"]
    assert packed_kwargs(position_ids=pos[None].expand(3, 1, -1))["cu_seq_lens_q"].tolist() == [0, 9, 24, 30]     # mRoPE (3, 1, N)
    kl = packed_kwargs(lengths=torch.tensor([9, 15, 6])); assert kl["cu_seq_lens_q"].tolist() == [0, 9, 24, 30] and kl["max_length_q"] == 15
    for p in ([[0, 0, 1, 2, 0, 0, 1]], [[0, 1, 2, 3]], [[5, 6, 0, 1, 2, 7, 8]]):                                    # 1-token samples, no restart, odd starts
        p = torch.tensor(p); cu = packed_kwargs(position_ids=p)["cu_seq_lens_q"]
        seg = find_packed_sequence_indices(p)[0]
        assert cu.diff().tolist() == torch.bincount(seg - seg[0]).tolist(), (p.tolist(), cu.tolist())
    with pytest.raises(ValueError): packed_kwargs(position_ids=torch.zeros(2, 5, dtype=torch.long))
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
    idx, cu, mx = _unpad(mask); assert idx.tolist() == [0, 1, 2, 4, 8, 9, 10, 11] and cu.tolist() == [0, 3, 4, 8] and mx == 4


def test_varlen_cpu_reference_matches_sdpa(monkeypatch):
    """--attn-impl varlen on CPU (per-segment SDPA reference with the kernel's semantics, fp32) equals sdpa: (a) padded batch unpadded
    inside the model — only real tokens reach the kernel, (b) pack_batch → cu_seqlens, (c) thinker.model on a packed row without cu
    kwargs → position_ids fallback, (d) KV-cache decode (prefill 9 → 1 → 4 tokens, causal aligned bottom-right),
    (e) a packed row without position restarts must differ (the segments are doing work), (f) thinker.model with a 2D padding
    mask → per-layer unpad."""
    import json
    from transformers import DynamicCache
    from vapasr.hf.varlen_attention import IMPL, is_varlen
    ms = _tiny("sdpa"); mv = _tiny("varlen"); mv.load_state_dict(ms.state_dict())
    assert is_varlen(mv.thinker) and not is_varlen(ms.thinker) and IMPL not in json.dumps(mv.config.to_dict(), default=str)   # checkpoints load back as sdpa
    calls = _count_kernel_calls(monkeypatch)
    lens = [19, 31, 11]; b = _batch(lens, [6, 12, 4], seed=3); x = _xs(b)
    o0, g0 = _loss_grads(ms, x)
    o1, g1 = _loss_grads(mv, x)                                                                       # (a)
    assert calls and set(calls) == {sum(lens)}, calls
    p = pack_batch(b); calls.clear(); o2, g2 = _loss_grads(mv, _xs(p))                                  # (b)
    assert calls and set(calls) == {sum(lens)}
    for o, g in ((o1, g1), (o2, g2)):
        assert int(o.n_labels) == int(o0.n_labels) > 0
        assert torch.allclose(o.loss, o0.loss, atol=1e-5, rtol=1e-5), (float(o.loss), float(o0.loss))
        for a_, b_ in zip(g, g0): assert torch.allclose(a_, b_, atol=1e-5, rtol=1e-4), _rel(a_, b_)
    E = torch.randn(1, sum(lens), 32, generator=torch.Generator().manual_seed(7)); pos = p["position_ids"]    # (c)
    hs = ms.thinker.model(inputs_embeds=E, position_ids=pos, use_cache=False).last_hidden_state
    hv = mv.thinker.model(inputs_embeds=E, position_ids=pos, use_cache=False).last_hidden_state
    assert torch.allclose(hv, hs, atol=1e-5), _rel(hv, hs)
    hw = mv.thinker.model(inputs_embeds=E, position_ids=torch.arange(E.shape[1])[None], use_cache=False).last_hidden_state   # (e)
    assert not torch.allclose(hw, hs, atol=1e-4)
    Ep = torch.randn(3, 10, 32, generator=torch.Generator().manual_seed(8)); am = torch.tensor([[1] * 10, [1] * 7 + [0] * 3, [1] * 4 + [0] * 6])   # (f) direct call with a padding mask: per-layer unpad
    hs = ms.thinker.model(inputs_embeds=Ep, attention_mask=am, use_cache=False).last_hidden_state
    calls.clear(); hv = mv.thinker.model(inputs_embeds=Ep, attention_mask=am, use_cache=False).last_hidden_state
    assert set(calls) == {21} and torch.allclose(hv[am.bool()], hs[am.bool()], atol=1e-5) and hv[~am.bool()].abs().max() > 0   # pad rows: residual stream only
    e = torch.randn(1, 14, 32, generator=torch.Generator().manual_seed(9)); res = []                   # (d)
    with torch.no_grad():
        for m in (ms, mv):
            cache = DynamicCache(); calls.clear()
            res.append([m.thinker(inputs_embeds=e[:, a_:b_], past_key_values=cache, use_cache=True).logits for a_, b_ in ((0, 9), (9, 10), (10, 14))])
    assert calls[-2:] == [4, 4] and 1 in calls                                                         # q rows per call: the cache path ran through varlen
    for a_, b_ in zip(*res): assert torch.allclose(b_, a_, atol=1e-5), _rel(b_, a_)


@pytest.mark.parametrize("reentrant", [False, True])
def test_varlen_gradient_checkpointing_keeps_segments(monkeypatch, reentrant):
    """Training setup (train(), gradient checkpointing as in semcommit_train / train_speed_bench): the cu kwargs the model builds once
    per forward reach every attention call of the forward and of the checkpoint recompute (2 × layers kernel calls, all with that very
    cu tensor — branch 1, not the position_ids fallback), and loss / grads equal padded sdpa for the padded batch (model-level unpad)
    and the packed batch. Reentrant checkpointing only supports .backward(), hence backward() instead of autograd.grad."""
    ms = _redraw_layers(_tiny("sdpa")); mv = _tiny("varlen"); mv.load_state_dict(ms.state_dict())
    for m in (ms, mv):
        m.train(); m.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant}); m.thinker.config.use_cache = False
    def run(m, x):
        for p_ in _params(m): p_.grad = None
        o = m(**x); o.loss.backward(); return o, [p_.grad.clone() for p_ in _params(m)]
    n_layers = len(mv.thinker.model.layers); lens = [19, 31, 11, 5]; b = _batch(lens, [6, 12, 4, 1], seed=3)
    rec = _spy_segments(monkeypatch); o0, g0 = run(ms, _xs(b)); assert not rec["kernel"]
    for name, x in (("padded", _xs(b)), ("packed", _xs(pack_batch(b)))):
        rec["kernel"].clear(); rec["built"].clear(); o, gr = run(mv, x)
        assert len(rec["built"]) == 1 and rec["built"][0].tolist() == [0, 19, 50, 61, 66], (name, rec["built"])
        assert len(rec["kernel"]) == 2 * n_layers, (name, len(rec["kernel"]))
        assert all(n == sum(lens) and cu is rec["built"][0] for n, cu in rec["kernel"]), (name, [(n, cu.tolist()) for n, cu in rec["kernel"]])
        assert torch.allclose(o.loss, o0.loss, atol=1e-6, rtol=1e-5), (name, float(o.loss), float(o0.loss))
        errs = [_rel(a_, b_) for a_, b_ in zip(gr, g0)]; assert max(errs) < 1e-5, (name, errs)


def test_varlen_model_unpad_keeps_column_positions():
    """_thinker_unpadded equals padded sdpa for right-, left- and hole-padded masks: positions are the original column indices
    (sdpa keeps positions under a padding mask), so RoPE distances across a hole are not shortened."""
    ms = _redraw_layers(_tiny("sdpa")); mv = _tiny("varlen"); mv.load_state_dict(ms.state_dict())
    E = torch.randn(3, 12, 32, generator=torch.Generator().manual_seed(4))
    for name, am in (("right", [[1] * 12, [1] * 9 + [0] * 3, [1] * 5 + [0] * 7]), ("left", [[1] * 12, [0] * 3 + [1] * 9, [0] * 7 + [1] * 5]),
                     ("hole", [[1] * 12, [1] * 4 + [0] * 2 + [1] * 6, [1] * 12])):
        am = torch.tensor(am); sel = am.bool()
        with torch.no_grad():
            hs = ms.thinker.model(inputs_embeds=E, attention_mask=am, use_cache=False).last_hidden_state
            hu = mv._thinker_unpadded(E, sel)
        assert _rel(hu[sel], hs[sel]) < 1e-5, (name, _rel(hu[sel], hs[sel]))
        assert hu[~sel].abs().max() == 0, name                                                      # pad rows are not computed


def test_varlen_row_segments_and_refusals(monkeypatch):
    """thinker.model without a mask on B > 1 rows: position_ids restarting inside a row give per-row segments, as sdpa's packed
    detection does per row (cu breaks at every row start and restart; (1, L) position_ids broadcast over the batch like transformers);
    contiguous rows give cu = arange·L. Pinned: an empty-cache prefill with restarting position_ids also runs per segment (the
    attention function cannot see the cache; sdpa would run plain causal there — no caller does this). A padding mask next to cu
    kwargs is refused instead of being silently ignored."""
    from transformers import DynamicCache
    from vapasr.hf.varlen_attention import packed_kwargs
    ms = _redraw_layers(_tiny("sdpa")); mv = _tiny("varlen"); mv.load_state_dict(ms.state_dict()); rec = _spy_segments(monkeypatch)
    E = torch.randn(2, 8, 32, generator=torch.Generator().manual_seed(5))
    for pos, want in ((torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 0, 1, 2, 3, 4]]), [0, 4, 8, 11, 16]),
                      (torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]), [0, 4, 8, 12, 16]),
                      (torch.arange(8)[None].expand(2, -1), [0, 8, 16])):
        hs = ms.thinker.model(inputs_embeds=E, position_ids=pos, use_cache=False).last_hidden_state
        rec["kernel"].clear(); hv = mv.thinker.model(inputs_embeds=E, position_ids=pos, use_cache=False).last_hidden_state
        assert rec["kernel"] and all(cu.tolist() == want for _, cu in rec["kernel"]), (pos.tolist(), [cu.tolist() for _, cu in rec["kernel"]])
        assert torch.allclose(hv, hs, atol=1e-5), (pos.tolist(), _rel(hv, hs))
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    hs = ms.thinker.model(inputs_embeds=E[:1], position_ids=pos, use_cache=False).last_hidden_state
    rec["kernel"].clear(); hv = mv.thinker.model(inputs_embeds=E[:1], position_ids=pos, past_key_values=DynamicCache(), use_cache=True).last_hidden_state
    assert rec["kernel"] and all(cu.tolist() == [0, 4, 8] for _, cu in rec["kernel"]) and torch.allclose(hv, hs, atol=1e-5), _rel(hv, hs)
    am = torch.tensor([[1] * 8, [1] * 5 + [0] * 3])
    with pytest.raises(NotImplementedError, match="padding mask"):
        mv.thinker.model(inputs_embeds=E, attention_mask=am, use_cache=False, **packed_kwargs(lengths=am.sum(1)))


def test_apply_speedups_varlen_allows_packing():
    """--attn-impl varlen + --pack-max-tokens is accepted (no O(N²) warning), registers the implementation and sets the thinker's text config."""
    import types
    from vapasr.speedup import apply_speedups
    from vapasr.hf.varlen_attention import IMPL, is_varlen
    m = _tiny("sdpa"); logs = []
    a = types.SimpleNamespace(attn_impl="varlen", pack_max_tokens=4096, batch_max_tokens=0, pack_max_bs=64, optim="adamw_torch_fused", deepspeed=None)
    rec = apply_speedups(m, a, log=logs.append, liger=False)
    assert rec["attn_impl"] == IMPL and is_varlen(m.thinker) and m.thinker.model.config._attn_implementation == IMPL
    assert not any("O(N²)" in str(l) for l in logs), logs
    rec = apply_speedups(m, types.SimpleNamespace(**dict(vars(a), attn_impl="sdpa", pack_max_tokens=0)), log=logs.append, liger=False)   # switching back works
    assert rec["attn_impl"] == "sdpa" and not is_varlen(m.thinker)


@_cuda
def test_varlen_kernel_matches_reference():
    """torch's FA2 varlen kernel (bf16) vs the fp32 per-segment reference: GQA 16/8, head_dim 128, fwd + bwd; and q shorter than k
    (KV-cache decode) with causal aligned bottom-right."""
    from vapasr.hf.varlen_attention import _flash_varlen, _reference_varlen
    dev = "cuda"; g = torch.Generator(device=dev).manual_seed(0); lens = [300, 17, 1200, 5, 800]; T = sum(lens)
    cu = torch.tensor([0] + lens, device=dev).cumsum(0).to(torch.int32)
    q, k, v = (torch.randn(T, h, 128, device=dev, generator=g).to(torch.bfloat16).requires_grad_() for h in (16, 8, 8))
    go = torch.randn(T, 16, 128, device=dev, generator=g).to(torch.bfloat16)
    o = _flash_varlen(q, k, v, cu, cu, max(lens), max(lens), True, 128 ** -0.5, 0.0); ga = torch.autograd.grad(o, (q, k, v), go)
    qf, kf, vf = (t.detach().float().requires_grad_() for t in (q, k, v))
    of = _reference_varlen(qf, kf, vf, cu, cu, True, 128 ** -0.5); gf = torch.autograd.grad(of, (qf, kf, vf), go.float())
    errs = [_rel(o, of)] + [_rel(a_, b_) for a_, b_ in zip(ga, gf)]; print("kernel vs fp32 reference (out, dq, dk, dv):", [round(e, 5) for e in errs])
    assert max(errs) < 1e-2, errs
    for lq in (3, 1):
        qs = torch.randn(lq, 16, 128, device=dev, generator=g).to(torch.bfloat16); ks, vs = (torch.randn(10, 8, 128, device=dev, generator=g).to(torch.bfloat16) for _ in range(2))
        cq, ck = (torch.tensor([0, n], device=dev, dtype=torch.int32) for n in (lq, 10))
        o = _flash_varlen(qs, ks, vs, cq, ck, lq, 10, True, None, 0.0); ref = _reference_varlen(qs.float(), ks.float(), vs.float(), cq, ck, True, None)
        assert _rel(o, ref) < 1e-2, (lq, _rel(o, ref))


@_cuda
def test_varlen_bf16_matches_padded_sdpa(monkeypatch):
    """Tiny model, fp32 weights, bf16 autocast (as in training): packed varlen and padded varlen (auto-unpad) vs padded sdpa —
    the same size of bf16 error as the exact packed-sdpa block-mask path."""
    from vapasr.speedup import set_attn_impl
    dev = "cuda"; m = _tiny("sdpa").to(dev)
    lens = [61, 203, 17, 150]; b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in _batch(lens, [(n - 2) // 2 for n in lens], seed=11).items()}
    p = pack_batch(b); x, xp = _xs(b), _xs(p)
    o0, g0 = _loss_grads(m, x, autocast=True)
    ops, gps = _loss_grads(m, xp, autocast=True)
    set_attn_impl(m, "varlen", log=lambda *_: None); calls = _count_kernel_calls(monkeypatch)
    opv, gpv = _loss_grads(m, xp, autocast=True); assert calls and set(calls) == {sum(lens)}
    calls.clear(); odv, gdv = _loss_grads(m, x, autocast=True); assert calls and set(calls) == {sum(lens)}
    rep = {}
    for name, (o, g) in dict(packed_sdpa=(ops, gps), packed_varlen=(opv, gpv), padded_varlen=(odv, gdv)).items():
        rep[name] = dict(loss=abs(float(o.loss) - float(o0.loss)) / abs(float(o0.loss)), grads=[round(_rel(a_, b_), 5) for a_, b_ in zip(g, g0)])
    print("bf16 relative error vs padded sdpa:", rep)
    for name in ("packed_varlen", "padded_varlen"):
        assert rep[name]["loss"] < 5e-3 and max(rep[name]["grads"]) < 3e-2, (name, rep)


@_cuda
def test_varlen_cuda_decode_matches_eager(monkeypatch):
    """KV-cache decode on GPU (prefill 9 → 1 → 4 tokens, bf16 autocast) through the varlen kernel vs an fp32 eager reference. Layer
    weights are redrawn with std 0.2 so attention dominates the residual stream (with the default 0.02 init bf16 logit rounding hides
    it): varlen has sdpa's error (on H200 sdpa dispatches to the same FA2 kernel here, so the two are bit-identical), while causal
    aligned top-left — wrong for q shorter than k — is ~100 % off, so the check can tell. The kernel really runs (q rows 9, 1, 4)."""
    import torch.nn.functional as F
    from transformers import DynamicCache
    from vapasr.speedup import set_attn_impl
    from vapasr.hf import varlen_attention as VA
    dev = "cuda"; m = _tiny("eager"); g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in m.thinker.model.layers.parameters():
            if p.dim() == 2: p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    m = m.to(dev); e = torch.randn(1, 14, 32, device=dev, generator=torch.Generator(device=dev).manual_seed(9))
    def run(autocast):
        cache = DynamicCache()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            return [m.thinker(inputs_embeds=e[:, a_:b_], past_key_values=cache, use_cache=True).logits.float() for a_, b_ in ((0, 9), (9, 10), (10, 14))]
    ref = run(False)
    set_attn_impl(m, "sdpa", log=lambda *_: None); sd = run(True)
    set_attn_impl(m, "varlen", log=lambda *_: None); calls = _count_kernel_calls(monkeypatch); vl = run(True)
    assert calls == [9, 9, 1, 1, 4, 4], calls
    kernel = VA._flash_varlen
    def top_left(q, k, v, cq, ck, mq, mk, causal, scale, dropout):                 # negative control: SDPA is_causal aligns top-left
        if mq == mk: return kernel(q, k, v, cq, ck, mq, mk, causal, scale, dropout)
        t = lambda x, r=1: x.float().transpose(0, 1).repeat_interleave(r, 0); r = q.shape[1] // k.shape[1]
        return F.scaled_dot_product_attention(t(q), t(k, r), t(v, r), is_causal=True, scale=scale).transpose(0, 1).to(q.dtype)
    monkeypatch.setattr(VA, "_flash_varlen", top_left); tl = run(True)
    err = {n: [round(_rel(a_, r), 5) for a_, r in zip(x, ref)] for n, x in (("sdpa", sd), ("varlen", vl), ("top_left", tl))}
    err["varlen_vs_sdpa"] = [round(_rel(a_, b_), 5) for a_, b_ in zip(vl, sd)]
    print("decode logits relative error vs fp32 eager:", err)
    assert max(err["varlen"]) < max(2e-2, 2 * max(err["sdpa"])), err
    assert min(err["top_left"][1:]) > 10 * max(err["varlen"]), err
