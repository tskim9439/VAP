"""Independent greedy mono streams sharing a dense KV-cache batch.

Each row advances through audio/text/NEXT independently. There are no gaps in
an active row's cache, so ordinary causal positions are identical to serial
decoding. Finished rows keep consuming dummy tokens; they cannot affect others.
"""
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache


@dataclass
class DecodeState:
    K: int
    cap: int
    max_flush: int
    max_total: int
    k: int = 0
    n: int = 0
    advance: bool = False
    done: bool = False
    forced: int = 0
    flush_rounds: int = 0
    emitted: list = field(default_factory=list)

    def consume(self, tid, next_id):
        """Return (next input kind, index), after processing current input."""
        if self.done:
            return "next", 0
        if self.advance:
            self.k += 1
            self.n = 0
            self.advance = False
            return ("audio", self.k) if self.k < self.K else ("empty", 0)
        if tid == next_id or self.n >= self.cap or len(self.emitted) >= self.max_total:
            self.forced += int(tid != next_id)
            if self.k >= self.K:
                self.flush_rounds += 1
                self.done = self.n == 0 or self.flush_rounds >= self.max_flush
            elif self.k == self.K - 1 and self.max_flush == 0:
                self.done = True
            self.advance = True
            return "next", 0
        self.emitted.append((self.k, tid))
        self.n += 1
        return "text", tid


@dataclass
class SpkGuide:
    """Stage 3 speaker-head guided decoding (model.spk_head, 2 slots).

    The head reads the hidden state at each audio chunk (slot A/B speaking). A text
    token emitted at chunk k ends at about chunk k - delay, so the head's opinion is
    the mean slot probability over chunks [k - delay - span + 1, k - delay]:
    one slot >= on and the other at least margin lower -> that slot, else no opinion
    (silence or overlap). insert: before a word-initial text token (word_start(tid),
    if given) whose slot differs from the current tag, emit the head's tag first.
    hold (off by default): when the model wants to switch tags but the head favors the
    current slot, take the best non-tag token instead -- this also blocks a tag opened
    only to emit the other slot's <EOT>, so it can cost EOT recall.
    """
    delay: int
    on: float = 0.5
    margin: float = 0.3
    span: int = 2
    insert: bool = True
    hold: bool = False
    word_start: object = None


@dataclass
class SwitchBias:
    """State-aware speaker-switch bias (V3 C3; decode only, off by default). Only rows whose row_modes[i] == 'xl'
    (cross-language prefix -- the caller decides) once a slot tag was emitted (cur), while the row is not advancing.
    Row state follows the emitted tokens with parse_emitted4's rules: cur = last slot tag, kind = 0 transcript / 1
    translation (<TRANS> -> 1, a slot tag -> 0), eot_open = <EOT> emitted and no slot tag since. On the greedy argmax tid:
      (i)   eot_open and (tid == <SPK_cur> or a transcript-kind word-initial text token)  -> b_eot
      (ii)  not eot_open, kind 1 and tid == <SPK_cur> (same slot re-opened after a translation segment) -> b_res
      (iii) not eot_open, kind 0 and tid a word-initial text token -> b_wi (not the token right after a slot tag:
            the model has just chosen that slot, and a flip there would only leave an empty segment)
    b > 0 is added to the logits of the other declared slot tags only (tags in that row's prefix, never a fixed index);
    if one of them then beats tid it replaces tid. Translation-kind word starts, mid-word tokens, the first tag and
    <NEXT_AUDIO>/<EMPTY_AUDIO>/<TRANS>/<EOT> are never touched. Serial stream_decode has no bias.
    """
    b_eot: float = 0.0
    b_res: float = 0.0
    b_wi: float = 0.0

    @classmethod
    def parse(cls, s):
        """'eot=B,res=B,wi=B' (missing keys 0, B >= 0) -> SwitchBias."""
        kw = {}
        for x in s.split(","):
            k, v = (y.strip() for y in x.split("=")); assert k in ("eot", "res", "wi") and float(v) >= 0, f"switch bias 'eot=B,res=B,wi=B' (B >= 0): {x!r}"
            kw["b_" + k] = float(v)
        return cls(**kw)

    @property
    def on(self):
        return max(self.b_eot, self.b_res, self.b_wi) > 0

    def policy(self):
        return dict(eot=self.b_eot, res=self.b_res, wi=self.b_wi, rows="xl")


def word_init_mask(tok, V):
    """(V,) bool: token decodes to text starting with a space (word-initial; CstDialogueDataset.word_start rule).
    Rows past the tokenizer (lm_head padding) are False. Build once per run."""
    n = min(int(V), len(tok)); m = torch.zeros(int(V), dtype=torch.bool)
    m[:n] = torch.tensor([tok.decode([t]).startswith(" ") for t in range(n)], dtype=torch.bool)
    return m


def _guide_slot(probs, k, g):
    j = min(k, len(probs)) - g.delay
    rows = [probs[t] for t in range(max(0, j - g.span + 1), j + 1) if 0 <= t < len(probs) and probs[t] is not None]
    if not rows:
        return None
    pa, pb = sum(r[0] for r in rows) / len(rows), sum(r[1] for r in rows) / len(rows)
    if max(pa, pb) < g.on or abs(pa - pb) < g.margin:
        return None
    return 0 if pa > pb else 1


@torch.inference_mode()
def decode_batch(model, feats, prefixes, *, next_bias=0.0, max_flush_rounds=None, spk_guide=None, switch_bias=None, row_modes=None, word_init=None, acts=None):
    """feats: list of (1,K,D) encoder outputs, prefixes: equal-length lists.

    Reuses features across delta sweeps. Uses the same cap/flush/blocked IDs as
    model.stream_decode; never changes a model's weights or tokenizer.
    spk_guide (SpkGuide): speaker-head guided tags (needs model.spk_head); states get
    guide_insert / guide_hold counts. Serial stream_decode has no guide.
    switch_bias (SwitchBias, None = off and skipped entirely): needs row_modes (one per row, 'xl' = biased) and
    word_init ((V,) bool, word_init_mask); states get sb_flips (replaced tags) and sb_points (decision points seen),
    dicts keyed eot/res/wi. Not combined with spk_guide.
    acts (E5, 2026-10-26): list of (K, slots) slot-activity tensors, one per row — added to the audio tokens via model.adapter_slot (chunk_embed act).
    """
    assert feats and len(feats) == len(prefixes) and (acts is None or len(acts) == len(feats))
    assert len({len(p) for p in prefixes}) == 1, "batch one language/delta at a time"
    emb = model.get_input_embeddings()
    dev = emb.weight.device
    ce = [(model.chunk_embed(f[None]) if acts is None else model.chunk_embed(f[None], acts[j][None]))[0].to(emb.weight.dtype) for j, f in enumerate(feats)]   # 활동이 없으면 예전 호출 그대로
    flush = model.config.max_flush_rounds if max_flush_rounds is None else max_flush_rounds
    states = [DecodeState(len(c), model.config.runaway_cap, flush, int(6 * len(c)) + 64) for c in ce]
    assert all(s.K > 0 for s in states)
    g = spk_guide
    if g is not None:
        assert getattr(model, "spk_head", None) is not None, "spk_guide needs model.spk_head (add_spk_head)"
        spk = [model.sp_ids["<SPK_A>"], model.sp_ids["<SPK_B>"]]
        eot = model.config.sem_registry.get("<EOT>", -1) if getattr(model.config, "sem_registry", None) else -1
        special = torch.zeros(logits_size(model), dtype=torch.bool, device=dev)
        special[[model.next_audio, model.empty_audio] + spk + ([eot] if eot >= 0 else [])] = True
        probs = [[None] * len(c) for c in ce]; cur = [None] * len(ce); inp = [0] * len(ce)   # inp: audio chunk index of the current input, else -1
        for s in states: s.guide_insert = s.guide_hold = 0
    sb = switch_bias
    if sb is not None:
        assert g is None, "switch_bias and spk_guide both rewrite tags -- use one"
        assert row_modes is not None and len(row_modes) == len(states) and word_init is not None and len(word_init) == logits_size(model), "switch_bias needs row_modes (per row) and word_init (V,)"
        tags = [model.sp_ids[t] for t in (getattr(model.config, "slot_tokens", None) or ["<SPK_A>", "<SPK_B>"])]
        eot_id, trans_id = model.sp_ids.get("<EOT>", -1), model.sp_ids.get("<TRANS>", -1)
        wi = [bool(x) for x in word_init.tolist()]
        for t in tags + [eot_id, trans_id, model.next_audio, model.empty_audio]:
            if t is not None and 0 <= t < len(wi): wi[t] = False                                  # specials are never text
        decl = [[j for j, t in enumerate(tags) if t in ps] for ps in map(set, prefixes)]          # declared slots = slot tags in the row's prefix
        sbs = [[None, 0, False, False] for _ in states]                                           # per row: cur, kind, eot_open, fresh (slot tag was the last emission)
        for s in states: s.sb_flips = dict(eot=0, res=0, wi=0); s.sb_points = dict(eot=0, res=0, wi=0)
    cache = DynamicCache()
    # The thinker wrapper applies the same model + lm_head. Only the last hidden
    # state is needed, avoiding vocabulary logits over the complete prefix.
    model.thinker.model(inputs_embeds=emb(torch.tensor(prefixes, device=dev)),
                        past_key_values=cache, use_cache=True)
    x = torch.stack([c[0] for c in ce])[:, None]
    while not all(s.done for s in states):
        h = model.thinker.model(inputs_embeds=x, past_key_values=cache, use_cache=True).last_hidden_state
        logits = model.thinker.lm_head(h[:, -1]).float()
        logits[:, model.blocked] = -torch.inf
        logits[:, model.next_audio] -= next_bias
        tids = logits.argmax(-1).tolist()
        if g is not None:
            act = torch.sigmoid(model.spk_head(h[:, -1]).float()).tolist()
            for i, s in enumerate(states):
                if inp[i] >= 0: probs[i][inp[i]] = act[i]
                if s.done or s.advance: continue
                tid = tids[i]; want = _guide_slot(probs[i], s.k, g)
                if want is None: continue
                if g.insert and not special[tid] and cur[i] != want and (g.word_start is None or g.word_start(tid)):
                    tids[i] = spk[want]; s.guide_insert += 1
                elif g.hold and tid in spk and cur[i] == want and spk.index(tid) != want:
                    alt = logits[i].masked_fill(special, -torch.inf); alt[model.next_audio] = logits[i, model.next_audio]
                    tids[i] = int(alt.argmax()); s.guide_hold += 1
        if sb is not None:
            for i, s in enumerate(states):
                c, kd, eo, fr = sbs[i]; tid = tids[i]
                if row_modes[i] != "xl" or c is None or s.done or s.advance or s.n >= s.cap or len(s.emitted) >= s.max_total: continue   # forced NEXT emits nothing
                case = ("eot" if tid == tags[c] or (kd == 0 and wi[tid]) else None) if eo else "res" if kd == 1 and tid == tags[c] else "wi" if kd == 0 and not fr and wi[tid] else None
                if case is None: continue
                s.sb_points[case] += 1; b = getattr(sb, "b_" + case); oth = [tags[j] for j in decl[i] if j != c]
                if b <= 0 or not oth: continue
                lo = logits[i, oth + [tid]].tolist(); j = max(range(len(oth)), key=lo.__getitem__)
                if lo[j] + b > lo[-1]: tids[i] = oth[j]; s.sb_flips[case] += 1
        nxt = []
        for i, (s, tid) in enumerate(zip(states, tids)):
            kind, value = s.consume(tid, model.next_audio)
            if g is not None:
                if kind == "text" and value in spk: cur[i] = spk.index(value)
                inp[i] = value if kind == "audio" else -1
            if sb is not None and kind == "text":
                q = sbs[i]; q[3] = value in tags
                if q[3]: q[:3] = [tags.index(value), 0, False]
                elif value == trans_id: q[1] = 1
                elif value == eot_id: q[2] = True
            if kind == "audio":
                nxt.append(ce[i][value])
            else:
                key = {"next": model.next_audio, "empty": model.empty_audio}.get(kind, value)
                nxt.append(emb.weight[key])
        x = torch.stack(nxt)[:, None]
    return states


def logits_size(model):
    return model.thinker.lm_head.weight.shape[0]
