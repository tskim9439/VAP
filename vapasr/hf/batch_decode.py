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
def decode_batch(model, feats, prefixes, *, next_bias=0.0, max_flush_rounds=None, spk_guide=None):
    """feats: list of (1,K,D) encoder outputs, prefixes: equal-length lists.

    Reuses features across delta sweeps. Uses the same cap/flush/blocked IDs as
    model.stream_decode; never changes a model's weights or tokenizer.
    spk_guide (SpkGuide): speaker-head guided tags (needs model.spk_head); states get
    guide_insert / guide_hold counts. Serial stream_decode has no guide.
    """
    assert feats and len(feats) == len(prefixes)
    assert len({len(p) for p in prefixes}) == 1, "batch one language/delta at a time"
    emb = model.get_input_embeddings()
    dev = emb.weight.device
    ce = [model.chunk_embed(f[None])[0].to(emb.weight.dtype) for f in feats]
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
        nxt = []
        for i, (s, tid) in enumerate(zip(states, tids)):
            kind, value = s.consume(tid, model.next_audio)
            if g is not None:
                if kind == "text" and value in spk: cur[i] = spk.index(value)
                inp[i] = value if kind == "audio" else -1
            if kind == "audio":
                nxt.append(ce[i][value])
            else:
                key = {"next": model.next_audio, "empty": model.empty_audio}.get(kind, value)
                nxt.append(emb.weight[key])
        x = torch.stack(nxt)[:, None]
    return states


def logits_size(model):
    return model.thinker.lm_head.weight.shape[0]
