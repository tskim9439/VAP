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


@torch.inference_mode()
def decode_batch(model, feats, prefixes, *, next_bias=0.0, max_flush_rounds=None):
    """feats: list of (1,K,D) encoder outputs, prefixes: equal-length lists.

    Reuses features across delta sweeps. Uses the same cap/flush/blocked IDs as
    model.stream_decode; never changes a model's weights or tokenizer.
    """
    assert feats and len(feats) == len(prefixes)
    assert len({len(p) for p in prefixes}) == 1, "batch one language/delta at a time"
    emb = model.get_input_embeddings()
    dev = emb.weight.device
    ce = [model.chunk_embed(f[None])[0].to(emb.weight.dtype) for f in feats]
    flush = model.config.max_flush_rounds if max_flush_rounds is None else max_flush_rounds
    states = [DecodeState(len(c), model.config.runaway_cap, flush, int(6 * len(c)) + 64) for c in ce]
    assert all(s.K > 0 for s in states)
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
        nxt = []
        for i, (s, tid) in enumerate(zip(states, tids)):
            kind, value = s.consume(tid, model.next_audio)
            if kind == "audio":
                nxt.append(ce[i][value])
            else:
                key = {"next": model.next_audio, "empty": model.empty_audio}.get(kind, value)
                nxt.append(emb.weight[key])
        x = torch.stack(nxt)[:, None]
    return states
