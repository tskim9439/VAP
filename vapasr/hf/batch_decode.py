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


@dataclass
class BlockState:
    """E9 slot-block decoding state for one row (cst_dialogue module doc E9). Input order per chunk: audio k -> tag 0 -> (outputs) ->
    tag 1 -> ... -> tag D-1 -> (outputs) -> audio k+1. The prediction after the audio is discarded (the next input is always tag 0).
    After a tag or an output, a stop (<NEXT_AUDIO>, a slot tag or <EMPTY_AUDIO>), the slot's per-chunk cap or the total cap moves
    on to the next tag (after the last slot: the next audio, or an <EMPTY_AUDIO> flush round once the audio is over). A flush round
    with no output, or max_flush rounds, ends the row. emitted gets (k, slot tag) before a slot's first output in a chunk, then
    (k, token) -- the same shape as DecodeState.emitted, so cst_eval.parse_emitted reads it unchanged.
    E13 (2026-11-02, cst_dialogue module doc E13; defaults keep E9): gate_lags > 0 opens slot s in round k only when act (Ka, N) bool has
    activity in [k - gate_lags + 1, k] or the slot has an utterance it opened (more <ONSET> than <EOT> emitted -- counted, since in translation mode an
    utterance's <EOT> follows its translation and may come after the next utterance's <ONSET>) -- the training gate
    (gate_rounds) plus the decoder's own open utterances. A round with no open slot is the audio alone. trans_id >= 0 (translation mode):
    each open slot's tag is followed by a <TRANS> input whose outputs are that slot's translation; emitted gets (k, slot tag), (k, <TRANS>)
    before them (cst_eval.parse_emitted4 reads it).
    E14 (2026-11-03, cst_dialogue module doc x_slot): x_id >= 0 adds the X box after the slot boxes of every round (input <SPK_5>, kind 2); its
    outputs are the speech of closed slots; emitted gets (k, x_id) before them (parse_emitted with the slot tags + (x_id,) reads it as slot D).
    X outputs never change the open-utterance counts."""
    K: int
    D: int
    cap: int
    max_flush: int
    max_total: int
    tags: tuple
    gate_lags: int = 0
    act: object = None
    trans_id: int = -1
    x_id: int = -1
    begin_id: int = -1
    eot_id: int = -1
    k: int = 0
    s: int = -1
    n: int = 0
    j: int = -1
    kind: int = 0
    n_round: int = 0
    n_total: int = 0
    done: bool = False
    forced: int = 0
    flush_rounds: int = 0
    n_tags: int = 0
    items: list = field(default_factory=list)
    open_: list = field(default_factory=list)
    emitted: list = field(default_factory=list)

    def open_slots(self):
        """Slots whose tag goes into round k (all D without a gate)."""
        if self.gate_lags <= 0:
            return list(range(self.D))
        if not self.open_:
            self.open_ = [0] * self.D
        a = self.act
        lo, hi = max(0, self.k - self.gate_lags + 1), min(a.shape[0], self.k + 1)
        return [s for s in range(self.D) if self.open_[s] > 0 or (s < a.shape[1] and hi > lo and bool(a[lo:hi, s].any()))]

    def _enter(self):
        self.s, self.kind = self.items[self.j]
        self.n = 0
        self.n_tags += int(self.kind == 0)
        return ("xtag", (self.k, self.s)) if self.kind == 2 else ("trans", (self.k, self.s)) if self.kind else ("tag", (self.k, self.s))

    def _next_round(self):
        if self.k >= self.K:
            self.flush_rounds += 1
            if self.n_round == 0 or self.flush_rounds >= self.max_flush:
                self.done = True
                return "pad", 0
        elif self.k == self.K - 1 and self.max_flush == 0:
            self.done = True
            return "pad", 0
        self.k += 1
        self.s = -1
        self.n_round = 0
        return ("audio", self.k) if self.k < self.K else ("empty", 0)

    def consume(self, tid, stop):
        """Prediction tid after the current input -> (next input kind, value): ('tag', (k, s)) | ('trans', (k, s)) | ('text', tid) | ('audio', k) | ('empty', 0) | ('pad', 0)."""
        if self.done:
            return "pad", 0
        if self.s < 0:
            self.items = [(s, x) for s in self.open_slots() for x in ((0, 1) if self.trans_id >= 0 else (0,))] + ([(self.D, 2)] if self.x_id >= 0 else [])
            self.j = 0
            return self._enter() if self.items else self._next_round()
        if tid in stop or self.n >= self.cap or self.n_total >= self.max_total:
            self.forced += int(tid not in stop)
            self.j += 1
            return self._enter() if self.j < len(self.items) else self._next_round()
        if self.n == 0:
            self.emitted.append((self.k, self.x_id if self.kind == 2 else self.tags[self.s]))
            if self.kind == 1:
                self.emitted.append((self.k, self.trans_id))
        self.emitted.append((self.k, int(tid)))
        if self.gate_lags > 0 and self.kind != 2 and tid in (self.begin_id, self.eot_id):
            self.open_[self.s] = self.open_[self.s] + 1 if tid == self.begin_id else max(0, self.open_[self.s] - 1)
        self.n += 1
        self.n_round += 1
        self.n_total += 1
        return "text", int(tid)


def block_stops(model, D, trans: bool = False):
    """E9 stop set: <NEXT_AUDIO>, <EMPTY_AUDIO> and every slot tag (tags are inputs, never outputs); E13 translation mode adds <TRANS> (also an input),
    E14 the X tag (model.x_id, also an input)."""
    x = int(getattr(model, "x_id", -1))
    return frozenset(model.slot_ids()) | {model.next_audio, model.empty_audio} | ({int(model.sp_ids["<TRANS>"])} if trans else set()) | ({x} if x >= 0 else set())


def block_gate(model, A):
    """E13 decoder gate from the tag activity A (K, N) (model.block_act): (gate_lags, act bool numpy) -- (0, None) when the model has no gate (config.slot_gate)."""
    import numpy as np
    gl = int(getattr(model.config, "slot_gate", 0) or 0)
    if gl <= 0:
        return 0, None
    assert A is not None, "gated slot-block decoding (config.slot_gate) needs the tag activity (act or the predicted-activity stream)"
    return gl, np.asarray(A.detach().float().cpu() if torch.is_tensor(A) else A) > 0.5


class ActGate:
    """Activity-gated slot-block decoding (diagnostic, 2026-10-30 -- wiki output-n3d-branch-20261029 6.5: does the LLM follow its slot activity input?).
    In round k the first output of slot s is turned into a stop when that slot has no activity (act (K, N) 0/1 -- the tag activity the decoder uses,
    model.block_act) near the audio it refers to: a text token covers frame f = k - delta, <BEGIN> the onset frame f = k - begin_lag (training rule,
    cst_eval.activity_from_events); the window is [f - back, f + fwd]. <EOT> and stops pass, and so does a window outside the activity (before the audio,
    late flush rounds). Decided on the first output only, so a slot's round is kept or dropped whole -- decode_blocks_batch (gate_tid) and
    realtime.realtime_block_decode apply the same rule. blocked counts the dropped first outputs {text, begin}."""

    def __init__(self, act, delta: int, begin_id: int, eot_id: int, back: int = 3, fwd: int = 1, begin_lag: int = 4):
        import numpy as np
        a = np.asarray(act.detach().float().cpu() if torch.is_tensor(act) else act) > 0.5
        self.K, self.N = a.shape
        self.cs = np.concatenate([np.zeros((1, self.N), np.int64), np.cumsum(a, 0)], 0)
        self.delta, self.begin_id, self.eot_id, self.back, self.fwd, self.begin_lag = int(delta), int(begin_id), int(eot_id), int(back), int(fwd), int(begin_lag)
        self.blocked = dict(text=0, begin=0)

    def allows(self, k, s, tid) -> bool:
        if tid == self.eot_id or s >= self.N:
            return True
        b = tid == self.begin_id
        f = int(k) - (self.begin_lag if b else self.delta)
        lo, hi = max(0, f - self.back), min(self.K, f + self.fwd + 1)
        if hi <= lo or self.cs[hi, s] > self.cs[lo, s]:
            return True
        self.blocked["begin" if b else "text"] += 1
        return False


def gate_tid(state, tid, stop, gate, stop_id):
    """BlockState prediction after the current input: a slot's first output (state.s >= 0, state.n == 0) that the gate drops becomes stop_id."""
    if gate is None or state.done or state.s < 0 or state.n or tid in stop or gate.allows(state.k, state.s, tid):
        return tid
    return stop_id


@torch.inference_mode()
def decode_blocks_batch(model, feats, prefixes, D, *, max_flush_rounds=None, acts=None, mix_off=False, gates=None, trans=False, gate_acts=None):
    """E9 slot-block greedy decoding, one input per row per step (BlockState) on a dense KV-cache batch.
    feats: list of (S,K,D) encoder outputs, prefixes: equal-length lists declaring D slots. The audio tokens use model.chunk_embed (with acts
    or the E6 predicted-activity stream, as decode_batch); each tag gets model.tag_embeds (tag row + that slot's recent activity, model.block_act).
    Caps: model.config.runaway_cap per slot per chunk, 6 K + 64 outputs per row, max_flush_rounds (default config).
    E10 (model.adapter_slotx): each tag also gets that slot's audio token (model.slot_audio_of — feats need the head-embedding stream);
    mix_off replaces the mixed audio token with model.adapter_mixnull (the chunk keeps its activity part).
    gates: per-row ActGate or None (activity-gated decoding -- a slot's round is dropped when its activity is off; the gate counts what it dropped).
    E13: a model with config.slot_gate opens only the slots its tag activity (or an open utterance) allows (BlockState gate_lags, block_gate);
    trans=True (translation-mode prefix) adds each open slot's <TRANS> input and its translation outputs.
    gate_acts (2026-11-03 gate diagnostics): per-row (K, N) activity the E13 gate opens on instead of the tag activity (e.g. expert probability > a lower
    threshold) -- the tag activity itself is unchanged."""
    assert feats and len(feats) == len(prefixes) and (acts is None or len(acts) == len(feats)) and (gates is None or len(gates) == len(feats))
    assert gate_acts is None or len(gate_acts) == len(feats)
    assert len({len(p) for p in prefixes}) == 1, "batch one language/delta at a time"
    emb = model.get_input_embeddings()
    dev = emb.weight.device
    mm = torch.ones(1, dtype=torch.bool, device=dev) if mix_off else None
    assert not mix_off or getattr(model, "adapter_mixnull", None) is not None, "mix_off needs an E10 model (adapter_mixnull)"
    ce = [(model.chunk_embed(f[None], mix_mask=mm) if acts is None else model.chunk_embed(f[None], acts[j][None], mix_mask=mm))[0].to(emb.weight.dtype) for j, f in enumerate(feats)]
    A = [model.block_act(f, acts[j] if acts is not None else None) for j, f in enumerate(feats)]
    X = [model.slot_audio_of(f, a) if getattr(model, "adapter_slotx", None) is not None else None for f, a in zip(feats, A)]
    flush = model.config.max_flush_rounds if max_flush_rounds is None else max_flush_rounds
    tags = tuple(model.slot_ids(D))
    stop = block_stops(model, D, trans)
    TR = int(model.sp_ids["<TRANS>"]) if trans else -1
    BG, EOT = int(model.sp_ids.get("<ONSET>", -1)), int(model.sp_ids["<EOT>"]); XI = int(getattr(model, "x_id", -1))   # E14 X 칸(없으면 −1)
    states = []
    for j, (c, a) in enumerate(zip(ce, A)):
        gl, ga = block_gate(model, a if gate_acts is None else gate_acts[j])
        states.append(BlockState(len(c), D, model.config.runaway_cap, flush, int((12 if trans else 6) * len(c)) + 64, tags, gate_lags=gl, act=ga,
                                 trans_id=TR, x_id=XI, begin_id=BG, eot_id=EOT))
    assert all(s.K > 0 for s in states)
    cache = DynamicCache()
    model.thinker.model(inputs_embeds=emb(torch.tensor(prefixes, device=dev)), past_key_values=cache, use_cache=True)
    x = torch.stack([c[0] for c in ce])[:, None]
    tag_cache = [{} for _ in states]                                                             # row -> {k: (D, H) tag embeddings}
    while not all(s.done for s in states):
        h = model.thinker.model(inputs_embeds=x, past_key_values=cache, use_cache=True).last_hidden_state
        logits = model.thinker.lm_head(h[:, -1]).float()
        logits[:, model.blocked] = -torch.inf
        tids = logits.argmax(-1).tolist()
        nxt = []
        for i, (s, tid) in enumerate(zip(states, tids)):
            if gates is not None:
                tid = gate_tid(s, tid, stop, gates[i], model.next_audio)
            kind, value = s.consume(tid, stop)
            if kind == "audio":
                nxt.append(ce[i][value])
            elif kind == "tag":
                k, j = value
                if k not in tag_cache[i]:
                    tag_cache[i] = {k: model.tag_embeds(k, A[i], D, X[i]).to(emb.weight.dtype)}
                nxt.append(tag_cache[i][k][j])
            elif kind == "trans":
                nxt.append(emb.weight[TR])
            elif kind == "xtag":
                nxt.append(emb.weight[XI])
            elif kind == "text":
                nxt.append(emb.weight[value])
            else:
                nxt.append(emb.weight[model.empty_audio if kind == "empty" else model.next_audio])
        x = torch.stack(nxt)[:, None]
    return states
