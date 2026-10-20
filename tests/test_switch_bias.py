"""V3 C3 상태 인지 교대 bias(vapasr/hf/batch_decode.SwitchBias, decode_batch(switch_bias=…)) — 각본 stub thinker 로 결정 지점
(i) <EOT> 뒤 · (ii) <TRANS> 뒤 같은 슬롯 재개 · (iii) 전사 word-start 에서만 다른 선언 슬롯 태그로 뒤집히는지(logit 차 < β 일 때만),
끄면(None · 전부 0 · 같은 언어 행) 토큰열이 같은지, 카운터, 평가 스크립트 배선(기본 끔 = 예전 출력 · --verify 거부)을 본다. CPU, 모델·tokenizer 다운로드 없음."""
import os, sys, types
import numpy as np, pytest
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if hasattr(transformers, "DynamicCache"): import vapasr.hf.batch_decode as BD
else:                                                                  # 구버전 transformers(로컬 base): stub 은 캐시를 쓰지 않는다 — import 때만 빈 클래스로 채우고,
    import vapasr.hf                                                   # 다른 시험이 이 모듈을 보지 않게 sys.modules·패키지 속성에서 다시 뺀다(_bd 가 이 파일 시험 동안만 넣는다)
    transformers.DynamicCache = type("DynamicCache", (), {})
    try: import vapasr.hf.batch_decode as BD
    finally: del transformers.DynamicCache; sys.modules.pop("vapasr.hf.batch_decode", None); vars(vapasr.hf).pop("batch_decode", None)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "experiments"))


@pytest.fixture(autouse=True)
def _bd(monkeypatch):
    monkeypatch.setitem(sys.modules, "vapasr.hf.batch_decode", BD)     # 평가 스크립트 run() 안의 import 도 이 모듈을 쓴다

V = 40; NEXT, EMPTY, A, B, C, EOT, TRANS = 30, 31, 32, 33, 34, 35, 36
W1, W2, W3, M1 = 11, 12, 13, 21                                        # W* = 단어 첫 토큰(디코드가 공백으로 시작), M1 = 단어 중간
WI = torch.zeros(V, dtype=torch.bool); WI[[W1, W2, W3]] = True
PRE = [1, 2, A, 3, B, 4]                                               # 'language <SPK_A>… <SPK_B>…' — 선언 슬롯 A·B
N = {NEXT: 5.0}
ALL = BD.SwitchBias(b_eot=10.0, b_res=10.0, b_wi=10.0)


class _Head:
    def __init__(s, th): s.th, s.weight = th, torch.zeros(V, 1)        # weight: logits_size
    def __call__(s, h): return torch.stack([s.th.row(i) for i in range(h.shape[0])])


class _Thinker:
    """각본 thinker: prefix 다음 t 번째 스텝에 행 i 의 logit = script[i][t]({tid: 값}, 나머지 −10). 각본이 끝나면 <NEXT_AUDIO>."""
    def __init__(s, script): s.script, s.t, s.lm_head = script, -1, _Head(s)
    def model(s, inputs_embeds=None, **_): s.t += 1; return types.SimpleNamespace(last_hidden_state=inputs_embeds)
    def row(s, i):
        x = torch.full((V,), -10.0)
        for k, v in (s.script[i][s.t - 1] if s.t - 1 < len(s.script[i]) else {NEXT: 10.0}).items(): x[k] = v
        return x


class _Model:
    def __init__(s, script, slots=("<SPK_A>", "<SPK_B>")):
        s.thinker, s.emb = _Thinker(script), torch.nn.Embedding(V, 2); s.next_audio, s.empty_audio, s.blocked = NEXT, EMPTY, torch.tensor([EMPTY])
        s.sp_ids = {"<NEXT_AUDIO>": NEXT, "<EMPTY_AUDIO>": EMPTY, "<SPK_A>": A, "<SPK_B>": B, "<SPK_3>": C, "<EOT>": EOT, "<TRANS>": TRANS}
        s.config = types.SimpleNamespace(max_flush_rounds=0, runaway_cap=8, slot_tokens=list(slots))
    def get_input_embeddings(s): return s.emb
    def chunk_embed(s, f): return f


def plan(*steps):
    """스텝 각본 — <NEXT_AUDIO> 가 이기는 스텝 뒤에는 출력이 버려지는 advance 스텝을 끼운다."""
    out = []
    for x in steps: out += [x] + ([{}] if max(x, key=x.get) == NEXT else [])
    return out


def dec(scripts, sb=None, modes=None, pre=PRE, slots=("<SPK_A>", "<SPK_B>"), K=3):
    n = len(scripts)
    return BD.decode_batch(_Model(scripts, slots), [torch.zeros(K, 2) for _ in range(n)], [list(pre)] * n,
                           **(dict(switch_bias=sb, row_modes=modes or ["xl"] * n, word_init=WI) if sb is not None else {}))


def toks(s):
    return [t for _, t in s.emitted]


def test_off_none_zero_and_same_language_rows_keep_tokens():
    g = np.random.default_rng(0); ids = [A, B, EOT, TRANS, NEXT, W1, W2, W3, M1]
    sc = [[{t: float(v) for t, v in zip(ids, g.normal(size=len(ids)))} for _ in range(300)] for _ in range(3)]
    ref = BD.decode_batch(_Model(sc), [torch.zeros(6, 2)] * 3, [PRE] * 3)                       # 인자 없음 = 예전 호출
    assert not hasattr(ref[0], "sb_flips") and sum(len(s.emitted) for s in ref) > 20 and {A, B, EOT, TRANS} <= {t for s in ref for t in toks(s)}
    for kw in (dict(switch_bias=None), dict(switch_bias=BD.SwitchBias(), row_modes=["xl"] * 3, word_init=WI),
               dict(switch_bias=ALL, row_modes=["same"] * 3, word_init=WI)):
        got = BD.decode_batch(_Model(sc), [torch.zeros(6, 2)] * 3, [PRE] * 3, **kw)
        assert [(s.emitted, s.forced, s.flush_rounds, s.k) for s in got] == [(s.emitted, s.forced, s.flush_rounds, s.k) for s in ref]
        assert all(sum(s.sb_flips.values()) == 0 for s in got) if kw["switch_bias"] is not None else not hasattr(got[0], "sb_flips")
    z = BD.decode_batch(_Model(sc), [torch.zeros(6, 2)] * 3, [PRE] * 3, switch_bias=BD.SwitchBias(), row_modes=["xl"] * 3, word_init=WI)
    assert sum(sum(s.sb_points.values()) for s in z) > 0                                          # 0 이어도 결정 지점은 센다
    assert not BD.SwitchBias().on and BD.SwitchBias(b_wi=0.5).on
    with pytest.raises(AssertionError): dec([sc[0]], BD.SwitchBias(b_eot=1.0), modes=["xl", "xl"])   # row_modes 는 행마다
    with pytest.raises(AssertionError): BD.decode_batch(_Model([sc[0]]), [torch.zeros(6, 2)], [PRE], switch_bias=ALL, row_modes=["xl"], word_init=WI[:10])


def test_wi_flips_word_start_only_when_gap_below_bias():
    sc = plan({A: 5.0}, {W1: 5.0, B: 4.9}, {M1: 5.0, B: 4.9}, {W2: 5.0, B: 4.0}, N)              # 태그 바로 뒤 W1 · 단어 중간 M1 은 건드리지 않는다
    base = toks(dec([sc])[0]); assert base == [A, W1, M1, W2]
    s = dec([sc], BD.SwitchBias(b_wi=1.5))[0]; assert toks(s) == [A, W1, M1, B] and s.sb_flips == dict(eot=0, res=0, wi=1) and s.sb_points == dict(eot=0, res=0, wi=1)
    s = dec([sc], BD.SwitchBias(b_wi=0.5))[0]; assert toks(s) == base and s.sb_flips["wi"] == 0 and s.sb_points["wi"] == 1
    s = dec([sc], BD.SwitchBias(b_eot=10.0, b_res=10.0))[0]; assert toks(s) == base                # 다른 결정 지점의 β 는 word-start 에 안 걸린다


def test_res_flips_same_slot_reopen_after_trans_only():
    sc = plan({A: 5.0}, {W1: 5.0}, {TRANS: 5.0, B: 4.9}, {W2: 5.0, B: 4.9}, {M1: 5.0, B: 4.9}, {A: 5.0, B: 4.0}, N)
    base = toks(dec([sc])[0]); assert base == [A, W1, TRANS, W2, M1, A]
    s = dec([sc], BD.SwitchBias(b_res=1.5, b_wi=10.0))[0]                                         # 번역 kind word-start(W2)·<TRANS> 는 그대로
    assert toks(s) == [A, W1, TRANS, W2, M1, B] and s.sb_flips == dict(eot=0, res=1, wi=0) and s.sb_points == dict(eot=0, res=1, wi=0)
    s = dec([sc], BD.SwitchBias(b_res=0.5, b_wi=10.0, b_eot=10.0))[0]; assert toks(s) == base and s.sb_points["res"] == 1


def test_eot_flips_reassert_and_word_start_after_eot():
    sc = plan({A: 5.0}, {W1: 5.0}, {EOT: 5.0, B: 4.9}, {NEXT: 5.0, B: 4.9}, {A: 5.0, B: 4.0}, N)     # <EOT>·<NEXT_AUDIO> 는 그대로, 다음 청크의 <SPK_A> 재선언
    base = toks(dec([sc])[0]); assert base == [A, W1, EOT, A]
    s = dec([sc], BD.SwitchBias(b_eot=1.5, b_wi=10.0))[0]; assert toks(s) == [A, W1, EOT, B] and s.sb_flips == dict(eot=1, res=0, wi=0)
    assert toks(dec([sc], BD.SwitchBias(b_eot=0.5, b_wi=10.0, b_res=10.0))[0]) == base
    sc = plan({A: 5.0}, {W1: 5.0}, {EOT: 5.0}, {W2: 5.0, B: 4.0}, N)                                # <EOT> 뒤 태그 없이 전사 word-start(놓친 교대 SILENT)
    assert toks(dec([sc])[0]) == [A, W1, EOT, W2]
    s = dec([sc], BD.SwitchBias(b_eot=1.5))[0]; assert toks(s) == [A, W1, EOT, B] and s.sb_points == dict(eot=1, res=0, wi=0)
    assert toks(dec([sc], BD.SwitchBias(b_wi=10.0))[0]) == [A, W1, EOT, W2]                         # <EOT> 뒤는 b_eot 만
    sc = plan({A: 5.0}, {W1: 5.0}, {TRANS: 5.0}, {W2: 5.0}, {EOT: 5.0}, {W3: 5.0, B: 4.9}, {A: 5.0, B: 4.0}, N)   # 번역 뒤 <EOT>: 번역 word-start 는 그대로, 재개 태그는 (i)
    s = dec([sc], BD.SwitchBias(b_eot=1.5))[0]; assert toks(s) == [A, W1, TRANS, W2, EOT, W3, B] and s.sb_flips == dict(eot=1, res=0, wi=0) and s.sb_points["eot"] == 1
    assert toks(dec([sc], BD.SwitchBias(b_res=10.0, b_wi=10.0))[0]) == [A, W1, TRANS, W2, EOT, W3, A]


def test_untouched_positions():
    first = plan({A: 5.0, B: 4.9}, {W1: 5.0, B: 4.9}, N)                                           # 첫 태그(cur 없음)
    pre_tag = plan({W1: 5.0, B: 4.9}, {A: 5.0, B: 4.9}, N)                                         # 첫 태그 앞 글자
    nxt = plan({A: 5.0}, {W1: 5.0}, {NEXT: 5.0, B: 4.9}, {EMPTY: 9.0, TRANS: 5.0, B: 4.9}, {EOT: 5.0, B: 4.9}, N)   # <NEXT_AUDIO>·<TRANS>·<EOT>(<EMPTY_AUDIO> 차단)
    for sc in (first, pre_tag, nxt):
        assert toks(dec([sc], ALL)[0]) == toks(dec([sc])[0])
    assert toks(dec([nxt])[0]) == [A, W1, TRANS, EOT]


def test_bias_goes_to_other_declared_slot_not_fixed_index():
    s3 = ("<SPK_A>", "<SPK_B>", "<SPK_3>")
    sc = plan({A: 5.0}, {W1: 5.0}, {W2: 5.0, C: 4.8, B: 3.0}, N)                                 # 선언 A·B: 선언 안 한 C 에는 bias 가 가지 않는다
    assert toks(dec([sc], BD.SwitchBias(b_wi=1.5), slots=s3)[0]) == [A, W1, W2]
    assert toks(dec([sc], BD.SwitchBias(b_wi=2.5), slots=s3)[0]) == [A, W1, B]
    sc = plan({B: 5.0}, {W1: 5.0}, {W2: 5.0, A: 4.0, B: 4.9}, N)                                 # 현재 B 면 다른 슬롯 = A
    assert toks(dec([sc], BD.SwitchBias(b_wi=1.5))[0]) == [B, W1, A]
    sc = plan({A: 5.0}, {W1: 5.0}, {W2: 5.0, B: 3.9, C: 4.2}, N)                                 # 셋 선언: 다른 슬롯 중 bias 뒤 가장 큰 것
    assert toks(dec([sc], BD.SwitchBias(b_wi=1.5), slots=s3, pre=PRE + [C])[0]) == [A, W1, C]
    sc = plan({A: 5.0}, {W1: 5.0}, {W2: 5.0, B: 4.0}, N)                                         # 한 슬롯만 선언 = 다른 슬롯 없음
    assert toks(dec([sc], BD.SwitchBias(b_wi=10.0), pre=[1, A, 2])[0]) == [A, W1, W2]


def test_counters_exact_along_flipped_path():
    sc = plan({A: 5.0}, {W1: 5.0, B: 4.9}, {W2: 5.0, B: 4.0}, {W3: 5.0, A: 4.9}, {TRANS: 5.0, A: 4.9}, {W1: 5.0, A: 4.9}, {B: 5.0, A: 4.0},
              {NEXT: 5.0, B: 4.9}, {W2: 5.0}, {M1: 5.0, B: 4.9}, {EOT: 5.0, B: 4.9}, N, {A: 5.0, B: 4.5}, {W1: 5.0, A: 4.9}, {W2: 5.0, A: 2.0}, N)
    assert toks(dec([sc])[0]) == [A, W1, W2, W3, TRANS, W1, B, W2, M1, EOT, A, W1, W2]
    s = dec([sc], BD.SwitchBias(b_eot=1.0, b_res=1.5, b_wi=1.5))[0]
    assert toks(s) == [A, W1, B, W3, TRANS, W1, A, W2, M1, EOT, B, W1, W2]                       # wi(B) → res(A) → eot(B), 마지막 word-start 는 차 3 > 1.5
    assert s.sb_flips == dict(eot=1, res=1, wi=1) and s.sb_points == dict(eot=1, res=1, wi=2)
    z = dec([sc], BD.SwitchBias())[0]; assert toks(z) == toks(dec([sc])[0]) and z.sb_points == dict(eot=0, res=0, wi=3) and z.sb_flips == dict(eot=0, res=0, wi=0)
    two = dec([sc, sc], BD.SwitchBias(b_eot=1.0, b_res=1.5, b_wi=1.5), modes=["same", "xl"])        # 행마다 따로(같은 언어 행은 그대로)
    assert toks(two[0]) == toks(dec([sc])[0]) and toks(two[1]) == toks(s) and two[0].sb_points == dict(eot=0, res=0, wi=0) and two[1].sb_flips == s.sb_flips


def test_parse_policy_and_word_init_mask():
    assert BD.SwitchBias.parse("eot=1.5,res=0.5,wi=0") == BD.SwitchBias(1.5, 0.5, 0.0) and BD.SwitchBias.parse(" wi = 1 ") == BD.SwitchBias(b_wi=1.0)
    assert BD.SwitchBias(1.0, 0.5, 0.0).policy() == dict(eot=1.0, res=0.5, wi=0.0, rows="xl")
    for bad in ("eot=-1", "foo=1", "eot", "eot=1;res=1"):
        with pytest.raises((AssertionError, ValueError)): BD.SwitchBias.parse(bad)
    tok = _Tok(); m = BD.word_init_mask(tok, len(tok) + 5)
    assert m.shape == (len(tok) + 5,) and m[27:53].all() and not m[1:27].any() and not m[53] and not m[len(tok):].any() and not m[tok.v["<SPK_A>"]]


# ── 평가 스크립트 배선(가짜 tokenizer · 결정적 stub 모델 · 합성 wav)
class _Tok:
    """가짜 byte-level tokenizer — 'a'–'z' = 1–26(단어 중간), ' a'–' z' = 27–52(단어 첫 토큰), 그 밖 글자 = 53, 특수 토큰 = 150000 부터."""
    def __init__(s):
        from vapasr.data.dialogue_tokens import PHASE2_SPECIALS
        from vapasr.uslm.interleave_data import SPECIAL_TOKENS
        s.v = {}; s.add_tokens(["<|audio_pad|>", "<TRANS>", "<asr_text>"] + SPECIAL_TOKENS + PHASE2_SPECIALS)
    def add_tokens(s, toks, special_tokens=True):
        for t in toks: s.v.setdefault(t, 150000 + len(s.v))
        s.inv = {v: k for k, v in s.v.items()}
    def convert_tokens_to_ids(s, t): return s.v[t]
    def __len__(s): return 150000 + len(s.v)
    def __call__(s, text, add_special_tokens=False, **kw):
        out, i = [], 0
        while i < len(text):
            sp = next((t for t in s.v if text.startswith(t, i)), None)
            if sp: out.append(s.v[sp]); i += len(sp)
            elif text[i] == " " and "a" <= text[i + 1: i + 2] <= "z": out.append(27 + ord(text[i + 1]) - 97); i += 2
            else: out.append(1 + ord(text[i]) - 97 if "a" <= text[i] <= "z" else 53); i += 1
        return {"input_ids": out}
    def convert_ids_to_tokens(s, ids):
        return [s.inv.get(t) or (chr(96 + t) if 1 <= t <= 26 else "Ġ" + chr(70 + t) if 27 <= t <= 52 else "x" if t == 53 else None) for t in map(int, ids)]
    def decode(s, ids): return "".join((p or "").replace("Ġ", " ") for p in s.convert_ids_to_tokens(ids))


class _SubHead:
    def __init__(s, W, sub, b, V): s.W, s.sub, s.b, s.weight = W, sub, b, torch.zeros(V, 1)
    def __call__(s, h):
        out = torch.full((h.shape[0], s.weight.shape[0]), -1e4); out[:, s.sub] = h.float() @ s.W.T + s.b; return out


class _LM(torch.nn.Module):
    """결정적 stub 모델 — 입력 이력(감쇠 합)의 tanh 를 작은 어휘(글자·두 슬롯 태그·<EOT>·<TRANS>·<NEXT_AUDIO>) logit 으로 사영(나머지 −1e4).
    encode = 80 ms 청크 통계 사영. 같은 입력이면 같은 출력이라 디코드 결과를 바이트 단위로 비교할 수 있다."""
    def __init__(s, tok, D=8):
        super().__init__(); g = torch.Generator().manual_seed(0); v = tok.v; Vt = len(tok)
        s.emb = torch.nn.Embedding(Vt, D)
        with torch.no_grad(): s.emb.weight.copy_(torch.randn(Vt, D, generator=g))
        sub = list(range(1, 54)) + [v["<SPK_A>"], v["<SPK_B>"], v["<EOT>"], v["<TRANS>"], v["<NEXT_AUDIO>"]]
        b = torch.zeros(len(sub)); b[-5:] = torch.tensor([1.5, 1.5, 1.0, 1.0, 2.5])
        s.P = torch.randn(4, D, generator=g); s.thinker = types.SimpleNamespace(model=s._fwd, lm_head=_SubHead(torch.randn(len(sub), D, generator=g) * 2, torch.tensor(sub), b, Vt))
        s.sp_ids = dict(v); s.next_audio, s.empty_audio = v["<NEXT_AUDIO>"], v["<EMPTY_AUDIO>"]; s.blocked = torch.tensor([v["<EMPTY_AUDIO>"]])
        s.config = types.SimpleNamespace(sp_ids=dict(v), slot_tokens=["<SPK_A>", "<SPK_B>"], blocked_ids=[v["<EMPTY_AUDIO>"]], max_flush_rounds=2, runaway_cap=6)
    def _fwd(s, inputs_embeds=None, **_):
        x = inputs_embeds.float(); s.h = x.mean(1) if x.shape[1] > 1 else 0.7 * s.h + x[:, 0]
        return types.SimpleNamespace(last_hidden_state=torch.tanh(s.h)[:, None])
    def get_input_embeddings(s): return s.emb
    def chunk_embed(s, f): return f
    def encode(s, w, n, K):
        K = int(K); x = torch.nn.functional.pad(w[0], (0, max(0, K * 1280 - w.shape[1])))[: K * 1280].reshape(K, 1280)
        return ((torch.stack([x.mean(1), x.std(1), x.abs().amax(1), x[:, :640].mean(1) - x[:, 640:].mean(1)], -1) * 4) @ s.P)[None]


def _wav(path, dur, seed):
    sf = pytest.importorskip("soundfile"); t = np.arange(int(dur * 16000)) / 16000; g = np.random.default_rng(seed)
    x = np.stack([0.3 * np.sin(2 * np.pi * f * t) * (1 + np.sin(2 * np.pi * t / (1.3 + i))) + 0.05 * g.standard_normal(len(t)) for i, f in enumerate((440, 660))], 1)
    sf.write(path, x.astype(np.float32), 16000)


def _dialogues(d):
    """corpus ko(한국어 c1)·xl(영·독 c3, meta.spk_lang) — 24 s, x·y 교대(1 s 쉼), 글자 토큰. 번역 타깃(행 id → {목표 언어: (ids, 시각)})도 돌려준다.
    c3: 8 s 창 두 개 모두 창 id 해시가 슬롯을 뒤집지 않는다 — cst_stage3_eval.window_ref 는 참조 흐름을 먼저 말한 순서로 만들어(교차 언어 슬롯 순서와 다르면 어긋난다) 그 순서가 같은 창만 쓴다."""
    from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef
    _wav(os.path.join(d, "st.wav"), 24.0, 0); tr = {}
    for corpus, cid, lang, sl in (("ko", "c1", "Korean", None), ("xl", "c3", "English+German", {"x": "English", "y": "German"})):
        utts = []
        for k in range(4):
            s = k * 6.0; utts.append(Utterance("x", s, s + 2.0, "ab", "ab", tokens=[(27, s + 1.0), (2, s + 1.5), (29, s + 2.0)], utt_id=f"x{k}"))
            utts.append(Utterance("y", s + 3.0, s + 5.0, "cd", "cd", tokens=[(30, s + 4.0), (4, s + 5.0)], utt_id=f"y{k}"))
        dl = Dialogue(conv_id=cid, corpus=corpus, lang=lang, split="train", duration_s=24.0, speakers=["x", "y"], utterances=utts,
                      channels={w: ChannelRef(path=os.path.join(d, "st.wav") + f"#ch{i}") for i, w in enumerate(["x", "y"])}, meta=dict(spk_lang=sl) if sl else {})
        open(os.path.join(d, f"{corpus}.refined.dialogues.jsonl"), "w").write(dl.to_json() + "\n")
        for u in utts: tr[f"{cid}|{u.speaker}|{u.utt_id}"] = {("German" if u.speaker == "x" else "English") if sl else "English": (np.array([40, 5, 41]), np.array([u.start + 1.0, u.end, u.end]))}
    return tr


def _sessions(d):
    """TAXI 형식 세션 3 개(독·영 두 화자, 엄격 교대) + 혼합 wav — 평가 데이터 아님(합성)."""
    import json
    os.makedirs(os.path.join(d, "mix"), exist_ok=True); out = []
    for j, dur in enumerate((5.0, 6.4, 4.8)):
        sid = f"s{j}"; _wav(os.path.join(d, "mix", f"{sid}.wav"), dur, j + 1)
        import soundfile as sf; x, _ = sf.read(os.path.join(d, "mix", f"{sid}.wav"), dtype="float32"); sf.write(os.path.join(d, "mix", f"{sid}.wav"), x.mean(1), 16000)
        turns = [dict(speaker="a" if k % 2 == 0 else "b", lang="German" if k % 2 == 0 else "English", start_s=k * 1.5, end_s=k * 1.5 + 1.2, transcript="ab cd") for k in range(int(dur // 1.5))]
        out.append(dict(session_id=sid, speakers=dict(a=dict(lang="German"), b=dict(lang="English")), duration_s=dur, turns=turns, mix=f"mix/{sid}.wav"))
    open(os.path.join(d, "sessions.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in out)); return os.path.join(d, "sessions.jsonl")


def run_script(mod, argv, monkeypatch, tok=None, tr=None):
    """mod.main() 을 stub 모델로 돌린다(load_model·load_ast_targets 바꿔치기) → stub 모델."""
    import collections, vapasr.hf.infer, vapasr.data.ast_dataset
    tok = tok or _Tok(); model = _LM(tok)
    monkeypatch.setattr(vapasr.hf.infer, "load_model", lambda *x, **k: (model, tok))
    if tr is not None: monkeypatch.setattr(vapasr.data.ast_dataset, "load_ast_targets", lambda *x, **k: (tr, collections.Counter(rows=len(tr)), "fp"))
    monkeypatch.setattr(sys, "argv", ["x"] + list(argv)); mod.main(); return model


def _ck(d):
    os.makedirs(d, exist_ok=True); open(os.path.join(d, "config.json"), "w").write("{}"); return d


def _read(path):
    return open(path, "rb").read()


def test_stage3_eval_refuses_verify_with_bias(tmp_path, monkeypatch):
    import cst_stage3_eval as E3, vapasr.hf.infer
    class Hit(Exception): pass
    def boom(*x, **k): raise Hit()
    monkeypatch.setattr(vapasr.hf.infer, "load_model", boom)
    base = ["run", "--checkpoint", _ck(str(tmp_path / "ck")), "--data", str(tmp_path), "--out", str(tmp_path / "o"), "--verify"]
    monkeypatch.setattr(sys, "argv", ["x"] + base + ["--switch-bias", "eot=1.5"])
    with pytest.raises(AssertionError, match="--verify"): E3.main()                             # 모델을 읽기 전에 거부
    monkeypatch.setattr(sys, "argv", ["x"] + base + ["--switch-bias", "eot=0,res=0,wi=0"])
    with pytest.raises(Hit): E3.main()                                                          # 전부 0 은 같은 토큰열이라 --verify 와 함께 허용
    monkeypatch.setattr(sys, "argv", ["x", "run", "--checkpoint", str(tmp_path / "ck"), "--data", str(tmp_path), "--out", str(tmp_path / "o"), "--oracle", "--switch-bias", "wi=1"])
    with pytest.raises(AssertionError): E3.main()                                               # 디코드 없는 경로와도 함께 쓰지 않는다
    import cst_stage4_eval as E4
    monkeypatch.setattr(sys, "argv", ["x", "run", "--checkpoint", str(tmp_path / "ck"), "--data", str(tmp_path), "--trans", "t", "--out", str(tmp_path / "o"), "--oracle", "--switch-bias", "wi=1"])
    with pytest.raises(AssertionError): E4.main()


def _rows(out):
    import json
    return [json.loads(l) for l in open(os.path.join(out, "predictions-rank0.jsonl"), encoding="utf-8")]


def stage3_args(d, out, *extra):
    return ["run", "--checkpoint", _ck(os.path.join(d, "ck")), "--data", d, "--out", out, "--corpora", "ko,xl", "--deltas", "2,4", "--limit-windows", "0",
            "--window", "8,8", "--hop", "4", "--batch-size", "4"] + list(extra)


def test_stage3_eval_switch_bias_default_off_parity(tmp_path, monkeypatch):
    import json, cst_stage3_eval as E3
    d = str(tmp_path); _dialogues(d)
    run_script(E3, stage3_args(d, str(tmp_path / "off")), monkeypatch)
    run_script(E3, stage3_args(d, str(tmp_path / "zero"), "--switch-bias", "eot=0,res=0,wi=0"), monkeypatch)
    run_script(E3, stage3_args(d, str(tmp_path / "on"), "--switch-bias", "eot=3,res=3,wi=3"), monkeypatch)
    off, zero, on = (_rows(str(tmp_path / k)) for k in ("off", "zero", "on"))
    assert len(off) > 4 and {r["corpus"] for r in off} == {"ko", "xl"} and not any("sb_flips" in r for r in off)
    assert "switch_bias" not in json.load(open(tmp_path / "off" / "config-rank0.json"))
    assert json.load(open(tmp_path / "on" / "config-rank0.json"))["switch_bias"] == dict(eot=3.0, res=3.0, wi=3.0, rows="xl")
    strip = lambda r: {k: v for k, v in r.items() if k not in ("sb_flips", "sb_points")}
    assert [strip(r) for r in zero] == off and all(sum(r["sb_flips"].values()) == 0 for r in zero)
    for a, b in zip(off, on):                                                                   # 같은 언어 창은 그대로, 교차 언어 창만 바뀐다
        if a["corpus"] == "ko": assert strip(b) == a and sum(b["sb_points"].values()) == 0
    assert sum(sum(r["sb_flips"].values()) for r in on if r["corpus"] == "xl") > 0 and any(a["emitted"] != b["emitted"] for a, b in zip(off, on) if a["corpus"] == "xl")


def stage4_args(d, out, *extra):
    return ["run", "--checkpoint", _ck(os.path.join(d, "ck")), "--data", d, "--trans", "t", "--out", out, "--corpora", "ko,xl", "--deltas", "2,4", "--limit-windows", "0",
            "--window", "8,8", "--hop", "4", "--batch-size", "4"] + list(extra)


def test_stage4_eval_switch_bias_default_off_parity(tmp_path, monkeypatch):
    import json, cst_stage4_eval as E4
    d = str(tmp_path); tr = _dialogues(d)
    run_script(E4, stage4_args(d, str(tmp_path / "off")), monkeypatch, tr=tr)
    run_script(E4, stage4_args(d, str(tmp_path / "zero"), "--switch-bias", "eot=0,res=0,wi=0"), monkeypatch, tr=tr)
    run_script(E4, stage4_args(d, str(tmp_path / "on"), "--switch-bias", "eot=3,res=3,wi=3"), monkeypatch, tr=tr)
    off, zero, on = (_rows(str(tmp_path / k)) for k in ("off", "zero", "on"))
    assert len(off) > 4 and {r["tgt_opt"] for r in off} == {"English", "xl"} and not any("sb_flips" in r for r in off)
    assert "switch_bias" not in json.load(open(tmp_path / "off" / "config-rank0.json"))
    assert json.load(open(tmp_path / "on" / "config-rank0.json"))["switch_bias"]["wi"] == 3.0
    strip = lambda r: {k: v for k, v in r.items() if k not in ("sb_flips", "sb_points")}
    assert [strip(r) for r in zero] == off and all(sum(r["sb_flips"].values()) == 0 for r in zero)
    for a, b in zip(off, on):
        if a["tgt_opt"] != "xl": assert strip(b) == a and sum(b["sb_points"].values()) == 0
    assert sum(sum(r["sb_flips"].values()) for r in on if r["tgt_opt"] == "xl") > 0


def taxi_args(sess, out, *extra):
    return ["run", "--checkpoint", "ck", "--sessions", sess, "--out", out, "--deltas", "2,4", "--max-window", "5", "--batch-size", "2"] + list(extra)


def test_taxi_run_switch_bias_default_off_parity(tmp_path, monkeypatch):
    import json, cst_stage4_taxi as T
    sess = _sessions(str(tmp_path))
    run_script(T, taxi_args(sess, str(tmp_path / "off")), monkeypatch)
    run_script(T, taxi_args(sess, str(tmp_path / "zero"), "--switch-bias", "eot=0,res=0,wi=0"), monkeypatch)
    run_script(T, taxi_args(sess, str(tmp_path / "on"), "--switch-bias", "eot=8,res=8,wi=8"), monkeypatch)                # stub 의 다른 태그 logit 은 멀다 — 배선 확인용 큰 값
    names = sorted(os.listdir(tmp_path / "off")); assert names == ["asr-d2.jsonl", "asr-d4.jsonl", "config.json", "hyp-d2.jsonl", "hyp-d4.jsonl"]
    assert sum(1 for _ in open(tmp_path / "off" / "asr-d4.jsonl")) > 0
    for n in names[:2] + names[3:]: assert _read(tmp_path / "off" / n) == _read(tmp_path / "zero" / n)
    c0, cz, c1 = (json.load(open(tmp_path / k / "config.json")) for k in ("off", "zero", "on"))
    assert "switch_bias" not in c0 and cz.pop("switch_bias") == dict(eot=0.0, res=0.0, wi=0.0, rows="xl") and cz == c0
    assert c1["switch_bias"]["eot"] == 8.0 and not os.path.exists(tmp_path / "off" / "switch-bias.json")
    sb = json.load(open(tmp_path / "on" / "switch-bias.json")); assert sum(sb["d4"]["flips"].values()) > 0 and sb["policy"] == c1["switch_bias"]
    assert any(_read(tmp_path / "off" / n) != _read(tmp_path / "on" / n) for n in ("asr-d2.jsonl", "asr-d4.jsonl", "hyp-d2.jsonl", "hyp-d4.jsonl"))
