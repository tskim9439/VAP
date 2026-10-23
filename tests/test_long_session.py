"""회의 통째 평가(cst_stage3_eval --window full, 2026-10-24): 비트 병렬 편집거리(cst_eval._ed_bits) = DP, LongThinker(블록 행렬곱 어텐션·KV 늘리기) = HF Qwen3
forward, encode_long(겹침 조각) = 통째 인코딩(청크 경계·수용 영역을 지킬 때만), full 창 = 대화마다 하나. CPU."""
import argparse, random, types
import pytest, torch

from vapasr.data.cst_eval import _ed_bits, ed
from vapasr.hf.long_session import HOP, LongThinker, encode_long


def dp(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def test_ed_bits_matches_dp():
    rng = random.Random(0)
    for _ in range(2000):
        A = rng.choice([2, 3, 7, 30]); a = [f"w{rng.randrange(A)}" for _ in range(rng.randrange(0, 70))]; b = [f"w{rng.randrange(A)}" for _ in range(rng.randrange(0, 70))]
        assert _ed_bits(a, b) == dp(a, b) == _ed_bits(b, a), (a, b)
    a = [f"w{rng.randrange(50)}" for _ in range(600)]; b = [x if rng.random() < 0.7 else f"w{rng.randrange(50)}" for x in a][40:]
    assert len(a) * len(b) > 250_000 and ed(a, b) == dp(a, b)                                  # 긴 열은 비트 병렬 경로 — 값은 같다


def tiny_qwen3():
    tr = pytest.importorskip("transformers")
    cfg = tr.Qwen3Config(vocab_size=97, hidden_size=48, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=12,
                         rope_theta=1e6, max_position_embeddings=4096, attn_implementation="eager")
    torch.manual_seed(0); m = tr.Qwen3ForCausalLM(cfg).eval()
    return m, types.SimpleNamespace(model=m.model, lm_head=m.lm_head)


@pytest.mark.parametrize("blk", [5, 4096])
def test_long_thinker_matches_hf(blk):
    m, th = tiny_qwen3(); N = 61; torch.manual_seed(1); E = torch.randn(N, 48)
    with torch.no_grad(): ref = m.model(inputs_embeds=E[None]).last_hidden_state[0]           # (N, D) — HF 인과 forward(norm 뒤)
    lt = LongThinker(th, blocked_ids=[3, 5], max_len=8, blk=blk); outs, s, steps = [], 0, [7] + [1, 2, 1, 1, 2] * 20
    for q in steps:
        q = min(q, N - s)
        if q <= 0: break
        h = lt.hidden(E[s: s + q], s); outs.append(th.model.norm(h)[0]); s += q
    got = torch.cat(outs); assert got.shape == ref.shape and lt.used == N and lt.cap >= N    # max_len 8 에서 몇 번 늘었다
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
    lt2 = LongThinker(th, blocked_ids=[3, 5], max_len=N, blk=blk); lt2.hidden(E[:N - 2], 0); tid = lt2.run(E[N - 2:], N - 2)
    lg = m.lm_head(ref[-1]).detach(); lg[[3, 5]] = float("-inf"); assert tid == int(lg.argmax())   # 두 위치 입력의 마지막 = 차단 뒤 argmax


class FakeEnc:
    """인코더 흉내: 프레임 t = 입력 프레임 [t − 10, 청크 끝] 에너지 합(청크 = 4 프레임, 번호는 입력 첫 프레임부터). 두 스트림(S = 2). 정수 샘플이라 합이 정확하다."""
    config = types.SimpleNamespace(encoder_right_context=3)
    def parameters(self): yield torch.zeros(1)
    def encode(self, w, wl, K):
        n, k = int(wl[0]), int(K[0]); x = w[0, :n].double(); nf = -(-n // HOP)
        e = torch.nn.functional.pad(x, (0, nf * HOP - n)).view(nf, HOP).sum(1); out = torch.zeros(1, 2, k, 3, dtype=torch.float64)
        for t in range(min(k, nf)):
            v = e[max(0, t - 10): min(nf, t // 4 * 4 + 4)].sum(); out[0, 0, t] = v; out[0, 1, t] = -v
        return out


def test_encode_long_matches_whole_only_when_aligned():
    torch.manual_seed(0); K = 53; wav = torch.randint(-3, 4, (K * HOP - 300,)).float().numpy(); f = FakeEnc()
    whole = f.encode(torch.from_numpy(wav)[None], torch.tensor([len(wav)]), torch.tensor([K]))[0]
    for core, ctx in ((8, 12), (16, 12), (4, 16), (60, 12)):
        assert torch.equal(encode_long(f, wav, K, core=core, ctx=ctx, tail=4), whole), (core, ctx)
    assert not torch.equal(encode_long(f, wav, K, core=8, ctx=4, tail=4), whole)              # 겹침 < 수용 영역(10) 이면 다르다
    with pytest.raises(AssertionError): encode_long(f, wav, K, core=6, ctx=12, tail=4)       # 청크 경계가 아닌 조각 시작은 거부


def test_full_window_one_per_dialogue(tmp_path):
    sf = pytest.importorskip("soundfile")  # noqa: F841 — make_multi 가 wav 를 쓴다
    from test_v2_slots import make_multi
    from test_dialogue_dataset import FakeTok
    from experiments.cst_stage3_eval import windows_of, win_conf, win_kw
    q, d = make_multi(tmp_path, n=4, dur=72.0)
    ns = argparse.Namespace(corpora="mtest", data=str(tmp_path), split=None, deltas=[4], limit_windows=0, backchannel=None, max_speakers=4, window="full", hop=10.0,
                            window_draws=0, declare=4)
    sets, items = windows_of(ns, FakeTok()); ds = sets["mtest"]
    assert [ds.items[i] for _, i in items] == [("m1", 0.0, 72.0)] and len(ds.slots(items[0][1])[1]) == 4
    s = ds.cst_sequence(items[0][1], 4); assert ds.problems_of(s) == [] and s["K"] == 900
    assert win_conf(ns) == {"max_speakers": 4, "declare": 4, "window": "full"} and "window_s" not in win_kw(ns)
    ns2 = argparse.Namespace(**{**vars(ns), "max_speakers": 2, "declare": None})                 # 2 슬롯: 네 명이 말하는 회의는 빠진다
    assert windows_of(ns2, FakeTok())[1] == []


def _stream(lt, E, chunks):
    """prefix(6) 뒤 청크마다 on_chunk → 오디오 1 위치 → 말 t 위치(모두 q = 1, 두 위치 입력은 청크 3 의 말에 한 번). 위치마다 norm 뒤 hidden."""
    outs, s = [lt.hidden(E[:6], 0)[0]], 6
    for k, t in enumerate(chunks):
        lt.on_chunk(k, s); outs.append(lt.hidden(E[s:s + 1], s)[0]); s += 1
        q = 2 if k == 3 and t >= 2 else 1
        for _ in range(0, t, q): outs.append(lt.hidden(E[s:s + q], s)[0]); s += q
    return torch.cat(outs), s


def test_long_thinker_window_before_eviction_equals_full():
    _, th = tiny_qwen3(); torch.manual_seed(2); E = torch.randn(120, 48); ch = [0, 2, 1, 4, 0, 3, 1, 0, 2, 5, 1, 0]
    a, n = _stream(LongThinker(th, [3], max_len=8, blk=5), E, ch); b, _ = _stream(LongThinker(th, [3], max_len=8, blk=5, context=100), E, ch)
    assert n < 120 and torch.allclose(a, b, atol=1e-5), (a - b).abs().max()                  # 창보다 짧은 동안은 문맥 전부와 같다


def test_long_thinker_window_compaction_and_prefix_rotation():
    _, th = tiny_qwen3(); torch.manual_seed(3); E = torch.randn(200, 48); ch = [1, 0, 3, 2, 0, 4, 1, 1, 0, 2, 3, 0, 1, 2, 0, 5, 1, 0, 2, 1]
    big = LongThinker(th, [3], max_len=4096, blk=5, context=4); a, n = _stream(big, E, ch)
    small = LongThinker(th, [3], max_len=8, blk=5, context=4); b, _ = _stream(small, E, ch)     # 작은 버퍼: 창을 prefix 뒤로 여러 번 당긴다
    assert big.lo > big.P and small.base > small.P and torch.allclose(a, b, atol=1e-6), (a - b).abs().max()
    full, _ = _stream(LongThinker(th, [3], max_len=8, blk=5), E, ch); assert not torch.allclose(full[-5:], a[-5:], atol=1e-3)   # 내보낸 뒤엔 다르다
    lt = small; x = torch.randn(2, 5, 12); c1, s1 = lt._rope(7, 8); c2, s2 = lt._rope(30, 31); c3, s3 = lt._rope(37, 38)
    r1 = x * c1 + _rh(x) * s1; assert torch.allclose(r1 * c2 + _rh(r1) * s2, x * c3 + _rh(x) * s3, atol=1e-5)   # 위치 7 회전 뒤 30 더 = 위치 37


def _rh(x):
    from vapasr.hf.realtime import _rotate_half
    return _rotate_half(x)


def test_attend_with_prefix_matches_naive():
    _, th = tiny_qwen3(); lt = LongThinker(th, [3], max_len=8, blk=4); torch.manual_seed(4)
    H, KV, HD = lt.H, lt.KV, lt.HD; q, P, W = 2, 5, 11; qs = torch.randn(H, q, HD); kp, vp = torch.randn(KV, P, HD), torch.randn(KV, P, HD)
    kw, vw = torch.randn(KV, W, HD), torch.randn(KV, W, HD); i0 = W - q; sc = HD ** -0.5
    got = lt._attend(qs, kw, vw, i0, sc, kp, vp)
    K = torch.cat([kp, kw], 1).repeat_interleave(H // KV, 0); V = torch.cat([vp, vw], 1).repeat_interleave(H // KV, 0)
    s = qs @ K.transpose(-1, -2) * sc; s[:, 0, P + i0 + 1:] = float("-inf")                      # 쿼리 0 은 자기 다음(쿼리 1 의 키)을 못 본다
    assert torch.allclose(got, torch.softmax(s, -1) @ V, atol=1e-6)
