"""E9 슬롯 블록 형식(2026-10-28): 데이터(block_streams·build_block_sequence·block_problems·슬롯별 마스크·선언 수 범위), 모델(add_slot_blocks·태그 활동·build·packing·forward),
디코더(BlockState·추측 prefill realtime_block_decode 가 한 토큰씩 디코드와 같다), 평가(parse_emitted <BEGIN>·emitted_from_blocks 오라클 왕복·activity_from_events). CPU."""
import importlib.util, pathlib, random
import numpy as np, pytest, torch
sf = pytest.importorskip("soundfile")
from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef
from vapasr.data.cst_dialogue import CstDialogueDataset, collate_cst, slot_activity, TAG_STRIDE
from vapasr.data.cst_eval import parse_emitted, emitted_from_blocks, emitted_from_sequence, activity_from_events
from vapasr.hf.batch_decode import BlockState, block_stops

_HERE = pathlib.Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", _HERE / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _tok():
    t = _load("test_dialogue_dataset").FakeTok(); t.add_tokens(["<DELAY_9>", "<DELAY_10>"]); return t


def _multi(tmp_path, n=3, dur=40.0):
    return _load("test_v2_slots").make_multi(tmp_path, n=n, dur=dur)


def _masked_dialogue(tmp_path):
    """a: 1–3 s(전사), 7–9 s(전사 없음 — quarantine), b: 4–6 s, 8–9.5 s(a 의 전사 없는 구간과 겹침)."""
    SR = 16000; dur = 14.0; p = str(tmp_path / "x.wav"); t = np.arange(int(dur * SR)) / SR
    sf.write(p, np.stack([0.3 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 660 * t)], 1).astype(np.float32), SR)
    u = [Utterance("a", 1.0, 3.0, "ab", "ab", tokens=[(11, 2.0), (12, 3.0)], utt_id="a1"), Utterance("a", 7.0, 9.0, "", "", tokens=None, utt_id="a2", flags=["quarantine"]),
         Utterance("b", 4.0, 6.0, "cd", "cd", tokens=[(13, 5.0), (14, 6.0)], utt_id="b1"), Utterance("b", 8.0, 9.5, "ef", "ef", tokens=[(15, 9.0), (16, 9.5)], utt_id="b2")]
    d = Dialogue(conv_id="c", corpus="t", lang="English", split="train", duration_s=dur, speakers=["a", "b"], utterances=u,
                 channels={"a": ChannelRef(path=p + "#ch0"), "b": ChannelRef(path=p + "#ch1")})
    q = str(tmp_path / "t.refined.dialogues.jsonl"); open(q, "w").write(d.to_json() + "\n"); return q


def test_layout_events_and_stop_labels(tmp_path):
    q, d = _multi(tmp_path)
    ds = CstDialogueDataset([q], _tok(), delays=(4, 10), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, slot_blocks=True, begin_weight=2.0, declare=4)
    assert len(ds) > 0
    NA = ds.csp.next_audio
    for i in range(len(ds)):
        cid, t0, L = ds.items[i]
        for dl in (4, 10):
            s = ds.cst_sequence(i, dl); assert s["blocks"] and ds.problems_of(s) == [], (ds.items[i], dl, ds.problems_of(s))
            K, D = s["K"], len(s["langs"]); assert D == 4 and s["D"] == 4
            assert sum(1 for inp, y in zip(s["is_input"], s["labels"]) if inp and y == NA) == (K + s["n_rounds"]) * D     # (라운드, 슬롯)마다 멈춤 하나(가림 없음)
            assert {p for p, w in enumerate(s["pos_weight"]) if w > 0} == set(s["begin_pos"]) and all(s["pos_weight"][p] == 2.0 for p in s["begin_pos"])
            eps = [e for e in ds.eps[cid] if e.end > t0 and e.start < t0 + L]
            starts = sorted(min(K, int((e.start - t0) / 0.08) + 4) for e in eps if e.start >= t0)
            ends = sorted(min(K, int((e.end - t0) / 0.08) + dl) for e in eps if e.end <= t0 + L)
            assert sorted(s["chunk_at"][p] for p in s["begin_pos"]) == starts and sorted(s["chunk_at"][p] for p in s["eot_pos"]) == ends
            assert all(s["tag_chunk"][p] >= 0 for p in range(len(s["ids"])) if s["ids"][p] in ds.spk_ids and s["is_input"][p] and p >= len(ds.cst_prefix(s["langs"], dl)))
            assert not any(s["ids"][p] == NA for p in range(len(s["ids"])))                    # <NEXT_AUDIO> 는 ids 에 없다(label 로만)
    with pytest.raises(ValueError): CstDialogueDataset([q], _tok(), delays=(2, 4), window_s=(20.0, 20.0), hop_s=5.0, max_speakers=4, slot_blocks=True)   # δ < <BEGIN> 지연
    with pytest.raises(ValueError): CstDialogueDataset([q], _tok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, max_speakers=4, slot_blocks=True, no_eot=True)


def test_slot_shuffle_training_only(tmp_path):
    """2026-10-30 활동 추종 학습: slot_shuffle — 학습 rng 에서 그 확률로 선언 슬롯 안 화자 자리를 무작위로(빈 자리는 가짜 화자). 이벤트열·활동 타깃·흔든 활동 입력이 같은 순서를
    따르고, 평가(rng 없음)와 0 이면 예전 그대로(난수도 더 뽑지 않는다)."""
    from vapasr.data.act_noise import ActNoise
    from vapasr.data.cst_dialogue import EMPTY_SLOT
    q, d = _multi(tmp_path)
    kw = dict(delays=(4, 10), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, slot_blocks=True, begin_weight=2.0, declare=4, spk_activity=True)
    base = CstDialogueDataset([q], _tok(), **kw); sh = CstDialogueDataset([q], _tok(), slot_shuffle=1.0, act_input=ActNoise(clean=1.0), **kw)
    real = lambda sp: [w for w in sp if not w.startswith(EMPTY_SLOT)]
    orders = set()
    for i in range(len(sh)):
        spk0 = base.slots(i, random.Random(3))[0]; assert sh.slots(i) == base.slots(i)                     # 평가(rng 없음)는 그대로
        for seed in range(6):
            spk1, langs1 = sh.slots(i, random.Random(seed)); orders.add(tuple(spk1))
            assert len(spk1) == len(langs1) == 4 and sorted(real(spk1)) == sorted(spk0) and len(set(spk1)) == 4
            s0, s1 = base.cst_sequence(i, 10, rng=random.Random(3)), sh.cst_sequence(i, 10, rng=random.Random(seed))
            assert s1["speakers"] == spk1 and sh.problems_of(s1) == []
            for w in spk0: assert s1["streams"][spk1.index(w)] == s0["streams"][spk0.index(w)]            # 같은 화자의 이벤트열이 자리만 바뀐다
            assert all(s1["streams"][j] == [] for j, w in enumerate(spk1) if w.startswith(EMPTY_SLOT))
    assert len(orders) > len(sh)                                                                          # 창마다 여러 순서가 나온다
    random.seed(7); a0 = base[0]["activity"]; random.seed(7); it = sh[0]; a1 = it["activity"]
    assert sorted(map(tuple, a1.T.tolist())) == sorted(map(tuple, a0.T.tolist())) and torch.equal(it["act_in"], a1)   # 활동 열이 같은 순서로 섞이고 흔든 입력도 같다
    assert it["n_spk"] == base[0]["n_spk"] == len(base.slots(0)[0])                                      # 가짜 화자는 화자 수에 안 센다
    z = CstDialogueDataset([q], _tok(), slot_shuffle=0.0, **kw); r1, r2 = random.Random(5), random.Random(5)
    assert z.slots(0, r1) == base.slots(0, r2) and r1.random() == r2.random()                            # 0 이면 난수를 더 뽑지 않는다
    with pytest.raises(ValueError): CstDialogueDataset([q], _tok(), slot_shuffle=1.5, **kw)


def test_slot_remap_training_only(tmp_path):
    """2026-10-30 활동 추종 학습: slot_remap — 학습 rng 에서 그 확률로 창 안 무음 틈 하나에서 슬롯 대응을 바꾼다. 틈 앞 episode 는 원래 슬롯, 뒤는 새 슬롯
    (이벤트열·활동 타깃·흔든 입력), 표본에 remap_k·remap_perm 이 붙고 collate 가 묶는다. 평가(rng 없음)와 0 이면 예전 그대로."""
    from vapasr.data.act_noise import ActNoise
    from vapasr.data.cst_dialogue import remap_perm
    q, d = _multi(tmp_path)
    kw = dict(delays=(4, 10), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, slot_blocks=True, begin_weight=2.0, declare=4, spk_activity=True)
    base = CstDialogueDataset([q], _tok(), **kw); rm = CstDialogueDataset([q], _tok(), slot_remap=1.0, act_input=ActNoise(clean=1.0), **kw)
    n_rm, hit = 0, None
    for i in range(len(rm)):
        cid, t0, L = rm.items[i]; K = int(round(L / 0.08))
        assert rm.cst_sequence(i, 10)["ids"] == base.cst_sequence(i, 10)["ids"]                          # 평가(rng 없음)는 그대로
        for seed in range(4):
            s = rm.cst_sequence(i, 10, rng=random.Random(seed)); assert rm.problems_of(s) == [], rm.problems_of(s)
            if "remap" not in s: continue
            n_rm += 1; tr, sp2 = s["remap"]; spk = s["speakers"]
            assert sorted(sp2) == sorted(spk) and sp2 != spk and t0 + 2.0 <= tr <= t0 + L - 2.0
            assert not any(e.start < tr < e.end for e in rm.eps[cid])                                     # 틈에 걸친 말이 없다
            for j, st_ in enumerate(s["streams"]):                                                       # 시험 대화의 전사 토큰 11+i·21+i = 화자 p_i
                for el in st_:
                    if 11 <= el[0] < 31: assert f"p{(el[0] - 11) % 10}" == (spk[j] if t0 + el[1] < tr else sp2[j])
            A = np.asarray(rm.slot_act_of(s, 4))
            for e in rm.eps[cid]:
                if t0 + 0.2 < e.start and e.end < t0 + L - 0.2:
                    f = int(((e.start + e.end) / 2 - t0) / 0.08); col = (spk if e.start < tr else sp2).index(e.speaker)
                    assert A[f, col] == 1 and A[f].sum() == 1
            hit = hit or (i, seed)
    assert n_rm > 0
    for i in range(len(rm)):                                                                             # __getitem__ 이 고른 바꾸기와 같은 값이 붙는지(같은 난수로 다시 만든다)
        random.seed(100 + i); it = rm[i]; random.seed(100 + i); s = rm.cst_sequence(i, random.choice(rm.delays), rng=random)
        if "remap" not in s: assert "remap_k" not in it; continue
        assert it["remap_k"] == int(round((s["remap"][0] - s["t0"]) / 0.08)) and it["remap_perm"].tolist() == remap_perm(s["speakers"], s["remap"][1], 4)
        assert torch.equal(it["act_in"], it["activity"]) and torch.equal(it["activity"], torch.tensor(rm.slot_act_of(s, 4), dtype=torch.float))
        random.seed(3); plain = CstDialogueDataset([q], _tok(), act_input=ActNoise(clean=1.0), **kw)[0]; out = collate_cst([it, plain])
        assert out["remap_k"].tolist() == [it["remap_k"], -1] and out["remap_perm"][1].tolist() == [0, 1, 2, 3] and torch.equal(out["remap_perm"][0], it["remap_perm"])
        assert out["n_remap"] == 1                                                                       # 학습 로그 remap_per_dlg_step
        break
    else: pytest.fail("__getitem__ 에서 바꾼 창이 하나도 없다")
    z = CstDialogueDataset([q], _tok(), slot_remap=0.0, **kw); r1, r2 = random.Random(5), random.Random(5)
    assert z.cst_sequence(0, 10, rng=r1)["ids"] == base.cst_sequence(0, 10, rng=r2)["ids"] and r1.random() == r2.random()   # 0 이면 난수를 더 뽑지 않는다
    with pytest.raises(ValueError): CstDialogueDataset([q], _tok(), slot_remap=1.5, **kw)
    with pytest.raises(ValueError): CstDialogueDataset([q], _tok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, max_speakers=4, slot_remap=0.5)   # 슬롯 블록 없이


def test_begin_not_after_first_word_at_float_boundary(tmp_path):
    """2026-10-28 서버 dry-run 회귀: 창 시작을 뺀 시각이 경계 바로 아래(100.24 − 100 = 0.2399…, /0.08 = 2.999…)이고 첫 글자 시각 = 발화 시작이면
    chunk_of(+1e-9)와 글자 버킷 int() 가 한 청크 어긋나 δ4 에서 <BEGIN> 이 첫 글자 뒤로 갔다 — k_min 을 버킷과 같은 식으로."""
    SR = 16000; dur = 130.0; p = str(tmp_path / "f.wav"); t = np.arange(int(dur * SR)) / SR
    sf.write(p, np.stack([0.3 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 660 * t)], 1).astype(np.float32), SR)
    u = [Utterance("a", 100.24, 101.04, "ab", "ab", tokens=[(11, 100.24), (12, 101.04)], utt_id="a1"),          # 첫 글자 시각 = 발화 시작
         Utterance("b", 102.16, 103.2, "cd", "cd", tokens=[(13, 102.16), (14, 103.2)], utt_id="b1"), Utterance("a", 104.08, 105.0, "ef", "ef", tokens=[(15, 104.08), (16, 105.0)], utt_id="a2")]
    d = Dialogue(conv_id="f", corpus="t", lang="English", split="train", duration_s=dur, speakers=["a", "b"], utterances=u,
                 channels={"a": ChannelRef(path=p + "#ch0"), "b": ChannelRef(path=p + "#ch1")})
    q = str(tmp_path / "t.refined.dialogues.jsonl"); open(q, "w").write(d.to_json() + "\n")
    ds = CstDialogueDataset([q], _tok(), delays=(4, 5), windows=[("f", 100.0, 8.0)], max_speakers=2, slot_blocks=True)
    assert (100.24 - 100.0) / 0.08 < 3 and int((100.24 - 100.0) / 0.08 + 1e-9) == 3                     # 경계 바로 아래(시험 전제)
    for dl in (4, 5):
        s = ds.cst_sequence(0, dl); assert ds.problems_of(s) == [], (dl, ds.problems_of(s))
        b = [s["chunk_at"][p_] for p_ in s["begin_pos"]]; w = {s["ids"][p_]: s["chunk_at"][p_] for p_ in range(len(s["ids"])) if not s["is_input"][p_] and s["ids"][p_] in (11, 13, 15)}
        assert b[0] <= w[11] and b[0] == 2 + 4                                                            # <BEGIN> 청크 = int(2.999…) + 4


def test_untranscribed_masks_only_that_slot(tmp_path):
    q = _masked_dialogue(tmp_path)
    kw = dict(delays=(4,), windows=[("c", 0.0, 12.0)], max_speakers=2)
    e9 = CstDialogueDataset([q], _tok(), slot_blocks=True, **kw); e6 = CstDialogueDataset([q], _tok(), **kw)
    s = e9.cst_sequence(0, 4); assert e9.problems_of(s) == [] and s["K"] == 150
    NA = e9.csp.next_audio; ids, lab, inp, tk, sa, ca = s["ids"], s["labels"], s["is_input"], s["tag_chunk"], s["slot_at"], s["chunk_at"]
    hidden_a = set(range(87, 121))                                                              # a 의 7–9 s: ⌊7/0.08⌋ … ⌈(9 + 0.4 + 0.24)/0.08⌉ − 1
    stops, cur = {}, None                                                                       # (슬롯, 청크) → 그 멈춤 label(다음 입력 위치)
    for p in range(len(ids)):
        if not inp[p] or (tk[p] < 0 and s["chunk_of"][p] < 0 and ids[p] != e9.csp.empty_audio): continue   # 출력·prefix
        if cur is not None: stops[cur] = lab[p]
        cur = (sa[p], tk[p]) if tk[p] >= 0 else None
    assert all(stops[(0, k)] == -100 for k in hidden_a) and all(stops[(0, k)] == NA for k in range(150) if k not in hidden_a)
    assert all(stops[(1, k)] == NA for k in range(150))                                          # b 는 가리지 않는다
    w = {ids[p]: lab[p] for p in range(len(ids)) if not inp[p] and ids[p] in (15, 16)}
    assert w == {15: 15, 16: 16}                                                                # b 의 겹친 말은 학습(E6 는 청크째 가린다)
    s6 = e6.cst_sequence(0, 4); assert {s6["ids"][p]: s6["labels"][p] for p in range(len(s6["ids"])) if s6["ids"][p] in (15, 16) and not s6["is_input"][p]} == {15: -100, 16: 16}
    ev = [(ids[p], lab[p], ca[p], sa[p]) for p in range(len(ids)) if not inp[p] and ids[p] in (e9.begin_id, e9.eot_id)]
    assert (e9.begin_id, e9.begin_id, 87 + 4, 0) in ev and (e9.eot_id, e9.eot_id, 112 + 4, 0) in ev   # 전사 없는 episode 의 <BEGIN>·<EOT> 는 학습
    assert len([x for x in ev if x[0] == e9.begin_id]) == 4 and len([x for x in ev if x[0] == e9.eot_id]) == 4


def test_oracle_roundtrip_words_match_e6_and_events_give_activity(tmp_path):
    q, d = _multi(tmp_path, n=4)
    kw = dict(delays=(4, 10), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, declare=4)
    e6 = CstDialogueDataset([q], _tok(), **kw); assert len(e6) > 0
    spk = e6.spk_ids; EOT = e6.eot_id; B = e6.begin_id
    for i in range(len(e6)):
        cid, t0, L = e6.items[i]; K = int(round(L / 0.08))
        for dl in (4, 10):
            q6 = e6.cst_sequence(i, dl); h6, _ = parse_emitted(emitted_from_sequence(q6["ids"], q6["is_input"], q6["chunk_of"], e6.csp.next_audio, e6.csp.empty_audio), spk, EOT)
            q9 = e6.block_sequence(i, dl); assert e6.problems_of(q9) == []                       # 끈 데이터셋에서도 부를 수 있다(평가 오라클)
            em = emitted_from_blocks(q9["ids"], q9["is_input"], q9["tag_chunk"], q9["slot_at"], spk); ev = []
            h9, _ = parse_emitted(em, spk, EOT, begin_id=B, events=ev)
            assert h9 == h6, (i, dl)                                                            # 글자·시각·슬롯이 E6 정답과 같다
            ha = activity_from_events(ev, K, 4, dl); ra = np.array(slot_activity(e6.eps[cid], q9["speakers"], t0, L, K, 4))
            assert (ha != ra).sum() <= 2 * len(e6.eps[cid]), (i, dl)                             # 구간 끝 반올림(프레임 하나)만 다르다


def test_block_state_reproduces_training_sequence(tmp_path):
    q, d = _multi(tmp_path, n=3)
    ds = CstDialogueDataset([q], _tok(), delays=(4, 10), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, slot_blocks=True, declare=4)
    stop = set(ds.spk_ids) | {ds.csp.next_audio, ds.csp.empty_audio}
    for i in range(len(ds)):
        for dl in (4, 10):
            s = ds.cst_sequence(i, dl); ids, inp = s["ids"], s["is_input"]; K, D = s["K"], len(s["langs"])
            st = BlockState(K, D, 100, 3, 10 ** 6, tuple(ds.spk_ids[:D])); p = next(j for j, c in enumerate(s["chunk_of"]) if c >= 0)
            while True:
                pred = ids[p + 1] if not inp[p + 1] else ds.csp.next_audio                        # 정답 다음 출력 또는 멈춤
                kind, v = st.consume(pred, stop)
                if kind == "pad": break
                p += 1
                if kind == "audio": assert inp[p] and s["chunk_of"][p] == v
                elif kind == "tag": assert inp[p] and ids[p] == ds.spk_ids[v[1]] and s["tag_chunk"][p] == v[0]
                elif kind == "text": assert not inp[p] and ids[p] == v
                else: assert kind == "empty" and inp[p] and ids[p] == ds.csp.empty_audio
            assert p == len(ids) - 2 and st.forced == 0                                         # 마지막 멈춤 label 을 싣는 <EMPTY_AUDIO> 는 넣지 않는다
            assert st.emitted == emitted_from_blocks(ids, inp, s["tag_chunk"], s["slot_at"], ds.spk_ids)


def _tiny_blocks(lags=6):
    t = _load("test_semcommit_tokens_loss"); m, tok = t.tiny_model(); m.add_semantic_tokens(tok, turn=True); m.enable_slots(tok, 4); m.add_slot_input(4)
    W = m.get_input_embeddings().weight
    with torch.no_grad(): W[151721] -= 3.0                                                       # 예약 행이 밀려난 상황
    bid = m.add_slot_blocks(tok, lags); return m, tok, bid


def test_add_slot_blocks_reinit_unblock_idempotent():
    m, tok, bid = _tiny_blocks(); c = m.config; W = m.get_input_embeddings().weight
    assert bid == 151721 and c.slot_blocks and c.begin_token == "<ONSET>" and c.tag_act_lags == 6 and bid not in c.blocked_ids and m.blocked.tolist() == c.blocked_ids
    assert "<ONSET>" not in (c.sem_reserved or []) and (W.detach()[bid] - W.detach()[:151705].mean(0)).abs().max() < 0.2
    assert m.adapter_tagact.in_features == 6 and not m.adapter_tagact.weight.any() and "adapter_tagact" in m.trainable_state()
    W1 = W.detach().clone(); assert m.add_slot_blocks(tok, 6) == bid and torch.equal(W.detach(), W1)
    with pytest.raises(AssertionError): m.add_slot_blocks(tok, 8)


def test_begin_unblocked_when_init_already_has_slot_blocks():
    """2026-10-30 E10 사고: 슬롯 블록 init(E9 final)에서 stage3_train 처럼 add_semantic_tokens(turn=True)를 다시 부르면 <ONSET> 이 다시 막힌다 —
    add_slot_blocks(이미 있음)가 풀어야 하고, 막힌 채 저장된 config 로 모델을 만들어도 풀려 있어야 한다."""
    from vapasr.hf import VapAsrForStreamingASR
    m, tok, bid = _tiny_blocks(); t = _load("test_semcommit_tokens_loss")
    m.add_semantic_tokens(tok, turn=True); assert bid in m.config.blocked_ids                     # stage3_train 의 순서 그대로 — 다시 막힌다
    m.enable_slots(tok, 4); m.add_slot_input(4); assert m.add_slot_blocks(tok, 6) == bid
    assert bid not in m.config.blocked_ids and bid not in m.blocked.tolist() and m.blocked.tolist() == m.config.blocked_ids
    m.config.blocked_ids = sorted(set(m.config.blocked_ids) | {bid})                                 # 수정 전에 저장된 체크포인트(E10 final)
    m2 = VapAsrForStreamingASR(m.config, thinker=t._Thinker()); assert bid not in m2.blocked.tolist() and bid not in m2.config.blocked_ids


def _batch(m, K=(5, 3)):
    """두 표본: [prefix 2][오디오 0][A][B][C][D][글자][오디오 1][A][B][C][D] … — 태그 위치 tag_chunk = 청크."""
    tags = m.slot_ids(4); rows, iss, cos, tcs = [], [], [], []
    for Kb in K:
        ids, ia, co, tc = [1, 2], [0, 0], [-1, -1], [-1, -1]
        for k in range(Kb):
            ids += [0] + tags + [50]; ia += [1, 0, 0, 0, 0, 0]; co += [k, -1, -1, -1, -1, -1]; tc += [-1, k, k, k, k, -1]
        rows.append(ids); iss.append(ia); cos.append(co); tcs.append(tc)
    L = max(len(r) for r in rows); pad = lambda xs, v: torch.tensor([x + [v] * (L - len(x)) for x in xs])
    b = dict(ids=pad(rows, 0), is_audio=pad(iss, 0).bool(), chunk_of=pad(cos, -1), tag_chunk=pad(tcs, -1), mask=pad([[1] * len(r) for r in rows], 0), K=torch.tensor(K))
    b["labels"] = torch.where(b["is_audio"] | (b["tag_chunk"] >= 0), torch.full_like(b["ids"], m.next_audio), b["ids"]); b["labels"][:, :3] = -100; b["labels"][b["mask"] == 0] = -100
    return b


def test_build_tag_activity_zero_init_values_and_packing():
    from vapasr.hf.packing import pack_batch
    m, tok, bid = _tiny_blocks(); m.eval(); b = _batch(m); Km = 5; torch.manual_seed(3)
    feats = torch.randn(2, 1, Km, 4); act = (torch.rand(2, Km, 4) > 0.5).float()
    E0 = m.build(feats, b["ids"], b["is_audio"], b["chunk_of"], act_in=act)
    assert torch.equal(m.build(feats, b["ids"], b["is_audio"], b["chunk_of"], act_in=act, tag_chunk=b["tag_chunk"]), E0)   # 0 초기화 → 그대로
    with torch.no_grad(): m.adapter_tagact.weight.copy_(torch.randn_like(m.adapter_tagact.weight))
    E1 = m.build(feats, b["ids"], b["is_audio"], b["chunk_of"], act_in=act, tag_chunk=b["tag_chunk"]); Wt = m.adapter_tagact.weight
    for r in range(2):
        for p in range(b["ids"].shape[1]):
            k = int(b["tag_chunk"][r, p])
            if k < 0: assert torch.equal(E1[r, p], E0[r, p]); continue
            s_ = m.slot_ids(4).index(int(b["ids"][r, p])); lag = torch.tensor([float(act[r, k - j, s_]) if k - j >= 0 else 0.0 for j in range(6)])
            assert torch.allclose(E1[r, p], E0[r, p] + Wt @ lag, atol=1e-5)
    pk = pack_batch(dict(b)); n = b["mask"].sum(1).tolist()
    assert int(pk["tag_chunk"].max()) >= TAG_STRIDE and pk["ids"].shape == (1, sum(n))
    Ep = m.build(feats, pk["ids"], pk["is_audio"], pk["chunk_of"], packed=True, act_in=act, tag_chunk=pk["tag_chunk"])
    assert torch.allclose(Ep[0], torch.cat([E1[r, : n[r]] for r in range(2)]), atol=1e-6)


def test_forward_trains_tag_adapter_and_ties_without_tags():
    m, tok, bid = _tiny_blocks(); b = _batch(m); torch.manual_seed(4); feats = torch.randn(2, 1, 5, 4); act = (torch.rand(2, 5, 4) > 0.5).float()
    x = {k: b[k] for k in ("ids", "is_audio", "chunk_of", "labels", "mask")}; m.train()
    out = m(**x, feats=feats, act_in=act, tag_chunk=b["tag_chunk"]); out.loss.backward()
    assert torch.isfinite(out.loss) and m.adapter_tagact.weight.grad is not None and m.adapter_tagact.weight.grad.abs().sum() > 0
    m.zero_grad(); out = m(**x, feats=feats, act_in=act); out.loss.backward()                   # 태그 없는 행(replay): 0 가중으로 묶인다
    assert m.adapter_tagact.weight.grad is not None and not m.adapter_tagact.weight.grad.any()


class _FakeLM:
    """GraphThinker·LongThinker 규약(run·run_all)의 가짜 LM: 위치마다 그 위치까지 넣은 입력(되돌려 덮어쓴 자리는 새 입력)의 해시로 다음 토큰을 고른다 — 인과 LM 과 같은 의존."""
    def __init__(self, embed, pick): self.embed, self.pick, self.at = embed, pick, {}
    def _feed(self, x, start):
        out = []
        for i in range(x.shape[0]):
            self.at[start + i] = x[i].detach().numpy().tobytes(); out.append(self.pick(hash(tuple(self.at[p] for p in range(start + i + 1)))))
        return out
    def run(self, x, start): return self._feed(x, start)[-1]
    def run_all(self, x, start): return self._feed(x, start)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_speculative_decode_equals_token_by_token(seed):
    from vapasr.hf.realtime import realtime_block_decode
    m, tok, bid = _tiny_blocks(); m.eval(); torch.manual_seed(seed); K, D = 30, 4
    with torch.no_grad(): m.adapter_tagact.weight.copy_(torch.randn_like(m.adapter_tagact.weight))
    feats = torch.randn(1, K, 4); act = (torch.rand(K, 4) > 0.6).float(); prefix = [5, 6, 7]; emb = m.get_input_embeddings().weight
    words = [50, 60, 70, bid, m.sp_ids["<EOT>"]]; NA = m.next_audio
    pick = lambda h: NA if (h % 10) < 6 + seed else words[(h // 10) % len(words)]
    m.config.runaway_cap = 3
    lm = _FakeLM(emb, pick); out, forced, ticks, rounds, _, nf = realtime_block_decode(m, lm, feats, prefix, D, act=act, max_flush_rounds=4)
    # 한 토큰씩: BlockState 로 같은 가짜 LM 을 돈다
    lm2 = _FakeLM(emb, pick); lm2.run_all(emb[torch.tensor(prefix)], 0); pos = len(prefix); ce = m.chunk_embed(feats[None])[0]
    st = BlockState(K, D, m.config.runaway_cap, 4, int(6 * K) + 64, tuple(m.slot_ids(D))); stop = block_stops(m, D); x = ce[0]
    while True:
        tid = lm2.run(x[None], pos); pos += 1; kind, v = st.consume(tid, stop)
        if kind == "pad": break
        x = ce[v] if kind == "audio" else m.tag_embeds(v[0], act, D)[v[1]] if kind == "tag" else emb[v] if kind == "text" else emb[m.empty_audio]
    assert out == st.emitted and forced == st.forced and rounds == st.flush_rounds and len(ticks) == K
    assert len(out) > 0 and nf < pos                                                             # 말이 있고, 추측 prefill 이 forward 를 줄였다


def test_activity_from_events_rules():
    ev = [("B", 0, 10), ("W", 0, 18), ("E", 0, 30), ("W", 1, 25), ("E", 1, 40), ("B", 2, 50), ("B", 2, 70), ("E", 3, 12)]
    a = activity_from_events(ev, 100, 4, delay=8)
    assert a[:, 0].nonzero()[0].tolist() == list(range(6, 23))                                    # <BEGIN> 10 − 4 … <EOT> 30 − 8
    assert a[:, 1].nonzero()[0].tolist() == list(range(17, 33))                                   # <BEGIN> 없이 <EOT>: 첫 글자 25 − 8 부터
    assert a[:, 2].nonzero()[0].tolist() == list(range(46, 100))                                  # 열린 채 다시 <BEGIN>: 이어서, 끝까지 열림
    assert a[:, 3].nonzero()[0].tolist() == list(range(0, 5))                                     # 앞 사건·글자 없는 <EOT>(창 앞에서 시작): 창 시작부터
    em = [(3, 151707), (3, 151721), (3, 50), (4, 151722), (5, 151708), (5, 60)]
    ev2 = []; hs, he = parse_emitted(em, (151707, 151708), 151722, begin_id=151721, events=ev2)
    assert hs == [[(0.32, 50)], [(0.48, 60)]] and he == [(0, 0.4)] and ev2 == [("B", 0, 3), ("W", 0, 3), ("E", 0, 4), ("W", 1, 5)]


def test_declare_range_and_items_collate(tmp_path):
    q, d = _multi(tmp_path, n=3, dur=40.0)
    ds = CstDialogueDataset([q], _tok(), delays=(4,), windows=[("m1", 0.0, 2.5), ("m1", 0.0, 20.0)], max_speakers=4, slot_blocks=True, declare_range=True,
                            spk_activity=True, act_input=_load("test_slot_input").ActNoise(clean=1.0))
    seen = {j: {len(ds.slots(j, random.Random(r))[1]) for r in range(60)} for j in range(2)}
    assert seen[0] == {1, 2, 3, 4} and seen[1] == {3, 4}                                          # 한 화자 창 1–4, 세 화자 창 3–4
    assert len(ds.slots(0)[1]) == 2                                                               # 평가(rng 없음): max(2, 화자 수)
    e6 = CstDialogueDataset([q], _tok(), delays=(4,), windows=[("m1", 0.0, 2.5)], max_speakers=4, declare_range=True)
    assert {len(e6.slots(0, random.Random(r))[1]) for r in range(60)} == {2, 3, 4}                  # E6 형식은 하한 2
    s1 = next(s for s in (ds.cst_sequence(0, 4, rng=random.Random(r)) for r in range(60)) if len(s["langs"]) == 1)
    assert ds.problems_of(s1) == [] and s1["ids"][: len(ds.cst_prefix(["English"], 4))] == ds.cst_prefix(["English"], 4)
    random.seed(0); items = [ds[j] for j in range(2)]
    for it in items: assert it["tag_chunk"].shape == it["ids"].shape and int(it["tag_chunk"].max()) >= 0 and it["act_in"].shape == (it["K"], 4)
    b = collate_cst(items); assert b["tag_chunk"].shape == b["ids"].shape and (b["tag_chunk"][b["mask"] == 0] == -1).all() and int(b["n_begin"]) >= 1


def test_act_gate_rules():
    """2026-10-30 활동 게이트: 글자는 프레임 k − δ, <BEGIN> 은 k − 4 근처 [−back, +fwd] 에 그 슬롯 활동이 없으면 막는다. <EOT>·활동 밖 창은 통과."""
    from vapasr.hf.batch_decode import ActGate
    act = np.zeros((40, 2), np.float32); act[10:15, 0] = 1; act[30:40, 1] = 1
    g = ActGate(torch.from_numpy(act), delta=8, begin_id=7, eot_id=9, back=3, fwd=1)
    assert g.allows(8 + 12, 0, 50) and g.allows(8 + 17, 0, 50) and not g.allows(8 + 18, 0, 50)              # 글자: 창 [f − 3, f + 1] 이 10–14 와 겹치나(f = 17 은 14–18, 18 은 15–19)
    assert g.allows(8 + 9, 0, 50) and not g.allows(8 + 8, 0, 50)                                             # f = 9 는 6–10, f = 8 은 5–9
    assert not g.allows(20, 1, 50) and g.allows(20, 1, 9)                                                    # <EOT> 는 늘 통과
    assert g.allows(4 + 29, 1, 7) and not g.allows(4 + 25, 1, 7) and g.allows(4 + 25, 0, 7) is False         # <BEGIN>: f = k − 4(29 는 26–30)
    assert g.allows(3, 0, 50) and g.allows(8 + 45, 1, 50) and g.allows(5, 3, 50)                             # 오디오 앞·flush 끝(창이 활동 밖)·활동 채널 밖 슬롯
    assert g.blocked == dict(text=3, begin=2)


@pytest.mark.parametrize("seed", [0, 1])
def test_gated_speculative_decode_equals_token_by_token(seed):
    """게이트를 건 추측 prefill(realtime_block_decode)이 BlockState 한 토큰씩 고리(decode_blocks_batch 와 같은 gate_tid)와 같고, 막힌 라운드·슬롯에는 말이 없다."""
    from vapasr.hf.batch_decode import ActGate, gate_tid
    from vapasr.hf.realtime import realtime_block_decode
    m, tok, bid = _tiny_blocks(); m.eval(); torch.manual_seed(seed); K, D, dl = 30, 4, 3
    with torch.no_grad(): m.adapter_tagact.weight.copy_(torch.randn_like(m.adapter_tagact.weight))
    feats = torch.randn(1, K, 4); act = (torch.rand(K, 4) > 0.6).float(); prefix = [5, 6, 7]; emb = m.get_input_embeddings().weight
    EOT = m.sp_ids["<EOT>"]; words = [50, 60, 70, bid, EOT]; NA = m.next_audio
    pick = lambda h: NA if (h % 10) < 4 + seed else words[(h // 10) % len(words)]
    m.config.runaway_cap = 3; mk = lambda: ActGate(act, dl, bid, EOT, back=1, fwd=0)
    g1 = mk(); out, forced, ticks, rounds, _, nf = realtime_block_decode(m, _FakeLM(emb, pick), feats, prefix, D, act=act, max_flush_rounds=4, gate=g1)
    lm2 = _FakeLM(emb, pick); lm2.run_all(emb[torch.tensor(prefix)], 0); pos = len(prefix); ce = m.chunk_embed(feats[None])[0]; g2 = mk()
    st = BlockState(K, D, m.config.runaway_cap, 4, int(6 * K) + 64, tuple(m.slot_ids(D))); stop = block_stops(m, D); x = ce[0]
    while True:
        tid = gate_tid(st, lm2.run(x[None], pos), stop, g2, NA); pos += 1; kind, v = st.consume(tid, stop)
        if kind == "pad": break
        x = ce[v] if kind == "audio" else m.tag_embeds(v[0], act, D)[v[1]] if kind == "tag" else emb[v] if kind == "text" else emb[m.empty_audio]
    assert out == st.emitted and forced == st.forced and rounds == st.flush_rounds and g1.blocked == g2.blocked and sum(g1.blocked.values()) > 0
    ungated, *_ = realtime_block_decode(m, _FakeLM(emb, pick), feats, prefix, D, act=act, max_flush_rounds=4)
    assert out != ungated                                                                                    # 게이트가 출력을 바꿨다
    tags = m.slot_ids(D); first = {}
    for j, (k, t) in enumerate(out):                                                                         # 라운드·슬롯 첫 출력은 게이트를 통과한 것만
        if t in tags: first[(k, tags.index(t))] = out[j + 1][1]
    assert first and all(mk().allows(k, s, t) for (k, s), t in first.items())


def test_stage3_eval_refuses_gate_without_decode(tmp_path, monkeypatch):
    """--act-gate 는 모델 디코드 전용, --follow-n3d 는 --from-predictions 전용 — 모델·tokenizer 를 읽기 전에 거부한다."""
    import sys
    sys.path.insert(0, str(_HERE.parent / "experiments")); import cst_stage3_eval as E3
    ck = tmp_path / "ck"; ck.mkdir(); (ck / "config.json").write_text("{}")
    base = ["x", "run", "--checkpoint", str(ck), "--data", str(tmp_path), "--out", str(tmp_path / "o")]
    for extra, msg in ((["--oracle", "--act-gate", "3,1"], "--act-gate"), (["--from-predictions", str(tmp_path), "--act-gate", "3,1"], "--act-gate"),
                       (["--follow-n3d"], "--follow-n3d"), (["--act-gate", "3"], "BACK,FWD")):
        monkeypatch.setattr(sys, "argv", base + extra)
        with pytest.raises(AssertionError, match=msg): E3.main()
