"""C2 2 회선 전화 교차 언어 합성(experiments/sim_build_dialogues.py --fto-clip·--intra-pause·--phone-per-speaker·--phone-same-frac·--name-suffix).
기본값이면 산출(세션 jsonl·mono·번역 타깃·stats)이 예전과 바이트 동일하고 난수 소비도 같다(편집 전 코드로 잰 골든 해시), tel 옵션의 타이밍·채널 불변식.
작은 합성 Phase 2 대화·S1 번역 타깃·잡음 뱅크를 tmp 에 만들어 빌더 main() 을 그대로 돌린다. CPU, 모델 없음."""
import gzip, hashlib, json, math, os, platform, random, sys
from pathlib import Path
import numpy as np, pytest
sf = pytest.importorskip("soundfile"); pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import sim_build_dialogues as sbd
from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef
from vapasr.data.sim_dialogue import Unit, SimConfig, TimingStats, synth_session, overlap_stats, turn_gaps
from vapasr.data.phone_aug import PhoneAugment

SR = 16000
TEL = ["--p-hold", "0.05", "--bc-scale", "0", "--fto-clip", "0.05,0.8", "--intra-pause", "0.1,0.24", "--jitter-db", "8", "--phone-per-speaker", "1.0",
       "--phone-same-frac", "0.1", "--name-suffix", "tel"]                                     # build_tel.sh 와 같은 tel 옵션
# 편집 전 코드(dfe1a8d)로 잰 기본 빌드 산출 해시 — _digest. 부동소수 합이 플랫폼마다 다를 수 있어 잰 환경에서만 비교한다(다른 곳은 명시 기본값 대조만).
GOLDEN_ENV = ("darwin", "arm64", "1.24.3")
GOLDEN = {"xl0": dict(jsonl="1cee0a9e1a36a4fa", n=5, mono="9eb05f9600630a6d", targets="b7b92ce9639446a1", stats="12d71198cff65ccb", args="6def9a200600ed4f"),
          "en0": dict(jsonl="cb2a39bd3a8cff15", n=5, mono="aa92338e85beb920", targets="e3b0c44298fc1c14", stats="956f76df4b441213", args="6def9a200600ed4f"),
          "xl3": dict(jsonl="383631642ccd56a4", n=7, mono="a0ad53bb9aff499d", targets="a0dd07ea4c3da753", stats="858a935f0e26768a", args="6def9a200600ed4f")}


# ── 고정 재료(골든 해시가 이 생성 규칙에 묶여 있다 — 바꾸지 않는다)
def _voice(g, dur, f0, amp, pause=None):
    n = int(round(dur * SR)); t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t + k) / k for k in range(1, 9)) * (0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t)) + 0.3 * g.standard_normal(n)   # 음절 변조 + 광대역(4 kHz 위) 성분
    x = amp * x / np.max(np.abs(x))
    if pause: x[int(pause[0] * SR): int(pause[1] * SR)] = 0.0
    ramp = np.minimum(1.0, np.minimum(np.arange(n), n - 1 - np.arange(n)) / (0.02 * SR))
    return (x * ramp).astype(np.float32)


def _dialogue(root, corpus, conv, lang, spks, seed, n_turns=24):
    r = random.Random(seed); g = np.random.default_rng(seed)
    f0 = {spks[0]: 110 + 10 * (seed % 5), spks[1]: 190 + 10 * (seed % 3)}; amp = {spks[0]: 0.3, spks[1]: 0.08}
    t = 0.5; s = 0; utts = []; segs = {x: [] for x in spks}
    for k in range(n_turns):
        hold = k > 0 and r.random() < 0.15
        if k and not hold: s = 1 - s
        L = 5.2 if k % 7 == 3 else (0.6 if k % 9 == 5 else r.uniform(1.0, 3.2))
        x = _voice(g, L, f0[spks[s]], amp[spks[s]], pause=(2.4, 2.8) if L > 5 else None)
        a, b = round(t, 3), round(t + L, 3); nt = 1 if L < 1 else max(2, int(L * 1.5))
        utts.append(Utterance(spks[s], a, b, f"w{k}", tokens=[(1000 + 37 * k + j, round(a + 0.15 + (L - 0.25) * (j + 1) / nt, 3)) for j in range(nt)], utt_id=f"u{k}"))
        segs[spks[s]].append((a, x)); t = b + (r.uniform(0.4, 1.0) if hold else r.uniform(-0.3, 0.9))
    dur = round(t + 1.0, 3); chans = {}
    for x in spks:
        ch = (1e-4 * g.standard_normal(int(dur * SR) + SR)).astype(np.float32)
        for a, seg in segs[x]: i0 = int(round(a * SR)); ch[i0: i0 + len(seg)] += seg
        p = root / f"{conv.replace(':', '_')}_{x}.wav"; sf.write(str(p), ch, SR, subtype="FLOAT"); chans[x] = ChannelRef(path=str(p))
    return Dialogue(conv_id=conv, corpus=corpus, lang=lang, split="train", duration_s=dur, speakers=list(spks), utterances=utts, channels=chans)


def _fixture(root):
    root = Path(root); (root / "audio").mkdir(parents=True); (root / "p2" / "splits").mkdir(parents=True); (root / "s1" / "xx").mkdir(parents=True)
    rows = []
    for corpus, lang, tgt, convs in (("otoSpeech", "English", "German", [("oto:1", ("0", "1")), ("oto:2", ("0", "1")), ("oto:3", ("0", "1"))]),
                                     ("kec", "German", "English", [("kec:r1", ("AS_001", "IS_002")), ("kec:r2", ("AS_003", "IS_004")), ("kec:r3", ("AS_005", "IS_006"))])):
        with open(root / "p2" / f"{corpus}.refined.dialogues.jsonl", "w", encoding="utf-8") as f:
            for j, (conv, spks) in enumerate(convs):
                d = _dialogue(root / "audio", corpus, conv, lang, spks, seed=11 * j + len(corpus)); f.write(d.to_json() + "\n")
                rows += [dict(id=f"{conv}|{u.speaker}|{u.utt_id}", tgt=tgt, tokens=[[5000 + q, t] for q, (_, t) in enumerate(u.tokens)]) for u in d.utterances]
    json.dump({"corpora": {"otoSpeech": {"heldout": ["oto:3"]}, "kec": {"heldout": ["kec:r3"]}}}, open(root / "p2" / "splits" / "v2.json", "w"))
    with gzip.open(root / "s1" / "xx" / "u-00000.jsonl.gz", "wt", encoding="utf-8") as f: f.write("".join(json.dumps(r) + "\n" for r in rows))
    g = np.random.default_rng(7); (root / "nb" / "audio").mkdir(parents=True); nrows = []
    for name, sec in (("n0", 3.0), ("n1", 1.3)):
        p = root / "nb" / "audio" / f"{name}.wav"; x = (0.1 * g.standard_normal(int(sec * SR))).astype(np.float32); sf.write(str(p), x, SR, subtype="PCM_16")
        nrows.append(dict(path=str(p), source="t", category="noise", n=len(x), dur=sec))
    (root / "nb" / "noise.jsonl").write_text("".join(json.dumps(r) + "\n" for r in nrows))
    h = np.zeros((2, 300), np.float16); h[0, 20] = 1.0; h[0, 120] = 0.3; h[1, 0] = 1.0; h[1, 200] = -0.2
    np.save(root / "nb" / "rir.npy", h); (root / "nb" / "rir.jsonl").write_text("".join(json.dumps(dict(i=i, source="t", n=300)) + "\n" for i in range(2))); (root / "nb" / "summary.json").write_text("{}")
    return root


def _argv(root, out, xl=True, n=5):
    base = ["--out", str(out), "--sessions", str(n), "--dur", "30,40", "--phase2", str(root / "p2"), "--split", str(root / "p2" / "splits" / "v2.json"),
            "--phase2-corpora", "otoSpeech", "--pool-weights", "otoSpeech:1", "--min-units", "3", "--workers", "1"]
    if not xl: return ["--lang", "English"] + base
    return ["--lang", "English", "--lang2", "German"] + base + ["--phase2-corpora2", "kec", "--pool-weights2", "kec:1", "--trans", str(root / "s1"),
                                                               "--noise-bank", str(root / "nb"), "--noise-p", "0.7", "--rir-p", "0.5"]


def _run(argv):
    old = sys.argv; sys.argv = ["sim_build_dialogues.py"] + [str(x) for x in argv]
    try: sbd.G.clear(); sbd.main()
    finally: sys.argv = old


def _digest(root, out, key):
    """산출 해시: 세션 jsonl(tmp 경로는 <T> 로)·mono npy 바이트·번역 타깃(압축 푼 내용 — gzip 헤더 시각 제외)·stats(sec·args 값 제외, args 키 목록은 포함)."""
    out = Path(out); H = lambda b: hashlib.sha256(b).hexdigest()[:16]
    jl = (out / f"{key}.refined.dialogues.jsonl").read_text(encoding="utf-8").replace(str(root), "<T>")
    mono = sorted((out / "mono" / key).glob("*.npy")); tg = sorted((out / f"targets-{key}").rglob("u-*.jsonl.gz"))
    st = json.load(open(out / f"{key}.stats.json")); args = sorted(st.pop("args")); st.pop("sec")
    return dict(jsonl=H(jl.encode()), n=len(jl.splitlines()), mono=H(b"".join(p.name.encode() + p.read_bytes() for p in mono)),
                targets=H(b"".join(p.relative_to(out).as_posix().encode() + gzip.open(p).read() for p in tg)), stats=H(json.dumps(st, sort_keys=True).encode()), args=H(json.dumps(args).encode()))


def _lines(out, key):
    return [json.loads(l) for l in open(Path(out) / f"{key}.refined.dialogues.jsonl", encoding="utf-8")]


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return _fixture(tmp_path_factory.mktemp("telfx"))


# ── 기본값 = 예전 빌드
def test_cli_defaults_are_simconfig_defaults():
    c = SimConfig(); assert sbd.TEL_DEFAULTS["fto_clip"] == c.fto and sbd.TEL_DEFAULTS["intra_pause"] == c.intra_pause


def test_default_build_matches_golden(fx, tmp_path):
    if (sys.platform, platform.machine(), np.__version__) != GOLDEN_ENV: pytest.skip(f"골든 해시는 {GOLDEN_ENV} 에서 쟀다")
    for name, argv, key in (("xl0", _argv(fx, tmp_path / "xl0"), "sim_en-de"), ("en0", _argv(fx, tmp_path / "en0", xl=False), "sim_en"),
                            ("xl3", _argv(fx, tmp_path / "xl3", n=7) + ["--seed", "3"], "sim_en-de")):
        _run(argv); assert _digest(fx, tmp_path / name, key) == GOLDEN[name], name


def test_explicit_defaults_equal_omitted(fx, tmp_path):
    """새 옵션을 기본값으로 명시해도(음수 범위는 = 로) 생략과 바이트 동일 — 세션·mono·번역 타깃·stats·args 키, tel·chan 없음."""
    for xl, key in ((True, "sim_en-de"), (False, "sim_en")):
        _run(_argv(fx, tmp_path / f"a{key}", xl=xl))
        _run(_argv(fx, tmp_path / f"b{key}", xl=xl) + ["--fto-clip=-2,3", "--intra-pause", "0.1,0.5", "--phone-per-speaker", "0", "--phone-same-frac", "0", "--jitter-db", "3"])
        da, db = _digest(fx, tmp_path / f"a{key}", key), _digest(fx, tmp_path / f"b{key}", key); assert da == db and da["n"] == 5
        st = json.load(open(tmp_path / f"b{key}" / f"{key}.stats.json")); assert "tel" not in st and not set(sbd.TEL_DEFAULTS) & set(st["args"])
        assert all("chan" not in d["meta"] for d in _lines(tmp_path / f"b{key}", key))


def test_phone_per_speaker_leaves_session_stream(fx, tmp_path):
    """화자별 채널 난수는 세션 rng 와 따로라 세션(발화·시각·조각·번역 타깃)은 기본 빌드와 같고 gains·chan·mono 만 바뀐다."""
    _run(_argv(fx, tmp_path / "a")); _run(_argv(fx, tmp_path / "p") + ["--phone-per-speaker", "1", "--phone-same-frac", "0.5"])
    A, P = _lines(tmp_path / "a", "sim_en-de"), _lines(tmp_path / "p", "sim_en-de"); assert len(A) == len(P) == 5
    for x, y in zip(A, P):
        mx, my = x.pop("meta"), y.pop("meta"); assert x == y and "chan" in my and "chan" not in mx
        assert {k: v for k, v in mx.items() if k != "gains"} == {k: v for k, v in my.items() if k not in ("gains", "chan")}
        assert all(abs(20 * math.log10(my["gains"][s] / mx["gains"][s])) < 0.1 for s in mx["gains"])     # 같은 jitter 난수(세션 rng) — PhoneAugment 가 RMS 를 맞춰 gain 도 거의 같다
        assert list(my["chan"]["spk"]) == ["a", "b"] and all(v["phone"] for v in my["chan"]["spk"].values())
    da, dp = _digest(fx, tmp_path / "a", "sim_en-de"), _digest(fx, tmp_path / "p", "sim_en-de"); assert da["targets"] == dp["targets"] and da["mono"] != dp["mono"]


class _NoChan(random.Random):
    def __init__(self, x=None):
        if isinstance(x, str) and ":chan" in x: raise AssertionError(f"채널 난수 {x}")
        super().__init__(x)


def test_p0_draws_no_channel_rng(fx, tmp_path, monkeypatch):
    """--phone-per-speaker 0 이면 채널 난수를 만들지 않는다(만들면 세션이 실패해 stats 에 남는다 — P = 1 이 대조)."""
    monkeypatch.setattr(sbd.random, "Random", _NoChan)
    _run(_argv(fx, tmp_path / "z") + ["--fto-clip", "0.05,0.8"]); st = json.load(open(tmp_path / "z" / "sim_en-de.stats.json"))
    assert st["sessions"] == 5 and st["failed"] == 0 and st["tel"]["chan"] is None
    _run(_argv(fx, tmp_path / "o", n=2) + ["--phone-per-speaker", "1"]); st = json.load(open(tmp_path / "o" / "sim_en-de.stats.json"))
    assert st["sessions"] == 0 and any("채널 난수" in e for e in st["errors"])


def test_option_validation(fx, tmp_path):
    for bad in (["--phone-same-frac", "0.1"], ["--phone-per-speaker", "1.5"], ["--fto-clip", "0.8,0.05"], ["--name-suffix", "Tel"], ["--intra-pause", "0.3"]):
        with pytest.raises(SystemExit): _run(_argv(fx, tmp_path / "v") + bad)
    with pytest.raises(SystemExit): _run(_argv(fx, tmp_path / "v", xl=False) + ["--name-suffix", "tel"])     # 접미사는 교차 언어만(같은 언어 키와 섞이지 않게)


# ── tel 빌드
def test_tel_build_end_to_end(fx, tmp_path):
    out = tmp_path / "t"; _run(_argv(fx, out, n=12) + TEL); key = "sim_en-de-tel"
    assert sorted(os.listdir(out)) == ["mono", f"{key}.refined.dialogues.jsonl", f"{key}.stats.json", f"targets-{key}"] and os.listdir(out / "mono") == [key]
    st = json.load(open(out / f"{key}.stats.json")); T = st["tel"]
    assert st["sessions"] == 12 and st["overlap_ratio_mean"] == 0 and st["bc_per_min"] == 0 and st["self_overlap"] == 0 and st["trans"]["rows"] > 0
    assert T["split_turns"] == 0 and T["change_gap_s"]["min"] >= 0.05 - 1e-3 and T["change_gap_s"]["hist"]["under"] == 0 and T["changes"] > 0
    assert T["chan"]["sessions"] == 12 and T["chan"]["phone"] == 24 and sum(T["level_diff_db"]["hist"]["n"]) == 12
    assert st["args"]["name_suffix"] == "tel" and st["args"]["fto_clip"] == [0.05, 0.8] and st["args"]["phone_per_speaker"] == 1.0
    for d in _lines(out, key):
        assert d["conv_id"].startswith(f"{key}_") and d["corpus"] == key and d["meta"]["bc"] == 0 and (out / "mono" / key / f"{d['conv_id']}.npy").exists()
        c = d["meta"]["chan"]; a, b = c["spk"]["a"], c["spk"]["b"]
        assert all(-23 - 8 - 1e-3 <= v["level_db"] <= -23 + 8 + 1e-3 for v in (a, b))
        if c["same"]: assert {k: v for k, v in a.items() if k != "level_db"} == {k: v for k, v in b.items() if k != "level_db"}


def test_tel_output_loads_in_cst_dataset(fx, tmp_path):
    """접미사 키·meta.chan 이 있는 tel 산출을 학습 데이터셋이 그대로 읽는다(교차 언어 슬롯, 받아쓰기·번역 모드 시퀀스 검사 0, mono 캐시)."""
    from test_dialogue_dataset import FakeTok
    from vapasr.data.ast_dataset import load_ast_targets
    from vapasr.data.cst_dialogue import CstDialogueDataset
    out = tmp_path / "t"; _run(_argv(fx, out, n=4) + TEL); key = "sim_en-de-tel"
    XT, _, _ = load_ast_targets([str(out / f"targets-{key}")]); tok = FakeTok(); tok.add_tokens(["<TRANS>"])
    ds = CstDialogueDataset([str(out / f"{key}.refined.dialogues.jsonl")], tok, delays=(2, 4), window_s=(20.0, 20.0), hop_s=10.0, seed=0, mono_cache_dir=str(out / "mono"),
                            trans=XT, ast_frac=0.5)
    assert ds.has_xl and len(ds) > 0
    for i in range(len(ds)):
        for dl in (2, 4):
            assert ds.problems_of(ds.cst_sequence(i, dl, tgt="xl")) == [] and ds.problems_of4(ds.cst4_sequence(i, dl, "xl")) == []
    random.seed(0); b = ds[0]; assert b["name"] == key and len(b["wav"]) == int(round(20.0 * SR))


def _units(prefix, n, rng):
    out = []
    for i in range(n):
        d = 0.7 if i % 5 == 0 else round(rng.uniform(1.0, 4.0), 3); k = 1 if d < 1 else 4
        out.append(Unit(key=prefix, order=i, path=f"/{prefix}/{i}.wav", src_offset=None, dur=d, tokens=[(100 + j, round(d * (j + 1) / (k + 1), 3)) for j in range(k)],
                        uid=f"{prefix}{i}", pool="t"))
    return out


def test_tel_timing_invariants_500_sessions():
    """tel 타이밍(--p-hold 0.05 --bc-scale 0 --fto-clip 0.05,0.8 --intra-pause 0.1,0.24): 겹침 0, 교대 간격 0.05–0.8 s, hold 비율 0.05 ± 0.02,
    턴 안 쉼 < 0.25 s 라 합성 턴 하나 = episode 하나, 맞장구 0. 같은 재료·기본 설정은 겹침·맞장구가 있다(대조)."""
    g = random.Random(5)
    st = TimingStats(fto=[round(g.uniform(-1.5, 3.0), 3) for _ in range(400)], hold=[round(g.uniform(0.3, 2.0), 3) for _ in range(100)], p_hold=0.55,
                     turn=[round(g.uniform(0.8, 8.0), 3) for _ in range(300)], bc_rate=0.08, bc_pos=[0.3, 0.5, 0.7])
    tel = SimConfig(p_hold=0.05, bc_scale=0.0, fto=(0.05, 0.8), intra_pause=(0.1, 0.24)); span = lambda u: (0.0, u.dur); nh = nc = ov0 = bc0 = 0
    for i in range(500):
        r = random.Random(i); us = [_units("A", 60, r), _units("B", 60, r)]
        d = synth_session(us, st, random.Random(f"s{i}"), r.uniform(30.0, 75.0), span, f"t{i}", "sim_en-de-tel", "English+German", cfg=tel)
        o, tg = overlap_stats(d), turn_gaps(d)
        assert o["overlap_s"] == 0 and o["self_overlap"] == 0 and d.meta["bc"] == 0
        assert tg["episodes"] == d.meta["turns"]                                                     # 턴 안 쉼 < GAP_S → 합성 턴 하나가 episode 하나
        assert all(0.05 - 1e-3 <= x <= 0.8 + 1e-3 for x in tg["change"]) and all(x >= 0.3 - 1e-3 for x in tg["hold"])
        assert all(x <= 0.24 + 1e-3 or x >= 0.3 - 1e-3 for x in tg["same"])                         # 같은 화자 쉼: 턴 안(≤ 0.24 s) 아니면 재개·상대 턴 너머
        nh += len(tg["hold"]); nc += len(tg["change"])
        if i < 50:
            r = random.Random(i); dd = synth_session([_units("A", 60, r), _units("B", 60, r)], st, random.Random(f"s{i}"), 60.0, span, f"d{i}", "sim_en-de", "English+German")
            ov0 += overlap_stats(dd)["overlap_s"] > 0; bc0 += dd.meta["bc"]
    assert abs(nh / (nh + nc) - 0.05) < 0.02 and nc > 3000 and ov0 > 10 and bc0 > 10, (nh, nc, ov0, bc0)


def test_turn_gaps():
    U = lambda s, a, b, i: Utterance(s, a, b, "w", tokens=[(1, b)], utt_id=f"{s}{i}")
    d = Dialogue(conv_id="g", corpus="t", lang="English", split="train", duration_s=20.0, speakers=["a", "b"],
                 utterances=[U("a", 0.5, 2.0, 0), U("a", 2.2, 3.0, 1), U("b", 3.3, 5.0, 2), U("b", 6.0, 7.0, 3), U("a", 6.8, 8.0, 4)])
    assert turn_gaps(d) == dict(change=[0.3, -0.2], hold=[1.0], same=[0.2, 3.8, 1.0], episodes=4)


# ── 화자별 전화 채널(mix_arrays·phone_fx)
def _mix_case(seed=0, n_utt=6):
    """화자 둘·발화 몇 개의 Dialogue + 발화 오디오(4 kHz 위 성분·시각 기준 펄스, 화자 레벨 다름)·말소리 구간 — mix_arrays 직접 시험."""
    g = np.random.default_rng(seed); utts = []; audio = {}; spans = {}; t = 0.5
    for k in range(n_utt):
        s = "ab"[k % 2]; L = 1.5 + 0.25 * k; uid = f"{s}{k}"; tt = np.arange(int(round(L * SR))) / SR
        x = (0.2 * np.sin(2 * np.pi * (150 if s == "a" else 230) * tt) + 0.1 * np.sin(2 * np.pi * 6000 * tt) + 0.05 * g.standard_normal(len(tt))) * (0.3 if s == "a" else 1.0)
        x[len(x) // 2: len(x) // 2 + 40] += 0.5; audio[uid] = x.astype(np.float32); spans[uid] = (0.0, L)
        utts.append(Utterance(s, round(t, 3), round(t + L, 3), "w", tokens=[(1, round(t + L - 0.1, 3))], utt_id=f"{s}:{uid}")); t += L + 0.4
    return Dialogue(conv_id="m", corpus="sim_en-de-tel", lang="English+German", split="train", duration_s=round(t + 0.5, 3), speakers=["a", "b"], utterances=utts), audio, spans


def _captured(aug, seed, i, speakers, same_frac=0.0):
    fx, chan = sbd.phone_fx(aug, seed, i, speakers, same_frac); raw, out = {}, {}
    def f(ch, spk): raw[spk] = ch.copy(); out[spk] = fx(ch, spk); return out[spk]
    return f, chan, raw, out


def _speech_rms(x, d, spk):
    rr = [(int(round(u.start * SR)), int(round((u.end - u.start) * SR))) for u in d.utterances if u.speaker == spk]
    return math.sqrt(sum(float(np.sum(x[a: a + m].astype(np.float64) ** 2)) for a, m in rr) / sum(m for _, m in rr))


def test_per_speaker_phone_band_phase_and_gain():
    """트랙마다: 4.2 kHz 위 에너지 < 2 %, 상호상관 지연 ±1 샘플(영위상), 처리된 트랙의 말소리 구간 레벨 = −23 dBFS + jitter(1 dB 안)."""
    d, audio, spans = _mix_case(); aug = PhoneAugment(p=1.0, mulaw_p=0.5)
    for jit in (0.0, 8.0):
        f, chan, raw, out = _captured(aug, 0, 3, d.speakers); lv = {}
        y, gains = sbd.mix_arrays(d, audio, spans, random.Random(9), jit, fx=f, levels=lv)
        r = random.Random(9); js = [r.uniform(-jit, jit) if jit else 0.0 for _ in d.speakers]                # mix_arrays 의 jitter 와 같은 세션 난수열
        for k, spk in enumerate(d.speakers):
            x, z = raw[spk], out[spk]; F = np.fft.rfftfreq(len(x), 1 / SR); X, Z = np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(z)) ** 2; hi = F > 4200
            assert X[hi].sum() > 0.1 * X.sum() and Z[hi].sum() < 0.02 * Z.sum()
            c = np.fft.irfft(np.fft.rfft(z) * np.conj(np.fft.rfft(x)), len(x)); assert abs(int(np.argmax(np.r_[c[-50:], c[:51]])) - 50) <= 1
            L = 20 * math.log10(gains[spk] * _speech_rms(z, d, spk)); assert abs(L - (-23.0 + js[k])) < 1.0 and abs(lv[spk] - (-23.0 + js[k])) < 1e-2
        assert chan["same"] is False and all(v["phone"] and 200 <= v["low"] <= 400 and 3000 <= v["high"] <= 3600 for v in chan["spk"].values())
        assert np.max(np.abs(y)) <= 0.98 + 1e-6


def test_same_frac_shares_channel_parameters():
    d, audio, spans = _mix_case(); aug = PhoneAugment(p=1.0, mulaw_p=0.5)
    f, chan, _, _ = _captured(aug, 0, 3, d.speakers, same_frac=1.0); sbd.mix_arrays(d, audio, spans, random.Random(1), 8.0, fx=f)
    assert chan["same"] and chan["spk"]["a"] == chan["spk"]["b"]                                        # 같은 시드 → 같은 대역·μ-law
    f2, chan2, _, _ = _captured(aug, 0, 3, d.speakers); sbd.mix_arrays(d, audio, spans, random.Random(1), 8.0, fx=f2)
    assert not chan2["same"] and chan2["spk"]["a"] != chan2["spk"]["b"] and chan2["spk"]["a"] == chan["spk"]["a"]   # 따로면 화자마다 다르다(첫 화자 시드는 같다)
    fr = sum(sbd.phone_fx(aug, 0, i, ["a", "b"], 0.1)[1]["same"] for i in range(3000)) / 3000; assert abs(fr - 0.1) < 0.02
    assert not any(sbd.phone_fx(aug, 0, i, ["a", "b"], 0.0)[1]["same"] for i in range(200))


def test_level_offset_scales_only_that_speaker():
    """--level-offset-db(진단): 같은 난수열에서 그 화자 gain 만 10^(dB/20) 배(피크 제한 전), 다른 화자·난수 소비는 그대로."""
    d, audio, spans = _mix_case(); lv0, lv1 = {}, {}
    r0, r1 = random.Random(5), random.Random(5)
    y0, g0 = sbd.mix_arrays(d, audio, spans, r0, 8.0, levels=lv0); y1, g1 = sbd.mix_arrays(d, audio, spans, r1, 8.0, levels=lv1, offsets={"a": -12.0})
    assert r0.random() == r1.random()                                                        # 난수를 더 뽑지 않는다
    assert abs(g1["a"] / g0["a"] - 10 ** (-12 / 20)) < 1e-3 and g1["b"] == g0["b"] and abs(lv1["a"] - (lv0["a"] - 12.0)) < 1e-2 and lv1["b"] == lv0["b"]
    y2, g2 = sbd.mix_arrays(d, audio, spans, random.Random(5), 8.0, offsets={"a": 0.0}); assert g2 == g0 and np.array_equal(y2, y0)


def test_level_offset_build(fx, tmp_path):
    """같은 인자 + --level-offset-db English:-12 → 같은 대화(발화·시각), 영어 화자 레벨만 −12 dB, meta.level_offset_db. 기본(끔)은 stats args 에 키가 없다."""
    _run(_argv(fx, tmp_path / "base")); _run(_argv(fx, tmp_path / "lo") + ["--level-offset-db", "English:-12"])
    A, B = _lines(tmp_path / "base", "sim_en-de"), _lines(tmp_path / "lo", "sim_en-de"); assert len(A) == len(B) > 0
    strip = lambda x: {k: v for k, v in x.items() if k != "meta"}
    for x, y in zip(A, B):
        assert strip(x) == strip(y) and y["meta"]["level_offset_db"] == {"a": -12.0, "b": 0.0}
        assert abs(y["meta"]["gains"]["a"] / x["meta"]["gains"]["a"] - 10 ** (-12 / 20)) < 2e-3 and y["meta"]["gains"]["b"] == x["meta"]["gains"]["b"]
    assert "level_offset_db" not in json.load(open(tmp_path / "base" / "sim_en-de.stats.json"))["args"]
    with pytest.raises(SystemExit): _run(_argv(fx, tmp_path / "bad") + ["--level-offset-db", "Klingon:-3"])

