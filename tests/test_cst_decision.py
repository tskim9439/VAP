"""V3-p1 C1·C5(vapasr/data/cst_dialogue.py) — 교대·머무름 결정 가중(decision_positions, change_weight·hold_weight, dec_kind)과
합성 창 EOT-문맥 dropout(drop_eot·eot_ghost·eot_drop_synth). 끄면(기본) 시퀀스·라벨·가중·난수 소비가 바뀌기 전과 같다(GOLDEN — 바꾸기 전 코드에서 기록).
모델 forward 결정 통계(dec_kind — 손실 불변)·packing·stage3 셋 키 분류·플래그. 모델·tokenizer 다운로드 없이 CPU."""
import dataclasses, hashlib, json, os, pathlib, random, subprocess, sys
import numpy as np, pytest, torch

from vapasr.data.cst_dialogue import (CstDialogueDataset, collate_cst, Recover, build_cst_sequence, build_cst4_sequence, decision_positions, read_back, read_back4,
                                      window_streams, window_streams4, dialogue_episodes)
from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef, chunk_of
from vapasr.data.interleave import Specials

sf = pytest.importorskip("soundfile")
from test_dialogue_dataset import FakeTok  # noqa: E402


class WTok(FakeTok):
    """FakeTok + decode: 홀수 id 는 단어 첫 토큰(' w'), 짝수는 단어 중간('w')."""
    def decode(self, ids): return "".join(" w" if int(t) % 2 else "w" for t in ids)


def tokw():
    t = WTok(); t.add_tokens(["<TRANS>"]); return t


# 15 s 묶음(o = 0·15·30·45): x 머무름(0.6 s 쉼) · y 교대(0.1 s) · x 가 y 끝 전에 시작(단어 중간 16 이 끼어든다) · 짧은 y · y 머무름 · x 교대
BLOCK = [("x", 0.0, 2.0, [(11, 1.0), (12, 2.0)]), ("x", 2.6, 3.4, [(13, 3.0), (15, 3.4)]), ("y", 3.5, 5.0, [(21, 4.2), (22, 4.6), (23, 5.0)]),
         ("x", 4.9, 6.0, [(16, 4.95), (17, 5.5), (18, 6.0)]), ("y", 6.1, 6.3, [(25, 6.3)]), ("x", 6.5, 8.0, [(19, 7.0), (20, 8.0)]),
         ("y", 9.0, 11.0, [(27, 10.0), (28, 11.0)]), ("y", 11.5, 12.0, [(29, 12.0)]), ("x", 12.1, 14.0, [(31, 13.0), (33, 14.0)])]


def make_dlg(tmp_path, name="c1", xl=False, synth=False):
    """60 s 2 인 대화(BLOCK × 4) + 행 번역 타깃. xl: meta.spk_lang(x 한국어 · y 영어), synth: meta.mask_eot(합성 — <EOT> 감독 없음)."""
    tmp_path.mkdir(parents=True, exist_ok=True); SR = 16000; dur = 60.0; p = str(tmp_path / f"{name}.wav"); t = np.arange(int(dur * SR)) / SR
    sf.write(p, np.stack([0.3 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 660 * t)], 1).astype(np.float32), SR)
    utts = []
    for b, o in enumerate((0.0, 15.0, 30.0, 45.0)):
        for j, (w, a, e, toks) in enumerate(BLOCK): utts.append(Utterance(w, o + a, o + e, "t", "t", tokens=[(tid, o + x) for tid, x in toks], utt_id=f"{w}{b}_{j}"))
    meta = dict(**({"spk_lang": {"x": "Korean", "y": "English"}} if xl else {}), **({"synthetic": "sim", "mask_eot": True} if synth else {}))
    d = Dialogue(conv_id=name, corpus="test", lang="Korean+English" if xl else "Korean", split="train", duration_s=dur, speakers=["x", "y"], utterances=utts,
                 channels={"x": ChannelRef(path=f"{p}#ch0"), "y": ChannelRef(path=f"{p}#ch1")}, meta=meta)
    q = str(tmp_path / f"{name}.refined.dialogues.jsonl"); open(q, "w").write(d.to_json() + "\n")
    tr = {}
    for u in utts:
        tg = ("English" if u.speaker == "x" else "Korean") if xl else "English"; ids = [41, 42] if u.speaker == "x" else [51, 52]
        tr[f"{name}|{u.speaker}|{u.utt_id}"] = {tg: (np.array(ids), np.array([u.start + (u.end - u.start) / 2, u.end]))}
    return q, tr


KINDS = dict(real=dict(xl=False, synth=False, ast_frac=0.5, recover=Recover(p=0.5)), synth=dict(xl=False, synth=True, ast_frac=0.5, recover=None),
             xl_synth=dict(xl=True, synth=True, ast_frac=0.8, recover=None), xl_real=dict(xl=True, synth=False, ast_frac=0.8, recover=None))


def make_ds(tmp_path, kind, **kw):
    k = KINDS[kind]; q, tr = make_dlg(tmp_path, f"c_{kind}", xl=k["xl"], synth=k["synth"])
    return CstDialogueDataset([q], tokw(), delays=(2, 4), window_s=(12.0, 20.0), hop_s=3.0, seed=1, decision_weight=2.0, trans=tr, ast_frac=k["ast_frac"],
                              recover=k["recover"], **kw)


def canon(x):
    if isinstance(x, dict): return [[repr(k), canon(v)] for k, v in sorted(x.items(), key=lambda kv: repr(kv[0]))]
    if isinstance(x, (set, frozenset)): return sorted((canon(v) for v in x), key=repr)
    if isinstance(x, (list, tuple)): return [canon(v) for v in x]
    if torch.is_tensor(x): return [str(x.dtype), x.tolist()]
    if isinstance(x, np.ndarray): return [str(x.dtype), x.tolist()]
    if dataclasses.is_dataclass(x): return canon(dataclasses.asdict(x))
    if isinstance(x, float): return repr(x)
    return x


def digest(x) -> str:
    return hashlib.sha256(json.dumps(canon(x), default=repr).encode()).hexdigest()


def golden_of(ds) -> dict:
    """끈 경로의 지문: 평가·점검(rng 없음 — 받아쓰기, 목표 언어마다 번역), 학습 rng 경로(같은 Random 을 이어 씀 + 끝 상태), __getitem__(전역 random) + collate."""
    det = [digest(ds.cst_sequence(i, d)) for i in range(len(ds)) for d in ds.delays]
    det += [digest(ds.cst_sequence(i, d, tgt=tg)) for i in range(len(ds)) for d in ds.delays for tg in ds.tgt_options(i)]
    r = random.Random(7); rng = [digest(ds.cst_sequence(i, d, rng=r)) for _ in range(2) for i in range(len(ds)) for d in ds.delays]
    random.seed(3); items = [ds[i] for i in range(len(ds))]; st = random.getstate()
    c = collate_cst(items[:3])
    from vapasr.hf.packing import pack_batch
    return dict(n=len(ds), det=digest(det), rng=digest(rng), rng_state=digest(r.getstate()), items=digest([{k: v for k, v in it.items() if k != "wav"} for it in items]),
                global_state=digest(st), collate=digest({k: v for k, v in c.items() if k != "wav"}), packed=digest({k: v for k, v in pack_batch(c).items() if k != "wav"}))


# ── slurm/submit-stage3-train.sh --dry-run (가짜 /soundai 뿌리 — 입력 검사를 통과할 빈 파일만 만든다)
ROOT = pathlib.Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "slurm" / "submit-stage3-train.sh"
CORPORA = ("aihub71631", "aihub134-1", "aihub134-2", "otoSpeech", "kec", "ztc", "nsc")


def submit_env(tmp_path):
    """가짜 뿌리 R(입력 파일)와 스크립트 사본(/soundai → R/soundai) → (R, 사본 경로)."""
    R = tmp_path / "R"; D = R / "soundai/users/tskim/VAPKT-data/data"
    for f in ["phase2/splits/v2.json", "semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637/main-words.list", "semcommit-work/asr-words-v1/lists/asr-words-20260929.list",
              "cv4-en-v1/lists/cv4-en-words-train.list", "yodas-de-v1/lists/de-words-train-tn2.list", "ast-mono-v1/targets-fix-gen-v3-bf16/pack/en-de/meta.json",
              "ast-mono-cv4en-v1/targets-fix-gen-v3-bf16/pack/en-de/meta.json", "noise-bank-v1/noise.jsonl"] + [f"phase2/{c}.refined.dialogues.jsonl" for c in CORPORA]:
        (D / f).parent.mkdir(parents=True, exist_ok=True); (D / f).write_text("x\n")
    init = R / "soundai/Model/VAPASR/semcommit-q06-s4-ast/final"; init.mkdir(parents=True); (init / "config.json").write_text("{}")
    (R / "sim/mono").mkdir(parents=True); (R / "td/pack").mkdir(parents=True); (R / "rs.list").write_text("x\n")
    sh = tmp_path / "repo/slurm/submit-stage3-train.sh"; sh.parent.mkdir(parents=True)
    sh.write_text(SUBMIT.read_text().replace("/soundai", f"{R}/soundai")); return R, sh


def run_submit(R, sh, args):
    """→ (종료 코드, stdout = 'sbatch …' 명령, stderr 요약) — 가짜 뿌리 경로는 <R> 로."""
    args = [x.replace("<R>", str(R)) for x in args]
    p = subprocess.run(["bash", str(sh)] + args, capture_output=True, text=True, env={**os.environ, "MAX_RESTART": "3"}, timeout=60)
    return p.returncode, p.stdout.replace(str(R), "<R>"), p.stderr.replace(str(R), "<R>")


SUBMIT_ARGS = [["--run", "x", "--dry-run"],
               ["--run", "st4-m2a", "--weights", "dlg_ko:1,dlg_en:1,dlg_de:1,sim_en-de:1,sim_ko:0.5,rep_ko:0.5,rep_en:0.5,rep_de:0.5,rep_chde:0.2", "--sim-data", "<R>/sim",
                "--phone-aug", "0.3", "--trans-lead-space", "--dlg-trans", "<R>/td", "--max-steps", "2500", "--after", "123", "--exclude", "hpc-102",
                "--replay-set", "rep_chde:German:@<R>/rs.list", "--extra", "--seed 0", "--dry-run"]]
SUBMIT_GOLDEN = ["6fca8c3940740c0ff167906bef2523943c9a5a7814b8915c59b402316cabd3e3", "bd0743f887781f02f14a7ffac9140f395ae1954e453b0573459e1b65b7d2899b"]   # 바꾸기 전 스크립트(dfe1a8d)


# ── (a) 끄면 바이트 동일: 바꾸기 전 코드(dfe1a8d)에서 기록한 지문
GOLDEN = {
    'real': dict(n=13, det='2f3d157c4359edb69fe4706e76c615dee108f6cb8c0f9ddc34078c96c9f7588f', rng='636051a2a31b2a4ace8dd980c019663f36773443e90769011818bb170e145ebc', rng_state='b7cdc5903271c27a17455809e9f888905a59381cc679c957e5623b3c406e510a', items='e4f47f854e03940a2fb57bc89a7c89d2fb41b40860628c8c2910e5ef498c552c', global_state='a990706848ab03cb485ea99269a2bf86f74a169786842173aa60ffc5e4d0637e', collate='f04717032ed9344f1d2f09e6ad97ce47a66f913e69cce41b481a31b144a28ba6', packed='dbefe8983e5451b78cb002ac24962af786f220b64c3cfc43867918daf2f67966'),
    'synth': dict(n=13, det='c240439e51989a82bdaeb5ab14118a1499318f3dab251b7e91820a6435e3dc2a', rng='e7b36f57b3cb6c5caa8080ee3731d91a7dfd5486322b3b49ae10437d1834ae58', rng_state='0aa8a48f147423b1539c71ddeb1523a49a009a5177c2ab08d4fbcec2604fa415', items='dbd7aa57727820ab7b5e177ee074eb2c2f225ea940815c2cfbd52b25c8b04dc1', global_state='75945406d2a4c4db62c20857d589622a16a09bff309dbbd540d526e71bfc89d6', collate='b293b1855a920dfca64ee12957941cc45232b9fa57ca012211b808a0a0426279', packed='ea7fad7f429374819d2a63d860fa10f40af332750f03db8e8a9e1d520e9e7a6b'),
    'xl_synth': dict(n=13, det='99997a7edc71955c70cdbff83ce1263e39b46f1a9cef5bf8325a1360f50f39db', rng='f523fe2d38eb15e20fbe1f8cd47e7ec8ec7fb376b19ef135b425eeead76b00f4', rng_state='11f96b4851f69810b0034c785ee856be5c254988836cf809e4d5cc1c2fad7a5a', items='d9e03e6454a177e5170282f0c7ed1074aad2a8a9532289ae0f77f7c2b8d99315', global_state='a4cf3f9dbebea4a26821411025e64b9ace1b50c1d4542415b9dda764d7ad4de1', collate='e4217540e0e8e54774c4cd8c171fa3a2cc3407b60cf31c41074c070b148bf904', packed='63880f1e7c5ccbb0da408358cf20fd13438585b80b5a37da2d545f07bc9563f5'),
    'xl_real': dict(n=13, det='009846059f736e9dea513ab13b509025c69c582916f0eb8b6f8b641d194df154', rng='86efa62964f59e29cb4fb89b40c0c9a4956629e4776394b53e3dd56d404e2d63', rng_state='11f96b4851f69810b0034c785ee856be5c254988836cf809e4d5cc1c2fad7a5a', items='680d8507edf4a58afb68a0d65513cbd2f9dc2556edbc91aa41f65818044d5be0', global_state='a4cf3f9dbebea4a26821411025e64b9ace1b50c1d4542415b9dda764d7ad4de1', collate='a143a8132cc9c496e7d71397fc8311898bde697914ff50f0606b1a278f10362d', packed='ccf6d303c0b66a67244033ae16df8f0a385cc9ecf6406a61e6a9cc01ab7d7a90'),
}


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_default_off_is_byte_identical(tmp_path, kind):
    assert golden_of(make_ds(tmp_path, kind)) == GOLDEN[kind]                                   # 새 인자를 주지 않음
    assert golden_of(make_ds(tmp_path / "z", kind, change_weight=0.0, hold_weight=0.0, eot_drop_synth=0.0)) == GOLDEN[kind]   # 0 을 명시


# ── C1 decision_positions(순수 함수) — 홀수 id = 단어 첫 토큰
NEXT, EMPTY, A, B, EOT, PAD, TR = 1000, 1001, 1002, 1003, 1004, 999, 1005
SP = Specials(next_audio=NEXT, empty_audio=EMPTY, spk=(A, B)); PRE = [1, A, 2, B, 3]               # prefix 의 슬롯 선언 태그는 문맥이 아니다
WS = lambda t: t % 2 == 1
ev = lambda t: (EOT, t, chunk_of(t + 0.24))


def at_tok(s, ps):
    return [(s["ids"][p], s["ids"][p + 1]) for p in ps]


def test_switch_and_hold_transcription():
    st = [[(11, 0.5), ev(1.1), (13, 2.6), (15, 2.9), ev(3.0), (17, 3.6), (18, 3.7)], [(21, 1.15), (23, 2.0), ev(2.1), (26, 3.65), (27, 4.0)]]
    eot = {(0, 1): 1.0, (1, 2): 1.0, (0, 4): 0.5}
    base = build_cst_sequence(st, eot, 60, PRE, PAD, SP, delay=2, decision_weight=2.0, word_start=WS)
    out = [t for t, inp in zip(base["ids"], base["is_input"]) if not inp and t != NEXT]
    assert out == [A, 11, B, 21, A, EOT, B, 23, EOT, A, 13, 15, EOT, 17, B, 26, A, 18, B, 27]     # B 가 A 의 <EOT> 청크에 먼저 → A 태그는 그 <EOT> 만을 위해 열린다
    sw, ho = decision_positions(base["ids"], base["is_input"], base["labels"], base["labels_alt"], base["src_at"], SP.spk, None, EOT, WS)
    assert at_tok(base, sw) == [(B, 21), (B, 23), (A, 13), (B, 27)] and [base["ids"][p] for p in ho] == [17]
    first = base["ids"].index(A, len(PRE)); opened = base["soft_pos"][0]
    assert first not in sw and base["ids"][opened: opened + 2] == [A, EOT] and opened not in sw                # 첫 태그·<SPK_A><EOT> 제외
    assert [base["ids"][p + 1] for p in range(len(PRE), len(base["ids"]) - 1) if base["ids"][p] in (A, B) and p not in sw and p != first and p != opened] == [26, 18]   # 단어 중간
    s = build_cst_sequence(st, eot, 60, PRE, PAD, SP, delay=2, decision_weight=2.0, word_start=WS, change_weight=3.0, hold_weight=1.5)
    assert {k: s[k] for k in base} == base | {"pos_weight": s["pos_weight"]} and s["n_switch"] == 4 and s["n_hold"] == 1
    assert [p for p, k in enumerate(s["dec_kind"]) if k == 1] == sw and [p for p, k in enumerate(s["dec_kind"]) if k == 2] == ho
    assert {p: w for p, w in enumerate(s["pos_weight"]) if w} == {**{p: 3.0 for p in sw}, **{p: 1.5 for p in ho}, **{d: 2.0 for d in base["soft_pos"]}}   # (e) soft 결정은 2.0 유지
    only = build_cst_sequence(st, eot, 60, PRE, PAD, SP, delay=2, decision_weight=2.0, word_start=WS, change_weight=3.0)
    assert only["dec_kind"] == s["dec_kind"] and {p for p, w in enumerate(only["pos_weight"]) if w} == set(sw) | set(base["soft_pos"])   # 머무름 0: 위치는 표시, 가중 없음
    assert build_cst_sequence(st, eot, 60, PRE, PAD, SP, delay=2, word_start=None, change_weight=3.0)["n_switch"] == 6   # word_start 없음: 단어 중간도 교대


def test_switch_and_hold_translation():
    st = [[(11, 0.5), (12, 1.0), (15, 3.1)], [(21, 1.6), (22, 2.0), (23, 2.6)], [(31, 0.6), (32, 1.0), ev(1.1), (35, 3.0)], [(41, 2.0), ev(2.1)]]   # 흐름 2·3 = A·B 번역
    s = build_cst4_sequence(st, {(2, 2): 1.0, (3, 1): 1.0}, 60, PRE, PAD, SP, TR, delay=2, word_start=WS, change_weight=3.0, hold_weight=1.5)
    out = [t for t, inp in zip(s["ids"], s["is_input"]) if not inp and t != NEXT]
    assert out == [A, 11, TR, 31, A, 12, TR, 32, EOT, B, 21, 22, TR, 41, EOT, B, 23, A, TR, 35, A, 15]
    assert at_tok(s, [p for p, k in enumerate(s["dec_kind"]) if k == 1]) == [(B, 21)]                     # B onset(<EOT> 뒤)만 교대
    assert at_tok(s, [p for p, k in enumerate(s["dec_kind"]) if k == 2]) == [(B, 23)]                     # 번역 <EOT> 뒤 같은 슬롯 재개 = 머무름
    tags = [p for p, t in enumerate(s["ids"]) if t in (A, B) and p >= len(PRE)]
    assert [(s["ids"][p + 1], s["dec_kind"][p]) for p in tags] == [(11, 0), (12, 0), (21, 1), (23, 2), (TR, 0), (15, 0)]   # 첫 태그·<TRANS> 뒤 재개·<SPK_A><TRANS> 제외
    o = build_cst4_sequence([[(11, 0.5)], [(21, 0.65)], [(31, 0.5), ev(0.6)], []], {(2, 1): 1.0}, 50, PRE, PAD, SP, TR, delay=2, word_start=WS, change_weight=3.0)
    d = o["soft_pos"][0]; assert o["ids"][d - 2: d + 2] == [B, 21, A, EOT] and o["dec_kind"][d - 2] == 1 and o["dec_kind"][d] == 0 and o["pos_weight"][d] == 0.0   # <SPK_x><EOT> 는 soft 결정 그대로


def test_flip_region_keeps_masks_and_weights_only_supervised_tags():
    from test_cst_dialogue import R, ST3, RC
    for rng, kind in ((R([0.0, 0.99], [0, 0, 0]), "missed/inrun"), (R([0.0, 0.0], [1, 0, 0]), "spurious/boundary")):
        s = build_cst_sequence(ST3, {}, 50, PRE, PAD, SP, delay=2, recover=RC, rng=rng, change_weight=3.0, hold_weight=1.5)
        f = s["flip"]; assert f["kind"] == kind
        assert all(s["labels"][p] == -100 and s["dec_kind"][p] == 0 and s["pos_weight"][p] == 0 for p in range(f["p_start"], f["p_end"]) if not s["is_input"][p])   # (f)
        assert all(s["labels"][p] == s["ids"][p] for p, k in enumerate(s["dec_kind"]) if k == 1)
    assert s["ids"][f["p_start"]] == B and s["dec_kind"][f["p_start"]] == 0                                  # 헛 태그(문맥에만)는 가중하지 않는다


# ── C1·C5 데이터셋(받아쓰기·번역 모드, 같은 언어·교차 언어, 실제·합성)
def seqs(ds, i, d, **kw):
    yield ds.cst_sequence(i, d, **kw)
    for tg in ds.tgt_options(i): yield ds.cst_sequence(i, d, tgt=tg, **kw)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_dataset_weights_pass_invariants(tmp_path, kind):
    off = make_ds(tmp_path, kind); on = make_ds(tmp_path / "w", kind, change_weight=3.0, hold_weight=1.5)
    n = dict(sw=0, ho=0, sw4=0, ho4=0)
    for i in range(len(on)):
        for d in on.delays:
            for a, b in zip(seqs(off, i, d), seqs(on, i, d)):
                assert on.problems_of(b) == [], (kind, i, d, b["tgt"], on.problems_of(b))      # (g) switch_w·hold_w 포함
                assert {k: b[k] for k in a if k != "pos_weight"} == {k: a[k] for k in a if k != "pos_weight"} and set(b) - set(a) == {"dec_kind", "n_switch", "n_hold", "dec_heads"}
                assert all(x == y or k in (1, 2) for x, y, k in zip(a["pos_weight"], b["pos_weight"], b["dec_kind"]))
                n["sw4" if b["tgt"] else "sw"] += b["n_switch"]; n["ho4" if b["tgt"] else "ho"] += b["n_hold"]
            r = random.Random(100 * i + d)
            for _ in range(3):
                s = on.cst_sequence(i, d, rng=r); assert on.problems_of(s) == [], (kind, i, d, s["flip"])   # 학습 경로(뒤집기·번역 모드·A/B 뒤집기)
    assert min(n.values()) > 0, n                                                               # 두 모드 모두 교대·머무름이 있다
    random.seed(0); b = [on[i] for i in range(4)]; assert all(x["dec_kind"].dtype == torch.int8 and x["dec_kind"].shape == x["ids"].shape for x in b)
    c = collate_cst(b); assert c["dec_kind"].shape == c["ids"].shape and c["n_switch"] == sum(x["n_switch"] for x in b) and c["n_hold"] == sum(x["n_hold"] for x in b)
    assert all(torch.equal(c["dec_kind"][j, : len(x["ids"])], x["dec_kind"]) and int(c["dec_kind"][j, len(x["ids"]):].abs().sum()) == 0 for j, x in enumerate(b))
    from vapasr.hf.packing import pack_batch
    p = pack_batch(c); assert p["dec_kind"].shape == p["ids"].shape and torch.equal(p["dec_kind"][0], torch.cat([x["dec_kind"] for x in b]))
    random.seed(0); c0 = collate_cst([off[i] for i in range(4)]); assert "dec_kind" not in c0 and "n_switch" not in c0 and "dec_kind" not in pack_batch(c0)
    with pytest.raises(ValueError): make_ds(tmp_path / "bad", kind, change_weight=-1.0)


@pytest.mark.parametrize("kind", ["synth", "xl_synth"])
def test_eot_drop_synthetic_windows(tmp_path, kind):
    ds = make_ds(tmp_path, kind); on = make_ds(tmp_path / "d", kind, eot_drop_synth=1.0); n_ghost = n_changed = 0
    for i in range(len(ds)):
        for d in ds.delays:
            for s0, s1 in zip(seqs(ds, i, d), seqs(ds, i, d, drop_eot=True)):
                four = bool(s1["tgt"]); nn = len(s1["streams"]) // 2 if four else len(ds.spk_ids)
                assert ds.problems_of(s1) == [] and ds.eot_id not in s1["ids"] and s1["n_eot"] == 0 and s1["win"]["eot_dropped"] == s1["win"]["eot"] == s0["win"]["eot"] > 0
                bad = read_back4(s1["ids"], s1["is_input"], ds.csp, ds.trans_id, ds.eot_id, n=nn)[2] if four else read_back(s1["ids"], s1["is_input"], ds.csp, ds.eot_id)[1]
                assert "dup_tag" not in bad and bad == []
                g0 = read_back4(s0["ids"], s0["is_input"], ds.csp, ds.trans_id, ds.eot_id, n=nn)[0] if four else [[t for t in x if t != ds.eot_id] for x in read_back(s0["ids"], s0["is_input"], ds.csp, ds.eot_id)[0]]
                g1 = read_back4(s1["ids"], s1["is_input"], ds.csp, ds.trans_id, ds.eot_id, n=nn)[0] if four else read_back(s1["ids"], s1["is_input"], ds.csp, ds.eot_id)[0]
                assert g0 == g1                                                                  # 흐름별 글자는 그대로, <EOT> 만 빠진다
                nx = lambda s: {s["chunk_at"][p]: s["labels"][p] for p, t in enumerate(s["ids"]) if t == ds.csp.next_audio and not s["is_input"][p] and s["chunk_at"][p] < s["K"]}
                assert text_labels(s0) == text_labels(s1)                                         # 글자 label 은 그대로
                ch = {k for k, v in nx(s0).items() if nx(s1)[k] != v}; gk = {s1["chunk_at"][p] for p in s1["ghost_pos"]}
                assert ch <= gk and all(nx(s1)[k] == -100 for k in gk) and len(s1["eot_ghost"]) == s1["win"]["eot_dropped"]   # 바뀐 <NEXT_AUDIO> 는 ghost 라운드뿐
                assert all(s0["labels"][p] == -100 for p in s0["eot_pos"])                         # 뺀 <EOT> 는 원래 감독이 없었다(합성)
                n_ghost += len(s1["ghost_pos"]); n_changed += len(ch)
            r0, r1 = random.Random(i), random.Random(i); a, b = ds.cst_sequence(i, d, rng=r0), on.cst_sequence(i, d, rng=r1)
            assert b["win"]["eot_dropped"] == b["win"]["eot"] and on.problems_of(b) == [] and a["win"].get("eot_dropped") is None   # p = 1: 늘 뺀다
    assert n_ghost > 0 and n_changed > 0
    random.seed(1); it = [on[i] for i in range(3)]; assert all(x["n_eot"] == 0 and x["n_eot_drop"] > 0 for x in it) and collate_cst(it)["n_eot_drop"] == sum(x["n_eot_drop"] for x in it)


def text_labels(s):
    """흐름별 글자 원소의 (tid, label) — <EOT> 를 빼면 흐름 안 index 가 밀리므로 순서로 맞춘다."""
    at = {(q[0], q[1]): p for p, q in enumerate(s["src_at"]) if q is not None}
    return [[(x[0], s["labels"][at[(k, j)]]) for j, x in enumerate(ev_) if len(x) == 2] for k, ev_ in enumerate(s["streams"])]


class CountRng(random.Random):
    """random() 호출 수를 센다(난수 소비 비교)."""
    def __init__(self, seed): super().__init__(seed); self.n = 0
    def random(self): self.n += 1; return super().random()


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_eot_drop_rng_consumption_and_real_dialogues_untouched(tmp_path, kind):
    off = make_ds(tmp_path, kind); half = make_ds(tmp_path / "h", kind, eot_drop_synth=0.5); synth = KINDS[kind]["synth"]
    for i in range(len(off)):
        for d in off.delays:
            r0, r1 = CountRng(i), CountRng(i); a = off.cst_sequence(i, d, rng=r0); b = half.cst_sequence(i, d, rng=r1)
            assert r1.n == r0.n + int(synth)                                                    # 합성 창만 한 번 더 뽑는다
            if not synth: assert b == a and r1.getstate() == r0.getstate()                      # (c) 실제 대화는 바이트 동일
            assert off.cst_sequence(i, d) == half.cst_sequence(i, d) and off.problems_of(half.cst_sequence(i, d, drop_eot=True)) == []   # rng 없는 평가·점검 경로는 뽑지 않는다
            if not synth: assert off.cst_sequence(i, d, drop_eot=True) == off.cst_sequence(i, d) | {"win": dict(off.cst_sequence(i, d)["win"], eot_dropped=0)}
    with pytest.raises(ValueError): make_ds(tmp_path / "bad", kind, eot_drop_synth=1.5)


def test_eot_drop_with_switch_weight(tmp_path):
    """(d) C1 + C5: <EOT> 없이 오는 교대 태그도 가중을 받는다."""
    ds = make_ds(tmp_path, "xl_synth", change_weight=3.0, hold_weight=1.5); n = 0
    for i in range(len(ds)):
        for d in ds.delays:
            for s in seqs(ds, i, d, drop_eot=True):
                assert ds.problems_of(s) == [] and ds.eot_id not in s["ids"] and s["n_hold"] == 0   # <EOT> 가 없으면 머무름 결정도 없다
                n += s["n_switch"]; assert all(s["pos_weight"][p] == 3.0 for p, k in enumerate(s["dec_kind"]) if k == 1)
    assert n > 0


@pytest.mark.parametrize("synth", [False, True])
def test_all_options_with_training_rng(tmp_path, synth):
    """뒤집기(recover p=1)·EOT-문맥 dropout(p=1)·교대·머무름 가중·번역 모드를 함께 켜도 학습 rng 경로의 불변식이 지켜진다."""
    q, tr = make_dlg(tmp_path, "c_all", synth=synth)
    ds = CstDialogueDataset([q], tokw(), delays=(2, 4), window_s=(12.0, 20.0), hop_s=3.0, seed=1, decision_weight=2.0, trans=tr, ast_frac=0.5,
                            recover=Recover(p=1.0), change_weight=3.0, hold_weight=1.5, eot_drop_synth=1.0)
    kinds, n_drop = set(), 0
    for i in range(len(ds)):
        for d in ds.delays:
            r = random.Random(1000 + 10 * i + d)
            for _ in range(8):
                s = ds.cst_sequence(i, d, rng=r); assert ds.problems_of(s) == [], (i, d, s["tgt"], s["flip"] and s["flip"]["kind"], ds.problems_of(s))
                kinds.add((bool(s["tgt"]), s["flip"] is not None)); n_drop += s["win"].get("eot_dropped", 0)
    assert kinds == {(False, True), (True, False)} and (n_drop > 0) == synth                     # 받아쓰기(뒤집기)·번역 모드 둘 다, 실제 대화는 빼지 않는다


def test_window_streams_drop_returns_ghosts():
    u = [Utterance("x", 0.0, 2.0, tokens=[(11, 1.0), (12, 2.0)], utt_id="x0"), Utterance("y", 2.5, 4.0, tokens=[(21, 3.0), (22, 4.0)], utt_id="y0"),
         Utterance("x", 4.5, 6.0, tokens=[(13, 5.0)], utt_id="x1")]
    real = Dialogue(conv_id="r", corpus="t", lang="German", split="train", duration_s=20.0, speakers=["x", "y"], utterances=u)
    syn = Dialogue(conv_id="s", corpus="t", lang="German", split="train", duration_s=20.0, speakers=["x", "y"], utterances=u, meta=dict(mask_eot=True))
    e_r, e_s = dialogue_episodes(real), dialogue_episodes(syn)
    assert window_streams(e_r, ["x", "y"], 0.0, 20.0, EOT, drop_eot=True)[:3] == window_streams(e_r, ["x", "y"], 0.0, 20.0, EOT)[:2] + (dict(window_streams(e_r, ["x", "y"], 0.0, 20.0, EOT)[2], eot_dropped=0),)
    st, eot_p, w, gh = window_streams(e_s, ["x", "y"], 0.0, 20.0, EOT, drop_eot=True)
    assert all(len(x) == 2 for ev_ in st for x in ev_) and eot_p == {} and w["eot"] == w["eot_dropped"] == 3 and len(window_streams(e_s, ["x", "y"], 0.0, 20.0, EOT)) == 3
    assert gh == [(0, chunk_of(2.24), [(0, 1)]), (1, chunk_of(4.24), [(1, 1)]), (0, chunk_of(6.24), [(0, 2)])]
    tr = lambda ep: ([(31, ep.end, False)], [], True)
    st4, _, info = window_streams4(e_s, ["x", "y"], 0.0, 20.0, EOT, tr, drop_eot=True)
    assert info["ghost"][0] == (0, chunk_of(2.24), [(0, 1), (2, 0)]) and info["st"]["eot_dropped"] == 3 and "ghost" not in window_streams4(e_s, ["x", "y"], 0.0, 20.0, EOT, tr)[2]
    s = build_cst_sequence(st, {}, 100, PRE, PAD, SP, delay=4, eot_ghost=gh)                    # ghost 청크 = max(k_min, 꼬리 방출 청크)
    want = [max(k, int(t / 0.08) + 4) for k, t in ((chunk_of(2.24), 2.0), (chunk_of(4.24), 4.0), (chunk_of(6.24), 5.0))]   # 버킷 규칙(build_interleaved_events)
    assert [s["chunk_at"][p] for p in s["ghost_pos"]] == want and all(s["ids"][p] == NEXT and s["labels"][p] == -100 for p in s["ghost_pos"])
    s2 = build_cst_sequence(st, {}, 60, PRE, PAD, SP, delay=4, eot_ghost=gh); assert len(s2["ghost_pos"]) == 2                 # K 60(4.8 s): 셋째(6 s 끝)는 flush — 그대로
    assert {k: v for k, v in s2.items() if k not in ("labels", "eot_ghost", "ghost_pos")} == {k: v for k, v in build_cst_sequence(st, {}, 60, PRE, PAD, SP, delay=4).items() if k != "labels"}



# ── no_eot(2026-10-23 — 턴 예측이 주 목표가 아니어서 <EOT> 를 정답·문맥 모두에서 뺀다, 머무름 기준점 = 뺀 <EOT> 의 청크 끝)
def test_window_streams_no_eot_ghosts_every_episode():
    u = [Utterance("x", 0.0, 2.0, tokens=[(11, 1.0), (12, 2.0)], utt_id="x0"), Utterance("y", 2.5, 4.0, tokens=[(21, 3.0), (22, 4.0)], utt_id="y0"),
         Utterance("x", 4.5, 6.0, tokens=[(13, 5.0)], utt_id="x1")]
    e_r = dialogue_episodes(Dialogue(conv_id="r", corpus="t", lang="German", split="train", duration_s=20.0, speakers=["x", "y"], utterances=u))
    st, eot_p, w, gh = window_streams(e_r, ["x", "y"], 0.0, 20.0, EOT, no_eot=True)                 # 실제 대화도 모두 뺀다
    assert eot_p == {} and w["eot"] == w["eot_dropped"] == 3 and all(len(x) == 2 for ev_ in st for x in ev_)
    assert gh == [(0, chunk_of(2.24), [(0, 1)]), (1, chunk_of(4.24), [(1, 1)]), (0, chunk_of(6.24), [(0, 2)])]
    tr = lambda ep: ([(31, ep.end, False)], [], True)
    assert window_streams4(e_r, ["x", "y"], 0.0, 20.0, EOT, tr, no_eot=True)[2]["ghost"][0] == (0, chunk_of(2.24), [(0, 1), (2, 0)])
    s = build_cst_sequence(st, {}, 100, PRE, PAD, SP, delay=4, eot_ghost=gh, no_eot=True, change_weight=3.0, hold_weight=1.5)
    assert all(s["ids"][p] == NEXT and s["labels"][p] == NEXT for p in s["ghost_pos"])          # 가리지 않는다(가르칠 <EOT> 대안이 없다)
    assert s["ghost_anchor"] == {p: g[0] for p, g in zip(s["ghost_pos"], gh)} and EOT not in s["ids"] and s["n_eot"] == 0
    assert [p for p, k in enumerate(s["dec_kind"]) if k == 2] == [] and s["n_switch"] == 2     # x 의 둘째 발화(13)는 y 뒤라 교대, 머무름은 없다
    u2 = [Utterance("x", 0.0, 2.0, tokens=[(11, 1.0), (12, 2.0)], utt_id="x0"), Utterance("x", 2.6, 3.4, tokens=[(13, 3.0), (15, 3.4)], utt_id="x1"),
          Utterance("y", 3.5, 5.0, tokens=[(21, 4.2), (22, 4.6)], utt_id="y0")]                 # x 가 0.6 s 쉬고 이어 말한다(머무름) → y 교대
    e2 = dialogue_episodes(Dialogue(conv_id="h", corpus="t", lang="German", split="train", duration_s=20.0, speakers=["x", "y"], utterances=u2))
    sa, pa, _ = window_streams(e2, ["x", "y"], 0.0, 20.0, EOT); sb, _, _, gb = window_streams(e2, ["x", "y"], 0.0, 20.0, EOT, no_eot=True)
    ref = build_cst_sequence(sa, pa, 100, PRE, PAD, SP, delay=4, change_weight=3.0, hold_weight=1.5)
    nb = build_cst_sequence(sb, {}, 100, PRE, PAD, SP, delay=4, eot_ghost=gb, no_eot=True, change_weight=3.0, hold_weight=1.5)
    kinds = lambda x: [(x["ids"][p], k) for p, k in enumerate(x["dec_kind"]) if k]
    assert kinds(nb) == kinds(ref) == [(13, 2), (B, 1)]                                           # 같은 자리에서 머무름(13)·교대(<SPK_B>)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_no_eot_drops_every_eot(tmp_path, kind):
    ref = make_ds(tmp_path, kind, change_weight=3.0, hold_weight=1.5); ds = make_ds(tmp_path / "n", kind, change_weight=3.0, hold_weight=1.5, no_eot=True)
    n = dict(hold=0, hold_ref=0, anchor=0, next_trained=0)
    for i in range(len(ds)):
        for d in ds.delays:
            for r, s in zip(seqs(ref, i, d), seqs(ds, i, d)):
                assert ds.problems_of(s) == [], (kind, i, d, s["tgt"], ds.problems_of(s))
                assert ds.eot_id not in s["ids"] and s["n_eot"] == 0 and not s["soft_pos"] and s["win"]["eot_dropped"] == s["win"]["eot"] == r["win"]["eot"]
                assert [[x[0] for x in ev_ if len(x) == 2] for ev_ in s["streams"]] == [[x[0] for x in ev_ if len(x) == 2] for ev_ in r["streams"]]
                assert text_labels(s) == text_labels(r)                                         # 글자·label 은 그대로, <EOT> 만 빠진다
                n["hold"] += s["n_hold"]; n["hold_ref"] += r["n_hold"]; n["anchor"] += len(s["ghost_anchor"])
                n["next_trained"] += sum(s["labels"][p] == s["ids"][p] for p in s["ghost_pos"])
            rr = random.Random(100 * i + d)
            for _ in range(3):
                s = ds.cst_sequence(i, d, rng=rr); assert ds.problems_of(s) == [] and ds.eot_id not in s["ids"], (kind, i, d, s["flip"])   # 학습 경로
    assert n["anchor"] > 0 and n["next_trained"] > 0 and n["hold"] > 0 and 0.5 * n["hold_ref"] <= n["hold"] <= 1.5 * n["hold_ref"], n
    random.seed(0); b = [ds[i] for i in range(4)]; assert all(x["n_eot"] == 0 and ds.eot_id not in x["ids"].tolist() for x in b)
    assert "dec_kind" in collate_cst(b)


@pytest.mark.parametrize("synth", [False, True])
def test_no_eot_with_all_options(tmp_path, synth):
    """뒤집기(recover p=1)·C5(p=1)·교대·머무름 가중·번역 모드와 함께 켜도 불변식이 지켜지고 <EOT> 가 남지 않는다."""
    q, tr = make_dlg(tmp_path, "c_all", synth=synth)
    ds = CstDialogueDataset([q], tokw(), delays=(2, 4), window_s=(12.0, 20.0), hop_s=3.0, seed=1, decision_weight=2.0, trans=tr, ast_frac=0.5,
                            recover=Recover(p=1.0), change_weight=3.0, hold_weight=1.5, eot_drop_synth=1.0, no_eot=True)
    kinds = set()
    for i in range(len(ds)):
        for d in ds.delays:
            r = random.Random(1000 + 10 * i + d)
            for _ in range(8):
                s = ds.cst_sequence(i, d, rng=r); assert ds.problems_of(s) == [] and ds.eot_id not in s["ids"], (i, d, s["tgt"], s["flip"] and s["flip"]["kind"], ds.problems_of(s))
                kinds.add((bool(s["tgt"]), s["flip"] is not None))
    assert kinds == {(False, True), (True, False)}


# ── 모델 forward: dec_kind 는 통계만(손실 불변)
def test_forward_decision_stats_do_not_change_loss():
    from test_semcommit_tokens_loss import tiny_model
    m, tok = tiny_model(); m.add_semantic_tokens(tok, turn=True); m.eval(); NX = m.config.sp_ids["<NEXT_AUDIO>"]; SA, SB = m.config.sp_ids["<SPK_A>"], m.config.sp_ids["<SPK_B>"]
    ids = torch.tensor([[1, 0, SA, 51, NX, 0, SB, 61, NX, 0, 53, NX]]); is_audio = ids == 0; chunk_of_ = torch.tensor([[-1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1]])
    labels = ids.clone(); labels[is_audio] = -100; labels[:, 0] = -100
    dk = torch.zeros_like(ids, dtype=torch.int8); dk[0, 6] = 1; dk[0, 10] = 2
    pw = torch.zeros(1, 12); pw[0, 6] = 3.0; pw[0, 10] = 1.5
    x = dict(ids=ids, is_audio=is_audio, chunk_of=chunk_of_, labels=labels, mask=torch.ones(1, 12, dtype=torch.long), feats=torch.randn(1, 1, 3, 4), pos_weight=pw, next_weight=0.3)
    a, b = m(**x), m(**x, dec_kind=dk)
    assert torch.equal(a.loss, b.loss) and a.top1_switch is None and a.n_switch is None
    with torch.no_grad():
        h = m.thinker.model(inputs_embeds=m.build(x["feats"], ids, is_audio, chunk_of_)).last_hidden_state[0]; lg = m.thinker.lm_head(h).float()
    p_sw, p_ho = int(lg[5].argmax()), int(lg[9].argmax())                                       # 위치 6·10 의 예측은 5·9 의 hidden
    special = set(m.config.sp_ids.values()) | set(m.config.blocked_ids)
    assert b.n_switch == 1 and b.top1_switch == float(p_sw == SB) and b.switch_miss == float(p_sw not in special) and b.switch_wrong_slot == float(p_sw == SA)
    assert b.n_hold == 1 and b.top1_hold == float(p_ho == 53) and abs(b.loss_switch - float(torch.nn.functional.cross_entropy(lg[5:6], torch.tensor([SB])))) < 1e-5
    c = m(**x, dec_kind=torch.zeros_like(dk)); assert torch.equal(c.loss, a.loss) and c.top1_switch is None and c.top1_hold is None   # 위치가 없으면 키도 None
    src = (ROOT / "vapasr/hf/trainer.py").read_text(); i = src.index("def compute_loss")             # Trainer 가 모델로 넘긴다(transformers.trainer import 없이 소스로 확인)
    assert '"dec_kind") if k in inputs}' in src[i: src.index("def log", i)]


def test_packed_forward_passes_dec_kind():
    """packing(ROW_KEYS 의 dec_kind)을 거쳐도 통계가 패딩 배치와 같고, 손실은 dec_kind 유무와 무관하다."""
    from test_packing import _tiny, _batch
    from vapasr.hf.packing import pack_batch
    m = _tiny("sdpa"); b = _batch([19, 31, 11], [6, 12, 4], seed=3); dk = torch.zeros_like(b["ids"], dtype=torch.int8)
    for i in range(3):
        lab = (b["labels"][i] != -100).nonzero()[:, 0].tolist(); dk[i, lab[1]] = 1; dk[i, lab[2]] = 2; dk[i, lab[3]] = 1
    run = lambda x: m(**{k: v for k, v in x.items() if k not in ("lang", "n_sem", "pack_n", "pack_kmax")})
    o0, o1, o2 = run(b), run(dict(b, dec_kind=dk)), run(pack_batch(dict(b, dec_kind=dk)))
    assert torch.equal(o0.loss, o1.loss) and torch.allclose(o1.loss, o2.loss, atol=1e-5, rtol=1e-5) and o0.n_switch is None
    assert o1.n_switch == o2.n_switch == 6 and o1.n_hold == o2.n_hold == 3 and pack_batch(dict(b, dec_kind=dk))["dec_kind"].shape == pack_batch(b)["ids"].shape
    for k in ("top1_switch", "switch_miss", "switch_wrong_slot", "top1_hold"): assert getattr(o1, k) == getattr(o2, k), k
    assert abs(o1.loss_switch - o2.loss_switch) < 1e-5


# ── stage3_train: 셋 키 분류·--set-phone·플래그
OLD_KEYS = ["dlg_ko", "dlg_en", "dlg_de", "dlg_mt", "sim_ko", "sim_en", "sim_de", "sim_en-de", "sim_en-ko", "sim_ko-en", "sim_de-en", "rep_ko", "rep_en", "rep_de", "rep_chde"]


def test_set_kind_keeps_old_classification():
    import re
    from vapasr.hf.stage3 import set_kind
    for k in OLD_KEYS + ["sim_en-de-", "sim_en-DE", "sim_en-de_tel", "simx_en-de", "sim_eng-de"]:
        assert (set_kind(k) == "xl") == bool(re.fullmatch(r"sim_[a-z]{2}-[a-z]{2}", k)), k     # 예전 정규식과 같은 분기
    assert [set_kind(k) for k in ("sim_en-de-tel", "sim_ko-en-tel2", "dlg_ko", "sim_ko", "rep_chde", "foo")] == ["xl", "xl", "dlg", "sim", "rep", None]


def test_parse_set_phone():
    from vapasr.hf.stage3 import parse_set_phone
    assert parse_set_phone("") == {} and parse_set_phone("sim_en-de-tel:0, dlg_en:0.3") == {"sim_en-de-tel": 0.0, "dlg_en": 0.3}
    for bad in ("rep_en:0.5", "sim_en-de", "sim_en-de:1.5", "dlg_en:0.1,dlg_en:0.2", ":0.1"):
        with pytest.raises(ValueError): parse_set_phone(bad)


def stage3(args, tmp_path):
    """stage3_train.py 를 인자 검사까지만(입력 파일 없이) 돌린다 → (종료 코드, 출력)."""
    p = subprocess.run([sys.executable, str(ROOT / "experiments/stage3_train.py"), "--init", str(tmp_path / "none"), "--out-dir", str(tmp_path / "o"),
                        "--data", str(tmp_path), "--split", str(tmp_path / "s.json")] + args, capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    return p.returncode, p.stdout + p.stderr


def test_stage3_train_flags(tmp_path):
    rc, out = stage3(["--help"], tmp_path)
    assert rc == 0 and all(f in out for f in ("--change-tag-weight", "--hold-tag-weight", "--decision-weight-sets", "--eot-drop-synth", "--eot-drop-sets", "--set-phone"))
    for args, msg in ((["--change-tag-weight", "-1"], "--change-tag-weight"), (["--eot-drop-synth", "1.5"], "--eot-drop-synth"), (["--set-phone", "rep_en:0.5"], "--set-phone"),
                      (["--set-phone", "sim_en-de-tel:0"], "--weights 에 없다"), (["--decision-weight-sets", "nope"], "invalid choice"), (["--eot-drop-sets", "all"], "invalid choice")):
        rc, out = stage3(args, tmp_path); assert rc != 0 and msg in out, (args, out[-500:])


# ── slurm/submit-stage3-train.sh
def test_submit_script_default_command_unchanged(tmp_path):
    for k, args in enumerate(SUBMIT_ARGS):
        R, sh = submit_env(tmp_path / str(k)); r = run_submit(R, sh, args)
        assert r[0] == 0 and hashlib.sha256(json.dumps(list(r)).encode()).hexdigest() == SUBMIT_GOLDEN[k], r


def test_submit_script_v3_flags(tmp_path):
    R, sh = submit_env(tmp_path); base = run_submit(R, sh, SUBMIT_ARGS[1])
    new = ["--change-tag-weight", "3", "--hold-tag-weight", "1.5", "--decision-weight-sets", "xl", "--eot-drop-synth", "0.5", "--eot-drop-sets", "synth",
           "--set-phone", "sim_en-de:0,dlg_en:0.2"]
    rc, out, err = run_submit(R, sh, SUBMIT_ARGS[1][:-1] + new + ["--dry-run"]); assert rc == 0, err
    want = " --change-tag-weight 3 --hold-tag-weight 1.5 --decision-weight-sets xl --eot-drop-synth 0.5 --eot-drop-sets synth --set-phone sim_en-de:0%2Cdlg_en:0.2"
    assert want in out and out.replace(want, "") == base[1]                                     # 새 인자만 덧붙는다
    for bad in (["--change-tag-weight", "-1"], ["--hold-tag-weight", "x"], ["--decision-weight-sets", "some"], ["--eot-drop-synth", "1.5"], ["--eot-drop-sets", "all"],
                ["--set-phone", "rep_en:0.5"], ["--set-phone", "sim_en-de:2"], ["--set-phone", "sim_ko-en:0"]):    # 마지막: --weights 에 없는 셋
        rc, out, err = run_submit(R, sh, SUBMIT_ARGS[1][:-1] + bad + ["--dry-run"]); assert rc != 0 and not out, (bad, err)


def test_unspaced_onset_is_a_decision(tmp_path):
    """앞 공백 없이 시작하는 단위(words 행 합성 발화의 첫 토큰 — 짝수 id)도 episode 첫 원소면 교대·머무름 결정이다(검토 2026-10-21: 영어 합성 단위 onset 이 가중에서 빠지던 결함).
    y 턴 첫 토큰 22 → 교대, x 가 쉼 뒤 같은 화자로 다시 시작하는 첫 토큰 14 → <EOT> 뒤 머무름. 단어 중간(짝수) 토큰은 여전히 아니다."""
    SR = 16000; dur = 12.0; w = str(tmp_path / "u.wav"); t = np.arange(int(dur * SR)) / SR
    sf.write(w, np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, 1).astype(np.float32), SR)
    U = [("x", 0.5, 2.0, [(11, 1.0), (13, 2.0)]), ("y", 2.1, 4.0, [(22, 3.0), (23, 4.0)]), ("x", 4.1, 6.0, [(15, 5.0)]), ("x", 6.6, 8.0, [(14, 7.0), (17, 8.0)])]
    utts = [Utterance(sp, a, e, "t", "t", tokens=toks, utt_id=f"u{j}") for j, (sp, a, e, toks) in enumerate(U)]
    d = Dialogue(conv_id="u", corpus="test", lang="Korean", split="train", duration_s=dur, speakers=["x", "y"], utterances=utts,
                 channels={"x": ChannelRef(path=f"{w}#ch0"), "y": ChannelRef(path=f"{w}#ch1")}, meta={})
    q = str(tmp_path / "u.refined.dialogues.jsonl"); open(q, "w").write(d.to_json() + "\n")
    ds = CstDialogueDataset([q], tokw(), delays=(2,), windows=[("u", 0.0, 10.0)], change_weight=3.0, hold_weight=1.5)
    s = ds.cst_sequence(0, 2); ids = s["ids"]; nxt = lambda p: next(ids[j] for j in range(p + 1, len(ids)) if not s["is_input"][j] and ids[j] != ds.csp.next_audio)
    sw = [p for p, k in enumerate(s["dec_kind"]) if k == 1]; ho = [p for p, k in enumerate(s["dec_kind"]) if k == 2]
    assert [nxt(p) for p in sw] == [22, 15] and [ids[p] for p in ho] == [14] and ds.problems_of(s) == []
    hd = set(); window_streams(ds.eps["u"], ["x", "y"], 0.0, 10.0, ds.eot_id, heads=hd)
    assert hd == {(0, 0), (1, 0), (0, 3), (0, 5)}                                                  # x: 11·13·<EOT>·15·<EOT>·14 … — episode 첫 원소만
    sw0, _ = decision_positions(ids, s["is_input"], s["labels"], s["labels_alt"], s["src_at"], ds.spk_ids, ds.trans_id, ds.eot_id, ds._ws())
    assert [nxt(p) for p in sw0] == [15]                                                            # heads 없이는 22 onset 을 놓친다(고치기 전 동작)
