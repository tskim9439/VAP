"""V3 R4 — 교차 언어 창 슬롯을 먼저 말한 순서로(xl_order 'arrival'): prefix 는 두 언어만(cst_head_text_xl), 슬롯 언어·번역 방향은 모델이 정한다.
기본('lang')은 예전과 같다. 평가: cst_stage4_eval.score4 xl_perm(두 슬롯 순열), cst_stage4_taxi.map_slots(창마다 슬롯 ↔ 언어 짝),
온셋 분류(cst_attr_turns.emitted_words 창별 슬롯 언어, taxi_v3_probe 가 run 의 짝을 받아 씀). CPU — cstbench 가 없으면 normalize 만 가짜로."""
import os, random, re, sys, types

import pytest

from vapasr.data.cst_dialogue import cst_head_text, cst_head_text_xl, window_speakers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
sf = pytest.importorskip("soundfile")
from test_cst_decision import make_ds, seqs  # noqa: E402


def test_head_text_xl_is_order_free():
    assert cst_head_text_xl(["German", "English"]) == cst_head_text_xl(["English", "German"]) == "language English German<asr_text>"
    assert cst_head_text_xl(["German", "English"], True) == "language English German\ntranslate auto<asr_text>"
    assert cst_head_text(["Korean", "Korean"]).startswith("language <SPK_A>Korean")                    # 같은 언어 형식은 그대로


@pytest.mark.parametrize("kind", ["xl_synth", "xl_real"])
def test_default_equals_lang(tmp_path, kind):
    a = make_ds(tmp_path / "a", kind); b = make_ds(tmp_path / "b", kind, xl_order="lang")
    for i in range(len(a)):
        for d in a.delays:
            for x, y in zip(seqs(a, i, d), seqs(b, i, d)): assert x["ids"] == y["ids"] and x["labels"] == y["labels"] and x["speakers"] == y["speakers"]


@pytest.mark.parametrize("kind", ["xl_synth", "xl_real"])
def test_arrival_slots_prefix_and_targets(tmp_path, kind):
    ds = make_ds(tmp_path, kind, xl_order="arrival", change_weight=3.0, hold_weight=1.5)
    n_tr = 0
    for i in range(len(ds)):
        cid, t0, L = ds.items[i]; first = window_speakers(ds.eps[cid], t0, L)
        r = random.Random(7); st = r.getstate(); spk, langs = ds.slots(i, r); assert r.getstate() == st      # arrival 은 난수를 뽑지 않는다
        if first: assert spk[0] == first[0]                                                          # 먼저 말한 화자 = A
        assert sorted(spk) == ["x", "y"] and langs == [{"x": "Korean", "y": "English"}[s] for s in spk]
        for d in ds.delays:
            for q in seqs(ds, i, d):
                assert ds.problems_of(q) == [], (i, d, q["tgt"], ds.problems_of(q))
                head = cst_head_text_xl(q["langs"], bool(q["tgt"]))
                want = ds._pre + ds.tok(head, add_special_tokens=False)["input_ids"] + [ds.delay_ids[d]]
                assert q["ids"][: len(want)] == want                                                  # 슬롯·언어를 묶지 않은 prefix
                if q["tgt"]: assert list(q["tgt"]) == [q["langs"][1], q["langs"][0]]; n_tr += 1         # 각 슬롯은 상대 언어로 번역
    assert n_tr > 0
    assert ds.cst_prefix(["Korean", "English"], 2, ["English", "Korean"]) == ds.cst_prefix(["English", "Korean"], 2, ["Korean", "English"])


def test_score4_xl_perm():
    from cst_stage4_eval import score4
    q = dict(langs=["English", "German"], tgt=["German", "English"])
    ref = [[("hello", 1.0), ("world", 1.2)], [("guten", 2.0), ("tag", 2.2)]]; hyp = [ref[1], ref[0]]          # 가설 슬롯이 뒤바뀜
    a = score4(None, q, ref, [[], []], [], hyp, [[], []], [], 10, lambda x: x)
    b = score4(None, q, ref, [[], []], [], hyp, [[], []], [], 10, lambda x: x, xl_perm=True)
    assert a["cp"] == a["slot"] > 0 and a["perm"] == [0, 1]
    assert b["cp"] == 0 and b["slot"] == a["slot"] and b["perm"] == [1, 0] and b["swap"]
    c = score4(None, q, ref, [[], []], [], ref, [[], []], [], 10, lambda x: x, xl_perm=True)
    assert c["cp"] == 0 and c["perm"] == [0, 1] and not c["swap"]                                    # 맞으면 항등(같으면 항등 우선)


@pytest.fixture
def norm(monkeypatch):
    """cstbench.evaluate.normalize — 없으면(로컬) 소문자·문장부호 제거로 대신한다(map_slots 는 normalize 만 쓴다)."""
    try:
        import cstbench.evaluate  # noqa: F401
    except ImportError:
        m, e = types.ModuleType("cstbench"), types.ModuleType("cstbench.evaluate"); e.normalize = lambda s, lang=None: re.sub(r"[^\w' ]+", " ", s.lower()).strip(); m.evaluate = e
        monkeypatch.setitem(sys.modules, "cstbench", m); monkeypatch.setitem(sys.modules, "cstbench.evaluate", e)


def test_map_slots(norm):
    from cst_stage4_taxi import map_slots, rule_slots
    rec = dict(turns=[dict(start_s=0.0, end_s=1.0, lang="German", transcript="Guten Tag"), dict(start_s=1.2, end_s=2.0, lang="English", transcript="Hello there"),
                      dict(start_s=50.0, end_s=51.0, lang="English", transcript="outside window")])
    de, en = [("guten", 0.9), ("tag", 1.0)], [("hello", 1.8), ("there", 2.0)]
    assert map_slots(rec, 0.0, 40.0, [de, en], ["English", "German"]) == ["German", "English"]       # 슬롯 0 이 독일어
    assert map_slots(rec, 0.0, 40.0, [en, de], ["English", "German"]) == ["English", "German"]
    assert map_slots(rec, 0.0, 40.0, [[], []], ["English", "German"]) == ["German", "English"]       # 같으면 창 안 첫 턴 언어가 슬롯 0
    assert map_slots(rec, 0.0, 40.0, [de + en, []], ["English", "German"]) == ["German", "English"]  # 한 슬롯에 다 몰려도 같다
    assert map_slots(rec, 45.0, 60.0, [[], []], ["English", "German"]) == ["English", "German"]      # 창 안 첫 턴이 영어
    assert map_slots(rec, 100.0, 120.0, [[], []], ["English", "German"]) == ["English", "German"]    # 턴이 없으면 langs 순서
    assert rule_slots(rec, 0.0, 40.0, ["English", "German"]) == ["German", "English"] and rule_slots(rec, 45.0, 60.0, ["English", "German"]) == ["English", "German"]


def test_emitted_words_window_langs():
    import cst_attr_turns as A
    from test_cst_attr_turns import WordTok
    tok = WordTok(); spk = tuple(tok.convert_tokens_to_ids(t) for t in ("<SPK_A>", "<SPK_B>")); tr, eot = tok.convert_tokens_to_ids("<TRANS>"), tok.convert_tokens_to_ids("<EOT>")
    em = [(0, spk[0]), (1, tok.convert_tokens_to_ids("hallo")), (2, spk[1]), (3, tok.convert_tokens_to_ids("hello"))]
    a, _ = A.emitted_words(tok, [(0.0, em), (10.0, em)], spk, tr, eot, ["German", "English"])
    b, _ = A.emitted_words(tok, [(0.0, em), (10.0, em, ["English", "German"])], spk, tr, eot, ["German", "English"])
    assert [x[1] for x in a] == ["German", "English", "German", "English"] and [x[1] for x in b] == ["German", "English", "English", "German"]   # 둘째 창만 창별 언어
    assert [x[:1] + x[2:] for x in a] == [x[:1] + x[2:] for x in b]


def test_probe_arrival_uses_window_slot_map(tmp_path, monkeypatch, norm):
    """먼저 말한 화자 = 슬롯 0 으로 낸 원출력을 arrival 모델(config xl_slot_order)로 probe 하면 — 언어 고정 슬롯 원출력을 예전 모델로 probe 한 것과
    asr·hyp·eot 행(정렬 후)·온셋 분류가 같다(슬롯 언어를 창마다 map_slots 로 짝지어 쓰므로)."""
    import importlib.util, json
    import cst_stage4_taxi as T
    import test_onset_classes as TO
    monkeypatch.setenv("CODE", TO.ROOT); tok = TO.WordTok()
    spec = importlib.util.spec_from_file_location("taxi_v3_probe_r4", os.path.join(TO.ROOT, TO.PROBE)); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    (tmp_path / "lang").mkdir(); (tmp_path / "arr").mkdir(); sess, em_lang = TO.probe_inputs(tmp_path / "lang", tok)
    first = lambda s: ([s["turns"][0]["speaker"]] + [x for x in sorted(s["speakers"]) if x != s["turns"][0]["speaker"]],)
    with monkeypatch.context() as m:
        m.setattr(T, "slots_of", lambda s: (first(s)[0], [s["speakers"][x]["lang"] for x in first(s)[0]]))
        sess2, em_arr = TO.probe_inputs(tmp_path / "arr", tok)
    S = [json.loads(l) for l in open(sess)]; assert any(T.slots_of(s)[0][0] != s["turns"][0]["speaker"] for s in S)   # 두 규칙이 다른 세션이 있다
    off = TO.run_probe(mod, monkeypatch, sess, TO.fresh(tmp_path / "o1", {}), tok, em_lang, True)

    class Arr(TO.FakeModel):
        def __init__(self, sp): super().__init__(sp); self.config.xl_slot_order = "arrival"
    monkeypatch.setattr(TO, "FakeModel", Arr)
    arr = TO.run_probe(mod, monkeypatch, sess2, TO.fresh(tmp_path / "o2", {}), tok, em_arr, True)
    rows = lambda b: sorted(b.decode().splitlines())
    for f in ("asr-d10.jsonl", "hyp-d10.jsonl", "eot-d10.jsonl"): assert rows(arr[f]) == rows(off[f]) and rows(off[f]), f
    ca, co = json.loads(arr["onset-classes-d10.json"]), json.loads(off["onset-classes-d10.json"])
    assert {k: v for k, v in ca.items() if k != "checkpoint"} == {k: v for k, v in co.items() if k != "checkpoint"} and ca["classified"] == len(TO.SCEN)
    assert json.loads(arr["probe.json"])["slot_map"] == "arrival" and "slot_map" not in json.loads(off["probe.json"])
    sm = json.loads(arr["slot-map.json"])["d10"]; assert sm["rule"] == len(S) and "swap" not in sm and "slot-map.json" not in off   # 각본은 첫 화자 = 슬롯 0
    assert sum(v for k, v in sm.items() if "/" in k) == len(S)
    sw = TO.run_probe(mod, monkeypatch, sess, TO.fresh(tmp_path / "o3", {}), tok, em_lang, True)                # 언어 고정 원출력 = 첫 화자가 슬롯 1 인 창이 규칙 위반
    # 규칙 위반 세션 = 언어 고정 순서의 슬롯 0 이 첫 화자가 아닌 곳(resume_bare·reassert·absorbed_eot). 두 화자 말이 같은 수로 한 슬롯에 몰린 앞 둘은 두 짝 비용이 같아
    # 규칙 짝(rule)으로 세고, 그 슬롯이 첫 화자(독일어) 쪽에 가까운 absorbed_eot(독 4 + 영 2)만 swap 이다
    bad = sorted(s["session_id"] for s in S if T.slots_of(s)[0][0] != s["turns"][0]["speaker"]); sm = json.loads(sw["slot-map.json"])["d10"]
    assert bad == ["on-absorbed_eot", "on-reassert", "on-resume_bare"] and sm["swap"] == 1 and sm["rule"] == len(S) - 1
    assert json.loads(arr["config.json"])["xl_slot_order"] == "arrival" and "xl_slot_order" not in json.loads(off["config.json"])
