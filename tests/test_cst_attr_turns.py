"""턴 결정 지표(experiments/cst_attr_turns.py) — 완벽한 가설·삼킨 턴·턴 안 헛교대·방향 분해·<EOT> 경계·bootstrap 재현,
cst_stage4_taxi score --turn-metrics(끄면 산출 그대로). 모델·tokenizer 다운로드 없이 CPU."""
import contextlib, io, json, os, subprocess, sys, types
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments")); sys.path.insert(0, os.path.join(ROOT, "cst-bench", "src"))
import cst_attr_turns as A  # noqa: E402
from cst_stage3_eval import words_of  # noqa: E402

OFF = 0.7                                                          # 가설 단어 = 참조 단어 끝 + OFF


class WordTok:
    """단어 하나 = 토큰 하나(byte-level piece 'Ġ단어' — 앞 공백), 특수 토큰('<…>')은 따로. 처음 보는 단어는 id 를 새로 준다(ASCII 만)."""
    def __init__(self): self.v, self.inv = {}, {}
    def _id(self, w):
        if w not in self.v: self.v[w] = 100 + len(self.v); self.inv[self.v[w]] = w
        return self.v[w]
    def __call__(self, text, add_special_tokens=False, **kw): return {"input_ids": [self._id(w) for w in text.split()]}
    def add_tokens(self, toks, special_tokens=True): [self._id(t) for t in toks]
    def convert_tokens_to_ids(self, t): return self._id(t)
    def convert_ids_to_tokens(self, ids): return [self.inv[int(i)] if self.inv[int(i)].startswith("<") else "Ġ" + self.inv[int(i)] for i in ids]
    def decode(self, ids): return "".join(" " + self.inv[int(i)] for i in ids if not self.inv[int(i)].startswith("<"))


TURNS = [("DP", "guten tag taxi zentrale"), ("CL", "hello i need a cab"), ("DP", "wohin bitte"), ("CL", "to the airport please"), ("DP", "ja danke"), ("CL", "thank you")]
LANG = {"DP": "German", "CL": "English"}


def session(sid="s0", turns=TURNS, t0=0.5, gap=0.3, wdur=0.4):
    """cstbench 세션(교차 언어 독·영): 턴마다 단어 수 × wdur 길이, 턴 사이 gap."""
    out, t = [], t0
    for k, (sp, txt) in enumerate(turns):
        d = wdur * len(txt.split()); out.append(dict(turn_id=f"t{k}", speaker=sp, lang=LANG[sp], start_s=round(t, 3), end_s=round(t + d, 3), transcript=txt)); t += d + gap
    return dict(session_id=sid, speakers={k: dict(lang=v, role=None) for k, v in LANG.items()}, duration_s=round(t + 0.5, 3), turns=out)


def ref_words(s):
    """(턴 번호, 단어, 단어 끝 시각) — cst_attr_turns.prep 과 같은 시각 규칙."""
    for k, tu in enumerate(s["turns"]):
        ws = tu["transcript"].split(); a, b = tu["start_s"], tu["end_s"]
        for j, w in enumerate(ws): yield k, j, w, a + (b - a) * (j + 1) / len(ws)


def asr_of(s, label=None):
    """완벽한 받아쓰기 asr 행(단어마다 시각 = 참조 끝 + OFF). label(턴, 단어 위치, 화자 언어) → 슬롯 언어(오배정 흉내)."""
    lab = label or (lambda k, j, lg: lg)
    return [dict(session=s["session_id"], lang=lab(k, j, s["turns"][k]["lang"]), t=round(t + OFF, 3), text=w) for k, j, w, t in ref_words(s)]


OTHER = {"German": "English", "English": "German"}


def swallow(*turns):
    return lambda k, j, lg: OTHER[lg] if k in turns else lg


def test_perfect_hypothesis():
    s = session(); r = A.turn_metrics([s], asr_of(s), B=50)
    assert r["toggle_recall"]["pct"] == 100.0 and r["toggle_recall"]["n"] == 5 and r["whole_wrong_change"]["turns"] == 0
    assert r["changes"] == dict(hyp=5, ref=5, ratio=1.0) and r["within_turn_switch"]["switches"] == 0 and r["misattributed"]["words"] == 0
    assert r["first_turn"]["acc_pct"] == 100.0 and r["offset_s"] == OFF and r["split"]["attr"] == 0.0 and r["slot_collapse"]["sessions"] == 0
    assert r["turns"]["change"]["correct"] == 5 and r["turns"]["first"]["correct"] == 1 and r["aligned_pct"] == 100.0


def test_swallowed_turn_is_whole_wrong_and_lowers_ratio():
    s = session(); r = A.turn_metrics([s], asr_of(s, swallow(3)), B=50)                            # 영어 턴 하나를 독일어 슬롯에(TAXI 몰림)
    assert r["whole_wrong_change"]["turns"] == 1 and r["turns"]["change"]["whole"] == 1 and r["misattributed"]["words"] == 4
    assert r["changes"]["hyp"] == 3 and r["changes"]["ratio"] < 1.0                                 # 교대 둘을 놓쳤다(들어갈 때·나올 때)
    assert r["toggle_recall"]["toggled"] == 3 and r["toggle_recall"]["transitions"] == {"prevOK>curOK/toggle": 3, "prevOK>curWRONG/no-toggle": 1, "prevWRONG>curOK/no-toggle": 1}
    assert r["error_onsets"]["onsets"] == {"speaker-change": 1} and r["error_onsets"]["recoveries"] == {"speaker-change": 1}
    assert r["split"]["agn"] == 0.0 and r["split"]["attr"] > 0                                       # 받아쓰기는 그대로, 귀속 몫만


def test_within_turn_false_switch_and_partial_types():
    s = session(); r = A.turn_metrics([s], asr_of(s, lambda k, j, lg: OTHER[lg] if (k, j) == (1, 2) else lg), B=50)   # 'need' 만 반대 슬롯
    assert r["within_turn_switch"]["switches"] == 2 and r["within_turn_switch"]["pairs"] == sum(len(t.split()) - 1 for _, t in TURNS)
    assert r["partial_words"] == dict(late=0, early=0, middle=1, multi=0) and r["toggle_recall"]["pct"] == 100.0 and r["changes"]["hyp"] == 7
    late = A.turn_metrics([s], asr_of(s, lambda k, j, lg: OTHER[lg] if (k, j) in ((1, 0), (1, 1)) else lg), B=10)    # 턴 첫머리 두 단어 = 교대를 늦게
    assert late["partial_words"]["late"] == 2 and late["toggle_recall"]["toggled"] == 4                # 앞 턴 끝과 새 턴 첫 단어가 같은 슬롯 → 그 교대는 놓침
    early = A.turn_metrics([s], asr_of(s, lambda k, j, lg: OTHER[lg] if (k, j) == (1, 4) else lg), B=10)               # 턴 끝 한 단어 = 일찍 바꿈
    assert early["partial_words"]["early"] == 1 and early["bootstrap"]["metrics"]["early_words"]["value"] > 0


def test_direction_split_by_new_speaker_language():
    s = session(); en = A.turn_metrics([s], asr_of(s, swallow(3)), B=10); de = A.turn_metrics([s], asr_of(s, swallow(2)), B=10)
    assert en["onset_miss"]["English"] == dict(n=3, miss=1, pct=33.33) and en["onset_miss"]["German"] == dict(n=1, miss=0, pct=0.0)   # 삼킨 턴 뒤 턴은 앞 턴이 틀려 빠진다
    assert de["onset_miss"]["German"]["miss"] == 1 and de["onset_miss"]["English"]["miss"] == 0
    assert en["short_turns"]["change_le3"]["n"] == 3 and en["short_turns"]["change_gt3"]["whole"] == 1   # 2 단어 교대 턴 셋, 'to the airport please' 4 단어


def test_wrong_slot_other_language_words_and_collapse():
    s = session()
    tr = {(3, 0): "zum", (3, 1): "flughafen"}                                                        # 삼킨 영어 턴의 두 단어를 독일어로 바꿔 씀(번역 누출)
    asr = [dict(x, text=tr.get((k, j), x["text"])) for x, (k, j, _, _) in zip(asr_of(s, swallow(3)), ref_words(s))]
    s2 = dict(s, turns=s["turns"] + [dict(turn_id="t6", speaker="DP", lang="German", start_s=13.0, end_s=14.0, transcript="zum flughafen")])
    r = A.turn_metrics([s2], asr, B=10)
    assert r["wrong_slot_lang"]["wrong_sub"] == 2 and r["wrong_slot_lang"]["other_lang"] == 2
    one = A.turn_metrics([s], asr_of(s, lambda k, j, lg: "German"), B=10)                           # 모든 말을 한 슬롯에(슬롯 붕괴)
    assert one["slot_collapse"]["sessions"] == 1 and one["changes"]["hyp"] == 0


def test_eot_at_boundaries_hit_and_miss():
    s = session(); asr = asr_of(s, swallow(3)); T = s["turns"]
    eot = [dict(session="s0", lang="German", t=round(T[0]["end_s"] + OFF + 0.1, 3)),                 # 교대 1(hit) 앞 슬롯 <EOT>
           dict(session="s0", lang="English", t=round(T[2]["end_s"] + OFF + 0.1, 3))]                # 교대 3(miss) 경계엔 다른 슬롯 <EOT>
    r = A.turn_metrics([s], asr, eot, B=10)["eot"]
    assert r["emitted"] == 2 and r["majority"]["hit"]["n"] == 4 and r["majority"]["miss"]["n"] == 1
    assert r["majority"]["hit"]["prev"] == 1 and r["majority"]["miss"]["other"] == 1 and r["majority"]["f_hit"] == 25.0 and r["majority"]["f_miss"] == 0.0
    assert r["toggle"]["miss"]["n"] == 2 and r["toggle"]["hit"]["n"] == 3


def test_bootstrap_reproducible_and_paired():
    S = [session(f"s{i}") for i in range(8)]
    good = [x for s in S for x in asr_of(s)]; bad = [x for i, s in enumerate(S) for x in asr_of(s, swallow(3) if i % 2 else swallow())]
    a = A.turn_metrics(S, bad, against=good, B=300, seed=0); b = A.turn_metrics(S, bad, against=good, B=300, seed=0); c = A.turn_metrics(S, bad, against=good, B=300, seed=1)
    assert a == b and a["bootstrap"]["seed"] == 0 and a["bootstrap"] != c["bootstrap"]
    m = a["bootstrap"]["metrics"]["toggle_recall"]; d = a["bootstrap"]["paired_diff"]["toggle_recall"]
    assert m["value"] == 80.0 and m["ci95"][0] < 80.0 < m["ci95"][1] and d["value"] == -20.0 and d["ci95"][1] < 0       # 같은 재표집의 차 — 놓친 교대만큼 낮다
    assert a["bootstrap"]["paired_diff"]["within_turn_switch"] == dict(value=0.0, ci95=[0.0, 0.0])


def test_same_language_sessions_are_skipped_and_cli_writes_aggregates_only(tmp_path):
    s = session(); same = dict(session(sid="same"), speakers={"DP": dict(lang="German", role=None), "CL": dict(lang="German", role=None)})
    r = A.turn_metrics([s, same], asr_of(s), B=10); assert r["skipped_same_lang"] == 1 and r["sessions"] == 1
    fs, fa, fo = tmp_path / "sessions.jsonl", tmp_path / "asr-d10.jsonl", tmp_path / "tm.json"
    fs.write_text(json.dumps(s) + "\n"); fa.write_text("".join(json.dumps(x) + "\n" for x in asr_of(s, swallow(3))))
    p = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "cst_attr_turns.py"), "--sessions", str(fs), "--asr", str(fa), "--boot", "20", "--out", str(fo)],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    txt = fo.read_text(); out = json.loads(txt); assert out["whole_wrong_change"]["turns"] == 1 and "toggle recall 60.0 %" in p.stdout
    assert "s0" not in txt.replace(str(tmp_path), "") and "airport" not in txt and "airport" not in p.stdout          # 집계만(세션 id·글자 없음)


def test_words_at_matches_words_of():
    tok = WordTok(); ids = tok("hello i need a cab")["input_ids"]; toks = [(0.08 * (k + 1), t, 10 + k) for k, t in enumerate(ids)]
    got = A.words_at(tok, toks); assert [(w, t) for w, t, _, _ in got] == words_of(tok, [(t, i) for t, i, _ in toks]) and [(a, b) for *_, a, b in got] == [(10 + k, 10 + k) for k in range(5)]


# ── cst_stage4_taxi score --turn-metrics: 끄면 산출·출력이 바꾸기 전과 같다
def base_module(rel, token, name):
    """token 을 처음 넣은 커밋의 부모(아직 커밋 전이면 HEAD) 판의 모듈 — 바꾸기 전 동작과 대조(git 이 없으면 None)."""
    try:
        h = subprocess.run(["git", "log", "--reverse", "--format=%H", "-S", token, "--", rel], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        src = subprocess.run(["git", "show", f"{h[0]}^:{rel}" if h else f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    if token in src: return None
    m = types.ModuleType(name); m.__file__ = os.path.join(ROOT, rel); exec(compile(src, m.__file__, "exec"), m.__dict__); return m


def score_out(mod, sess, out, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): mod.score(types.SimpleNamespace(sessions=str(sess), out=str(out), **kw))
    return {f: open(os.path.join(out, f), "rb").read() for f in sorted(os.listdir(out))}, buf.getvalue()


def score_fixture(d, trans=True):
    pytest.importorskip("sacrebleu")
    S = [session(f"s{i}") for i in range(3)]
    for s in S:
        for t in s["turns"]: t.update(translation=("übersetzung " + t["transcript"]) if trans else None, translation_lang=OTHER[t["lang"]] if trans else None)
    d.mkdir(); (d / "sessions.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in S))
    asr = [x for i, s in enumerate(S) for x in asr_of(s, swallow(3) if i == 1 else None)]
    hyp = [dict(session=x["session"], lang=OTHER[x["lang"]], t=x["t"] + 0.5, text=x["text"]) for x in asr]
    for dl in (4, 10):
        (d / f"asr-d{dl}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in asr)); (d / f"hyp-d{dl}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in hyp))
    return d / "sessions.jsonl"


def fresh(d, src, n=[0]):
    """d 를 입력 파일만 든 새 폴더로(이전 판은 옆 이름으로 옮겨 둔다 — 산출에 폴더 경로가 들어가므로 같은 경로에서 돌린다)."""
    if d.exists(): n[0] += 1; os.rename(d, f"{d}.prev{n[0]}")
    d.mkdir()
    for f, b in src.items(): (d / f).write_bytes(b)
    return d


@pytest.mark.parametrize("trans", [True, False])
def test_score_without_turn_metrics_is_unchanged(tmp_path, trans):
    import cst_stage4_taxi as T
    sess = score_fixture(tmp_path / "in", trans); src = {f: (tmp_path / "in" / f).read_bytes() for f in os.listdir(tmp_path / "in") if f != "sessions.jsonl"}
    d = tmp_path / "out"; runs = {}
    for name, kw in (("off", {}), ("flag_off", dict(turn_metrics=False)), ("on", dict(turn_metrics=True))): runs[name] = score_out(T, sess, fresh(d, src), **kw)
    p = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "cst_stage4_taxi.py"), "score", "--sessions", str(sess), "--out", str(fresh(d, src))],
                       capture_output=True, text=True, cwd=ROOT)                                             # CLI 기본값(플래그 없음)
    assert p.returncode == 0, p.stderr
    assert runs["off"] == runs["flag_off"] and ({f: open(d / f, "rb").read() for f in sorted(os.listdir(d))}, p.stdout) == runs["off"]
    files_off, out_off = runs["off"]; files_on, out_on = runs["on"]
    assert sorted(set(files_on) - set(files_off)) == ["turn-metrics-d10.json", "turn-metrics-d4.json"] and all(files_on[f] == b for f, b in files_off.items())
    assert out_on.startswith(out_off) and out_on[len(out_off):].count("턴 결정") == 2
    tm = json.loads(files_on["turn-metrics-d10.json"]); assert tm["whole_wrong_change"]["turns"] == 1 and tm["sessions"] == 3
    rep = json.loads(files_off["report-d10.json"]); assert bool(rep["directions"]) == trans                    # 번역 참조가 없으면 BLEU 방향이 빈다(받아쓰기·split 은 그대로)
    assert "split" in json.loads(files_off["asr-report-d10.json"])
    base = base_module("experiments/cst_stage4_taxi.py", "turn_metrics", "cst_stage4_taxi_base")
    if base is None: pytest.skip("git 기준판 없음")
    assert score_out(base, sess, fresh(d, src)) == runs["off"]                                              # 바꾸기 전 판과 파일·출력 바이트 동일
