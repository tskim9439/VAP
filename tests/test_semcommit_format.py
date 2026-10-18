"""출력 형식 플래그(정규형 ↔ 표기형, T5 outfmt-v1) — vapasr/data/output_format.py · vapasr/data/semcommit_written.py · SemCommitDataset(written_frac).

- 기본(written_frac=0)은 이 기능 전 코드와 바이트 동일: 시퀀스·항목·통계·난수 소비·반환 키·collate·est_lens 의 해시를 고정한다(GOLDEN — 기능 추가 전 코드로 뽑음).
- 표기형 행: KO = speechlm 원 tokens(display_text) 그대로 + 숫자 단어 digit hold, EN = pnc_text 단위를 lexical 단어에 정확히 맞춘 것.
- <SEM_END> 는 표기형에서도 '단어 마지막 토큰 바로 뒤' = 끝 구두점 뒤, 같은 청크. B/N 결정 위치는 그 다음 원소.
tokenizer 는 byte-level 장난감 BPE(ToyTok: Qwen 식 앞 공백 조각, 특수 토큰 id 151643– / 151705–). 실 Qwen tokenizer 검사는 SEMCOMMIT_TOKENIZER 가 있을 때만."""
import hashlib, json, os, random, re, subprocess, sys
import pytest
from vapasr.data.semcommit_words import _B2U, split_words, lexical_words
from vapasr.data.semcommit_dataset import SemCommitDataset, collate_semcommit, semcommit_fingerprint, fingerprint_diff
from vapasr.data.output_format import header_ids, header_text, output_formats_config, WRITTEN_INSTRUCTION, OUTFMT_PROTOCOL, NORMALIZED, WRITTEN
from vapasr.data import semcommit_written
from vapasr.data.semcommit_written import written_row, remap_labels, output_format_settings, number_readings, en_units, written_rules_sha256, WRITTEN_BUILDER
from vapasr.hf.packing import est_lens, pack_batch

_SPECIAL0 = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645, "<|audio_pad|>": 151676}
PIECES = ["안녕", "하세요", " 여기", " 화곡동", "인데요", "년이", "었대요", " 이천칠", " 네", " 맞아요", " 그래서", " 좋아요",
          "in", " in", "In", " two", " thousand", " and", " fourteen", " we", " adopted", " it", " wax", "man", " Wax", "-M", "arkey", " mark", "ey",
          " is", " law", " hello", " world", " so", " done", "hello", " yes", " the", " vote", " passed", " Hello", " World",
          "system", "\n", "assistant", "language", " English", " Korean", "<", "asr", "_text", ">", " Output", " format", ":", " written", " (", "punct",
          "uation", ",", " capital", "ization", " numer", "als", ").", "."]


def _bl(s: str) -> str:
    return "".join(_B2U[b] for b in s.encode("utf-8"))


class ToyTok:
    """Byte-level greedy BPE: 모든 바이트 + PIECES(가장 긴 일치), 특수 토큰은 먼저 잘라 낸다. HF tokenizer 의 쓰는 면만 흉내
    (__call__(str|list) · convert_ids_to_tokens · convert_tokens_to_ids · add_tokens · decode · get_vocab · special_ids)."""
    def __init__(self, pieces=PIECES):
        self.i2p = {b: _B2U[b] for b in range(256)}
        for p in pieces:
            q = _bl(p)
            if q not in self.i2p.values(): self.i2p[len(self.i2p)] = q
        self.p2i = {p: i for i, p in self.i2p.items()}; self.maxlen = max(len(p) for p in self.p2i)
        self.specials = dict(_SPECIAL0)
    @property
    def special_ids(self): return set(self.specials.values())
    def add_tokens(self, toks, special_tokens=True):
        for t in toks:
            if t not in self.specials: self.specials[t] = 151705 + sum(1 for v in self.specials.values() if v >= 151705)
    def convert_tokens_to_ids(self, t): return self.specials[t] if t in self.specials else self.p2i[t]
    def convert_ids_to_tokens(self, ids):
        inv = {v: k for k, v in self.specials.items()}; return [inv.get(int(i)) or self.i2p[int(i)] for i in ids]
    def _enc(self, text):
        s = _bl(text); out, i = [], 0
        while i < len(s):
            for n in range(min(self.maxlen, len(s) - i), 0, -1):
                if s[i:i + n] in self.p2i: out.append(self.p2i[s[i:i + n]]); i += n; break
        return out
    def __call__(self, text, add_special_tokens=False, **kw):
        if isinstance(text, (list, tuple)): return {"input_ids": [self(t)["input_ids"] for t in text]}
        rx = "(" + "|".join(re.escape(k) for k in sorted(self.specials, key=len, reverse=True)) + ")"; out = []
        for part in re.split(rx, text):
            if part: out += [self.specials[part]] if part in self.specials else self._enc(part)
        return {"input_ids": out}
    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.specials.items()}; out, buf = [], []
        for i in ids:
            if int(i) in inv:
                out.append(bytes(buf).decode("utf-8", "replace")); buf = []
                if not skip_special_tokens: out.append(inv[int(i)])
            else: buf += [{v: k for k, v in _B2U.items()}[c] for c in self.i2p[int(i)]]
        return "".join(out) + bytes(buf).decode("utf-8", "replace")
    def get_vocab(self): return {**self.p2i, **self.specials}


def stream_row(tok, sid, words_times, lang, set_name, raw_texts=None, lexical=True, tail=0.4, **extra):
    """words_times: [(공백 단어, [각 토큰 시각] 또는 끝 시각 float)] → words.jsonl 행(토큰 = 단어마다 앞 공백을 붙여 인코딩, words = split_words(+ lexical_words))."""
    ids, times = [], []
    for k, (w, tt) in enumerate(words_times):
        wi = tok((" " if k else "") + w)["input_ids"]
        if isinstance(tt, (int, float)): tt = [float(tt)] * len(wi)
        elif len(tt) != len(wi): tt = (list(tt) + [tt[-1]] * len(wi))[:len(wi)]
        ids += wi; times += list(tt)
    ws = split_words(ids, times, tok)
    if lexical: ws, why = lexical_words(ws); assert ws, why
    else: [w.setdefault("tags", []) for w in ws]
    for w in ws: w["seg"] = 0
    dur = round(times[-1] + tail, 3); raw = raw_texts or [" ".join(w["text"] for w in ws)]
    return dict(id=sid, set=set_name, lang=lang, K=int(round(dur * 12.5)), duration_s=dur, text=" ".join(w["text"] for w in ws),
                tokens=[[i, t] for i, t in zip(ids, times)], words=ws,
                segments=[dict(path=f"/x/{sid}.pcm", offset_s=0.0, silence_before_s=0.0, dur_s=dur, utt_id=sid, raw_text=r) for r in raw], **extra)


def labels_of(sid, lang, cands, turn_end=False):
    return dict(id=sid, lang=lang, candidates=[dict(after_word=i, grade=g, why="t") for i, g in cands], turn_end=turn_end)


def fixture_rows(tok):
    """speechlm KO(구두점·숫자 단어, display_text) · VoxPopuli EN(pnc_text: 2014·하이픈 복합어) · LibriSpeech EN(pnc 없음) · kspon 식 KO(display 없음)."""
    ko = stream_row(tok, "slm-ko-1", [("안녕하세요.", [0.30, 0.52, 0.52]), ("2007년이었대요.", [0.70, 0.78, 0.86, 0.95, 1.10, 1.30, 1.30]),
                                      ("여기", 1.62), ("화곡동인데요?", [1.90, 2.20, 2.20])], "Korean", "speechlm-partial",
                    raw_texts=["안녕하세요 이천칠년이었대요 여기 화곡동인데요"], display_text="안녕하세요. 2007년이었대요. 여기 화곡동인데요?")
    en = stream_row(tok, "vp-en-1", [("in", 0.40), ("two", 0.62), ("thousand", 0.90), ("and", 1.02), ("fourteen", 1.35), ("we", 1.60), ("adopted", 1.95),
                                     ("it", 2.10), ("waxman", 2.50), ("markey", 2.90), ("is", 3.05), ("law", 3.40)], "English", "voxpopuli-train",
                    lexical=False, pnc_text="In 2014, we adopted it. Waxman-Markey is law.")
    ls = stream_row(tok, "ls-en-1", [("hello", 0.30), ("world", 0.70), ("so", 1.00), ("done", 1.40)], "English", "librispeech-960", lexical=False)
    ks = stream_row(tok, "ks-ko-1", [("네", 0.40), ("맞아요", 0.90), ("그래서", 1.40), ("좋아요", 1.90)], "Korean", "kspon-train", lexical=False)
    labs = [labels_of("slm-ko-1", "Korean", [(0, "A"), (1, "A"), (2, "N"), (3, "B")]),
            labels_of("vp-en-1", "English", [(1, "N"), (4, "N"), (7, "A"), (8, "B"), (11, "A")]),
            labels_of("ls-en-1", "English", [(1, "A"), (2, "N"), (3, "A")]),
            labels_of("ks-ko-1", "Korean", [(1, "A"), (3, "A")])]
    return [ko, en, ls, ks], labs


def write_fixture(tmp_path, rows, labs, name="f"):
    wp, lp = tmp_path / f"{name}.words.jsonl", tmp_path / f"{name}.labels.jsonl"
    wp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    lp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in labs))
    return str(wp), str(lp)


def _h(x) -> str:
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


def normalized_digest(ds):
    """정규형 항목·시퀀스(모든 δ)·통계·est_lens 의 해시."""
    items = [dict(id=it["id"], K=it["K"], K0=it["K0"], dur=it["duration_s"], tokens=it["tokens"], marks=it["marks"], n_tok=it["n_tok"], n_sem=it["n_sem"], n_turn=it["n_turn"],
                  keys=sorted(it)) for it in ds.items]
    seqs = []
    for i in range(len(ds)):
        for d in ds.delays:
            s = ds.sequence(i, d)
            seqs.append(dict(ids=s["ids"], labels=s["labels"], pw=s["pos_weight"], inp=s["is_input"], ch=s["chunk_of"], dec=s["decision"], pre=ds.prefix(ds.items[i]["lang"], d),
                             tp=ds.target_problems(i, d)))
    return dict(items=_h(items), seqs=_h(seqs), stats=_h(sorted(ds.stats.items())), bad=_h(sorted(ds.bad.items())), est=_h([int(x) for x in est_lens(ds)]))


def getitem_digest(ds, seed=123):
    """__getitem__ 출력(키·δ·텐서)·collate 키·호출 뒤 난수 상태의 해시."""
    random.seed(seed); outs = [ds[i] for i in range(len(ds))]; after = random.random()
    rec = [dict(keys=sorted(o), delay=o["delay"], ids=o["ids"].tolist(), labels=o["labels"].tolist(), pw=o["pos_weight"].tolist(), aud=o["is_audio"].tolist(),
                ch=o["chunk_of"].tolist(), n=[o[k] for k in ("n_text", "n_sem", "n_turn", "n_B", "n_N", "n_decision_on_event", "K", "rounds", "n_flush")]) for o in outs]
    b = collate_semcommit(outs)
    return dict(out=_h(rec), after=round(after, 12), collate=_h(sorted(b)), collate_n=_h([b[k] for k in ("n_sem", "n_turn", "n_B", "n_N", "n_text")]))


GOLDEN_CONFIGS = {"sem": dict(delays=(2, 3, 4, 6, 8), turn_end=False), "turn": dict(delays=(2, 4, 6), turn_end=True, hangover_s=0.48)}
GOLDEN = {   # 기능 추가 전 코드(2026-09-29 작업 트리)로 뽑은 값 — SEMCOMMIT_PRINT_GOLDEN=1 pytest -s 로 다시 찍을 수 있다(바꾸면 기본 경로가 바뀐 것)
    "sem": {"items": "d0e6e97175c1b7af", "seqs": "14e288695482a460", "stats": "76afe96d27e87930", "bad": "4f53cda18c2baa0c", "est": "04cdeef40a40e276",
            "out": "c6159e5793c50d65", "after": 0.266553812531, "collate": "16aee3389bfde18b", "collate_n": "d649e0e31c3817d7"},
    "turn": {"items": "e4484b3e30e22547", "seqs": "8a3e3d7c13a5c5f4", "stats": "48852b7dcfeeeee0", "bad": "4f53cda18c2baa0c", "est": "96e825b1a6c7fd4e",
             "out": "e6e37fad60e23d5f", "after": 0.266553812531, "collate": "16aee3389bfde18b", "collate_n": "d649e0e31c3817d7"},
}


def build_golden_ds(tmp_path, cfg, **kw):
    tok = ToyTok(); rows, labs = fixture_rows(tok); wp, lp = write_fixture(tmp_path, rows, labs)
    return SemCommitDataset(wp, lp, tok, online=False, **GOLDEN_CONFIGS[cfg], **kw)


@pytest.mark.parametrize("cfg", sorted(GOLDEN_CONFIGS))
def test_golden_default_is_byte_identical(tmp_path, cfg):
    ds = build_golden_ds(tmp_path, cfg)
    got = dict(**normalized_digest(ds), **getitem_digest(ds))
    if os.environ.get("SEMCOMMIT_PRINT_GOLDEN"): print("GOLDEN", cfg, json.dumps(got))
    assert got == GOLDEN[cfg]


# ── 형식 머리(system 칸)
def test_header_ids_normalized_is_current_pre_and_written_inserts_instruction():
    tok = ToyTok(); pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n")["input_ids"]
    assert header_ids(tok, NORMALIZED) == pre and header_text(NORMALIZED) == "<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n"
    w = header_ids(tok, WRITTEN)
    assert w == pre[:3] + tok(WRITTEN_INSTRUCTION)["input_ids"] + pre[3:] and tok.decode(w) == header_text(WRITTEN)
    cfg = output_formats_config(tok)
    assert cfg == {"written": dict(instruction=WRITTEN_INSTRUCTION, header_ids=w, protocol=OUTFMT_PROTOCOL)}
    with pytest.raises(ValueError, match="unknown output format"): header_ids(tok, "spoken")


# ── KO 표기형 행 = 원 tokens(display_text) + digit hold
def test_ko_written_row_keeps_original_tokens_and_holds_digits():
    tok = ToyTok(); rows, _ = fixture_rows(tok); ko = rows[0]
    rw, wm = written_row(ko, tok)
    assert wm is None and rw["words"] == ko["words"] and [t[0] for t in rw["tokens"]] == [t[0] for t in ko["tokens"]]
    assert tok.decode([t[0] for t in rw["tokens"]]) == ko["display_text"]
    w1 = ko["words"][1]; assert w1["text"] == "2007년이었대요"                                    # 숫자 단어: 모든 토큰 = 단어 끝 시각(digit hold)
    assert [t[1] for t in rw["tokens"][w1["a"]:w1["b"]]] == [w1["end_time"]] * (w1["b"] - w1["a"])
    other = [j for j in range(len(ko["tokens"])) if not w1["a"] <= j < w1["b"]]
    assert [rw["tokens"][j] for j in other] == [ko["tokens"][j] for j in other]
    assert written_row(ko, tok, digit_hold=False) == (ko, None)
    assert written_row(dict(ko, display_text="전혀 다른 문장."), tok) == (None, "ko_display_mismatch")
    assert written_row(dict(ko, display_text="안녕하세요.  2007년이었대요.여기 화곡동인데요?"), tok)[0] is not None   # 공백 차이는 무시
    assert written_row({k: v for k, v in ko.items() if k != "display_text"}, tok) == (None, "ko_no_written_source")
    assert written_row(rows[3], tok) == (None, "ko_no_written_source")                          # kspon 식 행


# ── EN 표기형 행 = pnc 단위 ↔ lexical 단어 정확 정렬
def test_en_written_row_units_times_and_word_map():
    tok = ToyTok(); rows, labs = fixture_rows(tok); en = rows[1]
    rw, wm = written_row(en, tok)
    assert tok.decode([t[0] for t in rw["tokens"]]) == en["pnc_text"] == rw["text"]
    assert [(w["text"], w["tags"]) for w in rw["words"]] == [("In", []), ("2014", ["punct_comma"]), ("we", []), ("adopted", []), ("it", ["punct_final"]),
                                                               ("Waxman-Markey", []), ("is", []), ("law", ["punct_final"])]
    assert wm == [0, None, None, None, 1, 2, 3, 4, None, 5, 6, 7]
    lex_end = [w["end_time"] for w in en["words"]]
    assert [w["end_time"] for w in rw["words"]] == [lex_end[i] for i, j in enumerate(wm) if j is not None]   # 단위 끝 시각 = 그 단위 마지막 lexical 단어 끝
    w1 = rw["words"][1]; assert {t[1] for t in rw["tokens"][w1["a"]:w1["b"]]} == {lex_end[4]}             # '2014,' 전부 단위 끝(digit hold)
    ts = [t[1] for t in rw["tokens"]]; assert ts == sorted(ts)
    lw, _ = lexical_words(split_words([t[0] for t in rw["tokens"]], ts, tok))                          # 빌더 규칙으로 다시 나눠도 같은 단어
    assert [(w["text"], w["a"], w["b"]) for w in lw] == [(w["text"], w["a"], w["b"]) for w in rw["words"]]
    lab, dropped = remap_labels(labs[1], wm)
    assert [(c["after_word"], c["grade"]) for c in lab["candidates"]] == [(1, "N"), (4, "A"), (7, "A")] and dict(dropped) == {"N": 1, "B": 1}


def _en(tok, sid, lex, pnc, set_name="voxpopuli-train"):
    return stream_row(tok, sid, [(w, 0.3 + 0.25 * k) for k, w in enumerate(lex.split())], "English", set_name, lexical=False, **({"pnc_text": pnc} if pnc else {}))


@pytest.mark.parametrize("lex,pnc,set_name,want", [
    ("in twenty fourteen we met", "In 2014 we met.", "voxpopuli-train", ["In", "2014", "we", "met"]),                  # 연도 읽기
    ("we have twenty one percent", "We have 21%.", "yodas-en129", ["We", "have", "21"]),                               # yodas canon + percent(% 는 끝 구두점처럼 단어 span 끝)
    ("it is a layer many people agree", "It is a layer.Many people agree.", "yodas-en129-rg", ["It", "is", "a", "layer", "Many", "people", "agree"]),  # 붙임 오류 보정
    ("so — we won", "So — we won!", "voxpopuli-train", None),                                                              # lexical 에 '—' 가 있으면 정렬 불가
    ("so we won", "So — we won!", "voxpopuli-train", ["So", "we", "won"]),                                                  # 순수 구두점 토큰은 앞 단위에
])
def test_en_written_variants(lex, pnc, set_name, want):
    tok = ToyTok(); r = _en(tok, "x", lex, pnc, set_name); rw, wm = written_row(r, tok)
    if want is None: assert rw is None; return
    assert rw is not None, wm
    assert [w["text"] for w in rw["words"]] == want and tok.decode([t[0] for t in rw["tokens"]]) == re.sub(r"(?<=[a-z])([.?!])(?=[A-Z][a-z])", r"\1 ", pnc)


@pytest.mark.parametrize("pnc,surfaces,sem_word_text", [
    ("He said ( we won ) today.", ["He", "said", "(we", "won )", "today."], " said"),                  # 여는 괄호는 다음 단위로 → 'said' 뒤 SEM 이 '(' 뒤로 가지 않는다
    ('He said " we won " today.', ["He", "said", '"we', 'won "', "today."], " said"),                   # 짝수 번째 '"' = 여는 따옴표, 홀수 번째 = 닫는 따옴표(앞 단위)
    ("He said [ we won today. (", ["He", "said", "[we", "won", "today. ("], " said"),                   # 끝에 남은 여는 부호는 앞 단위로
    ("He said ( — we won today.", ["He", "said ( —", "we", "won", "today."], " said ( —"),              # 여는 부호 뒤에 다른 구두점 → 이전 규칙(앞 단위)
    ("( He said we won today.", ["( He", "said", "we", "won", "today."], " said"),                       # 첫 단위 앞 = 앞머리(기존 규칙 그대로)
])
def test_en_units_opening_marks_go_to_next_unit(pnc, surfaces, sem_word_text):
    assert [u["surface"] for u in en_units(pnc, "voxpopuli")] == surfaces
    tok = ToyTok(); r = _en(tok, "x", "he said we won today", pnc); rw, wm = written_row(r, tok)
    assert rw is not None, wm                                                                     # 빌더 규칙(lexical_words)으로 다시 나눠도 같은 단어(en_selfcheck 통과)
    assert [w["text"] for w in rw["words"]] == ["He", "said", "we", "won", "today"] and wm == [0, 1, 2, 3, 4]
    w1 = rw["words"][1]; assert tok.decode([t[0] for t in rw["tokens"][w1["a"]:w1["b"]]]) == sem_word_text   # 'said' 의 마지막 토큰 = SEM 자리


@pytest.mark.parametrize("lex,pnc,why", [
    ("we adopted it and it is law", "we adopted it and it is law.", "en_no_case"),
    ("the g twenty met", "The G20 met.", "en_align"),
    ("one eight percent", "1.8%.", "en_align"),
    ("hello world", None, "en_no_pnc"),
])
def test_en_written_rejects(lex, pnc, why):
    tok = ToyTok(); assert written_row(_en(tok, "x", lex, pnc), tok) == (None, why)


def test_number_readings_and_fake_tokenizer():
    assert ("two", "thousand", "and", "fourteen") in number_readings("2014") and ("twenty", "fourteen") in number_readings("2014")
    assert ("two", "thousand", "fourteen") in number_readings("2014") and ("five", "percent") in number_readings("5%")
    assert ("twenty", "first") in number_readings("21st") and number_readings("G20") == ()
    class NoPieces:                                                                              # 조각을 못 읽는 tokenizer → 표기형 없음(정규형만)
        def __call__(self, text, **kw): return {"input_ids": [1]}
    rows, _ = fixture_rows(ToyTok())
    assert written_row(rows[0], NoPieces()) == (None, "written_decode") and written_row(rows[1], NoPieces()) == (None, "written_decode")
    assert written_row(dict(rows[0], lang="Japanese"), ToyTok()) == (None, "written_lang")


WRITTEN_GOLDEN = {   # 표기형 행(written_row) 출력 해시 — 값이 바뀌면 규칙이 바뀐 것: WRITTEN_BUILDER 를 올리고 새 키로 다시 찍는다(SEMCOMMIT_PRINT_GOLDEN=1)
    "semcommit-written-v1": "1657b4fa36e00631",
}


def _written_golden_rows(tok):
    rows, _ = fixture_rows(tok)
    rows += [_en(tok, f"v{k}", lex, pnc, s) for k, (lex, pnc, s) in enumerate([
        ("in twenty fourteen we met", "In 2014 we met.", "voxpopuli-train"), ("we have twenty one percent", "We have 21%.", "yodas-en129"),
        ("it is a layer many people agree", "It is a layer.Many people agree.", "yodas-en129-rg"), ("so we won", "So — we won!", "voxpopuli-train"),
        ("he said we won today", "He said ( we won ) today.", "voxpopuli-train"), ("he said we won today", 'He said " we won " today.', "mnsc-train"),
        ("the g twenty met", "The G20 met.", "voxpopuli-train")])]
    return rows


def test_written_rows_golden():
    tok = ToyTok(); got = _h([written_row(r, tok) for r in _written_golden_rows(tok)])
    if os.environ.get("SEMCOMMIT_PRINT_GOLDEN"): print("WRITTEN_GOLDEN", WRITTEN_BUILDER, got)
    assert WRITTEN_GOLDEN.get(WRITTEN_BUILDER) == got, "표기형 규칙 출력이 바뀌었다 — WRITTEN_BUILDER 를 올리고 WRITTEN_GOLDEN 에 새 키로 찍는다"


def test_remap_labels_identity_and_range():
    lab = labels_of("a", "English", [(0, "A"), (2, "B")])
    assert remap_labels(lab, None) == (lab, {}) and remap_labels(None, [0]) == (None, {})
    with pytest.raises(ValueError, match="after_word_range"): remap_labels(lab, [0, None])


# ── 데이터셋: written_frac
def _ds(tmp_path, frac, cfg="sem", **kw):
    return build_golden_ds(tmp_path, cfg, written_frac=frac, **kw)


def test_frac_zero_builds_no_written_and_draws_no_random(tmp_path):
    ds = _ds(tmp_path, 0.0)
    assert all("written" not in it and "est_extra" not in it for it in ds.items) and all(ds.formats(i) == [NORMALIZED] for i in range(len(ds)))
    assert not any(k.startswith("written") for k in ds.stats)
    random.seed(7); x = ds[0]; after = random.random(); random.seed(7); random.choice(ds.delays); assert after == random.random()   # δ 한 번만 뽑는다
    assert "fmt" not in x and "n_written" not in x and "n_written" not in collate_semcommit([x])
    with pytest.raises(ValueError, match="no_written_form"): ds.sequence(0, 2, WRITTEN)
    with pytest.raises(ValueError, match=r"\[0, 1\]"): _ds(tmp_path, 1.5)


@pytest.mark.parametrize("cfg", sorted(GOLDEN_CONFIGS))
def test_frac_on_keeps_normalized_sequences(tmp_path, cfg):
    ds = _ds(tmp_path, 1.0, cfg); g = normalized_digest(ds)
    assert g["seqs"] == GOLDEN[cfg]["seqs"]                                                     # 정규형 항목·시퀀스(K 포함)는 플래그와 무관
    assert {it["id"] for it in ds.items if "written" in it} == {"slm-ko-1", "vp-en-1"}
    st = ds.stats
    assert (st["written_available"], st["written_en_no_pnc"], st["written_ko_no_written_source"], st["written_cand_dropped_N"], st["written_cand_dropped_B"]) == (2, 1, 1, 1, 1)
    assert st["written_hours"] == round(sum(it["duration0"] for it in ds.items if "written" in it) / 3600, 3) and "written_only_skipped" not in st


def _pieces_text(tok, ids):
    return tok.decode([t for t in ids if t < 151643])


def _ds_turn_labels(tmp_path, frac):
    """'turn' 설정 + labels turn_end=True(GOLDEN 픽스처 라벨은 turn_end=False 라 TURN 이 없다) → 스트림 끝 <EOT> 가 실제로 붙는다."""
    tok = ToyTok(); rows, labs = fixture_rows(tok); wp, lp = write_fixture(tmp_path, rows, [dict(x, turn_end=True) for x in labs], "turn")
    return SemCommitDataset(wp, lp, tok, online=False, written_frac=frac, **GOLDEN_CONFIGS["turn"])


@pytest.mark.parametrize("cfg,delay", [(c, d) for c in ("sem", "turn") for d in GOLDEN_CONFIGS[c]["delays"]])
def test_written_sequence_sem_after_punct_same_chunk(tmp_path, cfg, delay):
    ds = _ds(tmp_path, 1.0) if cfg == "sem" else _ds_turn_labels(tmp_path, 1.0); tok = ds.tok; sem = ds.sem_id
    rows, labs = fixture_rows(tok); raw = {r["id"]: (r, l) for r, l in zip(rows, labs)}
    for i, it in enumerate(ds.items):
        if "written" not in it: continue
        sw, sn = ds.sequence(i, delay, WRITTEN), ds.sequence(i, delay)
        assert ds.target_problems(i, delay, WRITTEN) == [] and ds.check_targets(i, delay, WRITTEN) and ds.target_problems(i, delay) == []
        assert sw["n_turn"] == sn["n_turn"] == int(cfg == "turn") and sw["turn_chunk"] == sn["turn_chunk"]      # TURN 청크 = 정규형
        pre = ds.prefix(it["lang"], delay, WRITTEN); P = len(pre)
        assert sw["ids"][:P] == pre and pre[:len(header_ids(tok, WRITTEN))] == header_ids(tok, WRITTEN)
        assert pre[len(header_ids(tok, WRITTEN)):] == ds.prefix(it["lang"], delay)[len(ds._pre):]           # 언어·δ 꼬리는 같다
        out = [t for t, inp in zip(sw["ids"], sw["is_input"]) if not inp and t not in (ds.sp.next_audio, ds.sp.empty_audio, sem, ds.turn_id)]
        want = "안녕하세요. 2007년이었대요. 여기 화곡동인데요?" if it["lang"] == "Korean" else "In 2014, we adopted it. Waxman-Markey is law."
        assert _pieces_text(tok, out) == want
        chunk = lambda s, p: max(s["chunk_of"][q] for q in range(p) if s["chunk_of"][q] >= 0)
        assert [chunk(sw, p) for p in sw["sem_pos"]] == [chunk(sn, p) for p in sn["sem_pos"]]          # SEM 청크 = 정규형
        for p in sw["sem_pos"]:
            assert tok.decode([sw["ids"][p - 1]]) in (".", "?")                                          # 끝 구두점 바로 뒤
            assert not sw["is_input"][p - 1] and sw["labels"][p] == sem
        row, lab = raw[it["id"]]; wm = written_row(row, tok)[1]                                      # 결정 청크 = 정규형(EN 단위 안쪽 후보는 버려져 짝이 없다)
        bn = sorted((c for c in lab["candidates"] if c["grade"] in "BN"), key=lambda c: c["after_word"])
        keep = [k for k, c in enumerate(bn) if wm is None or wm[c["after_word"]] is not None]
        assert [chunk(sw, p) for p, _ in sw["decision"]] == [chunk(sn, sn["decision"][k][0]) for k in keep]
        for p, g in sw["decision"]:
            assert g in ("B", "N", "B@TURN", "N@TURN") and not sw["is_input"][p]
            if g == "B": assert sw["labels"][p] == -100
            elif g == "N": assert sw["pos_weight"][p] == 1.0
            else: assert sw["labels"][p] == ds.turn_id and sw["pos_weight"][p] == 0.0                     # 결정 위치 = TURN 타깃 → 유지
        if it["lang"] == "Korean":                                                                 # B = '화곡동인데요?' → 결정 위치 바로 앞이 '?'
            pB = [p for p, g in sw["decision"] if g.startswith("B")][0]; assert tok.decode([sw["ids"][pB - 1]]) == "?"
            assert sw["n_sem"] == sn["n_sem"] == 2
        else:
            assert sw["n_sem"] == 2 and sn["n_sem"] == 2 and sw["n_B"] == 0 and sn["n_B"] == 1             # 'waxman'(B) 는 단위 안쪽 → 표기형에서 버림
            pN = [p for p, g in sw["decision"] if g == "N"]; assert [tok.decode([sw["ids"][p - 1]]) for p in pN] == [","]   # '2014,' 의 N → 쉼표 다음


def test_getitem_format_choice_collate_and_est(tmp_path):
    ds0 = _ds(tmp_path, 0.0); ds1 = _ds(tmp_path, 1.0)
    random.seed(0); xs = [ds1[i] for i in range(len(ds1))]
    for it, x in zip(ds1.items, xs):
        assert x["fmt"] == (WRITTEN if "written" in it else NORMALIZED) and x["n_written"] == int("written" in it)
        if "written" in it: assert x["ids"].tolist() == ds1.sequence(ds1.items.index(it), x["delay"], WRITTEN)["ids"]
    b = collate_semcommit(xs); assert b["n_written"] == 2
    e0, e1 = est_lens(ds0), est_lens(ds1)
    for it, a, c in zip(ds1.items, e0, e1):
        assert c == a + it.get("est_extra", 0) and (("written" in it) == ("est_extra" in it))
        if "written" in it:
            w = ds1._form(ds1.items.index(it), WRITTEN)
            assert it["est_extra"] >= len(w["tokens"]) + w["n_sem"] - len(it["tokens"]) - it["n_sem"] + len(header_ids(ds1.tok, WRITTEN)) - len(ds1._pre)
    half = _ds(tmp_path, 0.5); i_ko = [k for k, it in enumerate(half.items) if it["id"] == "slm-ko-1"][0]
    random.seed(1); n = sum(half[i_ko]["n_written"] for _ in range(400)); assert 150 < n < 250


def test_pack_batch_keeps_n_written(tmp_path):
    ds = _ds(tmp_path, 1.0); random.seed(0); b = collate_semcommit([ds[i] for i in range(len(ds))]); p = pack_batch(b)
    assert p["n_written"] == b["n_written"] == 2 and p["lang"] == b["lang"] and p["ids"].shape[0] == 1


def test_ko_textnorm_off_rejects_korean_written(tmp_path):
    with pytest.raises(ValueError, match="ko_textnorm=False"): _ds(tmp_path, 0.3, ko_textnorm=False)
    with pytest.raises(ValueError, match="ko_textnorm=False"): _ds(tmp_path, 0.3, ko_textnorm=False, langs=["Korean"])
    en = _ds(tmp_path, 0.3, ko_textnorm=False, langs=["English"]); assert {it["id"] for it in en.items if "written" in it} == {"vp-en-1"}
    assert len(_ds(tmp_path, 0.0, ko_textnorm=False).items) == 4                                  # 비율 0 이면 그대로(이 기능 전 동작)


def _late_punct_rows(tok):
    """KO speechlm 행 — 끝 구두점 토큰 시각이 앞 lexical 토큰보다 늦다(다른 청크). A 단어의 '.' 가 늦은 행 · 후보 없는 마지막 단어의 '?' 만 늦은 행."""
    a = stream_row(tok, "late-a", [("안녕하세요.", [0.30, 0.52, 0.60]), ("여기", 1.62), ("화곡동인데요?", [1.90, 2.20, 2.20])], "Korean", "speechlm-partial",
                   raw_texts=["안녕하세요 여기 화곡동인데요"], display_text="안녕하세요. 여기 화곡동인데요?")
    b = stream_row(tok, "late-b", [("안녕하세요.", [0.30, 0.52, 0.52]), ("여기", 1.62), ("화곡동인데요?", [1.90, 2.20, 2.60])], "Korean", "speechlm-partial",
                   raw_texts=["안녕하세요 여기 화곡동인데요"], display_text="안녕하세요. 여기 화곡동인데요?")
    ok = stream_row(tok, "ok-c", [("안녕하세요.", [0.30, 0.52, 0.52]), ("여기", 1.62), ("화곡동인데요?", [1.90, 2.20, 2.20])], "Korean", "speechlm-partial",
                    raw_texts=["안녕하세요 여기 화곡동인데요"], display_text="안녕하세요. 여기 화곡동인데요?")
    return [a, b, ok], [labels_of(x, "Korean", [(0, "A"), (1, "N")]) for x in ("late-a", "late-b", "ok-c")]


@pytest.mark.parametrize("cfg", sorted(GOLDEN_CONFIGS))
def test_written_dropped_when_chunks_would_differ(tmp_path, cfg):
    tok = ToyTok(); rows, labs = _late_punct_rows(tok); wp, lp = write_fixture(tmp_path, rows, labs, "late")
    d0, d1 = (SemCommitDataset(wp, lp, tok, online=False, written_frac=f, **GOLDEN_CONFIGS[cfg]) for f in (0.0, 1.0))
    assert {it["id"]: it.get("written_why") for it in d1.items} == {"late-a": "cand_time_shift", "late-b": "tail_shift", "ok-c": None}
    assert (d1.stats["written_cand_time_shift"], d1.stats["written_tail_shift"], d1.stats["written_available"]) == (1, 1, 1)
    assert [it["K"] for it in d1.items] == [it["K"] for it in d0.items]                           # K·정규형 시퀀스는 플래그와 무관(코드로 보장)
    assert normalized_digest(d1)["seqs"] == normalized_digest(d0)["seqs"]


def test_written_stats_count_remaining_items(tmp_path):
    for n in (1, 2, 3):
        ds = _ds(tmp_path, 1.0, max_items=n); st = ds.stats
        why = sum(v for k, v in st.items() if k.startswith("written_") and not k.startswith("written_cand_dropped_")
                  and k not in ("written_available", "written_hours", "written_tokens", "written_only_skipped"))
        assert why + st["written_available"] == len(ds) == n
        want = sum(it["written"].get("dropped", {}).get(g, 0) for it in ds.items if "written" in it for g in "ABN")
        assert sum(v for k, v in st.items() if k.startswith("written_cand_dropped_")) == want


def test_written_short_pool(tmp_path):
    tok = ToyTok(); rows, labs = fixture_rows(tok)
    ko = dict(rows[0], training_eligible=True, candidate_only=False, approval_fingerprint="fp-test"); assert ko["duration_s"] < 8.0
    wp, lp = write_fixture(tmp_path, [ko], [labs[0]], "short")
    ds = SemCommitDataset(wp, lp, tok, online=False, source_pool="speechlm-short", langs=["Korean"], written_frac=1.0, **GOLDEN_CONFIGS["sem"])
    assert len(ds) == 1 and ds.formats(0) == [NORMALIZED, WRITTEN]
    assert all(ds.target_problems(0, d, f) == [] for d in ds.delays for f in (NORMALIZED, WRITTEN))
    with pytest.raises(ValueError, match="non_speechlm_in_short_pool"):
        SemCommitDataset(*write_fixture(tmp_path, [ko, rows[3]], [labs[0], labs[3]], "short2"), tok, online=False, source_pool="speechlm-short", langs=["Korean"], written_frac=1.0)


def test_packing_est_extra_optional():
    class _DS:
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
    a = _DS([dict(K=10, tokens=[(1, 0.0)] * 5, n_sem=1), dict(K=20, tokens=[(1, 0.0)] * 7)])
    b = _DS([dict(K=10, tokens=[(1, 0.0)] * 5, n_sem=1, est_extra=9), dict(K=20, tokens=[(1, 0.0)] * 7)])
    assert est_lens(a).tolist() == [24 + 28 + 5 + 1, 24 + 48 + 7] and (est_lens(b) - est_lens(a)).tolist() == [9, 0]


# ── 지문: 비율이 0 이면 키가 없다(진행 중 run 재개), > 0 이면 output_format
def test_fingerprint_keys_only_when_enabled(tmp_path):
    tok = ToyTok(); rows, labs = fixture_rows(tok); wp, lp = write_fixture(tmp_path, rows, labs)
    base = semcommit_fingerprint(wp, lp, tok, turn_end=False, delays=[2, 3])
    off = output_format_settings(tok, {"English": 0.0, "Korean": 0.0}); assert off == {}
    assert semcommit_fingerprint(wp, lp, tok, turn_end=False, delays=[2, 3], **off) == base
    on = output_format_settings(tok, {"English": 0.6, "Korean": 0.3})
    fp = semcommit_fingerprint(wp, lp, tok, turn_end=False, delays=[2, 3], **on)
    assert [d.split(":")[0] for d in fingerprint_diff(base, fp)] == ["output_format"]
    of = fp["output_format"]
    assert of["protocol"] == OUTFMT_PROTOCOL and of["instruction"] == WRITTEN_INSTRUCTION and of["builder"] == WRITTEN_BUILDER and of["digit_hold"] is True
    assert of["written_frac"] == {"English": 0.6, "Korean": 0.3} and len(of["header_sha256"]) == 64 and of["rules_sha256"] == written_rules_sha256()


def test_rules_digest_tracks_rule_changes(monkeypatch):
    """WRITTEN_BUILDER 를 올리지 않은 규칙 변경(상수·textnorm·숫자 읽기)도 지문의 rules_sha256 을 바꾼다."""
    base = written_rules_sha256()
    for name, val in (("MIN_CASE_WORDS", 5), ("_OPEN", frozenset("([{"))):
        monkeypatch.setattr(semcommit_written, name, val); written_rules_sha256.cache_clear()
        try: assert written_rules_sha256() != base, name
        finally: monkeypatch.undo(); written_rules_sha256.cache_clear()
    orig = semcommit_written.target_en                                                            # textnorm 규칙 변경 흉내('percent' → 'per cent')
    monkeypatch.setattr(semcommit_written, "target_en", lambda T, c: orig(T, c).replace("percent", "per cent"))
    semcommit_written._key.cache_clear(); written_rules_sha256.cache_clear()
    try: assert written_rules_sha256() != base
    finally: monkeypatch.undo(); semcommit_written._key.cache_clear(); written_rules_sha256.cache_clear()
    assert written_rules_sha256() == base


@pytest.mark.parametrize("extra,msg", [(["--written-frac-ko", "0.3", "--no-ko-textnorm"], "--no-ko-textnorm"), (["--written-frac", "1.5"], "[0, 1]")])
def test_train_script_rejects_bad_written_args(tmp_path, extra, msg):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = subprocess.run([sys.executable, os.path.join(root, "experiments", "semcommit_train.py"), "--init", "qwen", "--train-words", "w", "--train-labels", "l",
                        "--out-dir", str(tmp_path / "o"), "--dry-run"] + extra, capture_output=True, text=True, cwd=root, timeout=300)
    assert p.returncode != 0 and msg in p.stderr and not (tmp_path / "o").exists()


# ── 실 Qwen3-ASR tokenizer(있을 때만): 머리 ids·지시문 길이·이어 붙임 동치
@pytest.mark.skipif(not os.environ.get("SEMCOMMIT_TOKENIZER"), reason="SEMCOMMIT_TOKENIZER(Qwen3-ASR tokenizer 디렉터리) 없음")
def test_real_qwen_header_ids():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["SEMCOMMIT_TOKENIZER"]); pre = header_ids(tok, NORMALIZED)
    assert pre == [151644, 8948, 198, 151645, 198, 151644, 77091, 198]
    ins = tok(WRITTEN_INSTRUCTION, add_special_tokens=False)["input_ids"]
    assert len(ins) == 14 and header_ids(tok, WRITTEN) == pre[:3] + ins + pre[3:]
    rows, _ = fixture_rows(tok)                                                                    # 실 tokenizer 로 만든 행도 표기형 selfcheck 를 통과한다
    assert tok.decode([t[0] for t in written_row(rows[0], tok)[0]["tokens"]]) == rows[0]["display_text"]
    assert tok.decode([t[0] for t in written_row(rows[1], tok)[0]["tokens"]]) == rows[1]["pnc_text"]
    rw, wm = written_row(_en(tok, "x", "he said we won today", "He said ( we won ) today."), tok)
    assert wm == [0, 1, 2, 3, 4] and [tok.decode([t[0] for t in rw["tokens"][w["a"]:w["b"]]]) for w in rw["words"]] == ["He", " said", " (we", " won )", " today."]
