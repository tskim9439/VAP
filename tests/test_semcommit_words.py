"""SEM_END v0 words.jsonl 빌더(vapasr/data/semcommit_words.py, experiments/semcommit_build_words.py) — 합성 데이터·가짜 byte-level tokenizer 로 CPU 몇 초.
실 Qwen tokenizer 검사는 SEMCOMMIT_TOKENIZER(tokenizer.json 또는 vocab.json·tokenizer.json 을 담은 디렉터리 — 디렉터리면 실 vocab.json 도 검사)가 있을 때만."""
import gzip, json, os, sys
from pathlib import Path
import pytest
from vapasr.data.kspon import normalize_kspon_v1
from vapasr.data import semcommit_words as sw

MXC = sw.MXC_PREFIX
ROOTS = {"LibriSpeech": "/data5/LibriSpeech", "KsponSpeech": "/data4/tskim/DBs/KsponSpeech/extracted"}

class FakeTok:
    """byte-level BPE 흉내: 단어 바이트(앞 공백 포함)를 chunk 바이트씩 잘라 piece 로 — KO 는 UTF-8 음절 중간에서 잘린다.
    단어의 앞 조각은 0.08 s 이른 시각(비균일 토큰 시각 → 단어 end_time = max)."""
    def __init__(self, chunk=4):
        self.chunk, self.p2i, self.i2p = chunk, {}, {}; self.special_ids = {151705}; self.i2p[151705] = "<NEXT_AUDIO>"
    def _id(self, b):
        p = "".join(sw._B2U[x] for x in b)
        if p not in self.p2i: i = 100 + len(self.p2i); self.p2i[p] = i; self.i2p[i] = p
        return self.p2i[p]
    def encode(self, words, ends, first=True):
        out = []
        for k, (w, t) in enumerate(zip(words, ends)):
            b = ((" " if (k or not first) else "") + w).encode("utf-8"); cuts = list(range(0, len(b), self.chunk))
            out += [[self._id(b[s:s + self.chunk]), t if m == len(cuts) - 1 else round(t - 0.08, 3)] for m, s in enumerate(cuts)]
        return out
    def convert_ids_to_tokens(self, ids): return [self.i2p.get(int(i)) for i in ids]

def make_rows(tok, sid, utts, lang="English", corpus="LibriSpeech", sub="test-clean", name="librispeech-test", mode="stream", tail=0.3, raws=None):
    """utts = [(utt_id, words, ends(스트림 절대), offset_s, dur_s)] → (aligned_row, streams_row)."""
    segs_a, segs_s, toks, prev = [], [], [], 0.0
    for j, (uid, words, ends, off, dur) in enumerate(utts):
        path = MXC + (f"LibriSpeech/{sub}/1/2/{uid}.flac" if corpus == "LibriSpeech" else f"KsponSpeech/{sub}/{uid}.pcm")
        toks += tok.encode(words, ends, first=(j == 0)); lex = " ".join(words)
        segs_a.append(dict(path=path, offset_s=off, silence_before_s=round(off - prev, 3), dur_s=dur))
        segs_s.append(dict(utt_id=uid, path=path, silence_before_s=round(off - prev, 3), offset_s=off, dur_s=dur, text=lex, lexical_text=lex,
                           raw_text=(raws[j] if raws else lex.upper()), display_source="none")); prev = off + dur
    D = round(prev + tail, 3); text = " ".join(w for u in utts for w in u[1])
    a = dict(name=name, id=sid, K=int(round(D * 12.5)), npy=None, lang=lang, subset=sub, mode=mode, tokens=toks, text=text, duration_s=D, segments=segs_a)
    s = dict(id=sid, corpus=corpus.lower(), split="test", subset=sub, mode=mode, lang=lang, duration_s=D, n_utts=len(utts), segments=segs_s, silence_after_s=tail)
    return a, s

LS_UTTS = [("1-2-0000", ["he", "hoped", "there", "would", "be", "stew", "for", "dinner"], [1.2, 1.6, 1.84, 2.0, 2.16, 2.56, 2.8, 3.2], 0.5, 3.0),
           ("1-2-0001", ["stuff", "it", "into", "you", "his", "belly", "counselled", "him"], [4.8, 5.04, 5.28, 5.52, 5.76, 6.08, 6.56, 6.8], 4.2, 2.8)]

# ───────────────────────── 경로 ─────────────────────────
def test_remap_path():
    ls = MXC + "LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"; ks = MXC + "KsponSpeech/KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000001.pcm"
    assert sw.remap_path(ls, ROOTS) == "/data5/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"
    assert sw.remap_path(ks, ROOTS) == "/data4/tskim/DBs/KsponSpeech/extracted/KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000001.pcm"
    assert sw.remap_path(sw.remap_path(ls, ROOTS), ROOTS).endswith(".flac")                                   # 이미 remap 된 경로는 그대로
    ev = MXC + "KsponSpeech/eval_clean/KsponSpeech_E00001.pcm"
    assert sw.remap_path(ev, ROOTS) == "/data4/tskim/DBs/KsponSpeech/extracted/eval_clean/KsponSpeech_E00001.pcm"
    assert sw.remap_path(ev, ROOTS, exists=lambda p: "KsponSpeech_eval/" in p).endswith("KsponSpeech_eval/eval_clean/KsponSpeech_E00001.pcm")
    with pytest.raises(FileNotFoundError): sw.remap_path(ev, ROOTS, exists=lambda p: False)
    with pytest.raises(ValueError): sw.remap_path("/lustre/x/LibriSpeech/a.flac", ROOTS)
    with pytest.raises(ValueError): sw.remap_path(MXC + "VoxPopuli/a.wav", ROOTS)

def test_remap_path_already_rooted_still_checks_exists():
    dev = "/data5/LibriSpeech/dev-clean/1/2/x.flac"                                                          # rack4 에 없는 split
    assert sw.remap_path(dev, ROOTS) == dev                                                                   # exists 없으면 그대로
    with pytest.raises(FileNotFoundError): sw.remap_path(dev, ROOTS, exists=lambda p: False)                  # 이미 rack4 경로여도 존재 확인
    assert sw.remap_path(dev, ROOTS, exists=lambda p: True) == dev
    ev = "/data4/tskim/DBs/KsponSpeech/extracted/eval_clean/KsponSpeech_E00001.pcm"
    assert sw.remap_path(ev, ROOTS, exists=lambda p: "KsponSpeech_eval/" in p) == "/data4/tskim/DBs/KsponSpeech/extracted/KsponSpeech_eval/eval_clean/KsponSpeech_E00001.pcm"
    assert sw.remap_path(dev, {"LibriSpeech": "/data5/LibriSpeech/", "KsponSpeech": "/k"}) == dev                # root 끝 '/' 허용

# ───────────────────────── 단어 분할 ─────────────────────────
def test_split_words_en_multi_utterance():
    tok = FakeTok(chunk=3); a, _ = make_rows(tok, "x", LS_UTTS)
    ids, ts = [t[0] for t in a["tokens"]], [t[1] for t in a["tokens"]]
    ws = sw.split_words(ids, ts, tok, a["text"])
    assert [w["text"] for w in ws] == a["text"].split() and [w["i"] for w in ws] == list(range(16))
    assert ws[0]["a"] == 0 and all(w1["a"] == w0["b"] for w0, w1 in zip(ws, ws[1:])) and ws[-1]["b"] == len(ids)
    assert [w["end_time"] for w in ws] == LS_UTTS[0][2] + LS_UTTS[1][2]                                       # 단어 끝 = 토큰 시각 max
    counselled = ws[14]; assert counselled["b"] - counselled["a"] == 4 and ts[counselled["a"]] == 6.48          # ' counselled' = 11 바이트 → 4 조각, 앞 조각은 이른 시각

def test_split_words_korean_bytes_and_fail_closed():
    tok = FakeTok(chunk=4); words = ["아", "몬", "소리야", "그건", "또"]; toks = tok.encode(words, [1.264, 1.504, 1.824, 2.304, 2.624])
    ids, ts = [t[0] for t in toks], [t[1] for t in toks]
    assert any(sw.piece_bytes(p)[:1] not in (b" ",) and (sw.piece_bytes(p)[0] & 0xC0) == 0x80 for p in tok.convert_ids_to_tokens(ids))   # 음절 중간에서 잘린 조각 존재
    ws = sw.split_words(ids, ts, tok, " ".join(words)); assert [w["text"] for w in ws] == words and ws[2]["end_time"] == 1.824
    assert sw.split_words(ids, ts, tok, "아 몬 소리 야 그건 또") is None                                        # text 불일치 → fail closed
    assert sw.split_words(ids + [151705], ts + [3.0], tok) is None                                            # 특수 토큰(바이트로도 읽히는 '<NEXT_AUDIO>') → None
    assert sw.split_words(ids + [99999], ts + [3.0], tok) is None                                             # 미지 id → None
    assert sw.split_words([], [], tok) is None
    with pytest.raises(ValueError): sw.split_words(ids, ts[:-1], tok)

def test_piece_vocab_reads_tokenizer_json_and_vocab_json(tmp_path):
    tok = FakeTok(); toks = tok.encode(["hello", "world"], [0.5, 0.9]); ids = [t[0] for t in toks]
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    tj = dict(model=dict(type="BPE", vocab=tok.p2i), added_tokens=[dict(id=151705, content="<NEXT_AUDIO>", special=True)])
    (tmp_path / "a" / "tokenizer.json").write_text(json.dumps(tj)); (tmp_path / "b" / "vocab.json").write_text(json.dumps(tok.p2i))
    (tmp_path / "b" / "added_tokens.json").write_text(json.dumps({"<NEXT_AUDIO>": 151705}))
    for v in (sw.PieceVocab(str(tmp_path / "a")), sw.PieceVocab(str(tmp_path / "a" / "tokenizer.json")), sw.PieceVocab(str(tmp_path / "b"))):
        assert [w["text"] for w in sw.split_words(ids, [t[1] for t in toks], v, "hello world")] == ["hello", "world"]
        assert 151705 in v.special_ids and sw.split_words(ids + [151705], [t[1] for t in toks] + [1.0], v) is None

def test_piece_vocab_vocab_json_with_model_and_added_tokens_pieces(tmp_path):
    """실 Qwen vocab.json 에는 'model'(id 2528)·'vocab' 같은 토큰이 있다 — tokenizer.json 판별이 그 키에 속으면 안 된다."""
    tok = FakeTok(); toks = tok.encode(["the", "model", "vocab"], [0.5, 0.9, 1.3]); ids = [t[0] for t in toks]
    vocab = dict(tok.p2i, model=2528, vocab=2529, added_tokens=2530); vocab[sw._B2U[32] + "model"] = 1614
    (tmp_path / "vocab.json").write_text(json.dumps(vocab)); (tmp_path / "added_tokens.json").write_text(json.dumps({"<NEXT_AUDIO>": 151705}))
    v = sw.PieceVocab(str(tmp_path / "vocab.json"))
    assert v.id2piece[2528] == "model" and v.id2piece[2530] == "added_tokens" and v.special_ids == {151705}
    assert [w["text"] for w in sw.split_words(ids, [t[1] for t in toks], v, "the model vocab")] == ["the", "model", "vocab"]
    assert [w["text"] for w in sw.split_words([2528, 1614], [0.2, 0.4], v)] == ["model", "model"]
    (tmp_path / "tj").mkdir()                                                                                 # tokenizer.json 의 vocab 에 'model' 이 있어도 같다
    (tmp_path / "tj" / "tokenizer.json").write_text(json.dumps(dict(model=dict(type="BPE", vocab=vocab), added_tokens=[dict(id=151705, content="<NEXT_AUDIO>")])))
    v2 = sw.PieceVocab(str(tmp_path / "tj")); assert v2.id2piece == v.id2piece and v2.special_ids == v.special_ids

# ───────────────────────── KO 태그 ─────────────────────────
@pytest.mark.parametrize("raw", ["아/ 몬 소리야, 그건 또. b/", "나는 악습은 원래 없어진다+ 없어져야 된다고 생각하긴 했는데 근데 그/ 약간 필요악으로 하나 정도쯤은 있어야 되거든. 물 뜨러 가고.",
                                 "b/ n/ 그래서 지호랑 계단 n/ 올라와서 b/ 막 위에 운동하는 기구 있대요. b/ 그서 그걸로 운동 할려구요. b/ n/", "한+ 한+ 한 시간에", "까* 오히려", "갈+ b/ 간다고",
                                 "(2월)/(이 월 )달에 가? 응.", "뭐/. 그래", "(7명)/(일곱 명)+ (7명)/(일곱 명)? 마지막엔", "(그래서)/(캐서*) 너는",
                                 "(2월)/(이 월 )달에+ 가", "(7*7)/(칠 칠) 이야", "(21, 1학점)/(이십 일+ 일 학점)", "(뭐)/(머)/ 그래", "일부러 손 들어서.b/",
                                 "(18학점)/(일 팔 학점)까지밖에+ (18학점)/(십 팔 학점)까지밖에 못 듣거든."])
def test_kspon_raw_tags_match_lexical_normalization(raw):
    assert [w for w, _ in sw.kspon_raw_tags(raw)] == normalize_kspon_v1(raw)[0].split()                        # 토큰 단위 적용 = 전체 문자열 정규화 후 split

def test_kspon_raw_tags_values():
    T = dict(sw.kspon_raw_tags("아/ 몬 소리야, 그건 또. b/"))
    assert T == {"아": ["filler"], "몬": [], "소리야": ["punct_comma"], "그건": [], "또": ["punct_final"]}
    assert sw.kspon_raw_tags("없어진다+ 없어져야 까* 오히려?") == [("없어진다", ["rep"]), ("없어져야", []), ("까", ["unclear"]), ("오히려", ["punct_final"])]
    assert sw.kspon_raw_tags("안 먹을래 b/.") == [("안", []), ("먹을래", ["punct_final"])]                   # 'b/.' 는 버리고 구두점은 직전 어절로
    assert sw.kspon_raw_tags("뭐/. 그래") == [("뭐", ["filler", "punct_final"]), ("그래", [])]
    assert sw.kspon_raw_tags("그 K-POP+ 노래가, 나왔는데.") == [("그", []), ("K", ["rep"]), ("POP", ["rep"]), ("노래가", ["punct_comma"]), ("나왔는데", ["punct_final"])]   # target_ko 의 [^\w\s]→공백
    assert sw.kspon_tags_for(["안", "먹을래", "b"], "안 먹을래 b/.") == (None, "kspon_tag_map_mismatch")

def test_kspon_raw_tags_dual_form_markers():
    """(철자)/(발음) 쌍 전체에 걸린 표지는 선택된 모든 어절에(kspon-full 에서 여러 어절 쌍 + 꼬리 표지 1,442 건, kspon-eval 버려진 형태의 '*' 111 건)."""
    assert sw.kspon_raw_tags("(7명)/(일곱 명)+ (7명)/(일곱 명)? 마지막엔") == [("일곱", ["rep"]), ("명", ["rep"]), ("일곱", []), ("명", ["punct_final"]), ("마지막엔", [])]
    assert sw.kspon_raw_tags("(그래서)/(캐서*) 너는") == [("그래서", ["unclear"]), ("너는", [])]                  # 버려진 발음형의 단어 끝 '*'
    assert sw.kspon_raw_tags("(2월)/(이 월 )달에+ 가") == [("이", ["rep"]), ("월", ["rep"]), ("달에", ["rep"]), ("가", [])]   # 한 raw 토큰 = 쌍 + 붙은 꼬리
    assert sw.kspon_raw_tags("(뭐)/(머)/ 그래") == [("뭐", ["filler"]), ("그래", [])]
    assert sw.kspon_raw_tags("(3G)/(쓰리 쥐)* 하나도") == [("쓰리", ["unclear"]), ("쥐", ["unclear"]), ("하나도", [])]
    assert sw.kspon_raw_tags("(7*7)/(칠 칠) 이야") == [("칠", []), ("칠", []), ("이야", [])]                   # 철자형의 곱셈 '*' 는 표지 아님
    assert sw.kspon_raw_tags("(21, 1학점)/(이십 일+ 일 학점)") == [("이십", []), ("일", ["rep"]), ("일", []), ("학점", [])]   # 선택된 형태 안 표지는 그 어절만
    assert sw.kspon_raw_tags("(7명)/(일곱 명)? 가") == [("일곱", []), ("명", ["punct_final"]), ("가", [])]      # 구두점은 마지막 어절만

def test_kspon_glued_marker():
    assert sw.kspon_raw_tags("일부러 손 들어서.b/") == [("일부러", []), ("손", []), ("들어서b", ["punct_final"])]   # 붙은 표지는 filler 가 아니다
    assert sw.kspon_glued_markers("일부러 손 들어서.b/") == ["들어서.b/"] and sw.kspon_glued_markers("그래서b/ 가") == ["그래서b/"]
    assert sw.kspon_glued_markers("아/ 뭐/. b/ 안 먹을래 b/. club/ K-POP+") == []                              # 단독 표지·filler·Latin 단어 filler 는 아님

# ───────────────────────── EN (LibriSpeech-PC) 태그 ─────────────────────────
def test_pnc_words_and_alignment():
    pc = "He hoped there would be stew for dinner, turnips and carrots; in thick, peppered, flour-fattened sauce. Stuff it!"
    pw = sw.pnc_words(pc); assert [w for w, _ in pw][-5:] == ["flour", "fattened", "sauce", "stuff", "it"]
    assert dict(pw)["dinner"] == ["punct_comma"] and dict(pw)["carrots"] == ["punct_comma"] and dict(pw)["sauce"] == ["punct_final"] and dict(pw)["flour"] == []
    lex = "he hoped there would be stew for dinner turnips and carrots in thick peppered flour fattened sauce stuff it".split()
    tags, r = sw.en_pnc_tags(lex, pc); assert r == 1.0 and tags[7] == ["punct_comma"] and tags[16] == ["punct_final"] and tags[18] == ["punct_final"]
    lex2 = list(lex); lex2[3] = "could"                                                                        # 치환 1 → 나머지는 정렬로 태그 유지
    tags2, r2 = sw.en_pnc_tags(lex2, pc); assert tags2 is not None and abs(r2 - 18 / 19) < 1e-9 and tags2[7] == ["punct_comma"] and tags2[3] == []
    assert sw.en_pnc_tags(lex[:10], pc) == (None, 10 / 19)                                                    # 일치율 < 0.9 → 태그 없음
    assert sw.pnc_words("Mister Hale's 'quite' — done... ok") == [("mister", []), ("hale's", []), ("quite", []), ("done", ["punct_final"]), ("ok", [])]   # 대시는 태그 아님
    assert sw.en_norm("Yo’") == "yo" and sw.en_norm("THERE'S,") == "there's"

def test_load_pnc_file_lines_and_list(tmp_path):
    rows = [dict(audio_filepath="test-clean/1/2/1-2-0000.flac", text="A b.", duration=1.0), dict(audio_filepath="test-clean/1/2/1-2-0001.flac", text="C, d!", duration=1.0)]
    (tmp_path / "l").mkdir(); (tmp_path / "j").mkdir()
    (tmp_path / "l" / "test-clean.json").write_text("\n".join(json.dumps(r) for r in rows) + "\n"); (tmp_path / "j" / "test-clean.json").write_text("\n  " + json.dumps(rows))   # 앞 공백 뒤 JSON list
    for d in ("l", "j"):
        assert sw.load_pnc_file(str(tmp_path / d / "test-clean.json")) == {"1-2-0000": "A b.", "1-2-0001": "C, d!"}
        idx = sw.PncIndex(str(tmp_path / d)); assert idx.lookup("1-2-0001", "test-clean") == "C, d!" and idx.lookup("1-2-0001", "test-other") is None

# ───────────────────────── segment 배정 ─────────────────────────
def test_assign_segments_count_time_fail():
    segs = [dict(offset_s=0.5, dur_s=3.0), dict(offset_s=4.2, dur_s=2.8)]
    ws = [dict(end_time=t) for t in (1.0, 3.6, 5.0, 7.1)]
    assert sw.assign_segments(ws, segs, [2, 2]) == ([0, 0, 1, 1], "count")
    assert sw.assign_segments(ws, segs, [3, 1]) == ([0, 0, 1, 1], "time")                                   # 단어 수가 시각과 모순 → 시각 창
    assert sw.assign_segments(ws, segs, [2, 3]) == ([0, 0, 1, 1], "time")                                   # 합 불일치(발화 누락 등) → 시각 창
    assert sw.assign_segments(ws + [dict(end_time=7.3)], segs, [2, 3]) == (None, "seg_assign_fail")          # 마지막 segment 끝 + 0.2 초과
    assert sw.assign_segments([dict(end_time=0.2)], segs[:1], [1]) == (None, "seg_assign_fail")              # segment 시작 전

# ───────────────────────── build_stream ─────────────────────────
def test_build_stream_librispeech_schema_and_pnc():
    tok = FakeTok(chunk=3); a, s = make_rows(tok, "ls-test-clean-1-2-00000", LS_UTTS); info = {}
    out = sw.build_stream(a, s, tok, ROOTS, info=info)
    assert set(out) == {"id", "set", "lang", "K", "duration_s", "text", "tokens", "segments", "words"} and out["set"] == "librispeech-test"
    assert out["tokens"] == a["tokens"] and out["K"] == a["K"] and out["duration_s"] == a["duration_s"] and info["seg_assign"] == "count"
    assert [g["path"] for g in out["segments"]] == ["/data5/LibriSpeech/test-clean/1/2/1-2-0000.flac", "/data5/LibriSpeech/test-clean/1/2/1-2-0001.flac"]
    assert [g["utt_id"] for g in out["segments"]] == ["1-2-0000", "1-2-0001"] and out["segments"][1]["raw_text"].startswith("STUFF")
    assert set(out["segments"][0]) == {"path", "offset_s", "silence_before_s", "dur_s", "utt_id", "raw_text"}
    assert [w["seg"] for w in out["words"]] == [0] * 8 + [1] * 8 and all(w["tags"] == [] for w in out["words"])
    assert set(out["words"][0]) == {"i", "text", "a", "b", "end_time", "seg", "tags"}
    json.dumps(out)                                                                                            # 직렬화 가능
    pnc = {"1-2-0000": "He hoped there would be stew for dinner.", "1-2-0001": "Stuff it into you, his belly counselled him."}
    info2 = {}; out2 = sw.build_stream(a, s, tok, ROOTS, pnc_lookup=pnc, info=info2)
    assert out2["pnc_text"] == pnc["1-2-0000"] + " " + pnc["1-2-0001"] and info2["pnc_ratios"] == [1.0, 1.0]
    assert [w["i"] for w in out2["words"] if w["tags"]] == [7, 11, 15] and out2["words"][11]["tags"] == ["punct_comma"]
    info3 = {}; out3 = sw.build_stream(a, s, tok, ROOTS, pnc_lookup={"1-2-0001": pnc["1-2-0001"]}, info=info3)     # 한 발화 PC 없음 → 그 발화만 태그 없음, pnc_text 없음
    assert "pnc_text" not in out3 and "pnc_missing" in info3["tag_notes"] and "pnc_text_partial" in info3["tag_notes"] and out3["words"][15]["tags"] == ["punct_final"]
    assert sw.build_stream(a, s, tok, None)["segments"][0]["path"].startswith(MXC)                           # roots=None → 경로 유지

def test_build_stream_drops():
    tok = FakeTok(chunk=3); a, s = make_rows(tok, "ls-test-clean-1-2-00000", LS_UTTS)
    assert sw.build_stream(a, dict(s, id="other"), tok, ROOTS) == (None, "streams_row_mismatch")
    assert sw.build_stream(dict(a, text=a["text"].replace("stew", "stow")), s, tok, ROOTS) == (None, "word_split_mismatch")
    assert sw.build_stream(dict(a, tokens=[]), s, tok, ROOTS) == (None, "empty_tokens")
    assert sw.build_stream(dict(a, tokens=a["tokens"][::-1]), s, tok, ROOTS) == (None, "tokens_not_sorted")
    assert sw.build_stream(dict(a, segments=a["segments"][:1]), s, tok, ROOTS) == (None, "segments_mismatch")
    assert sw.build_stream(dict(a, duration_s=6.0), s, tok, ROOTS) == (None, "word_after_stream_end")
    bad = [dict(g, path=g["path"].replace(MXC, "/lustre/x/")) for g in a["segments"]]
    assert sw.build_stream(dict(a, segments=bad), dict(s, segments=[dict(g, path=b["path"]) for g, b in zip(s["segments"], bad)]), tok, ROOTS) == (None, "unknown_path_prefix")
    assert sw.build_stream(a, s, tok, ROOTS, path_exists=lambda p: False) == (None, "audio_missing")
    # 발화 하나가 정렬에서 빠진 경우(단어 수 합 불일치) → 시각 창 배정
    s2 = dict(s, segments=[dict(s["segments"][0], lexical_text="he hoped there would be stew for dinner extra"), s["segments"][1]]); info = {}
    out = sw.build_stream(a, s2, tok, ROOTS, info=info); assert info["seg_assign"] == "time" and [w["seg"] for w in out["words"]] == [0] * 8 + [1] * 8

def test_build_stream_kspon_tags_and_stray():
    tok = FakeTok(chunk=4)
    utt = ("KsponSpeech_000001", ["아", "몬", "소리야", "그건", "또"], [1.264, 1.504, 1.824, 2.304, 2.624], 0.464, 3.148)
    a, s = make_rows(tok, "ks-train-01-KsponSpeech_000001", [utt], lang="Korean", corpus="KsponSpeech", sub="KsponSpeech_01/KsponSpeech_0001", name="kspon-full",
                     raws=["아/ 몬 소리야, 그건 또. b/"])
    info = {}; out = sw.build_stream(a, s, tok, ROOTS, info=info)
    assert [w["tags"] for w in out["words"]] == [["filler"], [], ["punct_comma"], [], ["punct_final"]] and info["tag_notes"] == []
    assert out["segments"][0]["path"] == "/data4/tskim/DBs/KsponSpeech/extracted/KsponSpeech_01/KsponSpeech_0001/KsponSpeech_000001.pcm"
    s_bad = dict(s, segments=[dict(s["segments"][0], raw_text="아/ 몬 소리야, 그건")]); info = {}                 # 1:1 실패 → 태그 없이 유지 + 사유
    out2 = sw.build_stream(a, s_bad, tok, ROOTS, info=info); assert all(w["tags"] == [] for w in out2["words"]) and info["tag_notes"] == ["kspon_tag_map_mismatch"]
    utt_b = ("KsponSpeech_E00866", ["안", "먹을래", "b"], [1.0, 1.4, 1.8], 0.5, 1.6)
    a3, s3 = make_rows(tok, "ks-eval_clean-utt-KsponSpeech_E00866", [utt_b], lang="Korean", corpus="KsponSpeech", sub="eval_clean", name="kspon-eval", mode="utt", raws=["안 먹을래 b/."])
    info = {}; assert sw.build_stream(a3, s3, tok, ROOTS, info=info) == (None, "stray_marker_letter") and info["drop_detail"] == ["안 먹을래 b"]
    utt_g = ("KsponSpeech_E00300", ["일부러", "손", "들어서b"], [1.0, 1.4, 1.8], 0.5, 1.6)                     # 붙은 표지 → lexical '들어서b'
    a4, s4 = make_rows(tok, "ks-eval_clean-utt-KsponSpeech_E00300", [utt_g], lang="Korean", corpus="KsponSpeech", sub="eval_clean", name="kspon-eval", mode="utt",
                       raws=["일부러 손 들어서.b/"])
    info = {}; assert sw.build_stream(a4, s4, tok, ROOTS, info=info) == (None, "glued_marker_letter") and info["drop_detail"] == ["들어서.b/"]
    utt_d = ("KsponSpeech_000002", ["일곱", "명", "일곱", "명", "마지막엔"], [1.0, 1.3, 1.8, 2.1, 2.6], 0.5, 2.4)   # 쌍 표지 → 모든 어절
    a5, s5 = make_rows(tok, "ks-train-01-KsponSpeech_000002", [utt_d], lang="Korean", corpus="KsponSpeech", sub="KsponSpeech_01/KsponSpeech_0001", name="kspon-full",
                       raws=["(7명)/(일곱 명)+ (7명)/(일곱 명)? 마지막엔"])
    assert [w["tags"] for w in sw.build_stream(a5, s5, tok, ROOTS)["words"]] == [["rep"], ["rep"], [], ["punct_final"], []]

# ───────────────────────── CLI ─────────────────────────
def _jl(path): return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

def _cli():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments")); import semcommit_build_words as cli; return cli

def _write_manifests(root: Path, tok, rows, streams_only=(), set_name="librispeech-test"):
    (root / set_name).mkdir(parents=True); (root / "original" / set_name).mkdir(parents=True)
    with gzip.open(root / set_name / "aligned-manifest.jsonl.gz", "wt", encoding="utf-8") as f:
        for a, _ in rows: f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(root / "original" / set_name / "streams.jsonl", "w", encoding="utf-8") as f:
        for k, (_, s) in enumerate(rows):
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            if k in streams_only: f.write(json.dumps(dict(s, id=s["id"] + "-qcdrop")) + "\n")                # 정렬 캐시에 없는 streams 행
    (root / "vocab.json").write_text(json.dumps(tok.p2i))

def test_cli_end_to_end_deterministic(tmp_path):
    cli = _cli(); tok = FakeTok(chunk=3); rows = []
    for k in range(6):
        utts = [(f"1-2-{k:02d}{u}", ws, [t + 10 * u for t in ends], off + 10 * u, dur) for u, (_, ws, ends, off, dur) in enumerate(LS_UTTS)]
        rows.append(make_rows(tok, f"ls-test-clean-1-2-{k:05d}", utts))
    a_bad, s_bad = make_rows(tok, "ls-test-clean-1-2-00099", LS_UTTS); rows.append((dict(a_bad, text=a_bad["text"] + " extra"), s_bad))   # 분할 불일치 → 제외
    rows.append(make_rows(tok, "ls-test-clean-utt-1-2-0000", LS_UTTS[:1], mode="utt"))
    rows.append(make_rows(tok, "ls-test-other-3-4-00000", LS_UTTS, sub="test-other"))
    _write_manifests(tmp_path, tok, rows, streams_only={1, 4})
    pnc = tmp_path / "pnc"; pnc.mkdir()
    (pnc / "test-clean.json").write_text("\n".join(json.dumps(dict(audio_filepath=f"test-clean/1/2/1-2-{k:02d}{u}.flac", text=t, duration=3.0))
                                                   for k in range(6) for u, t in enumerate(["He hoped there would be stew for dinner.", "Stuff it into you, his belly counselled him."])))
    base = ["--manifests", str(tmp_path), "--set", "librispeech-test", "--id-prefix", "ls-test-clean", "--mode", "stream", "--seed", "3", "--tokenizer", str(tmp_path / "vocab.json"),
            "--check-audio", "off"]                                                                            # 호스트와 무관(rack4 에는 /data5/LibriSpeech 가 있다)
    st = cli.main(base + ["--n", "3", "--out", str(tmp_path / "o1.jsonl"), "--stats", str(tmp_path / "s1.json")])
    cli.main(base + ["--n", "3", "--out", str(tmp_path / "o2.jsonl"), "--stats", str(tmp_path / "s2.json")])
    assert (tmp_path / "o1.jsonl").read_bytes() == (tmp_path / "o2.jsonl").read_bytes()                        # 결정적
    p = st["population"]; assert (p["aligned_rows"], p["streams_not_aligned"], p["candidates"], p["valid"], p["dropped"]) == (9, 2, 7, 6, {"word_split_mismatch": 1})
    o1 = _jl(tmp_path / "o1.jsonl"); assert len(o1) == 3 and st["shortfall"] == 0 and st["sample"]["words"] == 48
    order = [a["id"] for a, _ in rows]; assert [order.index(r["id"]) for r in o1] == sorted(order.index(r["id"]) for r in o1)   # manifest 순서
    cli.main(base + ["--n", "5", "--out", str(tmp_path / "o5.jsonl"), "--stats", str(tmp_path / "s5.json")])
    assert {r["id"] for r in o1} <= {r["id"] for r in _jl(tmp_path / "o5.jsonl")}                 # n 을 키우면 이전 표본 포함
    st9 = cli.main(base + ["--n", "9", "--pnc-dir", str(pnc), "--out", str(tmp_path / "o9.jsonl"), "--stats", str(tmp_path / "s9.json")])
    assert st9["shortfall"] == 3 and st9["sample"]["pnc_text"] == 6 and st9["sample"]["tags"]["punct_final"] == 12 and st9["sample"]["tags"]["punct_comma"] == 6
    s9 = json.loads(Path(tmp_path / "s9.json").read_text()); assert s9["check_audio"] == {"arg": "off", "checked": False} and s9["warnings"]
    assert s9["population"]["drop_examples"] == {"word_split_mismatch": [{"id": "ls-test-clean-1-2-00099"}]}
    with pytest.raises(SystemExit, match="--mode"):                                                            # --mode 없음 + stream·utt 혼재 → 종료
        cli.main(base[:6] + base[8:] + ["--n", "0", "--out", str(tmp_path / "oa.jsonl"), "--stats", str(tmp_path / "sa.json")])
    assert not (tmp_path / "oa.jsonl").exists()
    sto = cli.main(base[:6] + base[8:] + ["--id-prefix", "ls-test-other", "--out", str(tmp_path / "oo.jsonl"), "--stats", str(tmp_path / "so.json")])   # 한 mode 뿐이면 --mode 없이 OK
    assert sto["population"]["candidates"] == 1 and sto["mode"] is None

def test_cli_failures_and_stats_dir(tmp_path):
    cli = _cli(); tok = FakeTok(chunk=3)
    rows = [make_rows(tok, f"ls-test-clean-1-2-{k:05d}", [(f"1-2-{k:02d}{u}", ws, ends, off, dur) for u, (_, ws, ends, off, dur) in enumerate(LS_UTTS)]) for k in range(3)]
    _write_manifests(tmp_path, tok, rows)
    base = ["--manifests", str(tmp_path), "--set", "librispeech-test", "--mode", "stream", "--tokenizer", str(tmp_path / "vocab.json")]
    with pytest.raises(SystemExit, match="no rows match"):                                                    # 접두사 오타 → 출력 없이 종료
        cli.main(base + ["--id-prefix", "ls-test_clean", "--check-audio", "off", "--out", str(tmp_path / "x.jsonl"), "--stats", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.jsonl").exists() and not (tmp_path / "x.json").exists()
    st = cli.main(base + ["--check-audio", "off", "--out", str(tmp_path / "new1" / "o.jsonl"), "--stats", str(tmp_path / "new2" / "sub" / "s.json")])   # 새 디렉터리
    assert st["population"]["valid"] == 3 and json.loads(Path(tmp_path / "new2" / "sub" / "s.json").read_text())["population"]["valid"] == 3
    assert not list((tmp_path / "new2" / "sub").glob("*.tmp")) and not list((tmp_path / "new1").glob("*.tmp"))
    ls = tmp_path / "ls"                                                                                     # auto: root 가 있으면 존재 확인
    for k in range(2):
        for u in range(2): p = ls / "test-clean" / "1" / "2" / f"1-2-{k:02d}{u}.flac"; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"")
    st = cli.main(base + ["--ls-root", str(ls), "--out", str(tmp_path / "a.jsonl"), "--stats", str(tmp_path / "a.json")])
    assert st["check_audio"] == {"arg": "auto", "checked": True} and st["population"]["dropped"] == {"audio_missing": 1} and st["population"]["valid"] == 2
    assert st["warnings"] == [] and all(g["path"].startswith(str(ls)) for r in _jl(tmp_path / "a.jsonl") for g in r["segments"])
    st = cli.main(base + ["--ls-root", str(tmp_path / "nols"), "--out", str(tmp_path / "b.jsonl"), "--stats", str(tmp_path / "b.json")])   # root 없음 → 확인 안 함 + 경고
    assert st["check_audio"] == {"arg": "auto", "checked": False} and st["population"]["valid"] == 3 and "not checked" in st["warnings"][0]
    with pytest.raises(SystemExit, match="no valid streams"):                                                  # 유효 0 → stats 는 쓰고 종료
        cli.main(base + ["--ls-root", str(tmp_path / "nols"), "--check-audio", "--out", str(tmp_path / "c.jsonl"), "--stats", str(tmp_path / "c.json")])
    assert json.loads(Path(tmp_path / "c.json").read_text())["population"]["dropped"] == {"audio_missing": 3}

def test_cli_librispeech_dev_warning(tmp_path):
    cli = _cli(); tok = FakeTok(chunk=3); rows = [make_rows(tok, "ls-dev-clean-1-2-00000", LS_UTTS, sub="dev-clean", name="librispeech-dev")]
    _write_manifests(tmp_path, tok, rows, set_name="librispeech-dev")
    st = cli.main(["--manifests", str(tmp_path), "--set", "librispeech-dev", "--mode", "stream", "--tokenizer", str(tmp_path / "vocab.json"), "--check-audio", "off",
                   "--out", str(tmp_path / "o.jsonl"), "--stats", str(tmp_path / "s.json")])
    assert any("dev-clean/dev-other" in w for w in st["warnings"])

def test_cli_order_violation(tmp_path):
    cli = _cli(); tok = FakeTok(); r1 = make_rows(tok, "ls-test-clean-1-2-00000", LS_UTTS); r2 = make_rows(tok, "ls-test-clean-1-2-00001", LS_UTTS)
    _write_manifests(tmp_path, tok, [r1, r2])
    with gzip.open(tmp_path / "librispeech-test" / "aligned-manifest.jsonl.gz", "wt", encoding="utf-8") as f:          # aligned 순서가 streams 와 다름
        for a, _ in (r2, r1): f.write(json.dumps(a) + "\n")
    with pytest.raises(RuntimeError, match="not found"):
        cli.main(["--manifests", str(tmp_path), "--set", "librispeech-test", "--tokenizer", str(tmp_path / "vocab.json"), "--out", str(tmp_path / "o.jsonl"), "--stats", str(tmp_path / "s.json")])

# ───────────────────────── 실 Qwen tokenizer(선택) ─────────────────────────
@pytest.mark.skipif(not os.environ.get("SEMCOMMIT_TOKENIZER"), reason="SEMCOMMIT_TOKENIZER(Qwen3 tokenizer.json 또는 그 디렉터리) 미설정")
def test_real_qwen_tokenizer_rows():
    spec = os.environ["SEMCOMMIT_TOKENIZER"]; tok = sw.PieceVocab(spec)
    vj = os.path.join(spec, "vocab.json") if os.path.isdir(spec) else None
    if vj and os.path.isfile(vj):                                                                            # 실 vocab.json('model' 토큰 포함)도 읽혀야 한다
        v = sw.PieceVocab(vj); assert v.id2piece.get(2528) == "model"
        if os.path.isfile(os.path.join(spec, "tokenizer.json")): assert v.id2piece == tok.id2piece and v.special_ids == tok.special_ids
        tok = v
    ko = [[31079, 1.614], [140667, 2.094], [33704, 2.094], [79302, 3.854], [113, 3.854], [21329, 3.854], [17380, 3.854], [45130, 4.254], [120, 4.254], [40853, 4.254], [96137, 4.894]]
    ws = sw.split_words([t[0] for t in ko], [t[1] for t in ko], tok, "어 일단은 억지로 과장해서")                   # kspon-eval KsponSpeech_E00001 앞 11 토큰
    assert [(w["a"], w["b"], w["end_time"]) for w in ws] == [(0, 1, 1.614), (1, 3, 2.094), (3, 7, 3.854), (7, 11, 4.894)]
    en = [[383, 1.37], [25189, 1.77], [1052, 1.85], [1035, 2.01], [387, 2.17], [60343, 2.57]]
    assert [w["text"] for w in sw.split_words([t[0] for t in en], [t[1] for t in en], tok)] == "he hoped there would be stew".split()
