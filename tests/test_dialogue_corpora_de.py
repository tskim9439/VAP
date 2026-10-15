import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vapasr.data import dialogue_corpora as C  # noqa: E402


def test_kec_utterances_split_on_long_pause_and_mark_flags():
    words = [(0.0, 1.0, "<P>"), (1.0, 1.3, "was"), (1.3, 1.5, "<P>"), (1.5, 1.8, "[und]"), (1.8, 2.0, "<HÄSITATION>"),
             (2.0, 2.8, "<P>"),                                          # ≥ 0.5 s: 발화 끝
             (2.8, 3.0, "#Erfahrung#Erf"), (3.0, 3.1, "<LACHEN>"), (3.1, 3.5, "Haus"), (3.5, 4.5, "<ATMEN>"),
             (4.5, 4.9, "<UNVERSTÄNDLICH>")]
    us = C.kec_utterances(words, "AS_002", "AS_002_p1")
    assert [u.raw for u in us] == ["was und äh", "Erf Haus", ""]
    assert (us[0].start, us[0].end, us[0].flags) == (1.0, 2.0, None)
    assert us[1].flags == ["discontinued"] and us[1].start == 2.8 and us[1].end == 3.5   # 웃음(<0.5 s)은 구간을 넓히지 않고 끊지도 않는다
    assert us[2].flags == ["unintelligible"]                                           # 알아듣기 어려운 구간만 남은 발화: 시각은 유지, 감독 제외
    assert us[0].word_timing == [("was", 1.0, 1.3), ("und", 1.5, 1.8), ("äh", 1.8, 2.0)]


def test_kec_long_utterance_breaks_at_short_pause():
    words = [(i * 1.0, i * 1.0 + 0.8, f"w{i}") for i in range(25)]
    words = [x for i in range(25) for x in ([words[i]] + [(i + 0.8, i + 1.0, "<P>")])]
    us = C.kec_utterances(words, "IS_001", "p")
    assert len(us) >= 2 and all(u.end - u.start <= C.KEC_MAX_S + 1.0 for u in us)


def test_ztc_text_brackets_unintelligible_fragment():
    assert C.ztc_text("(und) dann (es ist) gut") == ("und dann es ist gut", set())
    raw, fl = C.ztc_text("das ??? da zwei- drei")
    assert raw == "das da drei" and fl == {"unintelligible", "fragment"}


def test_read_textgrid(tmp_path):
    tg = tmp_path / "x.TextGrid"
    tg.write_text('File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 2\ntiers? <exists>\nsize = 1\nitem []:\n'
                  '    item [1]:\n        class = "IntervalTier"\n        name = "TRN"\n        xmin = 0\n        xmax = 2\n        intervals: size = 2\n'
                  '        intervals [1]:\n            xmin = 0\n            xmax = 1.5\n            text = "er sagte ""ja"""\n'
                  '        intervals [2]:\n            xmin = 1.5\n            xmax = 2\n            text = ""\n', encoding="utf-8")
    assert C.read_textgrid(str(tg)) == {"TRN": [(0.0, 1.5, 'er sagte "ja"'), (1.5, 2.0, "")]}


def test_nsc_segments_drop_leak_keep_overlap():
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    import nsc_pseudo_label as N
    fps = N.FPS; T = 10 * fps
    e_a = np.full(T, -70.0); e_b = np.full(T, -70.0)
    e_a[1 * fps:3 * fps] = -30.0; e_b[1 * fps:3 * fps] = -56.0          # A 혼자 말함, B 채널엔 26 dB 아래 누설
    e_b[4 * fps:6 * fps] = -30.0; e_a[4 * fps:6 * fps] = -56.0          # B 혼자 말함, A 채널엔 누설
    e_a[7 * fps:8 * fps] = -32.0; e_b[7 * fps:8 * fps] = -31.0          # 둘이 동시에 말함
    sa, sb = N.segments_from_energy(e_a, e_b), N.segments_from_energy(e_b, e_a)
    assert [(round(s), round(e)) for s, e in sa] == [(1, 3), (7, 8)]
    assert [(round(s), round(e)) for s, e in sb] == [(4, 6), (7, 8)]


def test_kec_anonymized_name_dropped_from_text_not_span():
    us = C.kec_utterances([(0.0, 0.4, "die"), (0.4, 0.9, "Katze"), (0.9, 1.4, "NANA"), (1.4, 2.4, "<P>"), (2.4, 2.9, "NANANA")], "AS_002", "p")
    assert [(u.raw, u.start, u.end) for u in us] == [("die Katze", 0.0, 1.4)]          # 이름만 남은 발화는 사라지고 번호는 그대로
    assert us[0].utt_id == "p_00000"


def test_canon_de_fixed_guided_and_variants():
    t, st = C.canon_de("ich glaub des is nich so äh gibts n hund", "ich glaube das ist nicht so mhm gibt's einen hund")
    assert t == "ich glaube das ist nicht so mhm gibt's einen hund"
    assert st == {"fixed": 4, "guided": 3}                         # glaub·is·nich·gibts = 고정 대응, des·äh·n = 가설이 고른 후보
    assert C.canon_de("des mannes n hund", None)[0] == "des mannes n hund"       # 모호한 축약은 가설이 없으면 그대로(속격 des 보존)
    assert C.canon_de("das ist gross", "das ist groß")[0] == "das ist groß"
    assert C.canon_de("das ist groß", "das ist gross")[0] == "das ist groß"       # 반대 방향(ß → ss)은 하지 않는다
    assert C.canon_de("ich weiss das er kuckt ehm", "ich weiß dass er guckt", C.DE_CORPUS_CANON["ztc"])[0] == "ich weiß dass er guckt ähm"
    assert C.canon_de("ehm ja", None)[0] == "ehm ja"                          # 코퍼스 고유 대응은 그 코퍼스에만
    assert C.canon_de("ich find ein haus", "ich finde eine haus") == ("ich finde eine haus", {"variant": 2})   # 끝 e 탈락 복원은 가설을 따른다


def test_de_word_err_ignores_fillers():
    assert C.de_word_err("äh ich weiß nicht", "ich weiß nicht") == 0.0
    assert abs(C.de_word_err("ich weiß nicht", "ich weiß") - 1 / 3) < 1e-9


def test_nsc_number_expansion_for_teacher_agreement():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    import nsc_pseudo_label as N
    assert [N.num_de(n) for n in (0, 1, 7, 16, 21, 30, 99, 100, 101, 345, 1000, 2024, 21000)] == [
        "null", "eins", "sieben", "sechzehn", "einundzwanzig", "dreißig", "neunundneunzig", "einhundert", "einhunderteins", "dreihundertfünfundvierzig",
        "eintausend", "zweitausendvierundzwanzig", "einundzwanzigtausend"]
    assert N.expand_digits_de("6, 5, 9, 2 und Klasse 3") == "sechs, fünf, neun, zwei und Klasse drei"
    assert N.expand_digits_de("0172 1.500") == "null eins sieben zwei eintausendfünfhundert"


def test_make_split_v2_german_units(tmp_path):
    import json, subprocess
    def write(corpus, ds):
        with open(tmp_path / f"{corpus}.dialogues.jsonl", "w") as f:
            for d in ds: f.write(json.dumps(d) + "\n")
    write("kec", [dict(conv_id=f"kec:rec_{i:03d}_p{p}", speakers=[f"AS_{2*i}", f"IS_{2*i+1}"], duration_s=600, meta=dict(pair=f"rec_{i:03d}"))
                  for i in range(20) for p in (1, 2)])
    write("ztc", [dict(conv_id=f"ztc:{a}{b}{s}", speakers=[a, b], duration_s=300, meta={}) for a, b in (("AW", "AV"), ("AW", "AK"), ("LM", "LS"), ("JD", "TB"))
                  for s in "abc"])
    write("nsc", [dict(conv_id=f"nsc:m{i:03d}", speakers=["speaker", "interlocutor"], duration_s=400, meta=dict(interlocutor="aden" if i < 2 else "amsterdam"))
                  for i in range(10)])
    base = tmp_path / "v1.json"; base.write_text(json.dumps(dict(version="v1", frac=0.05, corpora=dict(ami=dict(heldout=["ami:x"])))))
    out = tmp_path / "v2.json"
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "experiments", "p2_make_split.py"), "--data", str(tmp_path), "--base", str(base),
                    "--corpora", "kec,ztc,nsc", "--frac", "0.1", "--version", "v2", "--out", str(out)], check=True, capture_output=True)
    sp = json.loads(out.read_text())["corpora"]
    assert sp["ami"] == dict(heldout=["ami:x"])                                             # v1 코퍼스는 그대로
    assert sp["kec"]["n_heldout"] == 4 and len({c.split("_p")[0] for c in sp["kec"]["heldout"]}) == 2   # 쌍 2/20, 그 쌍의 파트 전부
    held = set(sp["ztc"]["heldout"])
    assert ({"ztc:AWAVa", "ztc:AWAKa"} <= held) or not ({"ztc:AWAVa", "ztc:AWAKa"} & held)  # AW 를 공유하는 두 dyad 는 함께 움직인다
    assert sp["nsc"]["heldout"] == ["nsc:m000", "nsc:m001"]


def test_otospeech_merges_repeated_subsegments(tmp_path):
    import numpy as np, soundfile as sf
    d = tmp_path / "421"; d.mkdir(); (d / "metadata.json").write_text('{"task_id": "421"}')
    for c in (1, 2): sf.write(str(d / f"speaker_{c}_audio.wav"), np.zeros(16000 * 10, dtype=np.float32), 16000)
    srt = ("1\n00:00:01,000 --> 00:00:01,200\n[Filler] I like it, really.\n\n2\n00:00:01,201 --> 00:00:03,000\n[Normal Turn] I like it, really.\n\n"
           "3\n00:00:03,001 --> 00:00:03,100\n[Channel Bleed] noise\n\n4\n00:00:05,000 --> 00:00:05,400\n[Acknowledgement Backchannel] yeah\n\n"
           "5\n00:00:06,000 --> 00:00:06,300\n[Acknowledgement Backchannel] yeah\n\n6\n00:00:07,000 --> 00:00:08,000\n[Strong Floor Hold] it's, yeah\n\n"
           "7\n00:00:08,500 --> 00:00:08,700\n[Non-Speech Noise] mic bump\n")
    (d / "speaker_1_annotation_a.srt").write_text(srt); (d / "speaker_2_annotation_a.srt").write_text("")
    dl = C.load_otospeech_dialogue(str(d))
    assert [(u.start, u.end, u.raw, u.utt_id) for u in dl.utterances] == [(1.0, 3.0, "I like it, really.", "s1_00000"), (5.0, 5.4, "yeah", "s1_00003"), (6.0, 6.3, "yeah", "s1_00004"),
                                                                              (7.0, 8.0, "it's, yeah", "s1_00005")]   # 비음성 밖 라벨은 발화, 잡음은 아님
