"""데이터 확장 새 코퍼스(2026-10-26): NIA23 002_Meeting 읽기(발화 순번 wav·임시 사본 무시·mixed_pieces — 게인 없이 이어 붙인 mono),
Switchboard 읽기(파일명 시각의 자릿수 해석·2 채널 pieces). CPU."""
import json

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
from vapasr.data.dialogue_corpora import iter_nia23_meeting, iter_switchboard, swbd_times  # noqa: E402
from vapasr.data.dialogue_mix import mix_dialogue, mix_stems  # noqa: E402

SR = 16000


def test_nia23_reader_and_mixed_mono(tmp_path):
    root = tmp_path / "Training"; (root / "Jsons" / "Recording").mkdir(parents=True); (root / "wav" / "Recording").mkdir(parents=True)
    segs = [(0.0, 1.0, "S1", "예 시작하겠습니다"), (1.0, 2.5, "S2", "네 좋습니다"), (2.5, 3.0, "S1", "그럼")]
    js = dict(metadata=dict(title="MT01", domain="의료"), speaker=[dict(id="S1", role="사회자"), dict(id="S2", role="토론자")],
              utterance=[dict(id=f"MT01.1.{i}", start=f"{s:.3f}", end=f"{e:.3f}", speaker_id=w, original_form=t, form=t) for i, (s, e, w, t) in enumerate(segs, 1)])
    (root / "Jsons" / "Recording" / "MT01.json").write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    full = (np.random.default_rng(0).standard_normal(3 * SR) * np.array([0.05] * SR + [0.4] * (SR + SR // 2) + [0.05] * (SR // 2))).astype(np.float32)
    for i, (s, e, w, t) in enumerate(segs, 1):
        sf.write(root / "wav" / "Recording" / f"MT01_{i}.wav", full[int(s * SR): int(e * SR)], SR, subtype="FLOAT")
    sf.write(root / "wav" / "Recording" / ".azDownload-x-MT01_1.wav", np.zeros(10, np.float32), SR)     # 임시 사본(무시)
    (d,) = list(iter_nia23_meeting(str(root)))
    assert d.conv_id == "nia23mt:MT01" and d.lang == "Korean" and d.speakers == ["S1", "S2"] and d.meta["mixed_pieces"] and d.meta["roles"]["S1"] == "사회자"
    assert [u.raw for u in d.utterances] == [t for *_, t in segs] and all(not p[0].split("/")[-1].startswith(".") for c in d.channels.values() for p in c.pieces)
    y, m = mix_dialogue(d); g = m["gain"] * m["peak_scale"]
    assert m["mixed_pieces"] and np.allclose(y[: 3 * SR], full * g, atol=1e-6)                        # 화자별 게인 없이 원래 혼합 녹음 그대로(한 배율)
    with pytest.raises(ValueError): mix_stems(d)


def test_switchboard_times_and_reader(tmp_path):
    assert swbd_times("sw02440B_400303_401980375", 1.69875) == (400.303, 401.980375)
    assert swbd_times("sw03322A_18271375_191563125", 8.87) == (182.71375, 191.563125)
    assert swbd_times("sw02440B_400303_401980375", 5.0) is None and swbd_times("bad_name", 1.0) is None
    (tmp_path / "switchboard" / "TRAIN").mkdir(parents=True); rows = []
    for name, dur, txt in [("sw01000A_1500_3000", 1.5, "hello there"), ("sw01000B_2500_4000", 1.5, "<LAUGH> yeah"), ("sw01000A_5000_6500", 1.5, "<LAUGH>"),
                           ("sw01000B_7000_9000", 9.0, "bad time")]:
        p = f"switchboard/TRAIN/{name}.wav"; sf.write(tmp_path / p, np.zeros(int(min(dur, 2.0) * SR), np.float32), SR); rows.append(f"{p}|{txt}|16000|{dur}")
    csv = tmp_path / "train.csv"; csv.write_text("audio_path|script|sampling_rate|duration\n" + "\n".join(rows) + "\n")
    (d,) = list(iter_switchboard(str(csv), str(tmp_path)))
    assert d.conv_id == "swbd:sw01000" and d.speakers == ["sw01000A", "sw01000B"] and d.meta["bad_times"] == 1
    assert [(u.speaker, u.start, u.end, u.raw) for u in d.utterances] == [("sw01000A", 1.5, 3.0, "hello there"), ("sw01000B", 2.5, 4.0, "yeah")]   # 태그만인 발화는 빠진다
    assert [p[1] for p in d.channels["sw01000B"].pieces] == [2.5]
