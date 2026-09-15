"""대화 결합 합성 — 블록이 mutual silence 에서 잘리고, 화자가 원 대화별로 구별되며, EOT 가 mask 되고, 채널 조각이 원본 구간을 가리키는지."""
import numpy as np, pytest, torch
sf = pytest.importorskip("soundfile")
from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef, build_episodes
from vapasr.data.dialogue_stitch import stitch, _silence_points
from vapasr.data.dialogue_mix import mix_dialogue
from vapasr.data.dialogue_dataset import DialogueWindowDataset
from tests.test_dialogue_dataset import FakeTok
SR = 16000

def two_speaker(tmp_path, name, dur=60.0, f=(440, 660)):
    st = str(tmp_path / f"{name}.wav"); t = np.arange(int(dur * SR)) / SR
    sf.write(st, np.stack([0.3 * np.sin(2 * np.pi * f[0] * t), 0.3 * np.sin(2 * np.pi * f[1] * t)], 1).astype(np.float32), SR)
    utts = []
    for k in range(int(dur // 6)):
        s = k * 6.0; utts.append(Utterance("0", s, s + 2.0, "ab", "ab", tokens=[(11, s + 1.0), (12, s + 2.0)], utt_id=f"a{k}"))
        utts.append(Utterance("1", s + 3.0, s + 5.0, "cd", "cd", tokens=[(13, s + 4.0), (14, s + 5.0)], utt_id=f"b{k}"))
    return Dialogue(conv_id=name, corpus="test", lang="Korean", split="train", duration_s=dur, speakers=["0", "1"], utterances=utts, channels={"0": ChannelRef(path=st + "#ch0"), "1": ChannelRef(path=st + "#ch1")})

def test_stitch_structure(tmp_path):
    A = two_speaker(tmp_path, "A"); B = two_speaker(tmp_path, "B", f=(500, 700)); C = two_speaker(tmp_path, "C", f=(550, 750))
    assert 5.5 in _silence_points(A) and 1.0 not in _silence_points(A)
    S = stitch([A, B, C], seed=3, block_s=(12.0, 20.0))
    assert S.meta["mask_eot"] and S.meta["synthetic"] == "stitch" and len(S.speakers) == 6 and S.duration_s > 100
    assert all(u.start >= 0 and u.end <= S.duration_s + 1e-6 for u in S.utterances) and sum(1 for u in S.utterances if u.tokens) == len(S.utterances)
    # 같은 원 대화의 화자는 서로 다른 블록에서 재등장한다(순서 교대)
    firsts = {}; seen_again = 0
    for u in S.utterances:
        j = u.speaker.split(":")[0]
        if j in firsts and u.start - firsts[j] > 25: seen_again += 1
        firsts.setdefault(j, u.start)
    assert seen_again > 0
    # 채널 조각은 원본 구간을 4-튜플로 가리키고, 혼합이 된다
    assert all(len(pc) == 4 for ref in S.channels.values() for pc in ref.pieces)
    mono, meta = mix_dialogue(S); assert mono.shape[0] == int(round(S.duration_s * SR)) and not meta["missing_channels"]

def test_stitch_windows_mask_eot_and_can_reassign(tmp_path):
    dl = [two_speaker(tmp_path, n, f=(400 + 30 * i, 600 + 30 * i)) for i, n in enumerate("ABCD")]
    S = stitch(dl, seed=1, block_s=(10.0, 15.0)); p = str(tmp_path / "s.dialogues.jsonl"); open(p, "w").write(S.to_json() + "\n")
    ds = DialogueWindowDataset([p], FakeTok(), window_s=(30.0, 40.0), hop_s=10.0, delays=(4,), R=6, seed=0)
    assert len(ds) > 0
    eps, info = ds.window_episodes(ds.dlgs[S.conv_id], ds.items[0][1], ds.items[0][2])
    assert all(ep.p_end is None and ep.outcome == "synthetic" for ep in eps)
    s = ds[0]; assert s["soft_pos"].numel() == 0 and (s["labels"][s["ids"] == ds.sp.eot] == -100).all()
    # 8 명 세션 전체를 lazy-free R=6 으로 배정하면 재배정이 생긴다
    from vapasr.data.lane_alloc import allocate
    st = allocate(build_episodes(S), R=6); assert st.reassigned > 0 and st.exhausted == 0
