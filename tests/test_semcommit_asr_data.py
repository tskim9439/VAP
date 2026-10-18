"""ASR 정확도 개선 학습 옵션(2026-09-29) — 데이터 쪽: SEM 중립 ASR 행(sem_free)·풀 라우팅·속도 변형(vapasr/data/semcommit_dataset.py), collate sem_mask·packing. CPU."""
import importlib.util, json, math, pathlib
import numpy as np, pytest, torch
from vapasr.data.semcommit_dataset import SemCommitDataset, collate_semcommit
from vapasr.hf.packing import pack_batch

_HERE = pathlib.Path(__file__).parent
def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", _HERE / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
tsd = _load("test_semcommit_dataset")
FakeTok, make_row, make_labels, WORDS, chunk_at = tsd.FakeTok, tsd.make_row, tsd.make_labels, tsd.WORDS, tsd.chunk_at


def _write(tmp_path, rows, labels):
    wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"
    wp.write_text("".join(json.dumps(r) + "\n" for r in rows)); lp.write_text("".join(json.dumps(l) + "\n" for l in labels))
    return str(wp), str(lp)


# ── SEM 중립 ASR 행
def test_sem_free_unlabeled_rows_collate_and_pack(tmp_path):
    rows = [make_row("ls-1"), make_row("ls-2", WORDS[:3], tail=1.0), make_row("ls-3")]
    wp, lp = _write(tmp_path, rows, [make_labels("ls-1")])
    with pytest.raises(ValueError, match="함께"):
        SemCommitDataset(wp, lp, FakeTok(), online=False, allow_unlabeled=True, sem_free_unlabeled=True)
    assert len(SemCommitDataset(wp, lp, FakeTok(), delays=(4,), online=False)) == 1           # 기본: 라벨 없는 행은 건너뛴다(기존 그대로)
    ds = SemCommitDataset(wp, lp, FakeTok(), delays=(4,), online=False, turn_end=True, sem_free_unlabeled=True)
    assert len(ds) == 3 and ds.stats["sem_free"] == 2 and ds.stats["sem_free_hours"] > 0
    for i, it in enumerate(ds.items):
        s = ds.sequence(i, 4); assert not ds.target_problems(i, 4)
        if it.get("sem_free"): assert s["n_sem"] == s["n_turn"] == 0 and not s["decision"]     # 이벤트·B/N·턴 종료 없음
        else: assert s["n_sem"] > 0 and s["n_turn"] == 1
    xs = [ds[i] for i in range(3)]; b = collate_semcommit(xs)
    assert b["n_sem_free"] == 2
    for i, x in enumerate(xs):
        n = len(x["ids"]); assert bool(b["sem_mask"][i, :n].all()) == x["sem_free"] and not b["sem_mask"][i, n:].any()
    assert "sem_mask" not in collate_semcommit([x for x in xs if not x["sem_free"]])             # SEM 중립 행이 없으면 키도 없다
    p = pack_batch(b); assert p["sem_mask"].shape == p["ids"].shape and int(p["sem_mask"].sum()) == int(b["sem_mask"].sum())
    capped = SemCommitDataset(wp, lp, FakeTok(), delays=(4,), online=False, sem_free_unlabeled=True, sem_free_max_ratio=1.0)
    assert len(capped) == 2 and capped.stats["sem_free"] == 1 and capped.stats["sem_free_dropped"] == 1


def test_sem_free_routes_speechlm_rows_by_pool_but_labeled_mismatch_still_raises(tmp_path):
    long_, short = make_row("slm-long", tail=8.0), make_row("slm-short")
    for r in (long_, short): r.update(set="speechlm-partial", candidate_only=False, training_eligible=True, approval_fingerprint="a")
    wp, lp = _write(tmp_path, [long_, short], [])
    main = SemCommitDataset(wp, lp, FakeTok(), online=False, source_pool="main", sem_free_unlabeled=True)
    shp = SemCommitDataset(wp, lp, FakeTok(), online=False, source_pool="speechlm-short", sem_free_unlabeled=True)
    assert [it["id"] for it in main.items] == ["slm-long"] and main.stats["sem_free_other_pool"] == 1
    assert [it["id"] for it in shp.items] == ["slm-short"] and shp.stats["sem_free_other_pool"] == 1
    wp, lp = _write(tmp_path, [long_, short], [make_labels("slm-short")])
    with pytest.raises(ValueError, match="pool_mismatch"):                                   # 라벨 있는 행의 풀 불일치는 여전히 멈춘다
        SemCommitDataset(wp, lp, FakeTok(), online=False, source_pool="main", sem_free_unlabeled=True)
    e2 = make_row("ks-e2", lang="Korean")                                                     # speechlm 밖 SEM 중립 행(E2 코퍼스): main 에만, short 풀은 건너뛴다
    wp, lp = _write(tmp_path, [e2], [])
    assert len(SemCommitDataset(wp, lp, FakeTok(), online=False, source_pool="main", sem_free_unlabeled=True, ko_textnorm=False)) == 1
    s2 = SemCommitDataset(wp, lp, FakeTok(), online=False, source_pool="speechlm-short", sem_free_unlabeled=True, ko_textnorm=False)
    assert len(s2) == 0 and s2.stats["sem_free_other_pool"] == 1
    wp, lp = _write(tmp_path, [e2], [make_labels("ks-e2", lang="Korean")])
    with pytest.raises(ValueError, match="non_speechlm_in_short_pool"):
        SemCommitDataset(wp, lp, FakeTok(), online=False, source_pool="speechlm-short", sem_free_unlabeled=True, ko_textnorm=False)


# ── 속도 변형
@pytest.mark.parametrize("f", [0.9, 1.1])
def test_speed_perturb_scales_event_chunks_and_K(tmp_path, f):
    wp, lp = _write(tmp_path, [make_row()], [make_labels()])
    ds = SemCommitDataset(wp, lp, FakeTok(), delays=(2, 4), online=False, speed_perturb=(f,))
    it = ds.items[0]; ev = it["tokens"]
    for d in (2, 4):
        s, K, dur = ds.perturbed(0, d, f); s0 = ds.sequence(0, d)
        assert (s["n_sem"], s["n_B"], s["n_N"]) == (s0["n_sem"], s0["n_B"], s0["n_N"])
        sem_t = [x[1] for x in ev if len(x) == 2 and x[0] == ds.sem_id]
        for p, t in zip(sorted(s["sem_pos"]), sem_t):
            assert chunk_at(s, p) == int(t / f / 0.08) + d and s["ids"][p - 1] != ds.sp.next_audio   # SEM 은 1/f 시각의 청크, 단어 바로 뒤(같은 청크)
        assert K >= int(round(it["duration0"] / f / 0.08)) and int(round(dur * 12.5)) == K and s["sem_in_flush"] == 0
    assert it["est_extra"] >= 2 * (max(0, math.ceil(it["K"] / f) - it["K"]) + 2)            # packing 길이 추정 여유


@pytest.mark.parametrize("f", [0.9, 1.1])
def test_speed_perturb_getitem_audio_length_matches_K(tmp_path, f):
    pytest.importorskip("torchaudio")
    x = (0.2 * np.sin(np.arange(int(2.0 * 16000)) / 7.0) * 32767).astype("<i2"); x.tofile(tmp_path / "a.pcm")
    row = make_row(path=str(tmp_path / "a.pcm")); row["segments"][0]["dur_s"] = 2.0
    wp, lp = _write(tmp_path, [row], [make_labels()])
    ds = SemCommitDataset(wp, lp, FakeTok(), delays=(4,), online=True, speed_perturb=(f,))
    out = ds[0]; s, K, dur = ds.perturbed(0, out["delay"], f)
    assert out["speed"] == f and out["K"] == K and out["wav"].shape[0] == int(round(dur * 16000)) and out["ids"].tolist() == s["ids"]
    ds0 = SemCommitDataset(wp, lp, FakeTok(), delays=(4,), online=True)                      # 끄면(기본) 키가 없고 원래 K·길이
    o0 = ds0[0]; assert "speed" not in o0 and "sem_free" not in o0 and o0["K"] == ds0.items[0]["K"]


def test_noise_aug_hook_changes_audio_only(tmp_path):
    x = (0.2 * np.sin(np.arange(int(2.0 * 16000)) / 7.0) * 32767).astype("<i2"); x.tofile(tmp_path / "a.pcm")
    row = make_row(path=str(tmp_path / "a.pcm")); row["segments"][0]["dur_s"] = 2.0
    wp, lp = _write(tmp_path, [row], [make_labels()])
    calls = []
    def aug(w, rng): calls.append(len(w)); return w * 0.5
    ds0 = SemCommitDataset(wp, lp, FakeTok(), delays=(4,), online=True); ds = SemCommitDataset(wp, lp, FakeTok(), delays=(4,), online=True, noise_aug=aug)
    a, b = ds0[0], ds[0]
    assert calls == [a["wav"].shape[0]] and torch.allclose(b["wav"], a["wav"] * 0.5)
    assert b["ids"].tolist() == a["ids"].tolist() and b["K"] == a["K"] and torch.equal(b["labels"], a["labels"])   # 시퀀스·라벨 그대로


class OffTok(FakeTok):
    def __init__(self):
        super().__init__()
        for t, i in (("<|audio_start|>", 149990), ("<|audio_end|>", 149991), ("<|im_end|>", 149992)): self.v[t] = i


def test_offline_mode_items_and_asr_only(tmp_path):
    rows = [make_row("ls-1"), make_row("ls-2", WORDS[:3], tail=1.0)]
    wp, lp = _write(tmp_path, rows, [make_labels("ls-1")])
    ds = SemCommitDataset(wp, lp, OffTok(), delays=(4,), online=False, offline_frac=1.0)
    x = ds[0]; it = ds.items[0]; ev = it["tokens"]
    text = [t[0] for t in ev if len(t) == 2 and t[0] not in (ds.sem_id, ds.turn_id)]
    assert x["mode"] == "offline" and x["ids"].tolist() == ds._offline.sequence(text, it["K"], it["lang"])["ids"] and x["n_sem"] == 0
    assert int(x["is_audio"].sum()) == it["K"] and x["chunk_of"][x["is_audio"]].tolist() == list(range(it["K"]))
    assert collate_semcommit([ds[0], ds[1]])["n_offline"] == 2
    st = SemCommitDataset(wp, lp, OffTok(), delays=(4,), online=False, offline_frac=0.0)
    assert "mode" not in st[0] and "n_offline" not in collate_semcommit([st[0]])            # 끄면 키 없음(기존과 같다)
    with pytest.raises(ValueError, match="ignore_labels"):
        SemCommitDataset(wp, lp, OffTok(), online=False, ignore_labels=True)
    asr = SemCommitDataset(wp, lp, OffTok(), delays=(4,), online=False, ignore_labels=True, sem_free_unlabeled=True)
    assert len(asr) == 2 and asr.stats["sem_free"] == 2 and all(asr.sequence(i, 4)["n_sem"] == 0 for i in range(2))   # 라벨이 있어도 SEM 없음
