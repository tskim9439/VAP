"""English evaluation set VoxPopuli-Cleaned-AA: manifest builder (float32 WAV, coverage/revision/hashes), read_audio padding,
AA-WER comparison scoring (Whisper normalizer + ':00' + digit split, duration-weighted average) and summarize's aa_wer block."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")


def _cli():
    spec = importlib.util.spec_from_file_location("eval_cli", Path(__file__).parents[1] / "experiments/eval_single_turn_asr.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def _fake_vpaa(root: Path, n=2, bad_duration=False):
    (root / "audio").mkdir(parents=True)
    rows = []
    for i in range(n):
        x = (np.sin(np.arange(16000 + 800 * i) / 7.0) * 0.1).astype(np.float32)
        fn = f"u{i}.wav"
        sf.write(str(root / "audio" / fn), x, 16000, subtype="FLOAT")
        rows.append(dict(id=f"20140414-0900-PLENARY-19-en_20140414-22:40:11_{i}", gender="female",
                         duration=len(x) / 16000 + (1.0 if bad_duration and i == 0 else 0.0),
                         transcript=f"Thank you Mr President, item {i}.", language="en", url=f"audio/{fn}",
                         dataset="voxpopuli_en_test", file_name=fn))
    (root / "voxpopuli_cleaned_aa_v1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_prepare_voxpopuli_aa_manifest_and_read_audio(tmp_path):
    cli = _cli()
    src = _fake_vpaa(tmp_path / "vpaa")
    a = argparse.Namespace(voxpopuli_aa_root=str(tmp_path / "vpaa"), voxpopuli_aa_revision="rev-test", kspon_root=None, libri_root=None,
                           out=str(tmp_path / "out"), workers=2)
    cli.prepare(a)
    man = [json.loads(l) for l in (tmp_path / "out/manifest.jsonl").read_text().splitlines()]
    assert [r["id"] for r in man] == sorted(f"voxpopuli-aa/{r['id']}" for r in src)
    r0 = man[0]
    assert (r0["dataset"], r0["lang"], r0["audio_format"], r0["audio_subtype"], r0["sample_rate"]) == ("voxpopuli-aa-test", "English", "wav", "FLOAT", 16000)
    assert r0["reference"] == r0["raw_reference"] == "Thank you Mr President, item 0." and r0["session"] == "20140414-0900-PLENARY-19"
    cov = json.loads((tmp_path / "out/coverage.json").read_text())["coverage"]["voxpopuli-aa-test"]
    assert (cov["available"], cov["source_labels"], cov["revision"], cov["full_standard_set"]) == (2, 2, "rev-test", False)   # 표준 628 이 아님
    assert len(cov["jsonl_sha256"]) == 64 and len(cov["audio_sha256"]) == 64
    x = cli.read_audio(r0, 1.0)
    assert x.dtype == np.float32 and len(x) == r0["num_samples"] + 16000 and np.all(x[r0["num_samples"]:] == 0)
    ref, _ = sf.read(r0["path"], dtype="float32")
    np.testing.assert_array_equal(x[:len(ref)], ref)
    cli.prepare(a)                                                                        # 같은 입력 → 같은 manifest(다시 써도 통과)


def test_prepare_rejects_duration_mismatch_and_empty_sources(tmp_path):
    cli = _cli()
    _fake_vpaa(tmp_path / "vpaa", bad_duration=True)
    a = argparse.Namespace(voxpopuli_aa_root=str(tmp_path / "vpaa"), voxpopuli_aa_revision="r", kspon_root=None, libri_root=None,
                           out=str(tmp_path / "out"), workers=1)
    with pytest.raises(ValueError, match="Duration differs"):
        cli.prepare(a)
    with pytest.raises(SystemExit):
        cli.prepare(argparse.Namespace(voxpopuli_aa_root=None, voxpopuli_aa_revision="r", kspon_root=None, libri_root=None, out=str(tmp_path / "o2"), workers=1))


def test_aa_normalize_rules():
    from vapasr.data.aa_wer import aa_normalize
    assert aa_normalize("Thank you Mr. President, it's not secret.") == "thank you mister president it is not secret"
    assert aa_normalize("The colour grey, uh, colour.") == "the color gray color"                  # 영/미 철자·필러
    assert aa_normalize("at 7:00pm") == aa_normalize("at 7pm") == "at 7 pm"                        # ':00' 제거
    assert aa_normalize("call 1405 553 272") == aa_normalize("call 1405553272") == "call 1 4 0 5 5 5 3 2 7 2"   # 숫자 묶음 무시
    assert aa_normalize("twenty-one") == aa_normalize("21") == "2 1"
    assert aa_normalize("code 007") == aa_normalize("code zero zero seven") == "code 0 0 7"          # 선행 0 보존
    assert aa_normalize("0.5% of 2007") == "0 . 5 % of 2 0 0 7"                                       # 소수·연도는 그대로 두고 한 자리씩


def test_aa_counts_and_duration_weighted_summary():
    from vapasr.data.aa_wer import aa_counts, aa_summary
    c = aa_counts("the cat sat", "the bat sat down")
    assert (c["substitutions"], c["insertions"], c["deletions"], c["errors"], c["n_ref"]) == (1, 1, 0, 2, 3)
    rows = [dict(reference=" ".join(["w"] * 10), hypothesis=" ".join(["w"] * 9 + ["x"]), audio_s=10.0),    # WER 0.1
            dict(reference=" ".join(["v"] * 10), hypothesis=" ".join(["v"] * 5), audio_s=30.0),              # WER 0.5
            dict(reference="uh", hypothesis="hello", audio_s=5.0)]                                          # 정규화 뒤 빈 참조 → 가중 평균 제외
    s = aa_summary(rows)
    assert s["empty_ref"] == 1 and s["n_ref"] == 20 and s["errors"] == 1 + 5 + 1
    assert s["wer_duration_weighted"] == pytest.approx((10 * 0.1 + 30 * 0.5) / 40)
    assert s["wer_micro"] == pytest.approx(7 / 20)


def test_summarize_adds_aa_block_for_english_only(tmp_path):
    cli = _cli()
    out = tmp_path / "run"
    out.mkdir()
    (out / "config-rank0.json").write_text(json.dumps(dict(utterances=2, deltas=[4], world_size=1)))
    cnt = lambda e, n: dict(substitutions=e, deletions=0, insertions=0, n_ref=n, n_hyp=n, errors=e, rate=e / n)
    rows = [dict(id="a", dataset="voxpopuli-aa-test", lang="English", delta=4, audio_s=2.0, forced=0, reference="Mr President, 7:00pm",
                 hypothesis="mister president seven pm", metrics=dict(wer=cnt(1, 3))),
            dict(id="b", dataset="kspon-eval_clean", lang="Korean", delta=4, audio_s=1.0, forced=0, reference="안녕", hypothesis="안녕",
                 metrics=dict(cer_nospace=cnt(0, 2)))]
    (out / "predictions-rank0.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (out / "done-rank0.json").write_text("{}")
    cli.summarize(argparse.Namespace(out=str(out), allow_partial=False))
    rep = json.loads((out / "summary.json").read_text())
    assert rep["complete"] and set(rep["aa_wer"]) == {"voxpopuli-aa-test/delta-4"}
    aa = rep["aa_wer"]["voxpopuli-aa-test/delta-4"]
    assert aa["errors"] == 0 and aa["wer_duration_weighted"] == 0.0                                  # 'seven pm' = '7:00pm' 을 AA 정규화는 같게 본다
    assert rep["aa_scoring"]["aa_version"] == "aa-wer-approx-v1" and len(rep["aa_scoring"]["aa_code_sha256"]) == 64
