"""V3 R2m(같은 언어 셋 따로 교대·머무름 가중 — stage3.decision_weights, --same-lang-*-weight)와 체크포인트 평균(experiments/avg_checkpoints.py). CPU."""
import json, os, subprocess, sys

import pytest

from vapasr.hf.stage3 import decision_weights
from test_cst_decision import ROOT, SUBMIT_ARGS, run_submit, stage3, submit_env

KINDS = ("dlg", "sim", "xl")


def old_rule(kind, c, h, sets):                                                                # 바꾸기 전 dec_kw(stage3_train)의 규칙
    return dict(change_weight=c, hold_weight=h) if (c > 0 or h > 0) and (sets == "all" or kind == "xl") else {}


@pytest.mark.parametrize("sets", ["all", "xl"])
def test_decision_weights_default_is_old_rule(sets):
    for kind in KINDS:
        for c, h in ((0, 0), (3, 1.5), (3, 0), (0, 1.5)): assert decision_weights(kind, c, h, sets) == old_rule(kind, c, h, sets), (kind, c, h, sets)


@pytest.mark.parametrize("sets", ["all", "xl"])
def test_decision_weights_same_lang_override(sets):
    for kind in ("dlg", "sim"):
        assert decision_weights(kind, 3, 1.5, sets, 1.5, 1.5) == dict(change_weight=1.5, hold_weight=1.5)
        assert decision_weights(kind, 3, 1.5, sets, 1.5, None) == dict(change_weight=1.5, hold_weight=0.0)       # 빠진 쪽 0
        assert decision_weights(kind, 3, 1.5, sets, 0.0, 0.0) == {}                                             # 같은 언어 셋만 끈다
    assert decision_weights("xl", 3, 1.5, sets, 1.5, 1.5) == dict(change_weight=3, hold_weight=1.5)            # 교차 언어 셋은 그대로


def test_stage3_train_same_lang_flags(tmp_path):
    rc, out = stage3(["--help"], tmp_path)
    assert rc == 0 and "--same-lang-change-weight" in out and "--same-lang-hold-weight" in out
    for args in (["--same-lang-change-weight", "-1"], ["--same-lang-hold-weight", "-0.5"]):
        rc, out = stage3(args, tmp_path); assert rc != 0 and "--same-lang-change-weight·--same-lang-hold-weight ≥ 0" in out, out[-400:]


def test_submit_script_same_lang_flags(tmp_path):
    R, sh = submit_env(tmp_path); base = run_submit(R, sh, SUBMIT_ARGS[1])
    rc, out, err = run_submit(R, sh, SUBMIT_ARGS[1][:-1] + ["--same-lang-change-weight", "1.5", "--same-lang-hold-weight", "1.5", "--dry-run"]); assert rc == 0, err
    want = " --same-lang-change-weight 1.5 --same-lang-hold-weight 1.5"
    assert want in out and out.replace(want, "") == base[1] and "같은 언어 교대 가중 1.5" in err   # 새 인자만 덧붙는다
    for bad in (["--same-lang-change-weight", "-1"], ["--same-lang-hold-weight", "x"]):
        rc, out, err = run_submit(R, sh, SUBMIT_ARGS[1][:-1] + bad + ["--dry-run"]); assert rc != 0 and not out, (bad, err)


# ── 체크포인트 평균
def make_ck(d, a, pos, extra=None):
    import torch
    from safetensors.torch import save_file
    d.mkdir(parents=True)
    save_file({"w": torch.tensor(a, dtype=torch.float32), "pos": torch.tensor(pos, dtype=torch.int64), **(extra or {})}, str(d / "model.safetensors"), metadata={"format": "pt"})
    (d / "config.json").write_text(json.dumps({"tag": d.name})); (d / "optimizer.pt").write_bytes(b"x"); (d / "trainer_state.json").write_text("{}"); (d / "rng_state_0.pth").write_bytes(b"x")
    return d


def avg(args):
    p = subprocess.run([sys.executable, str(ROOT / "experiments/avg_checkpoints.py")] + [str(x) for x in args], capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout + p.stderr


def test_avg_checkpoints(tmp_path):
    torch = pytest.importorskip("torch"); pytest.importorskip("safetensors")
    from safetensors.torch import load_file
    cks = [make_ck(tmp_path / f"c{i}", [[1.0 * i, 2.0], [3.0, 4.0 + i]], [0, 1]) for i in range(3)]
    out = tmp_path / "run/final"; rc, log = avg(["--out", out, "--done"] + cks); assert rc == 0, log
    w = load_file(str(out / "model.safetensors"))
    assert torch.equal(w["w"], torch.tensor([[1.0, 2.0], [3.0, 5.0]])) and w["w"].dtype == torch.float32 and torch.equal(w["pos"], torch.tensor([0, 1]))
    assert json.loads((out / "config.json").read_text()) == {"tag": "c2"} and not any((out / f).exists() for f in ("optimizer.pt", "trainer_state.json", "rng_state_0.pth"))
    info = json.loads((out / "avg.json").read_text()); assert info["n"] == 3 and info["files"]["model.safetensors"]["parity"] and (tmp_path / "run/DONE").exists()
    rc, log = avg(["--out", out] + cks); assert rc != 0 and "이미 있다" in log                    # 덮어쓰지 않는다


@pytest.mark.parametrize("bad", ["int", "key", "shape"])
def test_avg_checkpoints_refuses_mismatch(tmp_path, bad):
    torch = pytest.importorskip("torch"); pytest.importorskip("safetensors")
    a = make_ck(tmp_path / "a", [1.0, 2.0], [0, 1])
    b = make_ck(tmp_path / "b", [3.0] * (3 if bad == "shape" else 2), [0, 2] if bad == "int" else [0, 1], {"x": torch.zeros(1)} if bad == "key" else None)
    rc, log = avg(["--out", tmp_path / "o", a, b]); assert rc != 0 and not (tmp_path / "o").exists(), log


def test_submit_script_afterok(tmp_path):
    """--afterok JOBID: init 이 아직 없어도 제출(체인), sbatch 에 --dependency=afterok — --after 와 함께면 거부, 없으면 init 검사 그대로."""
    R, sh = submit_env(tmp_path); a = SUBMIT_ARGS[1][:-1]; i = a.index("--after"); base = a[:i] + a[i + 2:]          # 골든 인자의 --after 쌍은 뺀다
    rc, out, err = run_submit(R, sh, base + ["--init", "<R>/none", "--afterok", "123", "--dry-run"]); assert rc == 0 and "--dependency=afterok:123" in out and "만들 예정" in err, err
    rc, out, err = run_submit(R, sh, base + ["--init", "<R>/none", "--dry-run"]); assert rc != 0 and "init 없음" in err
    rc, out, err = run_submit(R, sh, base + ["--afterok", "123", "--after", "5", "--dry-run"]); assert rc != 0 and not out
    rc, out, err = run_submit(R, sh, base + ["--afterok", "x1", "--dry-run"]); assert rc != 0 and not out


def test_semcommit_submit_exclude(tmp_path):
    """submit-semcommit-train-apex.sh --exclude: 기본(없음)은 명령에 --exclude 가 없고, 주면 sbatch 기본 제외(hpc-52·hpc-4)에 더해 붙는다."""
    R = tmp_path / "R"; snap = R / "snap"; snap.mkdir(parents=True)
    for f in ("main-words.list", "main-labels.list", "short-words.list", "short-labels.list"): (snap / f).write_text("x\n")
    (snap / "summary.json").write_text('{"parts": 1}'); init = R / "init"; init.mkdir(); (init / "config.json").write_text("{}")
    sh = tmp_path / "repo/slurm/submit-semcommit-train-apex.sh"; sh.parent.mkdir(parents=True)
    sh.write_text((ROOT / "slurm/submit-semcommit-train-apex.sh").read_text().replace("/soundai", f"{R}/soundai"))
    run = lambda extra: subprocess.run(["bash", str(sh), "--run", "x", "--snap", str(snap), "--init", str(init), "--dry-run"] + extra, capture_output=True, text=True, timeout=60)
    a, b = run([]), run(["--exclude", "hpc-102,slurmmxch200v5-hpc-50"])
    assert a.returncode == 0 and b.returncode == 0, (a.stderr, b.stderr)
    assert "--exclude" not in a.stdout and "--exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4,slurmmxch200v5-hpc-102,slurmmxch200v5-hpc-50" in b.stdout
    assert b.stdout.replace(" --exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4,slurmmxch200v5-hpc-102,slurmmxch200v5-hpc-50", "") == a.stdout
