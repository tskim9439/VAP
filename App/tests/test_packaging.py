"""설치·실행 스크립트 (setup.sh, run.sh) — 가중치·네트워크 없이.

dataless 가드(두 사본이 같고, 무엇도 읽기 전에 돈다), Python 버전 범위, macOS 하한, 끊긴 venv, 상대 --model, 저장소 밖 .pyc."""
import builtins
import contextlib
import io
import os
import platform
import shutil
import subprocess
import sys
import types

import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUP = os.path.join(APP, "setup.sh")
RUN = os.path.join(APP, "run.sh")
GUARD_OPENER = "\"$PY\" -I - \"$APP_DIR\" <<'PY' || exit 1"
CHECK_PY_OPENER = "\"$1\" -I - <<'PY'"
SF_DATALESS = 0x40000000
darwin = pytest.mark.skipif(sys.platform != "darwin", reason="macOS 전용 스크립트")


def _text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _heredoc(path, opener):
    t = _text(path)
    i = t.index(opener)
    start = t.index("\n", i) + 1
    return t[start:t.index("\nPY\n", start) + 1]


def _env(home, **kw):
    env = {k: v for k, v in os.environ.items() if k not in ("VAPKT_APP_MODEL", "PYTHON", "PYTHONPYCACHEPREFIX", "PYTHONDONTWRITEBYTECODE")}
    env["VAPKT_APP_HOME"] = str(home)
    env.update(kw)
    return env


def _fake_venv(home):
    """venv/bin/python = 지금 Python 을 부르는 작은 래퍼 (가드·모델 사전 점검은 표준 라이브러리만 쓴다)."""
    b = home / "venv" / "bin"
    b.mkdir(parents=True)
    py = b / "python"
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    return home


def _setup_python_ok():
    v = sys.version_info
    mac = platform.mac_ver()[0]
    return (sys.platform == "darwin" and platform.machine() == "arm64" and (3, 11) <= v < (3, 15)
            and mac and int(mac.split(".")[0]) >= 14)


# ── dataless 가드 ──────────────────────────────────────────────────────────────────────────────────

def test_guard_same_in_setup_and_run():
    g = _heredoc(SETUP, GUARD_OPENER)
    assert g == _heredoc(RUN, GUARD_OPENER)
    assert "SF_DATALESS = 0x40000000" in g and "brctl download" in g


def test_guard_runs_before_anything_reads_app_files():
    s = _text(SETUP)
    g = s.index(GUARD_OPENER)
    assert g < s.index('"$PY" -m venv') < s.index("-m pip install")
    assert g < s.index("-m pytest tests")
    assert s.index("/usr/bin/sw_vers") < s.index("CANDS=()")          # macOS 하한은 Python 고르기 전에
    r = _text(RUN)
    assert r.index(GUARD_OPENER) < r.index("engine\", \"checkpoint.py\"") < r.index("exec /usr/bin/caffeinate")


class _St:
    def __init__(self, st, flags):
        self._st, self.st_flags = st, flags

    def __getattr__(self, k):
        return getattr(self._st, k)


def _run_guard(monkeypatch, capsys, root, dataless):
    """가드를 프로세스 안에서 돌린다: dataless 집합의 경로만 st_flags 에 SF_DATALESS. (exit code, stderr, scandir 한 폴더들)."""
    real_stat, real_scandir = os.stat, os.scandir
    dl = {os.path.normpath(str(p)) for p in dataless}
    scanned = []

    def fake_stat(p, *a, **k):
        st = real_stat(p, *a, **k)
        return _St(st, st.st_flags | (SF_DATALESS if os.path.normpath(str(p)) in dl else 0))

    class _Entry:
        def __init__(self, e):
            self._e, self.name, self.path = e, e.name, e.path

        def stat(self, follow_symlinks=True):
            return fake_stat(self.path, follow_symlinks=follow_symlinks)

        def is_dir(self, follow_symlinks=True):
            return self._e.is_dir(follow_symlinks=follow_symlinks)

    @contextlib.contextmanager
    def fake_scandir(d):
        scanned.append(os.path.normpath(str(d)))
        with real_scandir(d) as it:
            yield [_Entry(e) for e in it]

    def no_open(*a, **k):
        raise AssertionError(f"가드가 파일을 열었다: {a}")

    code = compile(_heredoc(RUN, GUARD_OPENER), "dataless-guard", "exec")
    monkeypatch.setattr(os, "stat", fake_stat)
    monkeypatch.setattr(os, "scandir", fake_scandir)
    monkeypatch.setattr(sys, "argv", ["-", str(root)])
    monkeypatch.setattr(builtins, "open", no_open)
    monkeypatch.setattr(io, "open", no_open)
    rc = 0
    try:
        exec(code, {"__name__": "__main__"})
    except SystemExit as e:
        rc = e.code
    monkeypatch.undo()
    return rc, capsys.readouterr().err, scanned


def _tree(tmp_path):
    root = tmp_path / "App"
    for p in ("server.py", "engine/__init__.py", "web/app.js", "deep/x/y.py", "__pycache__/server.cpython-312.pyc"):
        f = root / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    return root


def test_guard_passes_clean_tree(monkeypatch, capsys, tmp_path):
    root = _tree(tmp_path)
    rc, err, scanned = _run_guard(monkeypatch, capsys, root, [])
    assert rc == 0 and err == ""
    assert os.path.normpath(str(root / "deep" / "x")) in scanned
    assert not any(s.endswith("__pycache__") for s in scanned)


def test_guard_refuses_dataless_file_and_dir_without_opening(monkeypatch, capsys, tmp_path):
    root = _tree(tmp_path)
    rc, err, scanned = _run_guard(monkeypatch, capsys, root, [root / "engine" / "__init__.py", root / "deep"])
    assert rc == 1
    assert "dataless" in err and "brctl download" in err and "iCloud 가 되는 네트워크" in err
    assert str(root / "engine" / "__init__.py") in err and str(root / "deep") + "/" in err
    assert os.path.normpath(str(root / "deep")) not in scanned          # dataless 폴더는 열지 않는다


def test_guard_refuses_dataless_root(monkeypatch, capsys, tmp_path):
    root = _tree(tmp_path)
    rc, err, scanned = _run_guard(monkeypatch, capsys, root, [root])
    assert rc == 1 and str(root) + "/" in err and scanned == []


# ── setup.sh: Python 버전 범위 (numpy 2.4.6·filelock 4.2.0 은 3.11+, aiohttp 3.14.3 은 3.14 까지) ─────────────

@pytest.mark.parametrize("ver, ok", [((3, 9, 6), False), ((3, 10, 0), False), ((3, 11, 4), True), ((3, 12, 14), True),
                                     ((3, 14, 4), True), ((3, 15, 0), False)])
def test_check_py_version_range(monkeypatch, capsys, ver, ok):
    src = _heredoc(SETUP, CHECK_PY_OPENER)
    assert src.count("sys.version_info") == 2
    code = compile(src.replace("sys.version_info", "_VER"), "check_py", "exec")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="0\n"))
    rc = 0
    try:
        exec(code, {"__name__": "__main__", "_VER": ver})
    except SystemExit as e:
        rc = e.code
    monkeypatch.undo()
    out = capsys.readouterr().out
    assert (rc == 0) == ok, out
    if not ok:
        assert "3.11" in out


@darwin
def test_setup_refuses_old_macos(tmp_path):
    app = tmp_path / "App"
    app.mkdir()
    s = _text(SETUP)
    assert s.count("/usr/bin/sw_vers -productVersion") == 1
    (app / "setup.sh").write_text(s.replace("/usr/bin/sw_vers -productVersion", "echo 13.6"))
    r = subprocess.run(["bash", str(app / "setup.sh"), "--no-test"], env=_env(tmp_path / "home"),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert "macOS 14 이상" in r.stderr and "13.6" in r.stderr
    assert "Python 선택" not in r.stdout and not (tmp_path / "home").exists()


# ── setup.sh: 기본 Python 이 지워진 venv 는 고치려 하지 않고 --recreate 를 알린다 ──────────────────────────────

@pytest.mark.skipif(not _setup_python_ok(), reason="arm64 Python 3.11–3.14 · macOS 14+ 에서만")
def test_setup_stops_on_dangling_venv(tmp_path):
    home = tmp_path / "home"
    venv = home / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /nonexistent/python@3.12/bin\n")
    os.symlink("/nonexistent/python@3.12/bin/python3.12", venv / "bin" / "python3.12")
    os.symlink("python3.12", venv / "bin" / "python")
    os.symlink("python3.12", venv / "bin" / "python3")
    r = subprocess.run(["bash", SETUP, "--no-test"], env=_env(home, PYTHON=sys.executable),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "기본 Python 이 없어졌습니다" in r.stderr and "--recreate" in r.stderr
    assert "/nonexistent/python@3.12/bin/python3.12" in r.stderr
    assert "venv 만들기" not in r.stdout and "패키지 설치" not in r.stdout     # venv 덧씌우기·pip 없음
    assert sorted(p.name for p in (venv / "bin").iterdir()) == ["python", "python3", "python3.12"]


# ── run.sh: 상대 --model 은 부른 곳 기준, 모델 사전 점검은 저장소 안에 .pyc 를 쓰지 않는다 ──────────────────────

@pytest.fixture
def run_copy(tmp_path):
    """run.sh 사본: 서버 대신 명령줄을 echo. engine/checkpoint.py 만 복사."""
    tmp = tmp_path.resolve()
    app = tmp / "App"
    (app / "engine").mkdir(parents=True)
    t = _text(RUN)
    assert t.count("exec /usr/bin/caffeinate -i ") == 1
    (app / "run.sh").write_text(t.replace("exec /usr/bin/caffeinate -i ", "exec echo SERVER "))
    shutil.copy(os.path.join(APP, "engine", "checkpoint.py"), app / "engine" / "checkpoint.py")
    home = _fake_venv(tmp / "home")
    final = tmp / "models" / "final"
    final.mkdir(parents=True)
    for n in ("config.json", "tokenizer.json", "model.safetensors"):
        (final / n).write_text("{}")
    return types.SimpleNamespace(app=app, home=home, final=final, run=str(app / "run.sh"))


@darwin
def test_run_sh_relative_model_flag(run_copy):
    c = run_copy
    r = subprocess.run(["bash", c.run, "--no-browser", "--model", "./final"], cwd=c.final.parent, env=_env(c.home),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert f"모델: {c.final}\n" in r.stdout
    assert f"SERVER {c.home}/venv/bin/python {c.app}/server.py --model {c.final} --no-browser --model ./final" in r.stdout
    assert not (c.app / "engine" / "__pycache__").exists()             # -I 여도 pycache_prefix 로
    assert list((c.home / "pycache").rglob("checkpoint*.pyc"))


@darwin
def test_run_sh_relative_model_env(run_copy):
    c = run_copy
    r = subprocess.run(["bash", c.run, "--no-browser"], cwd=c.final.parent, env=_env(c.home, VAPKT_APP_MODEL="./final"),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert f"SERVER {c.home}/venv/bin/python {c.app}/server.py --model {c.final} --no-browser" in r.stdout


@darwin
def test_run_sh_missing_relative_model_and_fake_from_elsewhere(run_copy, tmp_path):
    c = run_copy
    other = tmp_path.resolve() / "elsewhere"
    other.mkdir()
    r = subprocess.run(["bash", c.run, "--model", "./final"], cwd=other, env=_env(c.home), capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "모델을 찾지 못했습니다" in r.stderr and "SERVER" not in r.stdout
    r = subprocess.run(["bash", c.run, "--engine", "fake"], cwd=other, env=_env(c.home), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert f"SERVER {c.home}/venv/bin/python {c.app}/server.py --engine fake" in r.stdout


@darwin
def test_run_sh_dangling_venv_says_recreate(tmp_path):
    b = tmp_path / "home" / "venv" / "bin"
    b.mkdir(parents=True)
    os.symlink("/nonexistent/python3.12", b / "python")
    r = subprocess.run(["bash", RUN, "--engine", "fake"], env=_env(tmp_path / "home"), capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "--recreate" in r.stderr


def test_scripts_parse():
    for p in (SETUP, RUN):
        assert subprocess.run(["bash", "-n", p]).returncode == 0
