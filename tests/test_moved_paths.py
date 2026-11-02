"""공용 DB 가 옮긴 폴더(archive.MOVED)의 옛 manifest 경로를 읽는 쪽에서 찾기 — AI Hub 031 이름 변경(2026-11-01).
맥(APFS)은 정규형을 가리지 않고 찾으므로 NFC·NFD 고르기는 서버(Linux)에서만 실제로 갈린다. 여기서는 옮김·읽기·캐시·밖 경로 무 I/O 를 본다."""
import io, os, zipfile, unicodedata, numpy as np, pytest
sf = pytest.importorskip("soundfile")
from vapasr.data import archive
from vapasr.data.streams import load_utt_audio, SR

OLD, NEW = "031.방송콘텐츠 한국어-영어 통·번역 음성 데이터", "031.broadcasting_koen"
INNER = "01-1.정식개방데이터/Training/01.원천데이터"
ZIP = "TS_한-스_한국어음성_교양.zip"

@pytest.fixture
def moved(tmp_path, monkeypatch):
    """새 폴더(안쪽 한글 이름 NFD)에 zip(멤버 이름 "/" 시작 — AI Hub)과 wav. 옛 경로는 맨 위 폴더만 NFD·안쪽 NFC(실제 manifest 의 섞임)."""
    d = tmp_path / NEW / unicodedata.normalize("NFD", INNER); d.mkdir(parents=True)
    x = (0.3 * np.sin(np.arange(SR) / 8.0)).astype(np.float32); b = io.BytesIO(); sf.write(b, x, SR, format="WAV")
    with zipfile.ZipFile(d / unicodedata.normalize("NFD", ZIP), "w") as z: z.writestr("/a.wav", b.getvalue())
    sf.write(str(d / unicodedata.normalize("NFD", "가.wav")), x, SR)
    old = str(tmp_path / unicodedata.normalize("NFD", OLD))
    monkeypatch.setattr(archive, "_MOVED", archive._rules({old: str(tmp_path / NEW)})); monkeypatch.setattr(archive, "_RESOLVED", {})
    return old + "/" + unicodedata.normalize("NFC", INNER), x, str(tmp_path / NEW)

def test_old_zip_member_and_file_read_from_new_folder(moved):
    old_dir, x, new = moved
    for p in (f"{old_dir}/{ZIP}::a.wav", f"{old_dir}/가.wav"):
        y = load_utt_audio(p); assert len(y) == len(x) and float(np.abs(y - x).max()) < 1e-3
    assert sorted(archive._RESOLVED) == sorted([f"{old_dir}/{ZIP}", f"{old_dir}/가.wav"]) and all(r.startswith(new + "/") and os.path.exists(r) for r in archive._RESOLVED.values())

def test_missing_under_moved_folder_keeps_path_and_error(moved):
    old_dir = moved[0]; p = f"{old_dir}/없음.wav"
    assert archive.resolve_path(p) == p and p not in archive._RESOLVED        # 못 찾으면 기억하지 않고 원래 경로(원래 오류)
    with pytest.raises(Exception): load_utt_audio(p)

def test_paths_outside_moved_folders_need_no_io(moved, monkeypatch):
    monkeypatch.setattr(os.path, "exists", lambda p: (_ for _ in ()).throw(AssertionError(p)))
    for p in ("/soundai/DB/raw/aihub/033.x/a.zip", "/data/a.wav", moved[2] + "/x/a.zip"):   # 새 폴더 경로도 그대로
        assert archive.resolve_path(p) == p

def test_default_rule_matches_manifest_mixed_form():
    """기본 MOVED: 옛 manifest 는 맨 위 폴더만 NFD — NFC 로 비교해 잡힌다(I/O 전 단계까지)."""
    (old, new, head), = archive._rules(archive.MOVED)
    p = "/soundai/DB/raw/aihub/" + unicodedata.normalize("NFD", OLD) + "/" + INNER + "/" + ZIP
    assert p.startswith(head) and new == "/soundai/DB/raw/aihub/031.broadcasting_koen/" and unicodedata.normalize("NFC", "/".join(p.split("/")[:old.count("/") + 1])) == old
