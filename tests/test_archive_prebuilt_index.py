import io
import json
import tarfile

import pytest

from vapasr.data.archive import read_member
from vapasr.data.selection import digest


def test_prebuilt_tar_index_is_read_only_and_fail_closed(tmp_path, monkeypatch):
    archive = tmp_path / "audio.tar"
    with tarfile.open(archive, "w") as tf:
        payload = b"wave-bytes"
        info = tarfile.TarInfo("a.wav")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    with tarfile.open(archive, "r:") as tf:
        item = tf.getmember("a.wav")
        member = [item.offset_data, item.size]
    index = tmp_path / "indices"
    index.mkdir()
    monkeypatch.setenv("VAPASR_TAR_INDEX_ROOT", str(index))
    with pytest.raises(FileNotFoundError):
        read_member(str(archive), "a.wav")
    stat = archive.stat()
    identity = [str(archive.resolve()), stat.st_size, stat.st_mtime_ns]
    (index / (digest(identity) + ".json")).write_text(json.dumps(dict(identity=identity, members={"a.wav": member})))
    assert read_member(str(archive), "a.wav") == b"wave-bytes"
    assert list(index.iterdir()) == [index / (digest(identity) + ".json")]
