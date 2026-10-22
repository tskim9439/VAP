"""학습 자동 재개가 이어 받을 checkpoint-N(vapasr.hf.stage.last_complete_checkpoint) — 한 파일·조각 저장(1.7B) 모두, 저장 도중 끊긴 것은 건너뛴다. CPU·파일만."""
import json
from pathlib import Path

from vapasr.hf.stage import last_complete_checkpoint, weights_complete

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
WMAP = {"a.w": SHARDS[0], "b.w": SHARDS[1], "c.w": SHARDS[1]}


def ck(d, n, files=("model.safetensors",), state=True, index=None):
    c = d / f"checkpoint-{n}"; c.mkdir(parents=True)
    for f in files: (c / f).write_bytes(b"x")
    if index is not None: (c / "model.safetensors.index.json").write_text(index if isinstance(index, str) else json.dumps({"metadata": {}, "weight_map": index}))
    if state: (c / "trainer_state.json").write_text("{}")
    return c


def test_single_file_and_sharded(tmp_path):
    ck(tmp_path, 1000); assert last_complete_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-1000")
    ck(tmp_path, 2000, SHARDS, index=WMAP); assert last_complete_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-2000")   # 1.7B 조각 저장


def test_skips_incomplete(tmp_path):
    ck(tmp_path, 1000, SHARDS, index=WMAP)
    ck(tmp_path, 2000, SHARDS[:1], index=WMAP)                                                 # 조각 하나 없음
    ck(tmp_path, 3000, SHARDS, index=WMAP, state=False)                                        # trainer_state.json 없음(저장 도중 끊김)
    ck(tmp_path, 4000, SHARDS, index="{")                                                      # index 깨짐
    ck(tmp_path, 5000, SHARDS)                                                                 # index 없음
    ck(tmp_path, 6000, (), index={})                                                           # 빈 weight_map
    ck(tmp_path, 7000, SHARDS, index="[]")                                                     # index 형식이 다르다
    (tmp_path / "final").mkdir(); (tmp_path / "checkpoint-x").mkdir()
    assert last_complete_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-1000")
    assert weights_complete(str(tmp_path / "checkpoint-1000")) and not weights_complete(str(tmp_path / "checkpoint-2000"))


def test_numeric_order_and_empty(tmp_path):
    assert last_complete_checkpoint(str(tmp_path)) is None
    for n in (900, 10000, 2000): ck(tmp_path, n)
    assert last_complete_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-10000")   # 문자열이 아니라 숫자 순서


def test_trainers_use_shared_helper():
    for f in ("experiments/semcommit_train.py", "experiments/stage3_train.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert "from vapasr.hf.stage import last_complete_checkpoint" in src and "def last_complete_checkpoint" not in src, f
