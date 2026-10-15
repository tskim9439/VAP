"""Stage 3 학습 도우미(vapasr/hf/stage3.py) — 가중치 파싱, 가중 라운드로빈 순서·epoch 길이, collate 분기. CPU, 모델 없음."""
import pytest, torch
from vapasr.hf.stage3 import parse_weights, weighted_order, WeightedRoundRobinLoader, collate_stage3, file_digest, fingerprint_diff


def test_parse_weights():
    assert parse_weights("dlg_ko:1, rep_en:0.5,") == {"dlg_ko": 1.0, "rep_en": 0.5}
    for bad in ("dlg_ko", "dlg_ko:-1", ":1"):
        with pytest.raises(ValueError): parse_weights(bad)


def test_weighted_order_is_smooth_and_deterministic():
    w = {"a": 2.0, "b": 1.0, "c": 0.0, "d": 1.0}
    o = weighted_order(w, 40)
    assert o == weighted_order(w, 40) and "c" not in o and o.count("a") == 20 and o.count("b") == 10 and o.count("d") == 10
    assert max(len(x) for x in "".join(o).replace("a", " ").split()) <= 2                        # a 가 아닌 것이 3 번 넘게 이어지지 않는다(고르게 섞임)
    with pytest.raises(ValueError): weighted_order({"a": 0.0}, 3)


class _DS(torch.utils.data.Dataset):
    def __init__(self, name, n): self.name, self.n = name, n
    def __len__(self): return self.n
    def __getitem__(self, i): return (self.name, i)


class _Sampler:
    def __init__(self, n, bs): self.n, self.bs = n, bs; self.ep = 0
    def set_epoch(self, e): self.ep = e
    def __len__(self): return self.n // self.bs
    def __iter__(self): return iter([list(range(i, i + self.bs)) for i in range(0, self.bs * len(self), self.bs)])


def _factory(ds, seed=0, rank=0, world=1): return _Sampler(len(ds), 2)
_factory.pack = False


def test_weighted_loader_epoch_length_share_and_cycling():
    sets = {"dlg_ko": _DS("dlg_ko", 20), "dlg_de": _DS("dlg_de", 4), "rep_ko": _DS("rep_ko", 100)}
    L = WeightedRoundRobinLoader(sets, lambda m: 2, weights={"dlg_ko": 1, "dlg_de": 1, "rep_ko": 1}, anchor=["dlg_ko", "dlg_de"], num_workers=0,
                                 sampler_factory=_factory, collate_fn=lambda b: b[0][0])
    assert len(L) == round((10 + 2) / (2 / 3)) == 18                                         # 대화 셋 배치 12 = 비중 2/3
    got = list(L); assert len(got) == 18 and got.count("rep_ko") == 6 and got.count("dlg_de") == 6   # dlg_de(배치 2)는 새 epoch 순열로 반복
    d = L.describe(); assert d["steps_per_epoch"] == 18 and abs(d["share"]["rep_ko"] - 1 / 3) < 1e-3
    with pytest.raises(ValueError): WeightedRoundRobinLoader(sets, lambda m: 2, weights={"nope": 1}, anchor=["dlg_ko"], num_workers=0, sampler_factory=_factory)
    with pytest.raises(ValueError): WeightedRoundRobinLoader(sets, lambda m: 2, weights={"rep_ko": 1}, anchor=["dlg_ko"], num_workers=0, sampler_factory=_factory)


def test_collate_stage3_dispatch():
    base = dict(wav=torch.zeros(160), K=2, ids=torch.tensor([1, 2, 3]), is_audio=torch.tensor([False, True, False]), chunk_of=torch.tensor([-1, 0, -1]),
                labels=torch.tensor([-100, -100, 3]), pos_weight=torch.zeros(3), lang="Korean", delay=4, n_text=1, n_flush=0)
    d = collate_stage3([dict(base, labels_alt=torch.tensor([-100, -100, 5]), soft_w=torch.tensor([1.0, 1.0, 0.3]), n_eot=1, n_soft=1)])
    assert "sem_only_mask" in d and d["labels_alt"][0, 2] == 5 and abs(float(d["soft_w_full"][0, 2]) - 0.3) < 1e-6 and "sem_mask" not in d
    r = collate_stage3([dict(base, sem_free=True)])
    assert "labels_alt" not in r and "sem_only_mask" not in r and torch.equal(r["pos_weight"], torch.zeros(1, 3))


def test_fingerprint_helpers(tmp_path):
    f = tmp_path / "a.txt"; f.write_text("x")
    assert file_digest([str(f)]) == file_digest([str(f)]) != file_digest([str(f), str(tmp_path / "missing")])
    (tmp_path / "ck").mkdir(); (tmp_path / "ck" / "config.json").write_text('{"stage3": {"a": 1, "b": 2}}')
    assert fingerprint_diff(str(tmp_path / "ck"), {"a": 1, "b": 3, "c": 0}) == ["b", "c"] and fingerprint_diff(str(tmp_path), {"a": 1}) == ["<missing>"]
