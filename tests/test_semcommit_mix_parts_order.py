"""semcommit_mix_parts_order — English/Korean parts interleaved by main-pool hours (catch-up from finished parts, then alternate)."""
import json
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("mix", Path(__file__).resolve().parents[1] / "experiments" / "semcommit_mix_parts_order.py")
mix = importlib.util.module_from_spec(spec); spec.loader.exec_module(mix)


def _done(root, part, main_hours):
    d = root / "parts" / part; d.mkdir(parents=True)
    w = d / "words-main.jsonl"; w.write_text(json.dumps(dict(duration_s=main_hours * 3600)) + "\n")
    (d / "DONE.json").write_text(json.dumps(dict(files=dict(main=dict(words=str(w))))))


def test_catch_up_then_alternate(tmp_path):
    ko = tmp_path / "order.tsv"
    ko.write_text("".join(f"p{k}\tSRC{k % 2}\t10\n" for k in range(10)))
    _done(tmp_path, "p0", 2.0); _done(tmp_path, "p1", 2.0)                    # Korean already 4 h ahead, 2 h/part estimate
    en = tmp_path / "en"; en.mkdir()
    (en / "setA.tsv").write_text("".join(f"en-setA-{k:06d}\tsetA\t5\t1.0\n" for k in range(6)))
    (en / "setB.tsv").write_text("".join(f"en-setB-{k:06d}\tsetB\t5\t1.0\n" for k in range(3)))
    src = tmp_path / "map.json"; src.write_text(json.dumps({f"p{k}": {"source": f"SRC{k % 2}"} for k in range(10)}))
    order, s = mix.mix(ko, en, tmp_path, src)
    names = [p for p, *_ in order]
    assert "p0" not in names and "p1" not in names and len(names) == 8 + 9
    assert all(n.startswith("en-") for n in names[:5])                       # 4 h behind → English first
    en_h, ko_h, caught, worst = 0.0, 4.0, False, 0.0
    for p, _, _, h in order:
        if p.startswith("en-"): en_h += h
        else: ko_h += h
        caught = caught or en_h >= ko_h
        if caught and en_h < 9: worst = max(worst, abs(en_h - ko_h))
    assert worst <= 2.0 + 1e-9                                               # after catch-up: never more than one part (≤ 2 h) apart
    first_en = [n for n in names if n.startswith("en-")][:3]
    assert {n.split("-")[1] for n in first_en} == {"setA", "setB"}          # English sets interleaved by hours
    assert s["done_main_hours"] == {"English": 0.0, "Korean": 4.0} and s["balanced_until_part"] == max(i + 1 for i, n in enumerate(names) if n.startswith("en-"))
