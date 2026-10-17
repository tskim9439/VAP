"""화자 헤드 안내 디코딩의 판정(vapasr/hf/batch_decode._guide_slot) — 말 끝 시각(청크 k − δ) 근처 슬롯 확률로 한 슬롯을 고르거나 의견 없음. CPU."""
import pytest
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "DynamicCache"): pytest.skip("transformers 에 DynamicCache 없음(구버전)", allow_module_level=True)
from vapasr.hf.batch_decode import SpkGuide, _guide_slot


def test_guide_slot_reads_delayed_chunks_and_abstains():
    P = [[0.9, 0.1]] * 5 + [[0.1, 0.9]] * 5 + [[0.8, 0.7]] * 3 + [[0.1, 0.1]] * 3 + [None] * 2
    g = SpkGuide(delay=2, on=0.5, margin=0.3, span=2)
    assert _guide_slot(P, 6, g) == 0                    # 청크 3·4 → A
    assert _guide_slot(P, 9, g) == 1                    # 청크 6·7 → B
    assert _guide_slot(P, 6 + 1, g) is None             # 청크 4·5 평균 0.5·0.5 → 차이 없음
    assert _guide_slot(P, 13, g) is None                # 겹침(0.8·0.7) → 의견 없음
    assert _guide_slot(P, 16, g) is None                # 무음 → 의견 없음
    assert _guide_slot(P, 1, g) is None                 # 아직 말 끝 청크가 없다
    assert _guide_slot(P, 30, SpkGuide(delay=2)) is None and _guide_slot(P, 20, SpkGuide(delay=0, span=8)) is None   # flush: 마지막 청크 기준, 없는 칸은 건너뜀
    assert _guide_slot(P, 6, SpkGuide(delay=2, on=0.95)) is None
