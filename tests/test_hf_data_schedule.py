from collections import Counter

from vapasr.hf.data import proportional_order


def test_proportional_order_uses_every_batch_once():
    order = proportional_order({"en": 7, "ko": 3})
    assert Counter(order) == {"en": 7, "ko": 3}
    assert len(order) == 10
    assert max(len(run) for run in "".join("e" if x == "en" else "k" for x in order).split("k")) <= 3
