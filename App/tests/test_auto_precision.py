"""자동 정밀도 선택 규칙 (engine/__init__.py: choose_precision, predict_chunk_ms, auto_note) — 가중치 없이."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import AUTO_ORDER, AUTO_RULE, auto_note, choose_precision, predict_chunk_ms  # noqa: E402

# 구성요소 시간(ms): 기본 M4 실측 근처와 Pro 급 가정
M4 = {"th:fp32": (33.5, 38.0), "th:fp16": (18.9, 21.1), "en:fp32": 34.3, "en:fp16": 16.7}
PRO = {"th:fp32": (13.0, 15.0), "th:fp16": (7.0, 8.0), "en:fp32": 14.0, "en:fp16": 7.0}


def meas(table, calls):
    def m(th, en):
        calls.append((th, en))
        return (table[f"en:{en}"] if en else 0.0,) + table[f"th:{th}"]
    return m


def test_predict_formula():
    p = predict_chunk_ms(17.0, 19.0, 21.0)
    assert abs(p["p95"] - (17 + 2 * 21 + 13 * 19)) < 1e-9
    assert abs(p["mean"] - (17 + 1.73 * 21 + 2.85 * 19)) < 1e-9
    assert abs(p["burst"] - (17 + 21 + 23 * 19)) < 1e-9


def test_fast_machine_picks_exact_first_and_stops():
    calls = []
    choice, table, why = choose_precision(list(AUTO_ORDER), meas(PRO, calls))
    assert choice == ("fp32", "fp32") and why == "pass"
    assert calls == [("fp32", "fp32")] and len(table) == 1 and table[0]["ok"]     # 통과하면 더 재지 않는다


def test_base_m4_falls_back_to_fastest():
    calls = []
    choice, table, why = choose_precision(list(AUTO_ORDER), meas(M4, calls))
    assert why == "fallback_fastest" and choice == ("fp16", "fp16")
    assert [(r["thinker"], r["encoder"]) for r in table] == list(AUTO_ORDER) and not any(r["ok"] for r in table)


def test_mid_machine_keeps_fp32_thinker_with_fp16_encoder():
    mid = {"th:fp32": (15.5, 17.0), "th:fp16": (8.0, 9.0), "en:fp32": 24.0, "en:fp16": 12.0}
    # (fp32,fp32): p95 = 24+34+201.5 = 259.5 > 256 → 탈락, (fp32,fp16): 247.5 → 통과
    choice, table, why = choose_precision(list(AUTO_ORDER), meas(mid, []))
    assert choice == ("fp32", "fp16") and why == "pass" and len(table) == 2


def test_mean_and_burst_limits_apply():
    rule = dict(AUTO_RULE, p95_ms=1e9, mean_ms=1e9, burst_ms=100.0)          # burst 만 걸리게
    choice, table, why = choose_precision(list(AUTO_ORDER), meas(PRO, []), rule)
    assert why == "fallback_fastest" and choice == ("fp16", "fp16")
    rule = dict(AUTO_RULE, p95_ms=1e9, burst_ms=1e9, mean_ms=45.0)           # mean 만: PRO fp16/fp32 = 47.8, fp16/fp16 = 40.8
    choice, _, why = choose_precision(list(AUTO_ORDER), meas(PRO, []), rule)
    assert why == "pass" and choice == ("fp16", "fp16")


def test_thinker_only_candidates():
    calls = []
    choice, table, why = choose_precision([("fp32", None), ("fp16", None)], meas(M4, calls))
    # thinker 만: fp32 p95 = 76 + 435.5 > 256, fp16 = 42.2 + 245.7 = 287.9 > 256 → 가장 빠른 fp16
    assert choice == ("fp16", None) and why == "fallback_fastest" and calls == [("fp32", None), ("fp16", None)]


def test_auto_note_korean():
    choice, table, why = choose_precision(list(AUTO_ORDER), meas(M4, []))
    note = auto_note(dict(choice=dict(thinker=choice[0], encoder=choice[1]), reason=why, table=table, ms=1234.0))
    assert "fp16" in note and "가장 빠른" in note and "ms" in note
    assert auto_note(None) == ""


# ───────────────────────── q4 초안 기준 예측·측정 실패 처리 (F05·F16)
def test_predict_with_draft_formula():
    p = predict_chunk_ms(17.0, 16.6, 19.3, td=6.2, tv=20.2)
    c = 4 * 6.2 + 20.2
    assert abs(p["p95"] - (17 + 2 * 19.3 + 3 * c)) < 1e-9                       # ⌈13/5⌉ = 3 검증 라운드
    assert abs(p["burst"] - (17 + 19.3 + 5 * c)) < 1e-9                         # ⌈23/5⌉ = 5
    assert abs(p["mean"] - (17 + 1.73 * 19.3 + 2.85 * 6.2 + 0.73 * 20.2)) < 1e-9
    base = predict_chunk_ms(17.0, 16.6, 19.3)
    assert p["p95"] < base["p95"] and p["mean"] < base["mean"]


def test_draft_pays_rule():
    from engine import draft_pays
    assert draft_pays(16.6, 6.2, 20.2)                                          # 기본 M4 fp16: 45 ≤ 58
    assert not draft_pays(7.0, 6.5, 8.0)                                        # q4 스텝이 q=1 과 비슷한 GPU: 끈다


def test_draft_measurement_can_make_fp32_pass():
    """초안 없이 p95 가 넘던 fp32 가 초안 측정값으로는 통과할 수 있다 (가정한 Pro 급 값)."""
    t = {"th:fp32": (16.0, 18.0, 4.0, 18.5), "th:fp16": (8.0, 9.0, 3.0, 9.5), "en:fp32": 14.0, "en:fp16": 7.0}
    plain = lambda th, en: (t[f"en:{en}"],) + t[f"th:{th}"][:2]                 # noqa: E731
    drafted = lambda th, en: (t[f"en:{en}"],) + t[f"th:{th}"]                   # noqa: E731
    _, tab0, why0 = choose_precision(list(AUTO_ORDER), plain)
    assert tab0[0]["p95"] > 256 and not tab0[0]["ok"]
    choice, tab, why = choose_precision(list(AUTO_ORDER), drafted)
    assert choice == ("fp32", "fp32") and why == "pass" and tab[0]["draft"] and tab[0]["td_ms"] == 4.0


def test_measure_failure_skips_candidate():
    calls = []

    def m(th, en):
        calls.append((th, en))
        if th == "fp32":
            raise RuntimeError("[METAL] Insufficient Memory")
        return (PRO[f"en:{en}"],) + PRO[f"th:{th}"]
    choice, table, why = choose_precision(list(AUTO_ORDER), m)
    assert choice == ("fp16", "fp16") and why == "pass"
    assert [r.get("error", "")[:12] for r in table[:2]] == ["RuntimeError"] * 2 and not table[0]["ok"]
    rule = dict(AUTO_RULE, p95_ms=1.0)                                          # 아무것도 통과 못 함: 잰 것 중 가장 빠른 것
    choice, table, why = choose_precision(list(AUTO_ORDER), m, rule)
    assert why == "fallback_fastest" and choice == ("fp16", "fp16")
    note = auto_note(dict(choice=dict(thinker="fp16", encoder="fp16"), reason=why, table=table, ms=10.0))
    assert "측정 실패" in note and "Insufficient" in note


def test_all_measurements_fail_uses_default():
    def m(th, en):
        raise ValueError("KV 범위 밖")
    choice, table, why = choose_precision(list(AUTO_ORDER), m, default=("fp16", "fp16"))
    assert choice == ("fp16", "fp16") and why == "fallback_default" and len(table) == 4
    note = auto_note(dict(choice=dict(thinker="fp16", encoder="fp16"), reason=why, table=table, ms=1.0))
    assert "측정 실패 → 고정 기본값" in note


def test_bench_thinker_small_kv_shrinks_context():
    """KV 가 328 보다 작아도 측정이 죽지 않는다 (예전: start 320 + q 8 > 327)."""
    import pytest
    mx = pytest.importorskip("mlx.core")
    import test_decoder as T
    from engine import bench_draft, bench_thinker
    from engine.thinker import MlxThinker
    from engine.tokens import BLOCKED_EXPECTED
    for n in (327, 64, 30):
        th = MlxThinker({k: mx.array(v) for k, v in T.W_NP.items()}, T.CFG, BLOCKED_EXPECTED, dtype=mx.float32, max_len=n)
        t1, t8 = bench_thinker(th, iters=1)
        td, tv = bench_draft(th, th.make_draft(4), iters=1)
        assert t1 > 0 and t8 > 0 and td > 0 and tv > 0


def test_engine_rejects_tiny_kv_capacity_before_loading():
    import pytest
    import test_tokens as TT
    if not TT.HAVE_CK:
        pytest.skip("s4 체크포인트 config/토크나이저 없음")
    from engine import Engine
    with pytest.raises(ValueError, match="--kv-capacity"):
        Engine(TT.CK, kv_capacity=10)                                           # 가중치를 읽기 전에 (토크나이저만)


def test_auto_order_prefers_fp16_encoder_with_fp16_thinker():
    """fp16 thinker 끼리는 인코더 정밀도가 정확도에 차이 없음 (W6 §2.2) → 둘 다 통과하면 빠른 fp16/fp16."""
    assert AUTO_ORDER.index(("fp16", "fp16")) < AUTO_ORDER.index(("fp16", "fp32"))
    t = {"th:fp32": (33.5, 38.0, 7.2, 36.5), "th:fp16": (16.6, 19.3, 6.2, 20.2), "en:fp32": 34.3, "en:fp16": 16.7}   # 기본 M4 + 초안
    choice, table, why = choose_precision(list(AUTO_ORDER), lambda th, en: (t[f"en:{en}"],) + t[f"th:{th}"])
    assert choice == ("fp16", "fp16") and why == "pass" and [r["ok"] for r in table] == [False, False, True]
