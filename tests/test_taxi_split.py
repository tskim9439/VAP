"""cst_stage4_taxi.attr_split — 공통 정규화 cp·화자 무시 WER·귀속 몫(VibeVoice 대비 TAXI 분해와 같은 정의)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from cst_stage4_taxi import attr_split, simple_words  # noqa: E402

S = [dict(session_id="s1", turns=[dict(start_s=0.0, lang="German", transcript="Guten Tag, Taxi!"),
                                  dict(start_s=2.0, lang="English", transcript="Hello, I need a cab."),
                                  dict(start_s=5.0, lang="German", transcript="Wohin?")])]


def hyp(words):
    return [dict(session="s1", lang=lg, t=t, text=w) for lg, t, w in words]


def test_simple_words():
    assert simple_words("Hello, I need a cab.") == ["hello", "i", "need", "a", "cab"]


def test_perfect():
    h = hyp([("German", 1.0, "guten"), ("German", 1.1, "tag"), ("German", 1.2, "taxi"), ("English", 3.0, "hello"), ("English", 3.1, "i"),
             ("English", 3.2, "need"), ("English", 3.3, "a"), ("English", 3.4, "cab"), ("German", 6.0, "wohin")])
    assert attr_split(S, h) == (0, 0, 9)


def test_turn_in_wrong_slot_is_attribution_only():
    """영어 턴이 독일어 슬롯에 들어가면(TAXI 몰림) 화자 무시 오류는 0 이고 cp 만 남는다. 같은 짝 10(삽입 5 + 삭제 5) 대 뒤집은 짝 8(삭제 4 + 삽입 4) 중 작은 쪽."""
    h = hyp([("German", 1.0, "guten"), ("German", 1.1, "tag"), ("German", 1.2, "taxi"), ("German", 3.0, "hello"), ("German", 3.1, "i"),
             ("German", 3.2, "need"), ("German", 3.3, "a"), ("German", 3.4, "cab"), ("German", 6.0, "wohin")])
    cp, agn, n = attr_split(S, h)
    assert (agn, n) == (0, 9) and cp == 8


def test_swapped_slots_cost_nothing():
    """두 슬롯이 통째로 바뀌면 cp 순열이 바로잡는다."""
    h = hyp([("English", 1.0, "guten"), ("English", 1.1, "tag"), ("English", 1.2, "taxi"), ("German", 3.0, "hello"), ("German", 3.1, "i"),
             ("German", 3.2, "need"), ("German", 3.3, "a"), ("German", 3.4, "cab"), ("English", 6.0, "wohin")])
    assert attr_split(S, h) == (0, 0, 9)
