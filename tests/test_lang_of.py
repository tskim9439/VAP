"""manifest 이름 → 언어 판정. 학습 job 이 데이터 로딩 단계에서 KeyError 로 죽지 않도록 모든 manifest 이름을 고정한다(67126: aihub-bc-train → 'aihub')."""
import pytest
from vapasr.uslm.mono_data import lang_of

@pytest.mark.parametrize("name,lang", [
    ("librispeech-960", "English"), ("librispeech-dev", "English"), ("swbd-train", "English"), ("voxpopuli-train", "English"), ("yodas-en129", "English"), ("mnsc-1000", "English"),
    ("kspon-full", "Korean"), ("kspon-dev", "Korean"), ("nikl-1000", "Korean"), ("aihub71631-train", "Korean"), ("aihub71631-dev", "Korean"), ("aihub-bc-train", "Korean"),
    ("ls-train-utt-x", "English"), ("ahbc-train-x", "Korean"), ("ah71-dev-utt-x", "Korean"),
])
def test_lang_of(name, lang): assert lang_of(name) == lang

def test_unknown_raises():
    with pytest.raises(KeyError): lang_of("unknown-corpus")
