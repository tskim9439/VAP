"""final(오프라인) 모드 시퀀스 — Qwen3-ASR 원 형식(vapasr/uslm/model.py U0.5 브리지와 같다):

  <|im_start|>system\\n<|im_end|>\\n<|im_start|>user\\n<|audio_start|>[<|audio_pad|>×K]<|audio_end|><|im_end|>\\n<|im_start|>assistant\\nlanguage {Lang}<asr_text>{text}<|im_end|>

스트리밍(interleave, <DELAY_d>) 시퀀스와 같은 dict 키(ids · is_input · chunk_of · labels · pos_weight)를 낸다. <|audio_pad|> K 개 자리의 chunk_of = 0..K−1
(모델 build 가 인코더 청크 임베딩을 그 자리에 넣는다 — 인과 인코더라 스트리밍 중 계산한 특징을 그대로 쓸 수 있다). 라벨은 텍스트 토큰 + <|im_end|> 만.
텍스트 토큰은 스트리밍 이벤트열의 텍스트 토큰 그대로(이벤트 <SEM_END>·턴 종료 제외) — 같은 타깃 표기를 두 모드가 공유한다."""
from typing import Dict, List, Optional

OFFLINE_PRE = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
OFFLINE_MID = "<|im_end|>\n<|im_start|>assistant\n"


class OfflineFormat:
    """tokenizer 에서 형식 토큰 id 를 한 번 뽑아 둔다(언어별 머리 캐시)."""
    def __init__(self, tok):
        ids = {t: tok.convert_tokens_to_ids(t) for t in ("<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<|im_end|>")}
        bad = {t: i for t, i in ids.items() if not isinstance(i, int) or i < 0}
        if bad: raise ValueError(f"tokenizer 에 final 모드 형식 토큰이 없다: {bad}")
        self.audio_start, self.audio_end, self.audio_pad, self.im_end = (ids[t] for t in ("<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<|im_end|>"))
        self.pre = list(tok(OFFLINE_PRE, add_special_tokens=False)["input_ids"]); self.mid = list(tok(OFFLINE_MID, add_special_tokens=False)["input_ids"])
        self.tok = tok; self._lang: Dict[str, List[int]] = {}

    def lang_ids(self, lang: str) -> List[int]:
        if lang not in self._lang: self._lang[lang] = list(self.tok(f"language {lang}<asr_text>", add_special_tokens=False)["input_ids"])
        return self._lang[lang]

    def header(self, K: int, lang: str) -> List[int]:
        """오디오 블록 + 어시스턴트 머리(언어 강제)까지 — 추론 프롬프트."""
        return self.pre + [self.audio_start] + [self.audio_pad] * int(K) + [self.audio_end] + self.mid + self.lang_ids(lang)

    def sequence(self, text_ids: List[int], K: int, lang: str) -> dict:
        head = self.header(K, lang); a0 = len(self.pre) + 1
        ids = head + [int(t) for t in text_ids] + [self.im_end]
        is_input = [True] * len(head) + [False] * (len(ids) - len(head))
        chunk_of = [-1] * len(ids)
        for k in range(int(K)): chunk_of[a0 + k] = k
        labels = [(-100 if inp else t) for t, inp in zip(ids, is_input)]
        return dict(ids=ids, is_input=is_input, chunk_of=chunk_of, labels=labels, pos_weight=[0.0] * len(ids), n_text=len(text_ids), prompt_len=len(head))
