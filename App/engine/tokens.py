"""토큰 id·언어·지연·prefix 규칙 (decode-contract §a, §b, §f).

prefix 와 final(오프라인) 헤더는 체크포인트 토크나이저로 만들고, 아래 표(§b.2)와 다르면 즉시 실패한다
(토크나이저가 조금만 달라져도 parity 가 깨진다)."""
from __future__ import annotations

from dataclasses import dataclass

# 언어 이름(모델 prefix 에 그대로 들어감) → UI 라벨
LANGS: dict[str, str] = {"English": "영어", "Korean": "한국어", "German": "독일어"}
# 학습된 번역 방향만 (de↔ko 는 학습 안 됨)
PAIRS: list[tuple[str, str]] = [("English", "German"), ("English", "Korean"), ("German", "English"), ("Korean", "English")]
DELAYS: list[int] = [4, 5, 6, 7, 8, 9, 10]
# audio-ui-spec §2.1 [critic]: 전사 δ6, 번역은 짝마다 (ko→en δ8, 나머지 δ10)
DEFAULTS: dict = {"asr_delay": 6,
                  "translate_delay": {"English>German": 10, "English>Korean": 10, "German>English": 10, "Korean>English": 8}}

# ── 기대값 표 (decode-contract §a.1, §a.2, §b.2, §f) — 체크포인트 값과 대조만 한다
SP_EXPECTED = {"<NEXT_AUDIO>": 151705, "<EMPTY_AUDIO>": 151706, "<SPK_A>": 151707, "<SPK_B>": 151708,
               "<DELAY_1>": 151709, "<DELAY_2>": 151710, "<DELAY_3>": 151711, "<DELAY_4>": 151712, "<DELAY_5>": 151713,
               "<DELAY_6>": 151714, "<DELAY_7>": 151715, "<DELAY_8>": 151716, "<SPK_3>": 151717, "<SPK_4>": 151718,
               "<SPK_5>": 151719, "<SPK_6>": 151720, "<ONSET>": 151721, "<EOT>": 151722, "<SEM_END>": 151723,
               "<DELAY_9>": 151724, "<DELAY_10>": 151725, "<TRANS>": 151726}
QWEN_EXPECTED = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645, "<|audio_start|>": 151669,
                 "<|audio_end|>": 151670, "<|audio_pad|>": 151676, "<asr_text>": 151704}
BLOCKED_EXPECTED = [151644, 151645, 151669, 151670, 151676, 151704, 151706, 151709, 151710, 151711, 151712, 151713,
                    151714, 151715, 151716, 151717, 151718, 151719, 151720, 151721, 151722, 151724, 151725]
HEADER_EXPECTED = [151644, 8948, 198, 151645, 198, 151644, 77091, 198]          # <|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n
OFF_PRE_EXPECTED = [151644, 8948, 198, 151645, 198, 151644, 872, 198]           # ...<|im_start|>user\n
OFF_MID_EXPECTED = [151645, 198, 151644, 77091, 198]                            # <|im_end|>\n<|im_start|>assistant\n
_LANG_PIECES = {"English": [22574], "Korean": [42, 45195], "German": [32079]}


def expected_head(src: str, tgt: str | None) -> list[int]:
    """§b.2 표: language Ġ<SPK_A>L [\\n translate Ġ<SPK_A>T] <asr_text>."""
    ids = [11528, 220, 151707] + _LANG_PIECES[src]
    if tgt:
        ids += [198, 14045, 220, 151707] + _LANG_PIECES[tgt]
    return ids + [151704]


def head_text(src: str, tgt: str | None, spk: str = "<SPK_A>") -> str:
    """vapasr.data.ast_dataset.ast_head_text 와 같은 문자열."""
    return f"language {spk}{src}" + (f"\ntranslate {spk}{tgt}" if tgt else "") + "<asr_text>"


def check_mode(src: str, tgt: str | None, delay: int | None = None) -> None:
    if src not in LANGS:
        raise ValueError(f"지원하지 않는 원문 언어: {src!r} (가능: {', '.join(LANGS)})")
    if tgt is not None and (src, tgt) not in PAIRS:
        raise ValueError(f"학습되지 않은 번역 방향: {src}→{tgt} (가능: {', '.join(a + '→' + b for a, b in PAIRS)})")
    if delay is not None and delay not in DELAYS:
        raise ValueError(f"지원하지 않는 지연 δ{delay} (가능: {DELAYS[0]}–{DELAYS[-1]})")


@dataclass(frozen=True)
class SpecialIds:
    next: int; empty: int; spk_a: int; spk_b: int; trans: int; sem_end: int
    im_start: int; im_end: int; audio_start: int; audio_end: int; audio_pad: int; asr_text: int; endoftext: int
    delay: dict  # δ → id

    @classmethod
    def from_config(cls, cfg: dict, tok) -> "SpecialIds":
        """config.sp_ids 와 토크나이저 id 를 서로·표와 대조한다 (decode-contract 체크리스트)."""
        sp = dict(cfg["sp_ids"])
        bad = [f"{k}: config {sp.get(k)} / 기대 {v}" for k, v in SP_EXPECTED.items() if sp.get(k) != v]
        bad += [f"{k}: config {v} / 토크나이저 {tok.token_to_id(k)}" for k, v in sp.items() if tok.token_to_id(k) != v]
        bad += [f"{k}: 토크나이저 {tok.token_to_id(k)} / 기대 {v}" for k, v in QWEN_EXPECTED.items() if tok.token_to_id(k) != v]
        if bad:
            raise RuntimeError("특수 토큰 id 불일치 — 체크포인트/토크나이저가 s4 와 다릅니다:\n  " + "\n  ".join(bad))
        q = {k: tok.token_to_id(k) for k in QWEN_EXPECTED}
        return cls(next=sp["<NEXT_AUDIO>"], empty=sp["<EMPTY_AUDIO>"], spk_a=sp["<SPK_A>"], spk_b=sp["<SPK_B>"],
                   trans=sp["<TRANS>"], sem_end=sp["<SEM_END>"], im_start=q["<|im_start|>"], im_end=q["<|im_end|>"],
                   audio_start=q["<|audio_start|>"], audio_end=q["<|audio_end|>"], audio_pad=q["<|audio_pad|>"],
                   asr_text=q["<asr_text>"], endoftext=q["<|endoftext|>"],
                   delay={d: sp[f"<DELAY_{d}>"] for d in range(1, 11)})


def _byte_decoder() -> dict[str, int]:
    """GPT-2 bytes_to_unicode 의 역표 (byte-level BPE 문자 → 바이트)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class TokenTable:
    """체크포인트 토크나이저 + config 로 prefix·헤더·바이트 표를 만든다."""

    def __init__(self, tok, cfg: dict, vocab_rows: int):
        self.tok = tok
        self.sp = SpecialIds.from_config(cfg, tok)
        self.header = list(cfg["prefix_ids"])
        if self.header != HEADER_EXPECTED:
            raise RuntimeError(f"config.prefix_ids 가 표와 다릅니다: {self.header}")
        enc = lambda s: tok.encode(s, add_special_tokens=False).ids
        self.off_pre = enc("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n")
        self.off_mid = enc("<|im_end|>\n<|im_start|>assistant\n")
        if self.off_pre != OFF_PRE_EXPECTED or self.off_mid != OFF_MID_EXPECTED:
            raise RuntimeError(f"final 헤더 id 가 표와 다릅니다: pre {self.off_pre} mid {self.off_mid}")
        self._heads: dict = {}
        for src in LANGS:                         # 7 모드 × δ 전부 표와 대조
            for tgt in [None] + [b for a, b in PAIRS if a == src]:
                h = enc(head_text(src, tgt))
                if h != expected_head(src, tgt):
                    raise RuntimeError(f"prefix 불일치 {src}→{tgt}: {h} ≠ {expected_head(src, tgt)}")
                self._heads[(src, tgt)] = h
        self.n_tok = tok.get_vocab_size(with_added_tokens=True)
        if self.n_tok > vocab_rows:
            raise RuntimeError(f"토크나이저({self.n_tok})가 임베딩 행 수({vocab_rows})보다 큽니다")
        self._bytes = self._build_bytes(vocab_rows)
        # final 모드 차단 집합 (decode-contract §f; offline_decode.py:13-22 + EV:494-495)
        sp = self.sp
        ob = set(int(v) for v in cfg["sp_ids"].values()) | set(int(v) for v in (cfg.get("sem_registry") or {}).values())
        ob |= {sp.audio_start, sp.audio_end, sp.audio_pad, sp.im_start, sp.asr_text}
        ob.discard(sp.im_end)
        self.offline_blocked = sorted(ob - {sp.spk_a, sp.spk_b, sp.trans})

    def _build_bytes(self, rows: int) -> list[bytes]:
        # BPE 어휘(0..n_bpe-1)만 바이트를 갖는다. 추가 토큰(특수·비특수 모두, 151643..)과 그 밖 id 는 b""
        dec, n_bpe = _byte_decoder(), self.tok.get_vocab_size(with_added_tokens=False)
        out = [b""] * rows
        for t, i in self.tok.get_vocab(with_added_tokens=False).items():
            if i < min(rows, n_bpe):
                out[i] = bytes(dec[c] for c in t)
        return out

    def token_bytes(self, tid: int) -> bytes:
        """byte-level BPE 바이트. 추가/특수 토큰과 토크나이저 밖 id(151727..)는 b''."""
        return self._bytes[tid] if 0 <= tid < len(self._bytes) else b""

    def head_ids(self, src: str, tgt: str | None) -> list[int]:
        check_mode(src, tgt)
        return list(self._heads[(src, tgt)])

    def prefix_ids(self, src: str, tgt: str | None, delay: int) -> list[int]:
        """스트리밍 prefix = config.prefix_ids + 언어 줄 + <DELAY_δ> (§b.1)."""
        check_mode(src, tgt, delay)
        return self.header + self._heads[(src, tgt)] + [self.sp.delay[delay]]

    def offline_header(self, src: str, tgt: str | None, K: int) -> tuple[list[int], int]:
        """final 모드 헤더 (§f): pre + <|audio_start|> + pad×K + <|audio_end|> + mid + 언어 줄(δ 없음). → (ids, 오디오 시작 위치)."""
        ids = self.off_pre + [self.sp.audio_start] + [self.sp.audio_pad] * K + [self.sp.audio_end] + self.off_mid + self.head_ids(src, tgt)
        return ids, len(self.off_pre) + 1

    def decode(self, ids: list[int]) -> str:
        """tok.decode(ids, skip_special_tokens=True).strip() 와 같은 용도(바이트 이어붙이기)."""
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace").strip()
