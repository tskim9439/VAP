"""토큰 id → 화면 텍스트 증분 조립 (audio-ui-spec §4.4). 순수 파이썬, 세그먼트마다 새로 만든다.

- 태그 라우팅은 vapasr/data/ast_eval.py split_runs 와 같다: <SPK_A>/<SPK_B> → 원문(src), <TRANS> → 번역(tr),
  첫 태그 앞 토큰은 원문, 태그는 표시하지 않는다. 상태(run)는 라운드를 넘어 유지된다.
- byte-level BPE: 스트림마다 바이트 버퍼를 두고 완결된 UTF-8 문자까지만 내보낸다(한글 음절이 두 토큰에 걸칠 수 있음).
- 공백: 연속 공백은 한 칸, 빈 스트림의 앞 공백 제거, 끝 공백은 뒤에 글자가 올 때까지 보류.
  ⇒ "".join(모든 delta) == re.sub(r"\\s+", " ", 전체 디코드).strip()  (vapasr/hf/live.py, infer.py 와 같은 정규화)
- ASR 모드(keep_tr=False): 번역 구간 id 는 버리고 tr_dropped 로 센다(eval 은 run 0 만 가설로 쓴다).
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

_WS = re.compile(r"(\s+)")


def incomplete_tail_len(b: bytes) -> int:
    """b 끝의 '아직 덜 온' UTF-8 시퀀스 길이(0–3). 완성될 수 없는 바이트는 0 → 바로 replace 로 디코드된다."""
    n = len(b)
    for i in range(1, min(3, n) + 1):
        c = b[n - i]
        if 0x80 <= c <= 0xBF:            # 연속 바이트: 더 앞을 본다
            continue
        if 0xC2 <= c <= 0xDF: need = 2
        elif 0xE0 <= c <= 0xEF: need = 3
        elif 0xF0 <= c <= 0xF4: need = 4
        else: return 0                  # ASCII 또는 잘못된 선두 바이트
        return i if need > i else 0
    return 0


def decode_replace(b: bytes) -> tuple[str, int]:
    """errors="replace" 와 같은 결과 + 대체된 바이트 수."""
    try:
        return b.decode("utf-8"), 0
    except UnicodeDecodeError:
        pass
    out, rep = [], 0
    while b:
        try:
            out.append(b.decode("utf-8")); break
        except UnicodeDecodeError as e:
            out.append(b[:e.start].decode("utf-8")); out.append("\ufffd")
            rep += e.end - e.start; b = b[e.end:]
    return "".join(out), rep


class _Stream:
    """한 스트림(src 또는 tr)의 바이트 버퍼 + 공백 상태."""
    __slots__ = ("pending", "nonempty", "space", "parts", "n_ids", "replaced")

    def __init__(self):
        self.pending = b""; self.nonempty = False; self.space = False
        self.parts: list[str] = []; self.n_ids = 0; self.replaced = 0

    def _ws(self, s: str) -> str:
        """디코드된 글자열 → 공백 규칙을 적용한 표시 증분."""
        out = []
        for piece in _WS.split(s):
            if not piece: continue
            if piece.isspace():                           # re 의 \s 와 str.isspace 는 같은 문자 집합(전 코드포인트 확인)
                if self.nonempty: self.space = True       # 앞 공백은 버리고, 끝 공백은 보류
            else:
                if self.space: out.append(" "); self.space = False
                out.append(piece); self.nonempty = True
        d = "".join(out)
        if d: self.parts.append(d)
        return d

    def push(self, b: bytes) -> str:
        if not b: return ""
        buf = self.pending + b
        cut = len(buf) - incomplete_tail_len(buf)
        self.pending = buf[cut:]
        if cut == 0: return ""
        s, rep = decode_replace(buf[:cut]); self.replaced += rep
        return self._ws(s)

    def flush(self) -> str:
        """세그먼트 끝: 남은 불완전 바이트를 replace 로 디코드(센다). 보류된 끝 공백은 버린다."""
        if not self.pending: return ""
        s, rep = decode_replace(self.pending); self.pending = b""
        self.replaced += rep
        return self._ws(s)

    @property
    def text(self) -> str:
        return "".join(self.parts)


class TextAssembler:
    def __init__(self, token_bytes: Callable[[int], bytes], spk_ids: Iterable[int], trans_id: int, keep_tr: bool = True):
        self.token_bytes = token_bytes
        self.spk = frozenset(int(i) for i in spk_ids)
        self.trans_id = int(trans_id)
        self.keep_tr = keep_tr
        self.run = 0                      # 0 = 원문, 1 = 번역
        self.src, self.tr = _Stream(), _Stream()
        self.tr_dropped = 0
        self.finished = False

    def push(self, ids: Iterable[int]) -> tuple[str, str]:
        """한 라운드(또는 여러 라운드)의 id → (src_delta, tr_delta)."""
        bs = [b"", b""]
        for t in ids:
            t = int(t)
            if t in self.spk: self.run = 0; continue
            if t == self.trans_id: self.run = 1; continue
            if self.run == 1:
                if not self.keep_tr: self.tr_dropped += 1; continue
                self.tr.n_ids += 1
            else:
                self.src.n_ids += 1
            bs[self.run] += self.token_bytes(t)
        return self.src.push(bs[0]), self.tr.push(bs[1])

    def flush(self) -> tuple[str, str]:
        """남은 불완전 바이트를 마지막 증분으로 내보낸다(보통 빈 문자열). finish() 전에 불러 delta 로 보내면 된다."""
        return self.src.flush(), self.tr.flush()

    def finish(self) -> tuple[str, str, dict]:
        """(src_text, tr_text, stats). src_text == push()/flush() 증분들의 연결.
        flush() 를 따로 부르지 않았다면 그 증분은 stats["final_src"/"final_tr"] 에 담긴다(서버가 마지막 delta 로 보낸다)."""
        fs, ft = self.flush()
        self.finished = True
        stats = dict(replaced_bytes=self.src.replaced + self.tr.replaced, tr_dropped=self.tr_dropped,
                     n_src=self.src.n_ids, n_tr=self.tr.n_ids, final_src=fs, final_tr=ft)
        return self.src.text, self.tr.text, stats
