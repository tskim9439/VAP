"""가짜 엔진 (app-design §3.6): 실제 Engine/SegmentDecoder 와 같은 API·Round 필드·K/tail/flush/cap 규칙, MLX 가중치 없음.

결정적 토큰: 프레임 k 의 라운드는 δ 프레임 전(j = k − δ) 오디오가 유성(RMS ≥ −50 dBFS)이면 셈을 올리고,
words_every 번째마다 원문 단어 하나를 낸다(필요하면 <SPK_A> 태그). 번역 모드는 원문 2 단어마다 <TRANS> + 번역 단어,
첫 flush 라운드에 남은 번역과 마침표를 낸다. 토큰은 오디오 내용과 k 에만 의존하므로 feed 조각 크기와 무관하다.
일정·cap 은 engine/decoder.py 와 같다: 청크 c 는 4096 + 5120c 샘플에서 계산 가능, feed 는 k < K(n + MIN_TAIL) 까지만,
finish 는 꼬리를 MIN_TAIL 이상으로 올리고 K = int(round(n_total/16000·12.5)), k < K 만 디코드, flush 는 첫 빈 라운드에서 멈춤,
라운드 cap runaway_cap, 전체 cap 6·K_cap+64 (finish 안에서는 6K+64), KV 하드 클램프(Round.kv_full).
서버·UI·역압 테스트용(--engine fake). round_ms·chunk_ms 는 실행 중 바꿀 수 있다(E2: 지연 주입 후 해제).
load_ms·warmup_ms 는 느린 적재·워밍업을 흉내 낸다(라운드는 느려지지 않는다). on_token 은 decoder 와 같은 토큰 콜백.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np

SR = 16000
FRAME_N = 1280                       # 80 ms 오디오 프레임
MIN_TAIL = 1664                      # decoder.MIN_TAIL 과 같다(104 ms)
LANG_LABELS = {"English": "영어", "Korean": "한국어", "German": "독일어"}
LANGS = list(LANG_LABELS)
PAIRS = [("English", "German"), ("English", "Korean"), ("German", "English"), ("Korean", "English")]
DELAYS = [4, 5, 6, 7, 8, 9, 10]
DEFAULTS = {"asr_delay": 6,
            "translate_delay": {"English>German": 10, "English>Korean": 10, "German>English": 10, "Korean>English": 8}}
# 실제 s4 특수 id (decode-contract §a.1)
NEXT, EMPTY, SPK_A, SPK_B, SEM_END, TRANS = 151705, 151706, 151707, 151708, 151723, 151726
UNUSED = 151800                      # 151727–151935: 토큰 없는 행
HEADER = [151644, 8948, 198, 151645, 198, 151644, 77091, 198]
LANG_IDS = {"English": [22574], "German": [32079], "Korean": [42, 45195]}
DELAY_IDS = {4: 151712, 5: 151713, 6: 151714, 7: 151715, 8: 151716, 9: 151724, 10: 151725}

WORDS = {"English": "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split(),
         "German": "eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf".split(),
         "Korean": "하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열 날씨 안녕".split()}
TR_WORDS = {"English": "One two three four five six seven eight nine ten".split(),
            "German": "Eins Zwei Drei Vier Fünf Sechs Sieben Acht Neun Zehn".split(),
            "Korean": "하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 열".split()}


@dataclass
class Round:                          # engine/decoder.py 의 Round 와 같은 필드·순서 (MLX 를 끌어오지 않도록 여기서 정의)
    k: int
    ids: list[int]
    forced: bool
    is_tail: bool
    is_flush: bool
    need_samples: int
    enc_ms: float
    dec_ms: float
    kv_full: bool = False


def k_of(n_samples: int, frame_hz: float = 12.5) -> int:
    """평가와 같은 K = int(round(n/16000·12.5))."""
    return int(round(n_samples / SR * frame_hz))


def prefix_ids(src: str, tgt: str | None, delay: int) -> list[int]:
    """ast-v1 접두 (decode-contract §b.2 표와 같은 id)."""
    line = [11528, 220, SPK_A] + LANG_IDS[src]
    if tgt: line += [198, 14045, 220, SPK_A] + LANG_IDS[tgt]
    return HEADER + line + [151704, DELAY_IDS[delay]]


def _pieces(word: str, split: bool) -> list[bytes]:
    """단어(앞 공백 포함) → 바이트 조각. split 이면 2 바이트씩 잘라 UTF-8 문자를 토큰 경계에 걸치게 한다."""
    b = (" " + word).encode("utf-8")
    if not split: return [b]
    return [b[:3]] + [b[i:i + 2] for i in range(3, len(b), 2)]


_TRUE = lambda v: v not in ("0", "false", "no", "")


class FakeEngine:
    def __init__(self, model_dir: str | None = None, kv_capacity: int = 2048, precision: str = "auto",
                 encoder_precision: str = "auto", speculative: bool = True, *, hard_max_s: float = 30.0, tail_s: float = 1.0,
                 round_ms: float = 0.0, chunk_ms: float = 0.0, words_every: int = 4, hangul: bool = False,
                 runaway: bool = False, asr_trans: bool = False, specials: bool = False, voice_db: float = -50.0,
                 k_cap: int | None = None, runaway_cap: int = 32, max_flush_rounds: int = 8, check_thread: bool = True,
                 load_ms: float = 0.0, warmup_ms: float = 0.0):
        t0 = time.perf_counter()
        self.dir = model_dir
        self.kv_capacity = int(kv_capacity); self.speculative = bool(speculative)
        self.round_ms = float(round_ms); self.chunk_ms = float(chunk_ms)      # 실행 중 바꿔도 된다
        self.words_every = max(1, int(words_every)); self.hangul = hangul; self.runaway = runaway
        self.asr_trans = asr_trans; self.specials = specials; self.voice_db = voice_db
        self.k_cap = int(k_cap) if k_cap is not None else k_of(int(round((hard_max_s + tail_s) * SR)))
        self.runaway_cap = int(runaway_cap); self.max_flush_rounds = int(max_flush_rounds)
        self.check_thread = check_thread; self._tid = threading.get_ident()
        self.warmup_ms = float(warmup_ms)                                     # 느린 적재·워밍업 흉내 (종료·연결 시험용)
        if load_ms > 0: time.sleep(load_ms / 1000)
        self.langs, self.pairs, self.delays = list(LANGS), list(PAIRS), list(DELAYS)
        self.lang_labels = dict(LANG_LABELS)
        self.defaults = {"asr_delay": DEFAULTS["asr_delay"], "translate_delay": dict(DEFAULTS["translate_delay"])}
        self.spk_ids = (SPK_A, SPK_B); self.trans_id = TRANS
        # 고정 어휘: 모든 단어 조각에 id 1000.. 을 미리 매긴다(세션 순서와 무관)
        self._bytes: dict[int, bytes] = {}; self._ids: dict[bytes, int] = {}
        for table in (WORDS, TR_WORDS):
            for words in table.values():
                for w in words:
                    for sp in (False, True):
                        for p in _pieces(w, sp): self._add(p)
        self.dot_id = self._add(b".")
        self.info = dict(name="fake", dir=model_dir, precision="fake", precision_requested=precision, encoder_precision="fake",
                         kv_capacity=self.kv_capacity, load_ms=round((time.perf_counter() - t0) * 1000, 2),
                         speculative=self.speculative, mlx=None, device="cpu (fake)", runaway_cap=self.runaway_cap,
                         max_flush_rounds=self.max_flush_rounds, k_cap=self.k_cap)

    def _add(self, b: bytes) -> int:
        if b not in self._ids:
            i = 1000 + len(self._ids); self._ids[b] = i; self._bytes[i] = b
        return self._ids[b]

    def _check(self):
        if self.check_thread and threading.get_ident() != self._tid:
            raise RuntimeError("엔진은 만든 스레드(작업자)에서만 쓸 수 있습니다")

    def word_ids(self, word: str, split: bool) -> list[int]:
        return [self._ids[p] for p in _pieces(word, split)]

    def warmup(self) -> dict:
        self._check(); t0 = time.perf_counter()
        if self.warmup_ms > 0: time.sleep(self.warmup_ms / 1000)
        modes = [(s, None) for s in LANGS] + list(PAIRS)
        for s, t in modes: self.new_segment(s, t, DEFAULTS["asr_delay"])
        seg = self.new_segment("English", None, DEFAULTS["asr_delay"])
        seg.feed(np.zeros(2 * SR, np.float32)); seg.finish(SR)
        return dict(warmup_ms=round((time.perf_counter() - t0) * 1000, 2), encoder_ms=0.0, prefixes=len(modes), segments=1)

    def new_segment(self, src_lang: str, tgt_lang: str | None, delay: int, *, known_K: int | None = None,
                    on_token=None) -> "FakeSegmentDecoder":
        self._check()
        if src_lang not in LANGS: raise ValueError(f"지원하지 않는 원문 언어: {src_lang!r}")
        if tgt_lang is not None and (src_lang, tgt_lang) not in PAIRS: raise ValueError(f"학습되지 않은 번역 방향: {src_lang}→{tgt_lang}")
        if int(delay) not in DELAYS: raise ValueError(f"δ 는 4–10 이어야 합니다: {delay}")
        d = FakeSegmentDecoder(self, src_lang, tgt_lang, int(delay), known_K)
        d.on_token = on_token
        return d

    def token_bytes(self, tid: int) -> bytes:
        """가짜 어휘 조각의 바이트. 특수·추가 id 와 표 밖 id 는 b"" (스레드 무관, 순수 표 조회)."""
        return self._bytes.get(int(tid), b"")

    def close(self) -> None:
        pass

    @classmethod
    def from_spec(cls, spec: str = "", **kw) -> "FakeEngine":
        """CLI 용: "round_ms=120,hangul=1,words_every=3" → FakeEngine(...)."""
        conv = dict(round_ms=float, chunk_ms=float, words_every=int, hangul=_TRUE, runaway=_TRUE, asr_trans=_TRUE,
                    specials=_TRUE, voice_db=float, k_cap=int, runaway_cap=int, max_flush_rounds=int, kv_capacity=int,
                    hard_max_s=float, tail_s=float, check_thread=_TRUE, load_ms=float, warmup_ms=float)
        for part in filter(None, (s.strip() for s in spec.split(","))):
            k, _, v = part.partition("=")
            k = k.strip()
            if k not in conv: raise ValueError(f"알 수 없는 fake 엔진 옵션: {k}")
            kw[k] = conv[k](v.strip().lower())
        return cls(**kw)


class FakeSegmentDecoder:
    def __init__(self, eng: FakeEngine, src: str, tgt: str | None, delay: int, known_K: int | None = None):
        self.eng = eng; self.src = src; self.tgt = tgt; self.delay = delay
        self.prefix = prefix_ids(src, tgt, delay)
        self.kv_capacity = eng.kv_capacity
        self.pos = len(self.prefix)                     # 접두 prefill 뒤 KV 쓰기 위치
        self.samples_fed = 0; self.n_real: int | None = None; self.K: int | None = known_K
        self.spf = FRAME_N
        self.on_token = None                            # 토큰 콜백 (k, id): 토큰이 확정되는 즉시 (decoder 와 같은 훅, final 은 부르지 않는다)
        self.max_total = 6 * (known_K if known_K is not None else eng.k_cap) + 64
        self._x = np.zeros(SR * 4, np.float32)
        self._c = 0; self.k = 0; self.pending = False
        self.out: list[tuple[int, int]] = []            # (k, id) — 방출 전체
        self.forced = 0; self.flush_rounds = 0; self.kv_full = False; self.done = False
        self.stats = dict(tail_raised=0, overrun=0, k_mismatch=None)
        self._enc_at: dict[int, float] = {}
        self._run = None; self._nv = 0; self._w = 0; self._tw = 0; self._untr = 0
        tr = tgt if tgt else ("German" if src == "English" else "English")
        self._tr_lang = tr
        self._translate = tgt is not None or eng.asr_trans
        self._split_src = eng.hangul and src == "Korean"
        self._split_tr = eng.hangul and tr == "Korean"

    @property
    def positions(self) -> int:
        return self.pos

    # ---------- 토큰 계획 ----------
    def _append(self, x: np.ndarray):
        n = self.samples_fed + len(x)
        if n > len(self._x):
            nb = np.zeros(max(n, 2 * len(self._x)), np.float32); nb[:self.samples_fed] = self._x[:self.samples_fed]; self._x = nb
        self._x[self.samples_fed:n] = x; self.samples_fed = n

    def _voiced(self, j: int) -> bool:
        if j < 0: return False
        a, b = j * FRAME_N, min((j + 1) * FRAME_N, self.samples_fed)
        if b <= a: return False
        fr = self._x[a:b].astype(np.float64)
        return 10 * math.log10(float(np.mean(fr * fr)) + 1e-12) >= self.eng.voice_db

    def _tag(self, out: list[int], run: int):
        if self._run != run: out.append(SPK_A if run == 0 else TRANS); self._run = run

    def _src_word(self, out: list[int]):
        words = WORDS[self.src]; self._tag(out, 0)
        out += self.eng.word_ids(words[self._w % len(words)], self._split_src); self._w += 1; self._untr += 1
        if self.eng.specials and self._w % 5 == 0: out += [SEM_END, UNUSED]

    def _tr_words(self, out: list[int], final: bool = False):
        words = TR_WORDS[self._tr_lang]
        self._tag(out, 1)
        while self._untr > (0 if final else 1):        # 원문 2 단어마다 번역 1 단어(끝에서는 남은 것 모두)
            out += self.eng.word_ids(words[self._tw % len(words)], self._split_tr); self._tw += 1; self._untr -= 2
        self._untr = max(self._untr, 0)
        if final: out.append(self.eng.dot_id)

    def _plan(self, k: int, flush: bool) -> list[int]:
        """이 라운드에 낼 후보 id(태그 포함). 결정적: 오디오 내용·k·세그먼트 상태만 본다."""
        out: list[int] = []
        if flush:
            if k == self.K and self._translate and (self._untr or self._tw): self._tr_words(out, final=True)
            return out
        if self.eng.runaway:
            while len(out) < self.eng.runaway_cap + 8: self._src_word(out)
            return out
        if self._voiced(k - self.delay):
            self._nv += 1
            if self.eng.words_every == 1 or self._nv % self.eng.words_every == 1:
                self._src_word(out)
                if self._translate and self._untr >= 2: self._tr_words(out)
        return out

    # ---------- 라운드 ----------
    def _round(self, k: int, flush: bool = False) -> Round:
        t0 = time.perf_counter()
        if self.eng.round_ms > 0: time.sleep(self.eng.round_ms / 1000)
        need = self.samples_fed if flush else 4096 + 5120 * (k // 4)
        if not flush and self.n_real is not None: need = min(need, self.samples_fed)
        is_tail = flush or (self.n_real is not None and k * FRAME_N >= self.n_real)
        ids: list[int] = []; forced = kv_full = False
        if self.pos + 2 > self.kv_capacity:              # 하드 KV 클램프: 입력(NEXT+오디오)조차 못 쓴다
            forced = kv_full = True
        else:
            self.pos += 2 if self.pending else 1
            for tid in self._plan(k, flush):
                if len(ids) >= self.eng.runaway_cap or len(self.out) >= self.max_total: forced = True; break
                if self.pos + 2 > self.kv_capacity: forced = kv_full = True; break
                ids.append(tid); self.pos += 1; self.out.append((k, tid))
                if self.on_token is not None: self.on_token(k, tid)
            self.pending = True
        self.kv_full = self.kv_full or kv_full
        self.forced += forced
        return Round(k=k, ids=ids, forced=forced, is_tail=is_tail, is_flush=flush, need_samples=int(need),
                     enc_ms=self._enc_at.pop(k, 0.0), dec_ms=(time.perf_counter() - t0) * 1000, kv_full=kv_full)

    def _decode(self, upto: int) -> list[Round]:
        rounds = []
        while self.k < upto and not self.kv_full:
            rounds.append(self._round(self.k)); self.k += 1
        return rounds

    def _enc(self, k0: int):
        """청크 하나의 (가짜) 인코더 시간 → 그 청크 첫 프레임 라운드에 귀속."""
        t0 = time.perf_counter()
        if self.eng.chunk_ms > 0: time.sleep(self.eng.chunk_ms / 1000)
        self._enc_at[k0] = self._enc_at.get(k0, 0.0) + (time.perf_counter() - t0) * 1000

    def _check_open(self):
        self.eng._check()
        if self.done: raise RuntimeError("이미 끝난 세그먼트입니다")

    # ---------- 공개 API ----------
    def feed(self, pcm: np.ndarray) -> list[Round]:
        """16 kHz float32 → 새로 완성된 인코더 청크(4096 + 5120c 샘플)의 프레임 중 k < K(n + MIN_TAIL) 를 디코드."""
        self._check_open()
        x = np.asarray(pcm, np.float32).reshape(-1)
        self._append(x)
        while self.samples_fed >= 4096 + 5120 * self._c:
            self._enc(4 * self._c); self._c += 1
        return self._decode(min(4 * self._c, k_of(self.samples_fed + MIN_TAIL)))

    def finish(self, zero_tail_samples: int) -> list[Round]:
        """0 을 붙이고(최소 MIN_TAIL) K = int(round(n_total/16000*12.5)), 남은 k < K 프레임, 그다음 flush(첫 빈 라운드에서 멈춤)."""
        self._check_open()
        zt = int(zero_tail_samples)
        if zt < MIN_TAIL: self.stats["tail_raised"] = MIN_TAIL - zt; zt = MIN_TAIL
        self.n_real = self.samples_fed
        self._append(np.zeros(zt, np.float32))
        self.K = K = k_of(self.samples_fed)
        self.max_total = 6 * K + 64                       # finish 안에서는 정확한 전체 cap
        if self.k > K: self.stats["overrun"] = self.k - K
        if self.k < K: self._enc(self.k)                  # 남은 청크 + 마지막 짧은 스텝
        rounds = self._decode(K)
        for i in range(self.eng.max_flush_rounds):
            if self.kv_full: break
            r = self._round(K + i, flush=True); rounds.append(r); self.flush_rounds += 1
            if not r.ids: break
        self.done = True
        return rounds

    def final_redecode(self, max_new: int | None = None) -> tuple[list[int], float]:
        """가짜 final 모드: <SPK_A> 원문 전체 [+ <TRANS> 번역 전체 + "."]. (ids, ms)."""
        self.eng._check()
        if not self.done: raise RuntimeError("final_redecode 는 finish() 뒤에만 쓸 수 있습니다")
        t0 = time.perf_counter()
        src, run = [], 0                                  # split_runs: 원문 구간 id 만
        for _, t in self.out:
            if t in (SPK_A, SPK_B): run = 0
            elif t == TRANS: run = 1
            elif run == 0: src.append(t)
        ids = [SPK_A] + src
        if self.tgt:
            words = TR_WORDS[self._tr_lang]
            ids += [TRANS] + [i for n in range(self._tw) for i in self.eng.word_ids(words[n % len(words)], self._split_tr)]
            ids += [self.eng.dot_id]
        if self.eng.round_ms > 0: time.sleep(self.eng.round_ms / 1000)
        cap = max_new if max_new is not None else (self.K or 0) + 64
        return ids[:cap], (time.perf_counter() - t0) * 1000

    def final_job(self, max_new: int | None = None) -> "FakeFinalJob":
        """decoder.FinalJob 과 같은 꼴: step() 한 번에 토큰 하나 (첫 step 이 전체를 계산)."""
        self.eng._check()
        if not self.done: raise RuntimeError("final 재디코드는 finish() 뒤에만 쓸 수 있습니다")
        return FakeFinalJob(self, max_new)


class FakeFinalJob:
    """가짜 단계형 final 작업: 결과는 final_redecode 와 같고, step() 마다 id 하나씩 드러낸다 (서버 끼워 넣기 시험용)."""

    def __init__(self, seg: "FakeSegmentDecoder", max_new: int | None = None):
        self._seg, self._max_new = seg, max_new
        self._all: list[int] | None = None
        self.ids: list[int] = []; self.ms = 0.0; self.steps = 0; self.done = False

    def step(self) -> bool:
        if self.done: return True
        t0 = time.perf_counter()
        if self._all is None:
            self._all, _ = self._seg.final_redecode(self._max_new)
        if len(self.ids) < len(self._all):
            self.ids.append(self._all[len(self.ids)])
        self.steps += 1
        self.done = len(self.ids) >= len(self._all)
        self.ms += (time.perf_counter() - t0) * 1000
        return self.done

    def run(self) -> tuple[list[int], float]:
        while not self.step():
            pass
        return list(self.ids), self.ms
