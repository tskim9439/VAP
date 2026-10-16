"""SegmentDecoder: 세그먼트(발화) 하나의 스트리밍 디코드 (decode-contract §c, §f; thinker-spec §5).

규칙(병합 NEXT 형): prefix 1회(logits 버림) → 프레임 k 마다 [NEXT?, a_k] 입력 → argmax 가 NEXT 이거나
라운드 cap·전체 cap 에 닿을 때까지 토큰을 내고 먹인다 (NEXT 아닌 마지막 argmax 는 버리고 forced 로 센다) →
K 프레임 뒤 <EMPTY_AUDIO> flush 를 최대 max_flush 회, 빈 라운드에서 멈춘다.
투기 배치: 한 인코더 청크의 프레임들을 [NEXT?, a_k, NEXT, a_k+1, ...] 한 번의 forward 로 보고,
첫 방출 프레임까지 받아들인 뒤 KV 포인터를 되감는다(start 만 작게). 순차 greedy 와 수학적으로 같다.
초안(draft, thinker.Draft): 토큰을 낸 뒤 q4 초안이 다음 토큰 L 개를 제안하고 목표 forward 한 번 [tid, d1..dm] 으로 검증,
순차와 같은 검사(stop·cap·KV 클램프) 순서로 일치하는 만큼 받아들인다. 초안이 NEXT 로 끝나면 같은 인코더 그룹의 남은 프레임
[NEXT, a_ka, NEXT, a_ka+1, …] 을 검증 forward 에 덧붙여(라운드 끝 투기) 라운드가 거기서 끝나면 다음 배치 결과로 그대로 쓴다.
내는 id 는 언제나 목표 argmax 다 (logits 은 q>1 forward 라 비트 동일은 아님 — 투기 배치와 같은 등급, thinker-spec §5.2).
final(오프라인) 재디코드는 FinalJob 으로 forward 한 걸음씩 (별도 KV) — 서버가 스트리밍 사이에 끼워 넣는다.
lookahead > 0 이면 한 걸음이 스트리밍 출력을 초안으로 한 검증 forward (≤ 16 행)."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import mlx.core as mx

SR = 16000
MIN_TAIL = 1664          # 끝 영 꼬리 최소(104 ms): 이보다 길면 끝 결정 전에 디코드한 프레임은 모두 k < K
MAX_VERIFY_ROWS = 12     # 검증 forward 행 상한 (fp16 q=16 은 33 ms 로 뛴다; L=4 면 최대 5 + 2·3 = 11)
DRAFT_GUARD = (8, 0.4)   # 초안 검증 8 번 뒤 일치율 EMA 가 0.4 아래면 이 세그먼트는 초안 끔 (0 % 면 토큰당 ≈ 4·t_d + t_v)


def k_of(n_samples: int, frame_hz: float = 12.5) -> int:
    """평가와 같은 K = int(round(n/16000·12.5)) (round-half-even)."""
    return int(round(n_samples / SR * frame_hz))


@dataclass
class Round:
    k: int                 # 오디오 프레임 번호 (k ≥ K ⇒ flush 라운드)
    ids: list[int]         # 이번 라운드에 낸 id (태그 포함, <NEXT_AUDIO> 제외)
    forced: bool
    is_tail: bool          # 영 꼬리 안의 프레임이거나 flush
    is_flush: bool
    need_samples: int      # 이 라운드에 필요했던 세그먼트 샘플 수 (4096 + 5120·(k//4)); flush 는 n_total
    enc_ms: float          # 이 라운드에 귀속한 인코더 ms (청크의 첫 라운드가 청크 시간을 가짐)
    dec_ms: float
    kv_full: bool = False


class SegmentDecoder:
    """한 세그먼트. Engine.new_segment 가 만든다 (워커 스레드 전용). 생성 시 prefix 를 먹인다."""

    def __init__(self, thinker, adapter, prefix: list[int], sp, *, cap: int, max_flush: int, frame_hz: float = 12.5,
                 kv_capacity: int | None = None, stream=None, speculative: bool = True, group: int = 4,
                 k_cap: int = 388, max_total_per_chunk: float = 6.0, known_K: int | None = None,
                 tokens=None, mode: tuple | None = None, offline_block: mx.array | None = None,
                 draft=None, draft_len: int = 4, on_token=None, prefix_cache: "PrefixCache | None" = None,
                 final_lookahead: int = 0):
        """draft: thinker.Draft (None = 토큰마다 q=1 forward), on_token(k, id): 토큰이 확정되는 즉시 (워커 스레드에서; 가볍게,
        디코더를 건드리지 말 것 — 같은 토큰이 그 호출이 돌려주는 Round.ids 에도 있다), prefix_cache: Engine 의 PrefixCache,
        final_lookahead: final 재디코드 한 걸음의 초안 토큰 수 (0 = q=1)."""
        self.th, self.ad, self.stream, self.sp = thinker, adapter, stream, sp
        self.prefix = list(prefix)
        self.cap, self.max_flush, self.frame_hz = int(cap), int(max_flush), float(frame_hz)
        self.spf = int(round(SR / self.frame_hz))                     # 프레임당 샘플 1280
        self.kv_capacity = min(int(kv_capacity or thinker.max_len), thinker.max_len)
        self.speculative, self.group = bool(speculative), int(group)
        self.mtpc = float(max_total_per_chunk)
        self.K = known_K
        # 스트리밍 중 전체 cap 은 인과적으로 6·K_cap+64, finish 에서 정확한 6·K+64 (audio-ui-spec §1.6)
        self.max_total = int(self.mtpc * (known_K if known_K is not None else k_cap)) + 64
        self.tokens, self.mode, self.offline_block = tokens, mode, offline_block
        self.samples_fed = 0
        self.n_real: int | None = None
        self._frames: list[mx.array] = []
        self._ce: list[mx.array] = []                                 # 프레임별 adapter 출력 (thinker 입력 dtype)
        self._enc_at: dict[int, float] = {}
        self._enc_carry = 0.0
        self.k = 0
        self.pos = 0
        self.pending = False
        self.out: list[tuple[int, int]] = []
        self.forced = 0
        self.flush_rounds = 0
        self.kv_full = False
        self.done = False
        self.on_token = on_token
        self.draft, self.draft_len = draft, int(draft_len)
        self._draft_on = draft is not None and self.draft_len > 0
        self._acc_ema, self._acc_n = 1.0, 0
        self.final_lookahead = max(0, min(15, int(final_lookahead)))
        self._deferred: list[np.ndarray] = []                         # kv_full 뒤 인코더에 넣지 않은 PCM (features() 가 처리)
        self._deferred_tail: int | None = None
        self.stats = dict(forwards=0, prefix_ms=0.0, prefix_cached=False, tail_raised=0, overrun=0, k_mismatch=None,
                          verifies=0, drafted=0, accepted=0, draft_steps=0, res_tried=0, res_used=0, draft_off=0,
                          deferred_samples=0, on_token_errors=0)
        self._eN = thinker.embed([sp.next])[0]
        self._eE = thinker.embed([sp.empty])[0]
        t0 = time.perf_counter()
        if len(self.prefix) + 2 > self.kv_capacity:
            raise ValueError("KV 용량이 prefix 보다 작습니다")
        pc = prefix_cache if (prefix_cache is not None and prefix_cache.thinker is thinker) else None
        rows = pc.get(tuple(self.prefix)) if pc is not None else None
        if rows is not None:                                           # 같은 prefix 의 KV 행을 되돌린다 (forward 생략, 비트 동일)
            thinker.set_kv_rows(rows)
            self.stats["prefix_cached"] = True
        else:
            thinker.forward(thinker.embed(self.prefix), 0, sel=[])   # 마지막 prefix 위치의 logits 는 버린다
            if pc is not None:
                pc.put(tuple(self.prefix), thinker.kv_rows(len(self.prefix)))
        self.pos = len(self.prefix)
        self.stats["prefix_ms"] = (time.perf_counter() - t0) * 1000
        self._prefix_carry = self.stats["prefix_ms"]                   # 첫 Round 의 dec_ms 에 더한다 (compute·RTF 에 포함)

    # ── 공개 API
    @property
    def positions(self) -> int:
        return self.pos

    def feed(self, pcm: np.ndarray) -> list[Round]:
        """16 kHz float32 PCM 을 붙이고, 새로 완성된 인코더 청크의 프레임을 디코드한다."""
        self._check_open()
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if self.kv_full:                    # 더 디코드하지 않는다: 인코더도 미룬다 (final 이 features() 를 부를 때만 돌린다)
            if pcm.size:
                self._deferred.append(pcm.copy())
            self.samples_fed += int(pcm.size)
            self.stats["deferred_samples"] += int(pcm.size)
            return []
        t0 = time.perf_counter()
        fr = self.stream.feed(pcm) if pcm.size else None
        if fr is not None and fr.shape[0]:
            mx.eval(fr)
        self.samples_fed += int(pcm.size)
        self._add(fr, (time.perf_counter() - t0) * 1000)
        # 꼬리가 MIN_TAIL 이상 붙으므로 k < K(n + MIN_TAIL) 인 프레임은 최종 K 안이다 (실제로는 늘 전부 통과)
        return self._decode(min(len(self._ce), k_of(self.samples_fed + MIN_TAIL, self.frame_hz)))

    def finish(self, zero_tail_samples: int) -> list[Round]:
        """영 꼬리를 엔진 안에서 붙이고 K 를 정한 뒤 k < K 인 남은 프레임과 flush 라운드를 디코드한다."""
        self._check_open()
        zt = int(zero_tail_samples)
        if zt < MIN_TAIL:
            self.stats["tail_raised"] = MIN_TAIL - zt
            zt = MIN_TAIL
        self.n_real = self.samples_fed
        if self.kv_full:                    # 꼬리도 인코더에 넣지 않는다 (features() 에서), K 는 평가와 같은 식
            self._deferred_tail = zt
            self.samples_fed += zt
            self.K = k_of(self.samples_fed, self.frame_hz)
            self.max_total = int(self.mtpc * self.K) + 64
            self.done = True
            return []
        t0 = time.perf_counter()
        fr, K = self.stream.finish(zt)
        if fr is not None and fr.shape[0]:
            mx.eval(fr)
        self.samples_fed += zt
        Kx = k_of(self.samples_fed, self.frame_hz)
        if K != Kx:
            self.stats["k_mismatch"] = (K, Kx)
            K = Kx
        return self._finish(fr, (time.perf_counter() - t0) * 1000, K)

    def feed_frames(self, frames, enc_ms: float = 0.0) -> list[Round]:
        """인코더 출력 프레임을 직접 넣는다 (검증·테스트·파일 모드). known_K 가 있으면 그 안까지만 디코드."""
        self._check_open()
        self._add(mx.array(frames) if isinstance(frames, np.ndarray) else frames, enc_ms)
        lim = len(self._ce) if self.K is None else min(len(self._ce), self.K)
        return self._decode(lim)

    def finish_frames(self, K: int | None = None, n_total: int | None = None) -> list[Round]:
        """feed_frames 로 넣은 세그먼트를 K 로 마감한다 (K 를 모르면 known_K, 없으면 넣은 프레임 수)."""
        self._check_open()
        K = int(K if K is not None else (self.K if self.K is not None else len(self._ce)))
        if n_total is not None:
            self.samples_fed = int(n_total)
        return self._finish(None, 0.0, K)

    def features(self) -> mx.array:
        """(K, 1024) fp32 인과 인코더 특징 (finish 뒤; 그 전엔 지금까지의 프레임). kv_full 뒤 미룬 PCM 이 있으면 여기서 인코딩."""
        if self._deferred or self._deferred_tail is not None:
            self._encode_deferred()
        f = mx.concatenate(self._frames, axis=0) if self._frames else mx.zeros((0, 1024), mx.float32)
        return f[: self.K] if self.K is not None else f

    def final_job(self, max_new: int | None = None, lookahead: int | None = None) -> "FinalJob":
        """final(오프라인) 모드 재디코드를 한 걸음씩 진행하는 작업 (decode-contract §f, audio-ui-spec §5.2·§7).
        job.step() 한 번 = forward 한 번 → done. 서버는 스트리밍 일이 없을 때만 step() 하고 사이사이 입력 큐를 본다."""
        if not self.done or self.tokens is None or self.mode is None:
            raise RuntimeError("final 재디코드는 finish() 뒤에만 쓸 수 있습니다")
        return FinalJob(self, max_new, self.final_lookahead if lookahead is None else lookahead)

    def final_redecode(self, max_new: int | None = None) -> tuple[list[int], float]:
        """final 모드 재디코드를 끝까지 한 번에 (블로킹) → (ids, ms). 별도 KV 라 스트리밍 버퍼를 건드리지 않는다."""
        return self.final_job(max_new).run()

    # ── 내부
    def _check_open(self):
        if self.done:
            raise RuntimeError("이미 끝난 세그먼트입니다")

    def _encode_deferred(self):
        """kv_full 뒤 미룬 PCM(+꼬리)을 같은 스트림에 순서대로 넣는다 — 인코더 프레임은 패킷 크기와 무관하게 비트 동일."""
        pcm = np.concatenate(self._deferred) if self._deferred else np.zeros(0, np.float32)
        self._deferred = []
        frs = []
        if pcm.size:
            fr = self.stream.feed(pcm)
            if fr is not None and fr.shape[0]:
                frs.append(fr)
        if self._deferred_tail is not None:
            fr, _ = self.stream.finish(self._deferred_tail)
            self._deferred_tail = None
            if fr is not None and fr.shape[0]:
                frs.append(fr)
        have = sum(int(f.shape[0]) for f in self._frames)
        for fr in frs:
            if self.K is not None:
                fr = fr[: max(0, self.K - have)]                        # k ≥ K 프레임은 버린다 (_finish 와 같게)
            if fr.shape[0]:
                mx.eval(fr)
                self._frames.append(fr)
                have += int(fr.shape[0])
        if self.K is not None and have < self.K:                        # 모자라면 0 벡터 (_finish 와 같게)
            self._frames.append(mx.zeros((self.K - have, int(self.ad.w["net.0.weight"].shape[0])), mx.float32))

    def _emit(self, k: int, tid: int, ids: list[int]):
        """토큰 확정 (순차·초안 경로 공통의 유일한 곳). on_token 은 여기서 바로 — 다음 forward 전에."""
        self.out.append((k, tid))
        ids.append(tid)
        cb = self.on_token
        if cb is not None:
            try:
                cb(k, tid)
            except Exception:                                           # 소비자 오류로 디코드를 깨지 않는다
                self.stats["on_token_errors"] += 1

    def _add(self, fr, enc_ms: float):
        """프레임 추가 + 인코더 시간 귀속 (청크(4 프레임)의 첫 프레임에 균등 분배)."""
        self._enc_carry += enc_ms
        if fr is None or not fr.shape[0]:
            return
        base, n = len(self._ce), int(fr.shape[0])
        self._frames.append(fr)
        ce = self.ad(fr).astype(self.th.in_dtype)
        self._ce.extend(ce[i] for i in range(n))
        starts = [base + j for j in range(n) if (base + j) % 4 == 0] or [base]
        for s in starts:
            self._enc_at[s] = self._enc_at.get(s, 0.0) + self._enc_carry / len(starts)
        self._enc_carry = 0.0

    def _finish(self, fr, enc_ms: float, K: int) -> list[Round]:
        self.K = K
        self.max_total = int(self.mtpc * K) + 64                       # finish 안에서는 정확한 전체 cap
        if fr is not None and fr.shape[0] and len(self._ce) + fr.shape[0] > K:
            fr = fr[: max(0, K - len(self._ce))]                        # k ≥ K 프레임은 버린다
        self._add(fr, enc_ms)
        if len(self._ce) < K:                                           # 모자라면 0 벡터(adapter 전) — 실제로는 생기지 않음
            self._add(mx.zeros((K - len(self._ce), int(self.ad.w["net.0.weight"].shape[0])), mx.float32), 0.0)
        if self.k > K:
            self.stats["overrun"] = self.k - K                          # 꼬리가 짧아 K 이상을 이미 디코드함 (parity 깨짐 표시)
        rounds = [] if self.kv_full else self._decode(K)
        rounds += self._flush()
        if self._enc_carry and rounds:
            rounds[0].enc_ms += self._enc_carry
        self._enc_carry = 0.0
        self.done = True
        return rounds

    def _stop(self, tid: int, n: int) -> bool:
        return tid == self.sp.next or n >= self.cap or len(self.out) >= self.max_total

    def _round(self, k, ids, forced, dec_ms, kv_full=False, flush=False) -> Round:
        n_tot = self.samples_fed if (self.done or self.n_real is not None) else None
        need = 4096 + 5120 * (k // 4)
        if flush:
            need = self.samples_fed
        elif n_tot is not None:
            need = min(need, n_tot)
        tail = flush or (self.n_real is not None and k * self.spf >= self.n_real)
        enc = self._enc_at.pop(k, 0.0)
        if kv_full:                         # 마지막 Round: 아직 귀속 안 된 인코더 시간을 모두 (이후 Round 는 없다)
            enc += sum(self._enc_at.values()) + self._enc_carry
            self._enc_at.clear()
            self._enc_carry = 0.0
        dec_ms += self._prefix_carry
        self._prefix_carry = 0.0
        return Round(k=k, ids=ids, forced=bool(forced), is_tail=bool(tail), is_flush=flush, need_samples=int(need),
                     enc_ms=enc, dec_ms=dec_ms, kv_full=kv_full)

    def _batch(self, k0: int, k1: int):
        xs, sel = [], []
        for kk in range(k0, k1):
            if self.pending or xs:
                xs.append(self._eN)                                    # 앞 라운드를 닫는 NEXT
            xs.append(self._ce[kk])
            sel.append(len(xs) - 1)
        return xs, sel

    def _kv_round(self, k, flush=False) -> Round:
        self.forced += 1
        self.kv_full = True
        return self._round(k, [], True, 0.0, kv_full=True, flush=flush)

    def _decode(self, upto: int) -> list[Round]:
        """프레임 self.k .. upto-1 을 디코드 (투기 배치 또는 순차). carry = 직전 라운드의 검증 forward 에 덧붙여 둔
        오디오 행 결과 (start, 행 수, sel, ids, k1) — 같은 입력·같은 위치라 새 배치 forward 대신 쓴다."""
        rounds, carry = [], None
        while self.k < upto and not self.kv_full:
            k0 = self.k
            t0 = time.perf_counter()
            if carry is not None and carry[0] == self.pos and carry[4] <= upto:
                start, nxs, sel, ids, k1 = carry
                self.stats["res_used"] += 1
            else:
                k1 = min(upto, (k0 // self.group + 1) * self.group) if self.speculative else k0 + 1
                xs, sel = self._batch(k0, k1)
                if k1 - k0 > 1 and self.pos + len(xs) + 1 > self.kv_capacity:  # 배치가 안 들어가면 순차 규칙으로
                    k1 = k0 + 1
                    xs, sel = self._batch(k0, k1)
                if k1 - k0 == 1 and self.pos + 2 > self.kv_capacity:           # KV 하드 클램프
                    rounds.append(self._kv_round(k0))
                    break
                start = self.pos
                ids = self.th.forward(mx.stack(xs), start, sel)
                nxs = len(xs)
                self.stats["forwards"] += 1
            carry = None
            acc = next((j for j, t in enumerate(ids) if not self._stop(t, 0)), None)
            n_dec = len(ids) if acc is None else acc + 1
            share = (time.perf_counter() - t0) * 1000 / n_dec
            for j in range(n_dec if acc is None else acc):                 # 무방출 라운드들 (전체 cap 이면 forced)
                f = ids[j] != self.sp.next
                self.forced += f
                rounds.append(self._round(k0 + j, [], f, share))
            if acc is None:
                self.pos = start + nxs
                self.pending = True
                self.k = k1
            else:
                self.pos = start + sel[acc] + 1                              # 되감기: 프레임 k0+acc 의 오디오 행까지 유효
                self.pending = False
                t1 = time.perf_counter()
                ka = k0 + acc + 1
                em, f, kvf, carry = self._finish_round(k0 + acc, ids[acc], ka, k1 if self.speculative else ka)
                rounds.append(self._round(k0 + acc, em, f, share + (time.perf_counter() - t1) * 1000, kv_full=kvf))
                self.k = ka
        return rounds

    def _drafts(self, tid: int, L: int) -> tuple[list[int], bool]:
        """초안 최대 L 개 → (초안, NEXT 로 끝났는가). 다음 초안 스텝을 먼저 넣고 직전 결과를 읽는다 (낭비는 최대 한 스텝).
        초안은 스트리밍 KV 의 pos.. 에 근사 행을 쓴다 — 검증 forward 가 덮어쓰거나, 덮이지 않은 행은 읽히지 않는다."""
        th, D = self.th, self.draft
        out, x, prev = [], th.embed([tid]), None
        for j in range(L + 1):
            cur = None
            if j < L:
                cur = D.ids_lazy(x, self.pos + j)
                mx.async_eval(cur)
                self.stats["draft_steps"] += 1
            if prev is not None:
                d = prev.item()
                if d == self.sp.next:
                    return out, True
                out.append(d)
            if cur is None:
                break
            x = th.embed_lazy(cur)
            prev = cur
        return out, False

    def _draft_guard(self, matched: int, drafted: int):
        if not drafted:
            return
        self.stats["drafted"] += drafted
        self.stats["accepted"] += matched
        self._acc_ema = 0.75 * self._acc_ema + 0.25 * matched / drafted
        self._acc_n += 1
        if self._acc_n >= DRAFT_GUARD[0] and self._acc_ema < DRAFT_GUARD[1]:
            self._draft_on = False                                          # 이 세그먼트는 q=1 로 (ids 는 같고 시간만)
            self.stats["draft_off"] = 1

    def _finish_round(self, k: int, tid: int, ka: int = 0, kb: int = 0):
        """이미 아는 첫 argmax 로 라운드를 진행 → (낸 id, forced, kv_full, carry). ka..kb-1 = 같은 그룹의 남은 프레임.
        각 토큰 뒤: 초안이 있으면 검증 forward [tid, d1..dm](+ 초안이 NEXT 로 끝났으면 남은 오디오 행), 없으면 q=1 forward."""
        ids, n, carry = [], 0, None
        while True:
            if self._stop(tid, n):
                f = tid != self.sp.next
                self.forced += f
                self.pending = True
                return ids, f, False, carry
            if self.pos + 2 > self.kv_capacity:                               # KV 하드 클램프: 강제 종료
                self.forced += 1
                self.pending = True
                self.kv_full = True
                return ids, True, True, None
            self._emit(k, tid, ids)
            n += 1
            carry = None
            drafts, res = [], False
            if self._draft_on:
                L = min(self.draft_len, self.cap - n, self.max_total - len(self.out), self.kv_capacity - self.pos - 2)
                if L > 0:
                    drafts, res = self._drafts(tid, L)
                    self.stats["verifies"] += 1
                else:
                    res = True                      # 남은 자리가 없으면 이 예측으로 라운드가 끝난다 → 오디오 행을 덧붙여 둔다
            rows = [tid] + drafts
            nr, r = len(rows), kb - ka
            audio = (res and r > 0 and nr + 2 * r <= MAX_VERIFY_ROWS and self.pos + nr + 2 * r + 1 <= self.kv_capacity)
            x, sel = self.th.embed(rows), list(range(nr))
            if audio:
                xa = []
                for kk in range(ka, kb):
                    xa += [self._eN, self._ce[kk]]
                x = mx.concatenate([x, mx.stack(xa)], axis=0)
                sel += [nr + 2 * j + 1 for j in range(r)]
                self.stats["res_tried"] += 1
            start = self.pos
            preds = self.th.forward(x, start, sel)
            self.stats["forwards"] += 1
            m = 0                                                             # 목표 argmax 와 일치한 초안 수
            while m < len(drafts) and preds[m] == drafts[m]:
                m += 1
            self._draft_guard(m, len(drafts))
            for i in range(m):                                                # 순차와 같은 순서로 검사한 뒤 방출 (L 을 cap·전체 cap·
                self.pos += 1                                                 # KV 여유로 잘랐으므로 아래 두 검사는 실제로 걸리지 않는다 — 방어용)
                d = drafts[i]
                if self._stop(d, n):
                    f = d != self.sp.next
                    self.forced += f
                    self.pending = True
                    return ids, f, False, None
                if self.pos + 2 > self.kv_capacity:
                    self.forced += 1
                    self.pending = True
                    self.kv_full = True
                    return ids, True, True, None
                self._emit(k, d, ids)
                n += 1
            tid = preds[m]
            self.pos += 1
            if audio and m == nr - 1:                                        # 마지막 행 뒤 예측: 라운드가 끝나면 오디오 행이 유효
                carry = (start + nr, 2 * r, [2 * j + 1 for j in range(r)], preds[nr:], kb)

    def _flush(self) -> list[Round]:
        rounds = []
        for r in range(self.max_flush):
            if self.kv_full:
                break
            k = int(self.K) + r
            if self.pos + 2 > self.kv_capacity:
                rounds.append(self._kv_round(k, flush=True))
                break
            t0 = time.perf_counter()
            xs = [self._eN, self._eE] if self.pending else [self._eE]
            tid = self.th.forward(mx.stack(xs), self.pos)[0]
            self.stats["forwards"] += 1
            self.pos += len(xs)
            self.pending = False
            em, f, kvf, _ = self._finish_round(k, tid)
            self.flush_rounds += 1
            rounds.append(self._round(k, em, f, (time.perf_counter() - t0) * 1000, kv_full=kvf, flush=True))
            if not em:
                break
        return rounds


class PrefixCache:
    """prefix KV 행 캐시 (Engine 하나 = thinker 하나에 하나): prefix id 열 → 층별 K/V[:P] 깊은 복사, LRU.
    세그먼트 시작마다 하던 q=P prefix forward(fp16 ≈ 22 ms, fp32 ≈ 30 ms)를 행 복원(≈ 1 ms)으로 바꾼다 — 같은 입력의 같은
    forward 출력이라 비트 동일. 항목 하나 ≈ P·층·2·KVH·HD·itemsize (fp16 P=20: 2.3 MB)."""

    def __init__(self, thinker, size: int = 8):
        self.thinker, self.size = thinker, int(size)
        self._d: OrderedDict = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key: tuple):
        v = self._d.get(key)
        if v is None:
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return v

    def put(self, key: tuple, rows) -> None:
        self._d[key] = rows
        self._d.move_to_end(key)
        while len(self._d) > self.size:
            self._d.popitem(last=False)

    def clear(self) -> None:
        self._d.clear()

    def __len__(self) -> int:
        return len(self._d)


def stream_draft(seg: "SegmentDecoder") -> list[int]:
    """final 재디코드 초안 = 스트리밍 출력: 첫 화자 태그 + 원문 run (+ 번역이면 <TRANS> + 번역 run) + <|im_end|>."""
    sp = seg.sp
    ids = [t for _, t in seg.out]
    spk = next((t for t in ids if t in (sp.spk_a, sp.spk_b)), sp.spk_a)
    src, tr, run = [], [], 0
    for t in ids:
        if t in (sp.spk_a, sp.spk_b):
            run = 0
        elif t == sp.trans:
            run = 1
        else:
            (src if run == 0 else tr).append(t)
    d = [spk] + src
    if seg.mode and seg.mode[1]:
        d += [sp.trans] + tr
    return d + [sp.im_end]


class FinalJob:
    """final(오프라인) 모드 재디코드를 forward 한 번씩 진행한다 (decode-contract §f, audio-ui-spec §5.2·§7).

    step() 한 번 = 헤더 prefill 또는 토큰 forward 한 번 → done 여부. 서버는 스트리밍 일이 없을 때만 step() 을 부르고
    사이사이 입력 큐를 본다. 별도 KV(첫 step 에서 할당, 끝나면 해제)라 스트리밍 KV·다른 세그먼트와 섞여도 결과가 같다.
    lookahead > 0: prefill 뒤 한 걸음 = 스트리밍 출력(stream_draft)에서 prompt lookup 으로 고른 초안 ≤ lookahead 개를 한 forward
    [마지막 id, d1..dm] 로 검증 (≤ 16 행). <|im_end|>·상한 검사를 토큰마다 같은 순서로 하므로 id 는 q=1 걸음과 같다.
    워커 스레드에서만. 결과: ids (<|im_end|> 제외), ms (step 벽시계 합), steps."""

    def __init__(self, seg: "SegmentDecoder", max_new: int | None = None, lookahead: int = 0):
        self._seg = seg
        self.limit = int(seg.K) + 64 if max_new is None else int(max_new)      # ast-v1: K+64 (EV:501)
        self.ids: list[int] = []
        self.ms = 0.0
        self.steps = 0
        self.done = False
        self._kv = None
        self._pos = self._cap = 0
        self.lookahead = max(0, min(15, int(lookahead)))
        self.draft = stream_draft(seg) if self.lookahead else []
        self._dp = 0                                           # 초안 안의 직전 정렬 위치
        self.verifies = self.drafted = self.accepted = 0

    def _propose(self, room: int) -> list[int]:
        """생성 꼬리 n-gram (3→1) 이 초안에서 맞는 곳 중 직전 정렬 위치에 가장 가까운 곳의 다음 토큰들."""
        D, g = self.draft, self.ids
        for n in range(min(3, len(g)), 0, -1):
            tail = g[-n:]
            hits = [i + n for i in range(len(D) - n + 1) if D[i:i + n] == tail]
            if hits:
                e = min(hits, key=lambda x: (abs(x - self._dp), x))
                self._dp = e
                return D[e: e + min(self.lookahead, room)]
        return []

    def step(self) -> bool:
        if self.done:
            return True
        t0 = time.perf_counter()
        s, th = self._seg, self._seg.th
        if self._kv is None:                                   # 첫 step: 헤더 + 오디오 행 prefill (마지막 행 logits 를 쓴다)
            K = int(s.K)
            ids_h, a0 = s.tokens.offline_header(s.mode[0], s.mode[1], K)
            self._kv = th.new_kv(len(ids_h) + max(self.limit, 0) + 1)
            self._cap = int(self._kv[0][0].shape[2])
            if len(ids_h) > self._cap or self.limit <= 0:
                return self._end(t0)
            e = th.embed(ids_h)
            if K:
                e = mx.concatenate([e[:a0], s.ad(s.features()).astype(th.in_dtype), e[a0 + K:]], axis=0)
            preds, rows = th.forward(e, 0, block=s.offline_block, kv=self._kv), []
            self._pos = len(ids_h) - 1                         # 아래 루프가 행마다 +1
        else:
            room = min(self.limit - len(self.ids), self._cap - self._pos - 1)
            rows = [self.ids[-1]] + (self._propose(room - 1) if self.lookahead else [])
            preds = th.forward(th.embed(rows), self._pos, list(range(len(rows))), block=s.offline_block, kv=self._kv)
            self.verifies += len(rows) > 1
            self.drafted += len(rows) - 1
        self.steps += 1
        i = 0
        while True:                                            # preds[i] = 행 i 다음 토큰; 초안 행 i+1 과 같으면 이어서
            tid = preds[i]
            self._pos += 1
            if tid == s.sp.im_end:
                return self._end(t0)
            self.ids.append(int(tid))
            if len(self.ids) >= self.limit or self._pos + 1 > self._cap:
                return self._end(t0)
            if i + 1 < len(rows) and tid == rows[i + 1]:
                self.accepted += 1
                i += 1
                continue
            break
        self.ms += (time.perf_counter() - t0) * 1000
        return False

    def _end(self, t0: float) -> bool:
        self.ms += (time.perf_counter() - t0) * 1000
        self.done, self._kv = True, None
        return True

    def run(self) -> tuple[list[int], float]:
        while not self.step():
            pass
        return list(self.ids), self.ms
