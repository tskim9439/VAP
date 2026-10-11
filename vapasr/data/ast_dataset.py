"""ast-v1 출력 형식 데이터셋 — Stage 1 스트리밍 ASR(SemCommitDataset, ASR 전용) 위에 화자·언어 prefix 와 번역 흐름을 더한다
(wiki decision-ast-monotonic-data 결정 9, 2026-10-09 사용자 확정).

prefix(assistant 머리 뒤):  language <SPK_x>{원문 언어}[\\ntranslate <SPK_x>{목표 언어}]<asr_text><DELAY_d>
  - 화자 태그 x 는 항목마다 A/B 무작위(위치·언어 편향 방지). 지금 데이터는 한 화자 스트림이라 쌍이 하나다. ASR 만이면 translate 줄이 없다.
출력(청크 k = ⌊t/80 ms⌋ + δ, Stage 1 과 같은 버킷 규칙 — semcommit_dataset.build_interleaved_events):
  - 전사 토큰 = 원문 단어 끝 시각(words.jsonl tokens, 정규형). 번역 토큰 = stable prefix 확정 시각(experiments/ast_build_targets.py, 표기형).
  - 청크 안 정렬은 (시각, 흐름) — 같은 시각이면 전사가 먼저다. 구간이 바뀔 때만 태그를 넣는다: 전사 구간 <SPK_x>, 번역 구간 <TRANS>.
    첫 방출 앞에도 <SPK_x>. 예(δ4): <SPK_A> it <TRANS> Es <NEXT_AUDIO> … <SPK_A> values <TRANS> gemeinsame Werte <NEXT_AUDIO>
  - 번역 확정 시각은 그 행의 원문 단어 끝 시각 중 하나라 K(끝 패딩)는 전사 그대로다. flush(태그 없는 스트림 끝 방출)로 밀리면 target_problems 가 잡는다.
final(오프라인) 모드: 머리(language … <asr_text>) 뒤에 <SPK_x> 전사 [<TRANS> 번역] <|im_end|>.
<TRANS> 는 sp_ids 밖에서 trans_id 로 받는다 — SemCommitDataset 의 check_sem_ids 는 SEM 뒤 id 를 지연 확장만 허용한다.
번역 타깃(ast_targets): {행 id: {목표 언어: [(token id, 시각 s), ...]}} — load_ast_targets 가 ast_build_targets 산출물을 읽는다.
음량 증강(gain_aug, 2026-10-11 사용자 결정): FLEURS 영어 test 는 원본이 아주 작아(RMS 약 −60 dBFS) 스트리밍 모델이 통째로 침묵했다
(raw/sources/experiments/2026-10-11-q06-s2-ast-fleurs-ceiling). 학습 항목의 언어가 gain_aug_langs 면 확률 p 로 파형 RMS 를 [lo, hi] dBFS
의 무작위 목표로 맞춘다(잡음·잔향 뒤, 이득만 — 시각·K 그대로).
"""
import glob
import gzip
import json
import math
import random
from array import array
from collections import Counter
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .interleave import Specials
from .semcommit_dataset import SemCommitDataset, build_interleaved_events, last_event_chunk, SR
from .output_format import NORMALIZED
from ..uslm.interleave_data import CHUNK_S
from ..uslm.mono_data import build_mono_sequence

SPK_NAMES = ("<SPK_A>", "<SPK_B>")


def ast_head_text(src_lang: str, spk: str, tgt_lang: Optional[str] = None) -> str:
    """'language <SPK_x>English[\\ntranslate <SPK_x>German]<asr_text>' — Qwen3-ASR 원 형식의 language 줄에 화자 태그를 붙인 것."""
    return f"language {spk}{src_lang}" + (f"\ntranslate {spk}{tgt_lang}" if tgt_lang else "") + "<asr_text>"


def build_ast_sequence(ev_src: list, ev_tr: list, K: int, prefix: List[int], audio_pad: int, sp: Specials, spk_id: int, trans_id: int,
                       delay: int) -> dict:
    """전사 이벤트열(ev_src, (tid, t)) + 번역 이벤트열(ev_tr, 없으면 []) → 스트리밍 시퀀스. 오디오와 무관한 순수 함수.
    태그 규칙은 build_interleaved_events 의 흐름 태그 그대로다(흐름 0 = 전사 → spk_id, 흐름 1 = 번역 → trans_id, 바뀔 때만).
    → dict(ids, is_input, chunk_of, labels, pos_weight, n_rounds, n_flush, st, chunks, srcs, n_tr, n_tags, tr_in_flush)."""
    sp2 = Specials(next_audio=sp.next_audio, empty_audio=sp.empty_audio, spk=(int(spk_id), int(trans_id)))
    streams = [list(ev_src)] + ([list(ev_tr)] if ev_tr else [])
    chunks, st, srcs = build_interleaved_events(streams, (K - 0.5) * CHUNK_S, sp2, chunk_s=CHUNK_S, delay_frames=delay, max_per_chunk=0,
                                                add_spk_tags=True, with_src=True)
    ids, is_input, chunk_of, n_rounds, n_flush = build_mono_sequence(chunks, K, prefix, audio_pad, sp2, 0)
    labels = [(-100 if inp else t) for t, inp in zip(ids, is_input)]
    tr_in_flush = sum(1 for (k, _), sr in zip(chunks, srcs) if k >= K for q in sr if q is not None and q[0] == 1)
    n_tags = sum(1 for (k, em), sr in zip(chunks, srcs) for t, q in zip(em, sr) if q is None and t in (spk_id, trans_id))
    return dict(ids=ids, is_input=is_input, chunk_of=chunk_of, labels=labels, pos_weight=[0.0] * len(ids), n_rounds=n_rounds, n_flush=n_flush,
                st=st, chunks=chunks, srcs=srcs, n_tr=len(ev_tr or ()), n_tags=n_tags, tr_in_flush=tr_in_flush,
                n_sem=0, n_turn=0, n_B=0, n_N=0, turn_chunk=None, decision=[], sem_pos=[], turn_pos=[], n_decision_on_event=0,
                sem_in_flush=0, turn_in_flush=0)                                         # semcommit 시퀀스와 같은 키(학습 스크립트 dry-run·통계)


def load_ast_targets(paths: Sequence[str], keep_ids: Optional[set] = None) -> Tuple[Dict[str, Dict[str, tuple]], Counter, str]:
    """ast_build_targets 산출물 → ({행 id: {목표 언어: (ids, 시각)}}, 통계, 지문). 폴더에 pack/ 이 있으면 묶은 형태(npy, mmap)를 읽고
    (ids·시각 = 공유 배열의 조각), 없으면 targets-*/<direction>/u-*.jsonl.gz(또는 그 파일 목록)를 파싱한다(array). 지문(학습 지문에 넣는다)은
    묶은 형태면 방향별 행·토큰 수·묶은 시각, 아니면 파일 경로·크기 목록의 sha256. keep_ids 가 있으면 그 행만 읽는다. 같은 (행, 목표 언어)가 두 번 나오면 처음 것."""
    import hashlib, os
    import numpy as np
    out: Dict[str, Dict[str, tuple]] = {}; st = Counter()
    packed = [p for p in paths if os.path.isdir(os.path.join(p, "pack"))]
    if packed:                                                                       # 묶은 형태(ast_build_targets --pack): mmap 으로 노드 안 프로세스가 한 벌을 공유
        metas = []
        for p in packed:
            for d in sorted(glob.glob(os.path.join(p, "pack", "*", "meta.json"))):
                pd = os.path.dirname(d); meta = json.load(open(d)); metas.append(meta)
                ids = np.load(os.path.join(pd, "ids.npy"), mmap_mode="r"); ts = np.load(os.path.join(pd, "times.npy"), mmap_mode="r")
                off = np.load(os.path.join(pd, "offsets.npy")); rows = open(os.path.join(pd, "rows.txt"), encoding="utf-8").read().split("\n")[:-1]
                assert len(rows) + 1 == len(off) == meta["rows"] + 1, f"pack 불일치: {pd}"
                st["files"] += 1
                for j, rid in enumerate(rows):
                    st["rows"] += 1
                    if keep_ids is not None and rid not in keep_ids: st["not_in_dataset"] += 1; continue
                    dd = out.setdefault(rid, {})
                    if meta["tgt"] in dd: st["dup"] += 1; continue
                    a0, a1 = int(off[j]), int(off[j + 1]); dd[meta["tgt"]] = (ids[a0:a1], ts[a0:a1]); st[f"tgt_{meta['tgt']}"] += 1
        fp = hashlib.sha256(json.dumps(sorted((m["direction"], m["rows"], m["tokens"], m["packed"]) for m in metas)).encode()).hexdigest()
        paths = [p for p in paths if p not in packed]
        if not paths: return out, st, fp
    files: List[str] = []
    for p in paths:
        files += sorted(glob.glob(f"{p}/**/u-*.jsonl.gz", recursive=True)) if not p.endswith(".gz") else [p]
    fp = hashlib.sha256(json.dumps([(f, os.path.getsize(f)) for f in files]).encode()).hexdigest()
    st["files"] += len(files)
    for f in files:
        for line in gzip.open(f, "rt", encoding="utf-8"):
            r = json.loads(line); st["rows"] += 1
            if keep_ids is not None and r["id"] not in keep_ids: st["not_in_dataset"] += 1; continue
            d = out.setdefault(r["id"], {})
            if r["tgt"] in d: st["dup"] += 1; continue
            d[r["tgt"]] = (array("q", (int(t[0]) for t in r["tokens"])), array("d", (float(t[1]) for t in r["tokens"])))
            st[f"tgt_{r['tgt']}"] += 1
    return out, st, fp


def apply_gain(wav: np.ndarray, target_rms_db: float, peak_cap: float = 0.99) -> Tuple[np.ndarray, float]:
    """파형 RMS 를 target_rms_db(dBFS)로 맞춘다(이득만). 봉우리가 peak_cap 을 넘으면 그만큼 이득을 줄인다. 무음이면 그대로. → (파형, 이득 dB)"""
    x = np.asarray(wav, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    if rms < 1e-8: return x, 0.0
    g = 10.0 ** (target_rms_db / 20.0) / rms; peak = float(np.abs(x).max())
    if peak * g > peak_cap: g = peak_cap / peak
    return (x * g).astype(np.float32), 20.0 * math.log10(g)


class AstStreamDataset(SemCommitDataset):
    """SemCommitDataset(ASR 전용: ignore_labels + sem_free_unlabeled, 표기형 끔) + ast-v1 prefix·번역 흐름.
    ast_targets: load_ast_targets 결과(없으면 화자·언어 prefix 만 붙인 ASR). ast_frac: 번역 타깃이 있는 항목을 이 확률로 번역 모드로 낸다
    (목표 언어가 여럿이면 균등하게 하나). trans_id: <TRANS> id. 화자 태그는 sp_ids 의 <SPK_A>/<SPK_B>.
    gain_aug: (p, lo_db, hi_db) — online 학습에서 gain_aug_langs 항목을 확률 p 로 RMS [lo, hi] dBFS 에 맞춘다(apply_gain). None 이면 끔(난수를 더 뽑지 않는다)."""

    def __init__(self, *args, ast_targets: Optional[Dict[str, Dict[str, tuple]]] = None, ast_frac: float = 0.5, trans_id: Optional[int] = None,
                 gain_aug: Optional[Sequence[float]] = None, gain_aug_langs: Sequence[str] = ("English",), **kw):
        if float(kw.get("written_frac", 0) or 0) > 0: raise ValueError("ast-v1 은 표기형(written_frac)과 함께 쓰지 않는다 — 전사는 정규형")
        if not (kw.get("ignore_labels") and kw.get("sem_free_unlabeled")): raise ValueError("ast-v1 은 ASR 전용(ignore_labels + sem_free_unlabeled)")
        if trans_id is None: raise ValueError("trans_id(<TRANS>) 가 필요하다")
        if not 0.0 <= float(ast_frac) <= 1.0: raise ValueError(f"ast_frac {ast_frac} 는 [0, 1] 범위")
        if gain_aug is not None:
            gain_aug = tuple(float(x) for x in gain_aug)
            if len(gain_aug) != 3 or not 0.0 <= gain_aug[0] <= 1.0 or not gain_aug[1] <= gain_aug[2] <= 0.0:
                raise ValueError(f"gain_aug (p, lo_db, hi_db): 0 ≤ p ≤ 1, lo ≤ hi ≤ 0 dBFS — {gain_aug}")
        super().__init__(*args, **kw)
        self.trans_id = int(trans_id); self.ast_frac = float(ast_frac)
        self.gain_aug = gain_aug; self.gain_aug_langs = frozenset(gain_aug_langs)
        self.spk_ids = tuple(int(self.sp_ids[n]) for n in SPK_NAMES)
        targets = ast_targets or {}; n_ast = Counter()
        for it in self.items:
            tg = targets.get(it["id"])
            t_end = float(it["tokens"][-1][1]) if it["tokens"] else 0.0                     # 이벤트열은 시간순(check_words)
            extra = 4                                                                               # 화자 태그(prefix + 첫 방출) — ASR 항목
            if tg:
                it["ast"] = {}
                for lang, (ids, ts) in sorted(tg.items()):
                    if len(ts) and float(ts[-1]) > t_end + 1e-6:                                # 시각은 오름차순 — 마지막만 보면 된다
                        n_ast["clamped_tokens"] += sum(1 for t in ts if t > t_end + 1e-6)
                        ts = array("d", (min(float(t), t_end) for t in ts))                       # 확정 시각은 원문 단어 끝 이내여야 한다
                    it["ast"][lang] = (ids, ts); n_ast[f"items_{lang}"] += 1; n_ast["tr_tokens"] += len(ids)
                extra = 16 + 3 * max(len(v[0]) for v in it["ast"].values())                         # translate 줄 + 번역 토큰 + 구간 태그(보수적)
            it["est_extra"] = int(it.get("est_extra", 0)) + extra
        self.stats.update({f"ast_{k}": v for k, v in n_ast.items()}); self.stats["ast_items"] = sum(1 for it in self.items if "ast" in it)

    # ── 시퀀스
    def head_ids(self, lang: str, spk: int, tgt: Optional[str], delay: Optional[int]) -> List[int]:
        """스트리밍 prefix(delay 있음) 또는 final 머리 뒤 언어 줄(delay=None)."""
        c = self.__dict__.setdefault("_ast_head_cache", {}); key = (lang, spk, tgt, delay)
        if key not in c:
            ids = self.tok(ast_head_text(lang, SPK_NAMES[spk], tgt), add_special_tokens=False)["input_ids"]
            c[key] = (self._pre + ids + [self.sp_ids[f"<DELAY_{delay}>"]]) if delay is not None else ids
        return list(c[key])

    def _events(self, it: dict, tgt: Optional[str], f: float = 1.0) -> Tuple[list, list]:
        ev = [(x[0], x[1] / f) for x in it["tokens"]]
        if not tgt: return ev, []
        ids, ts = it["ast"][tgt]
        return ev, [(int(i), float(t) / f) for i, t in zip(ids, ts)]

    def ast_sequence(self, i: int, delay: int, spk: int = 0, tgt: Optional[str] = None, f: float = 1.0) -> Tuple[dict, int, float]:
        """(시퀀스, K, 오디오 길이 s). f ≠ 1 은 속도 변형(perturbed 와 같은 K 규칙: 전사·번역 시각 t → t/f)."""
        it = self.items[i]; ev, tr = self._events(it, tgt, f)
        if f == 1.0: K, dur = it["K"], it["duration_s"]
        else:
            K0 = max(1, int(round(it["duration0"] / f / CHUNK_S)))
            need = max(last_event_chunk(ev, d) for d in self.delays) + self.tail_margin if self.pad_tail else 0
            K = max(K0, need); dur = it["duration0"] / f if K == K0 else K * CHUNK_S
        s = build_ast_sequence(ev, tr, K, self.head_ids(it["lang"], spk, tgt, delay), self.audio_pad, self.sp, self.spk_ids[spk], self.trans_id, delay)
        return s, K, dur

    def ast_offline_sequence(self, i: int, K: int, spk: int = 0, tgt: Optional[str] = None) -> dict:
        """final 모드: 오디오 K 청크 → <SPK_x> 전사 [<TRANS> 번역] <|im_end|> (머리에 화자·언어 줄)."""
        if "_offline" not in self.__dict__:                                                     # offline_frac = 0 이면 기반 클래스가 만들지 않는다(검사·평가용)
            from .offline_seq import OfflineFormat
            self._offline = OfflineFormat(self.tok)
        o = self._offline; it = self.items[i]; ev, tr = self._events(it, tgt)
        head = o.pre + [o.audio_start] + [o.audio_pad] * int(K) + [o.audio_end] + o.mid + self.head_ids(it["lang"], spk, tgt, None)
        body = [self.spk_ids[spk]] + [x[0] for x in ev] + ([self.trans_id] + [x[0] for x in tr] if tr else [])
        ids = head + body + [o.im_end]; a0 = len(o.pre) + 1
        is_input = [True] * len(head) + [False] * (len(ids) - len(head)); chunk_of = [-1] * len(ids)
        for k in range(int(K)): chunk_of[a0 + k] = k
        return dict(ids=ids, is_input=is_input, chunk_of=chunk_of, labels=[(-100 if inp else t) for t, inp in zip(ids, is_input)],
                    pos_weight=[0.0] * len(ids), st=SimpleNamespace(tokens=len(ev) + len(tr), overflow_tokens=0), n_flush=0, n_rounds=0,
                    n_tr=len(tr), n_tags=1 + bool(tr), tr_in_flush=0)

    def prefix(self, lang: str, delay: int, fmt: str = NORMALIZED) -> List[int]:
        """검사·dry-run 용 기본 prefix: 화자 A, 번역 없음."""
        return self.head_ids(lang, 0, None, delay)

    def sequence(self, i: int, delay: int, fmt: str = NORMALIZED) -> dict:
        """검사용 기본 형태: 화자 A, 번역 없음(ASR)."""
        return self.ast_sequence(i, delay, 0, None)[0]

    def target_problems(self, i: int, delay: int, fmt: str = NORMALIZED) -> List[str]:
        """ASR(화자 A)과 번역 타깃마다(화자 B) 시퀀스를 만들어 검사: 전사·번역 토큰 순서가 이벤트열과 같은가, 구간 태그가 규칙대로인가
        (첫 방출과 흐름이 바뀔 때만, 전사 앞 <SPK_x>, 번역 앞 <TRANS>), 번역이 flush 로 밀리지 않았는가, 이벤트 토큰(SEM/턴)이 없는가."""
        it = self.items[i]; P = []
        if any(x[0] in (self.sem_id, self.turn_id) or len(x) > 2 for x in it["tokens"]): P.append("events_in_asr_item")
        for spk, tgt in [(0, None)] + [(1, t) for t in sorted(it.get("ast", {}))]:
            s, K, _ = self.ast_sequence(i, delay, spk, tgt); name = f"{SPK_NAMES[spk]}/{tgt or 'asr'}"
            ev, tr = self._events(it, tgt)
            # 출력 토큰열(라벨 위치, NEXT 제외)을 구간 규칙으로 다시 읽는다: 태그가 흐름을 정하고, 흐름마다 토큰 순서가 이벤트열과 같아야 한다
            src, trn, run, bad = [], [], None, 0
            for t, inp in zip(s["ids"], s["is_input"]):
                if inp or t == self.sp.next_audio: continue
                if t == self.spk_ids[spk]: bad += run == 0; run = 0; continue          # 같은 구간 태그가 연달아 나오면 위반
                if t == self.trans_id: bad += run == 1; run = 1; continue
                if t in self.spk_ids or run is None: bad += 1; continue                 # 다른 화자 태그 · 태그 없이 시작
                (src if run == 0 else trn).append(t)
            if src != [x[0] for x in ev]: P.append(f"src_order:{name}")
            if trn != [x[0] for x in tr]: P.append(f"tr_order:{name}")
            if bad: P.append(f"tag:{name}:{bad}")
            if s["tr_in_flush"]: P.append(f"tr_in_flush:{name}:{s['tr_in_flush']}")
        return P

    # ── 항목
    def __getitem__(self, i):
        delay = random.choice(self.delays); err = None
        for r in range(self.max_audio_retries):
            j = (i + r) % len(self.items); it = self.items[j]
            try: wav = self.audio(it); break
            except (ValueError, OSError, RuntimeError) as e:
                err = e; self._audio_fail = getattr(self, "_audio_fail", 0) + 1
                if self._audio_fail <= 5: print(f"  ! 항목 {it['id']} 오디오 조립 실패({type(e).__name__}: {str(e)[:120]}) → 이웃 항목 (누적 {self._audio_fail})", flush=True)
        else: raise RuntimeError(f"오디오 조립 {self.max_audio_retries} 회 연속 실패: {err}")
        spk = random.randrange(2)
        tgt = random.choice(sorted(it["ast"])) if "ast" in it and random.random() < self.ast_frac else None
        f = random.choice(self.speed_perturb) if self.speed_perturb else 1.0
        s, K, dur = self.ast_sequence(j, delay, spk, tgt, f)
        if f != 1.0: wav = self.speed_wav(wav, f, int(round(dur * SR)), it["id"])
        offline = self.offline_frac > 0 and random.random() < self.offline_frac
        if offline: s = self.ast_offline_sequence(j, K, spk, tgt)
        if self.noise_aug is not None and self.online: wav = self.noise_aug(wav, random)
        if self.gain_aug is not None and self.online and it["lang"] in self.gain_aug_langs and random.random() < self.gain_aug[0]:
            wav, _ = apply_gain(wav, random.uniform(self.gain_aug[1], self.gain_aug[2]))       # 잡음·잔향 뒤 전체 음량(SNR 그대로)
        out = dict(wav=torch.from_numpy(wav), ids=torch.tensor(s["ids"]), is_audio=torch.tensor([c >= 0 for c in s["chunk_of"]]), chunk_of=torch.tensor(s["chunk_of"]),
                   labels=torch.tensor(s["labels"]), pos_weight=torch.tensor(s["pos_weight"], dtype=torch.float32), lang=it["lang"], delay=delay, name=it["name"], id=it["id"],
                   n_text=s["st"].tokens, overflow=s["st"].overflow_tokens, n_flush=s["n_flush"], K=K, rounds=s["n_rounds"],
                   n_sem=0, n_turn=0, n_B=0, n_N=0, n_decision_on_event=0, sem_in_flush=0, turn_in_flush=0,
                   sem_free=True, ast=int(tgt is not None), tgt=tgt or "", spk=spk, n_tr=s["n_tr"])
        if self.speed_perturb: out["speed"] = f
        if self.offline_frac > 0: out["mode"] = "offline" if offline else "stream"
        return out
