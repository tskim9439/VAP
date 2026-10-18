"""Semantic commit 학습 데이터셋 — Phase 1 mono 스트림에 <SEM_END>(와 선택적으로 턴 종료 <EOT>)를 끼우고 B/N 결정 위치를 표시한다
(raw/inbox/streaming_asr_semantic_commit_plan.md §21 시퀀스 · §23 등급).

입력(모듈 간 공용 형식, 스트림당 한 줄):
  words.jsonl  {id, set, lang, K, duration_s, text, tokens:[[tid, end_s], ...] (스트림 절대 시각·시간순 = aligned manifest 토큰 그대로),
                segments:[{path, offset_s, silence_before_s, dur_s, utt_id, raw_text}], words:[{i, text, a, b, end_time, seg, tags}]}  tokens[a:b] = 단어의 BPE 토큰
  labels.jsonl {id, lang, candidates:[{after_word, grade 'A'|'B'|'N', why, ...}], turn_end}
시퀀스 = Phase 1 mono(vapasr/uslm/mono_data.build_mono_sequence, 같은 prefix·<DELAY_d>) + 이벤트:
  <SEM_END>  A 후보 단어 i 의 마지막 토큰 바로 뒤, 같은 청크. 시각 = 그 토큰 시각 → 같은 버킷, 청크 안 (t, spk) 안정 정렬이 목록 순서를 지킨다(text < SEM < TURN).
  턴 종료  스트림 끝 1 회(발화 끝 proxy, 선택 — 기본 끔: Phase 1 은 <SEM_END> 만). 토큰은 Phase 2 와 같은 <EOT>(v0.2 의 <TURN_END> 는 폐기).
             명시 청크 이벤트: k_turn = max(마지막 단어/SEM 방출 청크, int((t_last + hangover_s)/0.08)).
             스트림을 무음으로 늘려 K' = max(K, k_turn(δmax) + tail_margin) → TURN 은 실제(무음) 오디오 위에서 <NEXT_AUDIO> 와 경쟁하고 <EMPTY_AUDIO> flush 에는 절대 안 간다.
  B 후보 → SEM 없음 + 결정 위치 라벨 -100.  N 후보 → SEM 없음 + 결정 위치 pos_weight = hardneg_weight(기본 1.0; NEXT 의 0.3/0.15 대신).
  결정 위치 = 단어 마지막 토큰 **다음 원소**의 시퀀스 index. 모델은 h[pos-1](= 그 토큰)로 labels[pos] 를 예측하므로 '단어를 낸 직후의 결정' 이다.
  결정 위치가 이벤트 타깃이면(마지막 단어가 B/N 이고 TURN 이 같은 청크 바로 뒤 — δ=6 이면 거의 항상) 가리거나 덮어쓰지 않는다: TURN 라벨·turn_weight 를 그대로 두고
  decision 에 (p, 'B@TURN'|'N@TURN') 으로 남기고 n_decision_on_event 로 센다(plan Rule 14 '그게 그러니까... <TURN_END>' — 불완전한 끝에서도 TURN 을 배운다;
  모델 forward 는 pos_weight 가 이벤트 가중을 이기므로 이 충돌을 데이터셋 책임으로 둔다).
  비후보 단어 끝은 암묵적 음성(기본 가중).
M(청크당 텍스트 한도) > 0 은 v0 계약 밖: SEM 도 한도에 세어져 단어와 다른 청크로 밀릴 수 있다(target_problems 가 'sem_detached' 로 잡는다; 학습 스크립트는 --M 0 만 받는다).
build_interleaved(vapasr/data/interleave.py) 는 고치지 않는다. build_interleaved_events 가 같은 버킷팅(k = min(n, int(t/0.08)+δ), 청크 안 (t, spk) 안정 정렬,
M 이월, flush)을 재현하면서 명시 청크 이벤트와 방출 출처(src)를 더한다 — 이벤트가 없으면 build_interleaved 와 출력·통계가 같다(tests/test_semcommit_dataset.py 가 고정).
출력 형식(outfmt-v1, vapasr/data/output_format.py): 기본 = 정규형(빈 system 칸). written_frac > 0 이면 표기형이 있는 항목(semcommit_written.written_row —
  KO speechlm display_text · EN pnc_text 정렬)을 그 확률로 표기형(system 칸 = 고정 지시문, 구두점·대소문자·숫자 타깃)으로 낸다. <SEM_END>·B/N 규칙은 같다
  (단어 마지막 토큰 = 끝 구두점 뒤). 후보 단어 끝 시각·마지막 방출 청크가 정규형과 다른 표기형은 버리므로(written_cand_time_shift · written_tail_shift)
  두 형식의 SEM·결정 청크가 같고 K 는 정규형 그대로다. written_frac = 0 이면 항목·시퀀스·난수 소비·반환 키가 이 기능 전과 바이트 동일
  (tests/test_semcommit_format.py GOLDEN)."""
import hashlib, json, math, os, random
from array import array
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np, torch
from torch.utils.data import Dataset
from .interleave import InterleaveStats, Specials
from .semcommit_tokens import SEM_SPECIALS, BASE_VOCAB, TURN_TOKEN, add_semantic_specials, assert_frozen_semantic
from ..uslm.interleave_data import specials_of, CHUNK_S
from ..uslm.mono_data import build_mono_sequence, collate_streams
from .semcommit_tn import normalize_ko_row
from .semcommit_written import written_row, remap_labels
from .output_format import header_ids, NORMALIZED, WRITTEN

SR = 16000

def check_sem_ids(ids: Dict[str, int]) -> None:
    """SEM 이 Phase 2 registry 를 밀어내지 않았는지(가짜 tokenizer 에도 적용되는 구조 검사): SEM 이 모든 Phase 1/2 id 보다 뒤, 턴 종료 <EOT> 는 Phase 2 registry 안.
    실제 Qwen tokenizer(<NEXT_AUDIO> = BASE_VOCAB)면 semcommit_tokens.assert_frozen_semantic 로 동결 id(<EOT> 151722, <SEM_END> 151723 등)까지 대조한다."""
    from .semcommit_tokens import DELAY_EXT                                            # 지연 확장 <DELAY_9/10> 은 SEM 뒤(semcommit_tokens.add_delay_tokens)
    s = ids["<SEM_END>"]
    assert s > max(v for k, v in ids.items() if k not in SEM_SPECIALS and k not in DELAY_EXT) and TURN_TOKEN in ids and ids[TURN_TOKEN] < s, f"SEM/턴 종료 id 순서 위반: {ids}"
    assert all(ids[k] > s for k in DELAY_EXT if k in ids), f"지연 확장 id 는 SEM 뒤여야 한다: {ids}"
    if ids.get("<NEXT_AUDIO>") == BASE_VOCAB: assert_frozen_semantic(ids)

def add_semcommit_specials(tok) -> Dict[str, int]:
    """semcommit_tokens.add_semantic_specials(Phase 1 + Phase 2 + SEM, 이 순서) + check_sem_ids → {name: id}."""
    ids = add_semantic_specials(tok); check_sem_ids(ids); return ids

# ── 1. 버킷팅 래퍼 ────────────────────────────────────────────────────────────────────────────────────────────
def build_interleaved_events(streams: List[list], duration_s: float, sp: Specials, chunk_s: float = CHUNK_S, delay_frames: int = 2,
                             max_per_chunk: int = 0, add_spk_tags: bool = True, with_src: bool = False):
    """build_interleaved 와 같은 규약 + 명시 청크 이벤트.
    streams[s] 원소: (tid, t_end) = 시각 버킷 토큰(텍스트·SEM), (tid, t, k_min) = 명시 청크 이벤트(TURN) — 그 스트림의 앞선 원소가 모두 방출된 뒤,
    k_min 이상인 첫 청크에서 앞선 원소 바로 뒤(같은 스트림의 뒤 원소보다 앞)에 낸다. 청크당 한도 M 에는 세지 않는다. 끝까지 못 나가면 flush(EMPTY 앞).
    → (chunks, stats) 또는 with_src 면 (chunks, stats, srcs): srcs[c][e] = chunks[c][1][e] 의 출처 (stream, 목록 index), 특수 토큰은 None."""
    n_chunks = int(math.ceil(duration_s / chunk_s)); buckets: List[list] = [[] for _ in range(n_chunks + 1)]
    explicit: List[List[Tuple[int, int, int]]] = []
    for s, toks in enumerate(streams):
        ex = []
        for j, x in enumerate(toks):
            if len(x) > 2 and x[2] is not None: ex.append((j, int(x[2]), int(x[0]))); continue
            tid, t_end = x[0], x[1]
            k = min(n_chunks, int(t_end / chunk_s) + delay_frames); buckets[k].append((t_end, s, tid, j))
        explicit.append(ex)
    done = [[False] * len(t) for t in streams]; first = [0] * len(streams)            # first[s] = 아직 안 나간 첫 원소 index
    def mark(s, j):
        done[s][j] = True
        while first[s] < len(done[s]) and done[s][first[s]]: first[s] += 1
    def inject(emit, src, k, st, last_spk):
        """준비된 명시 이벤트를 emit 안 제자리에 끼운다 → (삽입 수, last_spk)."""
        n = 0; changed = True
        while changed:
            changed = False
            for s, ex in enumerate(explicit):
                while ex and (k is None or ex[0][1] <= k) and first[s] >= ex[0][0]:
                    j, _, tid = ex.pop(0)
                    before = [p for p, q in enumerate(src) if q is not None and q[0] == s and q[1] < j]
                    after = [p for p, q in enumerate(src) if q is not None and q[0] == s and q[1] > j]
                    if before: p = before[-1] + 1
                    elif after: p = after[0]                                          # 뒤 원소 앞(그 앞의 화자 태그 뒤)
                    else:
                        p = len(emit)
                        if add_spk_tags and k is not None and s != last_spk: emit.append(sp.spk[s]); src.append(None); p += 1; last_spk = s
                    emit.insert(p, tid); src.insert(p, (s, j)); mark(s, j); st.tokens += 1; n += 1; changed = True
        return n, last_spk
    out, srcs = [], []; st = InterleaveStats(chunks=n_chunks); backlog: list = []; last_spk = None
    for k in range(n_chunks):
        pending = sorted(backlog + buckets[k], key=lambda x: (x[0], x[1])); backlog = []
        emit, src, n_txt = [], [], 0
        for item in pending:
            if max_per_chunk and n_txt >= max_per_chunk: backlog.append(item); continue
            t_end, s, tid, j = item
            if add_spk_tags and s != last_spk: emit.append(sp.spk[s]); src.append(None); last_spk = s
            emit.append(tid); src.append((s, j)); n_txt += 1; st.tokens += 1; mark(s, j)
        n_ev, last_spk = inject(emit, src, k, st, last_spk); n_txt += n_ev
        st.overflow_tokens += len(backlog); st.max_backlog = max(st.max_backlog, len(backlog)); st.extra_delay_frames += len(backlog)
        st.per_chunk_hist[n_txt] = st.per_chunk_hist.get(n_txt, 0) + 1
        emit.append(sp.next_audio); src.append(None); out.append((k, emit)); srcs.append(src)
    backlog += buckets[n_chunks]
    if backlog or any(explicit):                                                       # 이벤트가 없으면 build_interleaved 의 `if backlog:` 와 같다
        st.tokens += len(backlog); st.overflow_tokens += len(buckets[n_chunks])
        fl = sorted(backlog, key=lambda x: (x[0], x[1])); emit = [x[2] for x in fl]; src = [(x[1], x[3]) for x in fl]
        for x in fl: mark(x[1], x[3])
        inject(emit, src, None, st, last_spk)
        out.append((n_chunks, emit + [sp.empty_audio])); srcs.append(src + [None])
    return (out, st, srcs) if with_src else (out, st)

# ── 2. 이벤트 토큰열 ─────────────────────────────────────────────────────────────────────────────────────────
def check_words(words: List[dict], tokens: List[Tuple[int, float]]) -> None:
    """words.jsonl 계약 검사(어기면 ValueError): 토큰 시간순, 단어 i 의 tokens[a:b] 가 비지 않고 겹치지 않으며 증가, end_time = 마지막 토큰 시각."""
    for j in range(1, len(tokens)):
        if tokens[j][1] < tokens[j - 1][1]: raise ValueError(f"tokens_unsorted: at {j}")
    prev_b = 0
    for i, w in enumerate(words):
        a, b = int(w["a"]), int(w["b"])
        if int(w.get("i", i)) != i: raise ValueError(f"word_index: {w.get('i')} != position {i}")
        if not (prev_b <= a < b <= len(tokens)): raise ValueError(f"word_span: word {i} [{a},{b}) prev end {prev_b}, n_tok {len(tokens)}")
        if abs(float(w["end_time"]) - tokens[b - 1][1]) > 1e-6: raise ValueError(f"word_end_time: word {i} {w['end_time']} != last token time {tokens[b - 1][1]}")
        prev_b = b

def build_semcommit_tokens(words: List[dict], tokens, labels_row: Optional[dict], sem_id: int, turn_id: int, turn_end: bool = True,
                           hangover_s: float = 0.48, chunk_s: float = CHUNK_S):
    """→ (event_tokens, decision_marks).
    event_tokens: 텍스트 토큰 (tid, t) 순서 그대로 + A 후보 단어 i 의 마지막 토큰(tokens[b_i - 1]) 바로 뒤 (sem_id, t_i) (t_i = words[i].end_time = 그 토큰 시각),
                  turn_end 면 끝에 명시 청크 이벤트 (turn_id, t_last + hangover_s, int((t_last + hangover_s)/chunk_s)) — 실제 청크는 래퍼가
                  max(이 값, 앞선 원소의 방출 청크) 로 정한다(δ 마다 다름).
    decision_marks: [(event_tokens 안에서 B/N 단어 마지막 토큰의 index, 'B'|'N')] — 결정 위치는 그 원소 다음.
    labels_row 가 None 이면 후보 없음(비라벨 스트림), labels_row['turn_end'] 가 False 면 TURN 없음. 계약 위반은 ValueError."""
    toks = [(int(t[0]), float(t[1])) for t in tokens]; check_words(words, toks)
    by_word: Dict[int, str] = {}
    for c in (labels_row or {}).get("candidates", []):
        i, g = int(c["after_word"]), c["grade"]
        if g not in ("A", "B", "N"): raise ValueError(f"bad_grade: {g!r}")
        if not 0 <= i < len(words): raise ValueError(f"after_word_range: {i} of {len(words)} words")
        if i in by_word: raise ValueError(f"dup_candidate: after_word {i}")
        by_word[i] = g
    at_tok = {int(words[i]["b"]) - 1: g for i, g in by_word.items()}
    ev: list = []; marks: List[Tuple[int, str]] = []
    for j, (tid, t) in enumerate(toks):
        ev.append((tid, t)); g = at_tok.get(j)
        if g == "A": ev.append((int(sem_id), t))
        elif g is not None: marks.append((len(ev) - 1, g))
    if turn_end and (labels_row or {}).get("turn_end", True) and words:
        t_last = max(float(w["end_time"]) for w in words); tt = t_last + hangover_s
        ev.append((int(turn_id), tt, int(tt / chunk_s)))
    return ev, marks

def last_event_chunk(ev: list, delay: int, chunk_s: float = CHUNK_S) -> int:
    """M=0 일 때 마지막 원소가 방출될 청크(스트림 길이 제한 없음). 명시 이벤트 = max(k_min, 앞선 원소 청크)."""
    k = -1
    for x in ev: k = max(k, int(x[2])) if len(x) > 2 and x[2] is not None else max(k, int(x[1] / chunk_s) + delay)
    return k

def last_emit_chunk(ev: list, delay: int, sp: Specials, M: int = 0) -> int:
    """마지막 방출 청크. M=0 은 공식(last_event_chunk), M>0 은 이월 때문에 공식이 틀리므로 충분히 긴 스트림으로 래퍼를 돌려 잰다."""
    k = last_event_chunk(ev, delay)
    if not M: return k
    big = k + len(ev) + 8                                                                # 청크마다 ≥1 개는 나가므로 이월은 len(ev) 청크 안에 끝난다
    chunks, _, srcs = build_interleaved_events([ev], (big - 0.5) * CHUNK_S, sp, delay_frames=delay, max_per_chunk=M, add_spk_tags=False, with_src=True)
    return max(c for (c, _), sr in zip(chunks, srcs) if any(q is not None for q in sr))

def build_semcommit_sequence(ev: list, marks, K: int, prefix: List[int], audio_pad: int, sp: Specials, delay: int, M: int = 0,
                             hardneg_weight: float = 1.0, sem_id: Optional[int] = None, turn_id: Optional[int] = None) -> dict:
    """이벤트 토큰열 → mono 시퀀스(ids·is_input·chunk_of·labels·pos_weight) + 결정 위치·이벤트 통계. 오디오와 무관한 순수 함수."""
    chunks, st, srcs = build_interleaved_events([ev], (K - 0.5) * CHUNK_S, sp, chunk_s=CHUNK_S, delay_frames=delay, max_per_chunk=M, add_spk_tags=False, with_src=True)
    ids, is_input, chunk_of, n_rounds, n_flush = build_mono_sequence(chunks, K, prefix, audio_pad, sp, M)
    order = [q for (_, em), sr in zip(chunks, srcs) for q in sr if q is not None]              # 방출 순서의 출처
    pos = [p for p in range(len(ids)) if not is_input[p] and ids[p] != sp.next_audio]
    assert len(pos) == len(order) and all(ids[p] == ev[q[1]][0] for p, q in zip(pos, order)), "방출 출처 ↔ 시퀀스 위치 대응 실패"
    pos_of = {q[1]: p for p, q in zip(pos, order)}; at = {p: q[1] for p, q in zip(pos, order)}          # 이벤트열 index ↔ 시퀀스 위치
    explicit = lambda x: len(x) > 2 and x[2] is not None
    labels = [(-100 if inp else t) for t, inp in zip(ids, is_input)]; pos_weight = [0.0] * len(ids); decision = []
    for j, g in marks:
        p = pos_of[j] + 1; assert p < len(ids) and not is_input[p], f"결정 위치 {p} 가 입력 위치"
        q = at.get(p)
        if q is not None and (explicit(ev[q]) or (sem_id is not None and ev[q][0] == sem_id)):         # 결정 위치 = 이벤트 타깃 → 라벨·가중 유지, 따로 센다
            decision.append((p, f"{g}@{'TURN' if explicit(ev[q]) else 'SEM'}")); continue
        if g == "B": labels[p] = -100
        else: pos_weight[p] = float(hardneg_weight)
        decision.append((p, g))
    flush0 = next((p for p in range(len(ids)) if is_input[p] and ids[p] == sp.empty_audio and chunk_of[p] < 0), len(ids))
    sem_pos = [pos_of[j] for j, x in enumerate(ev) if sem_id is not None and x[0] == sem_id and len(x) == 2]
    turn_pos = [pos_of[j] for j, x in enumerate(ev) if turn_id is not None and x[0] == turn_id and len(x) > 2]
    turn_chunk = next((k for (k, em), sr in zip(chunks, srcs) for q in sr if q is not None and len(ev[q[1]]) > 2), None)
    return dict(ids=ids, is_input=is_input, chunk_of=chunk_of, labels=labels, pos_weight=pos_weight, n_rounds=n_rounds, n_flush=n_flush, st=st, chunks=chunks, srcs=srcs,
                pos_of=pos_of, decision=decision, sem_pos=sem_pos, turn_pos=turn_pos, turn_chunk=turn_chunk,
                n_sem=len(sem_pos), n_turn=len(turn_pos), n_B=sum(g == "B" for _, g in decision), n_N=sum(g == "N" for _, g in decision),    # 실제로 적용된 B 가림·N 가중
                n_decision_on_event=sum("@" in g for _, g in decision), sem_in_flush=sum(p > flush0 for p in sem_pos), turn_in_flush=sum(p > flush0 for p in turn_pos))

# ── 3. 데이터셋 ─────────────────────────────────────────────────────────────────────────────────────────────
def _paths(x: Union[str, Sequence[str]]) -> List[str]:
    return [p for s in ([x] if isinstance(x, str) else list(x)) for p in str(s).split(",") if p]

def _read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def _pack_events(ev: list) -> tuple:
    """이벤트열 → (ids, 시각, k_min(-1 = 시각 버킷)) 배열 3 개 — 표기형 사본을 원소당 tuple 대신 24 B 로 보관(항목 수백만 × rank × worker)."""
    return (array("q", (int(x[0]) for x in ev)), array("d", (float(x[1]) for x in ev)),
            array("q", (int(x[2]) if len(x) > 2 and x[2] is not None else -1 for x in ev)))

def _unpack_events(p: tuple) -> list:
    return [(i, t) if k < 0 else (i, t, k) for i, t, k in zip(*p)]

FP_PROTOCOL = "semcommit-train-v1"                                                         # v1: 턴 종료 = <EOT>(선택, 기본 끔), <TURN_END> 폐기
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def vocab_sha256(tok) -> Optional[str]:
    """tokenizer 어휘 전체(piece→id, 추가 토큰 포함)의 sha256 — 파일 바이트가 아니라 내용 지문이라 run 디렉토리에 재직렬화된 tokenizer.json 도 같은 값. get_vocab 없으면 None."""
    try: v = tok.get_vocab()
    except Exception: return None
    return hashlib.sha256(json.dumps(sorted(v.items()), ensure_ascii=False).encode()).hexdigest()

def semcommit_fingerprint(words, labels, tok, **settings) -> dict:
    """학습 데이터를 모양 짓는 설정(turn_end·hangover_s·tail_margin·pad_tail·M·delays·가중 …) + words/labels 파일(순서대로)·tokenizer 어휘 지문.
    학습 스크립트가 config.semcommit 으로 체크포인트에 넣는다 → 재개 때 fingerprint_diff 로 대조, 평가가 학습 패딩 설정을 복원. JSON 왕복 형태(tuple → list)."""
    fp = dict(protocol=FP_PROTOCOL, **settings, words_sha256=[sha256_file(p) for p in _paths(words)], labels_sha256=[sha256_file(p) for p in _paths(labels)], vocab_sha256=vocab_sha256(tok))
    return json.loads(json.dumps(fp))

def fingerprint_diff(saved: Optional[dict], cur: dict) -> List[str]:
    """저장된 fingerprint(체크포인트 config.semcommit) vs 지금 → 다른 키 ['key: 저장 → 지금', ...]. 저장본이 없으면 ['<fingerprint 없음>'](이 코드 이전 체크포인트)."""
    if not saved: return ["<fingerprint 없음>"]
    cur = json.loads(json.dumps(cur))
    def short(v): return [short(x) for x in v] if isinstance(v, list) else (v[:12] + "…") if isinstance(v, str) and len(v) > 16 else v
    return [f"{k}: {short(saved.get(k))!r} → {short(cur.get(k))!r}" for k in sorted(set(saved) | set(cur)) if saved.get(k) != cur.get(k)]

def checkpoint_fingerprint_diff(ckpt_dir: str, cur: dict) -> List[str]:
    """재개할 체크포인트 디렉토리의 config.json['semcommit'] vs 지금(fingerprint_diff). config.json 이 없으면 '<fingerprint 없음>'."""
    p = os.path.join(ckpt_dir, "config.json"); saved = None
    if os.path.exists(p):
        with open(p) as f: saved = json.load(f).get("semcommit")
    return fingerprint_diff(saved, cur)

class SemCommitDataset(Dataset):
    """words.jsonl + labels.jsonl(id 로 짝) → Phase 1 mono 시퀀스 + SEM/TURN + B/N 결정 위치. __getitem__ 은 MonoStreamDataset 과 같은 키 + pos_weight·이벤트 통계.
    online=True: segments 로 assemble_stream(패딩된 duration 까지 끝 무음 합성). online=False: 같은 길이의 0 파형(시퀀스 검사·CPU 드라이런 전용 — 학습 금지).
    turn_end: 턴 종료 <EOT> 를 함께 학습(기본 False — Phase 1 은 <SEM_END> 만).
    langs: 언어 필터(스크립트가 EN/KO 셋을 나눌 때). path_map: segment path 접두어 치환 {old: new}. pad_tail: TURN 이 없어도 마지막 방출이 flush 에 안 가게 늘린다.
    짝 통계(stats): words_rows(언어 필터 뒤 words 행) · labeled(그중 labels 짝이 있는 행) · labels_without_words(어느 words 행과도 짝이 없는 label id — 다른 셋·표본의 labels)
      · dup_label_rows · decision_on_event(B/N 결정 위치가 TURN 타깃에 떨어지는 (항목, δ) 수) / decision_on_event_items. bad['lang_mismatch'] = label.lang ≠ words.lang(제외).
    tok_check: 앞쪽 words 행 이만큼을 이 tokenizer 로 다시 단어 분할(semcommit_words.split_words — words 빌더와 같은 함수)해 단어 text·[a,b) 가 같은지 본다.
      다르면 ValueError(words.jsonl 의 토큰 id 가 이 tokenizer 에서 다른 단어가 된다). tokenizer 에 convert_ids_to_tokens 가 없으면(가짜) 건너뛰고 stats['tok_check_skipped'].
    ko_textnorm: Korean 행의 타깃을 asr-tn-v1 로(vapasr/data/semcommit_tn.normalize_ko_row) — speechlm 행은 구두점 토큰이 붙어 정렬돼 있다.
      바뀐 행 수 stats['ko_textnorm_changed'], 숫자를 원본 전사(segments[].raw_text)의 읽는 형태로 바꾼 행 stats['ko_digit_spoken'],
      제외 사유 bad['tn_digit'(대응 불확실)|'tn_charset'|'tn_word_split'|'tn_word_empty'|'tn_decode'|'tn_label_index'].
    written_frac: 표기형이 있는 항목을 __getitem__ 에서 이 확률로 표기형으로 낸다(기본 0 = 끔: 표기형을 만들지 않고 난수도 더 뽑지 않는다 — 이 기능 전과 바이트 동일).
      표기형은 정규형이 통과한 행에서만 만든다(v1: 정규형이 실패한 행은 표기형만으로도 쓰지 않는다 → 정규형 항목 집합·시퀀스는 플래그와 무관).
      KO 는 normalize_ko_row **전** 원 행·원 라벨에서(semcommit_written.written_row) — 그래서 ko_textnorm=False 와 KO 표기형은 함께 못 쓴다(ValueError:
      정규형 타깃도 원 display 토큰이 되어 두 형식 머리가 같은 타깃을 갖는다). 항목에 written(압축 이벤트열·marks·n_tok·n_sem·n_turn[·dropped])과
      est_extra(표기형 토큰·지시문 머리 증가분 — packing.est_lens 가 더한다)가 붙는다. K 는 정규형 필요 청크 그대로다: 표기형의 후보 단어 끝 시각이
      정규형과 다르거나(written_cand_time_shift — KO 끝 구두점 토큰 시각이 늦은 행) 어느 δ 에서든 마지막 방출 청크가 더 늦으면(written_tail_shift) 표기형을 버린다.
      stats(written_only_skipped 를 빼고 max_items 절단 뒤 남은 항목 기준): written_available · written_hours · written_tokens · written_<사유>(표기형 없음,
      항목의 written_why) · written_cand_dropped_<A|B|N>(EN 단위 안쪽 후보). written_only_skipped = 정규형이 실패해 버린 행 중 표기형 원천이 있던 행(절단 전 행 수).
      __getitem__ 은 fmt·n_written 을 더 낸다(비율 > 0 일 때만). sequence/target_problems/check_targets 의 fmt 인자, formats(i) = 그 항목의 형식들.
    sem_free_unlabeled: labels 짝이 없는 행을 'SEM 중립' ASR 항목으로 쓴다(allow_unlabeled 의 'SEM 음성' 과 반대, 함께 못 쓴다). 이벤트·B/N 없이 텍스트·NEXT 만 학습하고,
      collate 가 그 행에 sem_mask 를 내 모델이 <SEM_END>·턴 종료 logit 을 softmax 에서 뺀다(→ 'SEM 아님' 신호가 없다). 턴 종료·표기형은 만들지 않는다.
      speechlm 파트의 words.jsonl 은 두 풀(8 s 기준)이 섞여 있어, 라벨 없는 행의 풀 불일치는 멈추지 않고 건너뛴다(stats sem_free_other_pool) — 같은 파일을 main·short 에 준다.
      sem_free_max_ratio: SEM 중립 항목 수 ≤ 비율 × 라벨 항목 수(섞은 순서 앞에서; stats sem_free_dropped). stats sem_free · sem_free_hours.
      __getitem__ 은 sem_free 를 더 낸다(sem_free_unlabeled 일 때만).
    speed_perturb: 배율 목록(예 (0.9, 1.0, 1.1)). __getitem__ 마다 하나를 고르고, 1 이 아니면 오디오를 리샘플하고(speed_wav) 이벤트 시각을 1/배율로 늘려 시퀀스·K 를
      다시 만든다(perturbed). 빈 목록(기본)이면 난수를 더 뽑지 않는다. 느린 배율이 시퀀스를 늘리므로 항목 est_extra 에 여유를 더한다(packing 길이 추정).
      __getitem__ 은 speed 를 더 낸다(켰을 때만).
    offline_frac: 항목마다 이 확률로 final(오프라인) 모드 — Qwen3-ASR 원 형식(오디오 전체 → 전사, vapasr/data/offline_seq). 같은 모델이 스트리밍(δ)과 final 을 함께 배운다.
      정규형만(표기형은 스트리밍), 이벤트 없음. __getitem__ 은 mode('stream'|'offline'), collate 는 n_offline 을 더 낸다. 0(기본)이면 난수를 더 뽑지 않는다.
    ignore_labels: labels 를 무시하고 모든 행을 SEM 중립으로(ASR 전용 Stage 1; sem_free_unlabeled 와 함께).
    noise_aug: vapasr/data/noise_aug.NoiseAugment — 오디오에 잔향(직접음 정렬, 지연 없음)·가산 잡음(SNR 범위)을 건다. 시퀀스·K 는 그대로. None(기본)이면 끔."""
    def __init__(self, words_jsonl, labels_jsonl, tok, sp_ids: Optional[Dict[str, int]] = None, delays=(2, 3, 4, 6), turn_end: bool = False,
                 hangover_s: float = 0.48, hardneg_weight: float = 1.0, max_items: Optional[int] = None, seed: int = 0, online: bool = True,
                 allow_unlabeled: bool = False, max_per_chunk: int = 0, langs: Optional[Sequence[str]] = None, path_map: Optional[Dict[str, str]] = None,
                 tail_margin: int = 2, pad_tail: bool = True, max_audio_retries: int = 8, tok_check: int = 32, ko_textnorm: bool = True,
                 source_pool: Optional[str] = None, written_frac: float = 0.0, sem_free_unlabeled: bool = False,
                 sem_free_max_ratio: Optional[float] = None, speed_perturb: Sequence[float] = (), noise_aug=None, offline_frac: float = 0.0,
                 ignore_labels: bool = False):
        if source_pool not in (None, "main", "speechlm-short"):
            raise ValueError(f"unknown source_pool: {source_pool}")
        if sem_free_unlabeled and allow_unlabeled: raise ValueError("allow_unlabeled(라벨 없음 = SEM 음성)와 sem_free_unlabeled(라벨 없음 = SEM 중립)는 함께 쓸 수 없다")
        self.sem_free_unlabeled = bool(sem_free_unlabeled)
        self.speed_perturb = tuple(float(x) for x in speed_perturb)
        if any(not 0.5 <= x <= 2.0 for x in self.speed_perturb): raise ValueError(f"speed_perturb 배율은 [0.5, 2] 범위: {self.speed_perturb}")
        self.tail_margin, self.pad_tail = int(tail_margin), bool(pad_tail)
        self.noise_aug = noise_aug                                                             # vapasr/data/noise_aug.NoiseAugment(잡음·잔향) 또는 None
        if not 0.0 <= float(offline_frac) <= 1.0: raise ValueError(f"offline_frac {offline_frac} 는 [0, 1] 범위")
        self.offline_frac = float(offline_frac)
        if self.offline_frac > 0:
            from .offline_seq import OfflineFormat
            self._offline = OfflineFormat(tok)
        if ignore_labels and not sem_free_unlabeled: raise ValueError("ignore_labels(ASR 전용)는 sem_free_unlabeled 와 함께 쓴다 — 모든 행이 SEM 중립")
        self.ignore_labels = bool(ignore_labels)
        if not 0.0 <= float(written_frac) <= 1.0: raise ValueError(f"written_frac {written_frac} 는 [0, 1] 범위")
        if float(written_frac) > 0 and not ko_textnorm and (not langs or "Korean" in langs):
            raise ValueError("ko_textnorm=False 와 Korean 표기형(written_frac > 0)은 함께 쓸 수 없다 — 정규형 타깃도 원 display 토큰(구두점·숫자)이 된다. "
                             "langs 에서 Korean 을 빼거나 ko_textnorm=True")
        self.written_frac = float(written_frac)
        self.tok = tok; self.sp_ids = dict(sp_ids) if sp_ids is not None else add_semcommit_specials(tok); check_sem_ids(self.sp_ids)
        self.sp = specials_of(self.sp_ids); self.sem_id, self.turn_id = self.sp_ids["<SEM_END>"], self.sp_ids[TURN_TOKEN]
        self.delays = tuple(int(d) for d in delays); self.M = max_per_chunk; self.online = online; self.turn_end = turn_end; self.hangover_s = hangover_s
        self.hardneg_weight = hardneg_weight; self.max_audio_retries = max_audio_retries
        self.audio_pad = tok.convert_tokens_to_ids("<|audio_pad|>"); self._pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        assert isinstance(self.audio_pad, int) and self.audio_pad >= 0, "tokenizer 에 <|audio_pad|> 없음(Qwen3-ASR tokenizer 필요)"
        head_extra = len(header_ids(tok, WRITTEN)) - len(self._pre) if self.written_frac > 0 else 0          # 표기형 머리(지시문) 토큰 증가분
        labels: Dict[str, dict] = {}; self.stats = Counter()
        for p in _paths(labels_jsonl):
            for r in _read_jsonl(p):
                if r["id"] in labels: self.stats["dup_label_rows"] += 1; continue
                labels[r["id"]] = r
        self.items: List[dict] = []; self.bad: Counter = Counter(); seen = set(); tok_bad: list = []
        can_split = callable(getattr(tok, "convert_ids_to_tokens", None))
        for p in _paths(words_jsonl):
            for r in _read_jsonl(p):
                seen.add(r["id"])
                if langs and r["lang"] not in langs: continue
                lab = None if self.ignore_labels else labels.get(r["id"])
                if r.get("set") == "speechlm-partial" and source_pool in ("main", "speechlm-short"):
                    if r.get("training_eligible") is not True or r.get("candidate_only") is not False or not r.get("approval_fingerprint"):
                        raise ValueError(f"unapproved_speechlm_word: {r['id']}")
                    is_short = float(r["duration_s"]) < 8.0
                    if is_short != (source_pool == "speechlm-short"):
                        if lab is None and self.sem_free_unlabeled:                           # SEM 중립 ASR 파트의 words.jsonl 은 두 풀이 섞여 있다 → 길이로 나눈다(같은 파일을 두 풀에 준다)
                            self.stats["sem_free_other_pool"] += 1; continue
                        raise ValueError(f"speechlm_pool_mismatch: {r['id']} duration={r['duration_s']} pool={source_pool}")
                elif source_pool == "speechlm-short":
                    if lab is None and self.sem_free_unlabeled:                               # speechlm 밖 SEM 중립 ASR 행(E2 코퍼스 등)은 main 풀에서만 쓴다
                        self.stats["sem_free_other_pool"] += 1; continue
                    raise ValueError(f"non_speechlm_in_short_pool: {r['id']}")
                self.stats["words_rows"] += 1; self.stats["labeled"] += lab is not None
                if lab is None and not allow_unlabeled and not self.sem_free_unlabeled: self.stats["skipped_unlabeled"] += 1; continue
                sem_free = lab is None and self.sem_free_unlabeled                            # 라벨 없음 → 텍스트·NEXT 만 배우고 <SEM_END> 는 softmax 에서 뺀다(sem_mask)
                if lab is not None and lab.get("lang") not in (None, r["lang"]): self.bad["lang_mismatch"] += 1; continue
                if not r.get("words"): self.bad["no_words"] += 1; continue
                r0, lab0 = r, lab                                                              # 표기형은 KO 정규화 전 원 행·원 라벨에서 만든다
                if ko_textnorm and r["lang"] == "Korean":                                    # KO 타깃 → asr-tn-v1(구두점 토큰 제거; digit·허용 밖 문자 행 제외) — semcommit_tn
                    r2 = normalize_ko_row(r, tok)
                    if isinstance(r2, tuple): self.bad[r2[1]] += 1; self._written_only(r0); continue
                    self.stats["ko_textnorm_changed"] += r2 is not r
                    self.stats["ko_digit_spoken"] += r2 is not r and any(ch.isdigit() for w in r["words"] for ch in w["text"])   # 숫자 → 원본 전사의 읽는 형태
                    if "word_map" in r2 and lab is not None:                                   # 숫자 단어가 여러 단어가 됨 → 라벨 단어 index 를 새 순서로
                        wm = r2["word_map"]
                        if any(not 0 <= c["after_word"] < len(wm) for c in lab.get("candidates", [])): self.bad["tn_label_index"] += 1; self._written_only(r0); continue
                        lab = dict(lab, candidates=[dict(c, after_word=wm[c["after_word"]]) for c in lab.get("candidates", [])])
                    r = r2
                try: ev, marks = build_semcommit_tokens(r["words"], r["tokens"], lab, self.sem_id, self.turn_id, turn_end=turn_end and not sem_free, hangover_s=hangover_s)
                except ValueError as e: self.bad[str(e).split(":")[0]] += 1; self._written_only(r0); continue          # 메시지 앞머리 = 사유 코드
                except KeyError as e: self.bad[f"missing_key:{e}"] += 1; self._written_only(r0); continue
                if can_split and self.stats["tok_check"] < tok_check:                          # tokenizer 내용 관문(계약을 통과한 행만; 빌더와 같은 분할)
                    from .semcommit_words import split_words
                    self.stats["tok_check"] += 1
                    try: ws = split_words([int(t[0]) for t in r["tokens"]], [float(t[1]) for t in r["tokens"]], tok)
                    except Exception as e: ws = f"{type(e).__name__}: {e}"
                    want = [(w["text"], int(w["a"]), int(w["b"])) for w in r["words"]]
                    got = [(w["text"], w["a"], w["b"]) for w in ws] if isinstance(ws, list) else None
                    if got is not None and got != want:                                     # speechlm words: 가장자리 구두점을 태그로 뗀 lexical 단어(같은 빌더 규칙)
                        from .semcommit_words import lexical_words
                        lw, _ = lexical_words([dict(w) for w in ws]); got = [(w["text"], w["a"], w["b"]) for w in lw] if lw else None
                    if got != want:
                        tok_bad.append((r["id"], " ".join(w[0] for w in want[:8]), ws if not isinstance(ws, list) else " ".join(w["text"] for w in ws[:8])))
                K0 = int(r["K"]); has_turn = any(len(x) > 2 for x in ev)
                need = max(last_emit_chunk(ev, d, self.sp, max_per_chunk) for d in self.delays) + tail_margin if (has_turn or pad_tail) else 0
                wr = self._written_form(r0, lab0, r, ev, turn_end, hangover_s, has_turn) if self.written_frac > 0 and not sem_free else None   # 꼬리가 정규형보다 늦은 표기형은 버린다 → K 는 정규형 그대로
                K = max(K0, need); dur = float(r["duration_s"]) if K == K0 else K * CHUNK_S
                segs = [dict(s) for s in r["segments"]]
                for s in segs:
                    for old, new in (path_map or {}).items():
                        if s["path"].startswith(old): s["path"] = new + s["path"][len(old):]; break
                n_sem = sum(1 for x in ev if len(x) == 2 and x[0] == self.sem_id)
                self.items.append(dict(name=r.get("set", "semcommit"), id=r["id"], lang=r["lang"], K=K, K0=K0, duration_s=dur, duration0=float(r["duration_s"]), segments=segs,
                                       tokens=ev, marks=marks, n_tok=len(r["tokens"]), n_sem=n_sem, n_turn=int(has_turn), text=r.get("text", ""),
                                       **({"sem_free": True} if sem_free else {})))
                if isinstance(wr, str): self.items[-1]["written_why"] = wr                      # 표기형 없음 사유(통계는 절단 뒤에 센다)
                elif wr is not None:                                                           # 표기형 사본(압축) + packing 길이 추정 증가분
                    ev_w, marks_w, n_tok_w, dropped = wr; n_sem_w = sum(1 for x in ev_w if len(x) == 2 and x[0] == self.sem_id)
                    self.items[-1]["written"] = dict(tokens=_pack_events(ev_w), marks=marks_w, n_tok=n_tok_w, n_sem=n_sem_w, n_turn=int(has_turn),
                                                     **({"dropped": dict(sorted(dropped.items()))} if dropped else {}))
                    self.items[-1]["est_extra"] = max(0, len(ev_w) + n_sem_w - len(ev) - n_sem + head_extra)
        if tok_bad: raise ValueError(f"tokenizer_mismatch: {len(tok_bad)}/{self.stats['tok_check']} 행의 토큰 id 가 이 tokenizer 에서 다른 단어가 된다(words.jsonl 을 만든 tokenizer 와 다름) 예: {tok_bad[:2]}")
        if not can_split: self.stats["tok_check_skipped"] = 1
        self.stats["labels_without_words"] = sum(1 for k in labels if k not in seen)
        random.Random(seed).shuffle(self.items)
        if sem_free_max_ratio is not None:                                                    # SEM 중립 항목 ≤ 비율 × 라벨 항목(섞은 순서의 앞에서) — SEM 신호가 묽어지지 않게
            n_lab = sum(1 for it in self.items if not it.get("sem_free")); cap = int(float(sem_free_max_ratio) * n_lab); kept = []; n_free = 0
            for it in self.items:
                if it.get("sem_free"):
                    n_free += 1
                    if n_free > cap: self.stats["sem_free_dropped"] += 1; continue
                kept.append(it)
            self.items = kept
        if max_items: self.items = self.items[:max_items]
        if self.speed_perturb:                                                                # 느린 배율이 시퀀스를 늘린다 → packing 길이 추정에 보수적 여유(2 × 늘어난 청크)
            lo = min(self.speed_perturb)
            for it in self.items: it["est_extra"] = int(it.get("est_extra", 0)) + 2 * (max(0, math.ceil(it["K"] / lo) - it["K"]) + 2)
        for i, it in enumerate(self.items):                                                   # 통계는 남은 항목 기준
            self.stats["padded"] += int(it["K"] > it["K0"]); self.stats["pad_chunks"] += it["K"] - it["K0"]; self.stats["sem"] += it["n_sem"]; self.stats["turn"] += it["n_turn"]
            if it.get("sem_free"): self.stats["sem_free"] += 1; self.stats["sem_free_hours"] += it["duration0"] / 3600
            self.stats["B"] += sum(g == "B" for _, g in it["marks"]); self.stats["N"] += sum(g == "N" for _, g in it["marks"]); self.stats["words_tokens"] += it["n_tok"]
            ev = it["tokens"]
            if any(j + 1 < len(ev) and len(ev[j + 1]) > 2 for j, _ in it["marks"]):          # B/N 원소 바로 뒤가 TURN → δ 에 따라 결정 위치 = TURN 타깃(정확한 수는 시퀀스로)
                n = sum(self.sequence(i, d)["n_decision_on_event"] for d in self.delays)
                self.stats["decision_on_event"] += n; self.stats["decision_on_event_items"] += int(n > 0)
        if "sem_free_hours" in self.stats: self.stats["sem_free_hours"] = round(self.stats["sem_free_hours"], 3)
        if self.written_frac > 0:                                                             # 표기형 가용량·없음 사유·버린 후보(남은 항목 기준)
            ws = [it for it in self.items if "written" in it]
            self.stats["written_available"] += len(ws); self.stats["written_tokens"] += sum(it["written"]["n_tok"] for it in ws)
            self.stats["written_hours"] = round(sum(it["duration0"] for it in ws) / 3600, 3)
            for it in self.items:
                if "written_why" in it: self.stats[f"written_{it['written_why']}"] += 1
                for g, n in it.get("written", {}).get("dropped", {}).items(): self.stats[f"written_cand_dropped_{g}"] += n
        self.stats["items"] = len(self.items)

    def _written_only(self, r0: dict) -> None:
        """정규형이 실패해 버린 행 중 표기형 원천이 있던 행 수(v1 은 표기형만으로 쓰지 않는다). 항목이 되기 전 행이라 max_items 절단 전 수다."""
        if self.written_frac > 0 and (r0.get("display_text") or r0.get("pnc_text")): self.stats["written_only_skipped"] += 1

    def _written_form(self, r0: dict, lab0: Optional[dict], r: dict, ev: list, turn_end: bool, hangover_s: float, has_turn: bool):
        """표기형 (이벤트열, marks, n_tok, 버린 후보 Counter) 또는 사유 문자열(표기형 없음 → 항목 written_why). 라벨은 원 단어 index 기준(lab0) → word_map 으로 옮긴다.
        r·ev = 이 행의 정규형(KO 는 normalize_ko_row 뒤 — 숫자 단어가 여러 단어가 되면 r['word_map']). 두 형식의 청크 일치를 여기서 강제한다:
          cand_time_shift  남는 후보(A/B/N) 단어의 끝 시각 ≠ 정규형 단어 끝 시각(→ SEM·결정 청크가 달라진다; KO 끝 구두점 토큰 시각이 늦은 행)
          tail_shift       어느 δ 에서든 표기형 마지막 방출 청크 > 정규형(→ K·flush 가 달라진다)"""
        rw, wm = written_row(r0, self.tok)
        if rw is None: return wm
        try:
            lab_w, dropped = remap_labels(lab0, wm)
            ev_w, marks_w = build_semcommit_tokens(rw["words"], rw["tokens"], lab_w, self.sem_id, self.turn_id, turn_end=turn_end, hangover_s=hangover_s)
        except ValueError as e: return str(e).split(":")[0]
        except KeyError as e: return f"missing_key:{e}"
        if any(len(x) > 2 for x in ev_w) != has_turn: return "turn_mismatch"
        tn_map = r.get("word_map") if r is not r0 else None                                   # 원 단어 index → 정규형 단어 index(그 숫자 단어의 마지막)
        for c in (lab0 or {}).get("candidates", []):                                          # 범위는 build_semcommit_tokens 가 이미 검사했다
            i = int(c["after_word"]); j = i if wm is None else wm[i]
            if j is None: continue                                                            # EN 단위 안쪽(버린 후보)
            if abs(float(rw["words"][j]["end_time"]) - float(r["words"][tn_map[i] if tn_map else i]["end_time"])) > 1e-6: return "cand_time_shift"
        if any(last_emit_chunk(ev_w, d, self.sp, self.M) > last_emit_chunk(ev, d, self.sp, self.M) for d in self.delays): return "tail_shift"
        return ev_w, marks_w, len(rw["tokens"]), dropped

    def formats(self, i: int) -> List[str]:
        """항목 i 가 가진 출력 형식(정규형은 항상, 표기형은 written_frac > 0 이고 만들 수 있었을 때)."""
        return [NORMALIZED] + ([WRITTEN] if "written" in self.items[i] else [])

    def _form(self, i: int, fmt: str = NORMALIZED) -> dict:
        """형식별 이벤트열 보기 {tokens, marks, n_sem, n_turn, ...}. 정규형 = 항목 그 자체(기존 경로 그대로)."""
        it = self.items[i]
        if fmt == NORMALIZED: return it
        if fmt != WRITTEN: raise ValueError(f"unknown output format: {fmt!r}")
        w = it.get("written")
        if w is None: raise ValueError(f"no_written_form: {it['id']}")
        return dict(w, tokens=_unpack_events(w["tokens"]))

    def __len__(self): return len(self.items)
    def prefix(self, lang: str, delay: int, fmt: str = NORMALIZED) -> List[int]:
        """MonoStreamDataset.prefix 와 같다: system/assistant 머리 + 'language {Lang}<asr_text>' + <DELAY_d>. (lang, δ) 별로 한 번만 토큰화.
        fmt='written' 은 머리만 표기형 지시문 머리(output_format.header_ids) — 캐시 키 (lang, δ, 'written'), 정규형 키·값은 그대로."""
        c = self.__dict__.setdefault("_prefix_cache", {})
        if fmt != NORMALIZED:
            key = (lang, delay, fmt)
            if key not in c: c[key] = header_ids(self.tok, fmt) + self.tok(f"language {lang}<asr_text>", add_special_tokens=False)["input_ids"] + [self.sp_ids[f"<DELAY_{delay}>"]]
            return list(c[key])
        if (lang, delay) not in c: c[(lang, delay)] = self._pre + self.tok(f"language {lang}<asr_text>", add_special_tokens=False)["input_ids"] + [self.sp_ids[f"<DELAY_{delay}>"]]
        return list(c[(lang, delay)])
    def sequence(self, i: int, delay: int, fmt: str = NORMALIZED) -> dict:
        it = self.items[i]; f = self._form(i, fmt)
        return build_semcommit_sequence(f["tokens"], f["marks"], it["K"], self.prefix(it["lang"], delay, fmt), self.audio_pad, self.sp, delay, self.M,
                                        hardneg_weight=self.hardneg_weight, sem_id=self.sem_id, turn_id=self.turn_id)
    def audio(self, it: dict) -> np.ndarray:
        """(T,) float32 @16 kHz, T = round(duration_s·SR) — 패딩된 길이까지 assemble_stream 이 끝 무음을 합성한다(마지막 발화 뒤 20 ms 배경을 늘림)."""
        if not self.online: return np.zeros(int(round(it["duration_s"] * SR)), np.float32)
        from .streams import assemble_stream
        return assemble_stream(dict(id=it["id"], duration_s=it["duration_s"], segments=it["segments"]))
    def target_problems(self, i: int, delay: int, fmt: str = NORMALIZED) -> List[str]:
        """학습 전 시퀀스 불변식 위반 목록(빈 목록 = 통과):
        text_order(라벨 위치 텍스트 ≠ 참조 토큰열) · event_count(SEM/TURN 수) · turn_in_flush · sem_in_flush(패딩한 항목) ·
        sem_detached(SEM 이 앞 원소 = 단어 마지막 토큰 바로 뒤가 아님 — 사이에 NEXT/오디오, 즉 다른 청크; M>0 에서 생긴다) ·
        event_target_altered(SEM/TURN 위치의 라벨이 가려졌거나 pos_weight 로 덮임) · decision(B 결정 위치가 안 가려짐 / N 가중 ≠ hardneg / 이벤트 위 결정이 '@' 로 안 남음).
        fmt='written' 이면 표기형 시퀀스(지시문 머리 + 표기형 이벤트열)에 같은 검사."""
        s = self.sequence(i, delay) if fmt == NORMALIZED else self.sequence(i, delay, fmt)                  # 정규형 호출 모양은 그대로(sequence 를 바꿔 끼우는 테스트)
        it = self.items[i]; f = self._form(i, fmt); ev = f["tokens"]; ids, lab, pw, pos_of = s["ids"], s["labels"], s["pos_weight"], s["pos_of"]
        skip = {self.sp.next_audio, self.sp.empty_audio, self.sem_id, self.turn_id}; P = []
        got = [t for t, inp in zip(ids, s["is_input"]) if not inp and t not in skip]
        if got != [x[0] for x in ev if x[0] not in (self.sem_id, self.turn_id)]: P.append("text_order")
        if (s["n_sem"], s["n_turn"]) != (f["n_sem"], f["n_turn"]): P.append(f"event_count: sem {s['n_sem']}/{f['n_sem']} turn {s['n_turn']}/{f['n_turn']}")
        if s["turn_in_flush"]: P.append("turn_in_flush")
        if s["sem_in_flush"] and it["K"] > it["K0"]: P.append("sem_in_flush")
        P += [f"sem_detached@{j}" for j, x in enumerate(ev) if len(x) == 2 and x[0] == self.sem_id and not (j > 0 and pos_of[j] == pos_of[j - 1] + 1)]
        P += [f"event_target_altered@{p}" for p in s["sem_pos"] + s["turn_pos"] if lab[p] != ids[p] or pw[p]]
        for p, g in s["decision"]:
            on_ev = ids[p] in (self.sem_id, self.turn_id)
            if on_ev != ("@" in g) or (g == "B" and lab[p] != -100) or (g == "N" and pw[p] != float(self.hardneg_weight)): P.append(f"decision@{p}:{g}")
        return P
    def check_targets(self, i: int, delay: int, fmt: str = NORMALIZED) -> bool:
        """target_problems 가 비었는가(학습 스크립트의 시작 전 관문)."""
        return not self.target_problems(i, delay, fmt)
    def perturbed(self, i: int, delay: int, f: float, fmt: str = NORMALIZED) -> Tuple[dict, int, float]:
        """속도 변형(배율 f; > 1 = 빠르게·짧게, Kaldi 식 리샘플이라 피치도 바뀐다)의 시퀀스: 이벤트 시각 t → t/f(명시 청크 이벤트 = 턴 종료는 새 시각으로 청크를 다시),
        K' = max(round(원 길이/f · 12.5), 모든 δ 의 마지막 방출 청크 + tail_margin) → (시퀀스, K', 오디오 길이 초). 결정 위치(marks)는 이벤트열 index 라 그대로다."""
        it = self.items[i]; fm = self._form(i, fmt)
        ev = [(x[0], x[1] / f) if len(x) == 2 else (x[0], x[1] / f, int(x[1] / f / CHUNK_S)) for x in fm["tokens"]]
        K0 = max(1, int(round(it["duration0"] / f / CHUNK_S))); has_turn = any(len(x) > 2 for x in ev)
        need = max(last_emit_chunk(ev, d, self.sp, self.M) for d in self.delays) + self.tail_margin if (has_turn or self.pad_tail) else 0
        K = max(K0, need); dur = it["duration0"] / f if K == K0 else K * CHUNK_S
        s = build_semcommit_sequence(ev, fm["marks"], K, self.prefix(it["lang"], delay, fmt), self.audio_pad, self.sp, delay, self.M,
                                     hardneg_weight=self.hardneg_weight, sem_id=self.sem_id, turn_id=self.turn_id)
        return s, K, dur
    @staticmethod
    def speed_wav(wav: np.ndarray, f: float, n: int, key: str = "") -> np.ndarray:
        """wav 를 배율 f 로 빠르게(f > 1) 또는 느리게(f < 1): SR·f 로 녹음된 것으로 보고 SR 로 리샘플(길이 ≈ len/f) → n 샘플에 맞춘다.
        모자라면(꼬리 필요 청크가 원 패딩/f 보다 1–2 청크 긴 경우) 마지막 20 ms 배경을 streams.silence_like 로 늘린다(assemble_stream 끝 무음 합성과 같다)."""
        import torchaudio.functional as AF
        from .streams import silence_like, _rng
        x = AF.resample(torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)), orig_freq=int(round(SR * f)), new_freq=SR).numpy()
        if len(x) >= n: return np.ascontiguousarray(x[:n], dtype=np.float32)
        return np.concatenate([x, silence_like(x[-320:] if len(x) >= 320 else np.zeros(320, np.float32), n - len(x), _rng(f"{key}:speed{f}"))]).astype(np.float32)
    def __getitem__(self, i):
        delay = random.choice(self.delays); err = None
        for r in range(self.max_audio_retries):                                          # 오디오 조립 실패 → 이웃 항목(재귀 대신 유한 반복)
            j = (i + r) % len(self.items); it = self.items[j]
            try: wav = self.audio(it); break
            except (ValueError, OSError, RuntimeError) as e:
                err = e; self._audio_fail = getattr(self, "_audio_fail", 0) + 1
                if self._audio_fail <= 5: print(f"  ! 항목 {it['id']} 오디오 조립 실패({type(e).__name__}: {str(e)[:120]}) → 이웃 항목 (누적 {self._audio_fail})", flush=True)
        else: raise RuntimeError(f"오디오 조립 {self.max_audio_retries} 회 연속 실패: {err}")
        fmt = WRITTEN if self.written_frac > 0 and "written" in it and random.random() < self.written_frac else NORMALIZED   # 0 이면 난수를 더 뽑지 않는다
        f = random.choice(self.speed_perturb) if self.speed_perturb else 1.0                  # 속도 변형이 꺼져 있으면 난수를 더 뽑지 않는다(기존 run 과 같은 난수열)
        if f != 1.0:
            s, K, dur = self.perturbed(j, delay, f, fmt); wav = self.speed_wav(wav, f, int(round(dur * SR)), it["id"])
        else:
            s = self.sequence(j, delay) if fmt == NORMALIZED else self.sequence(j, delay, fmt); K = it["K"]
        offline = self.offline_frac > 0 and fmt == NORMALIZED and random.random() < self.offline_frac   # final 모드(오디오 전체 → 전사). 0 이면 난수를 더 뽑지 않는다
        if offline: s = self.offline_sequence(j, K, fmt)
        if self.noise_aug is not None and self.online: wav = self.noise_aug(wav, random)          # 잡음·잔향(오디오만; 시각·K 그대로). 끄면 난수를 더 뽑지 않는다
        out = dict(wav=torch.from_numpy(wav), ids=torch.tensor(s["ids"]), is_audio=torch.tensor([c >= 0 for c in s["chunk_of"]]), chunk_of=torch.tensor(s["chunk_of"]),
                   labels=torch.tensor(s["labels"]), pos_weight=torch.tensor(s["pos_weight"], dtype=torch.float32), lang=it["lang"], delay=delay, name=it["name"], id=it["id"],
                   n_text=s["st"].tokens - s["n_sem"] - s["n_turn"], overflow=s["st"].overflow_tokens, n_flush=s["n_flush"], K=K, rounds=s["n_rounds"],
                   n_sem=s["n_sem"], n_turn=s["n_turn"], n_B=s["n_B"], n_N=s["n_N"], n_decision_on_event=s["n_decision_on_event"], sem_in_flush=s["sem_in_flush"], turn_in_flush=s["turn_in_flush"])
        if self.written_frac > 0: out.update(fmt=fmt, n_written=int(fmt == WRITTEN))
        if self.sem_free_unlabeled: out["sem_free"] = bool(it.get("sem_free", False))
        if self.speed_perturb: out["speed"] = f
        if self.offline_frac > 0: out["mode"] = "offline" if offline else "stream"
        return out

    def offline_sequence(self, i: int, K: int, fmt: str = NORMALIZED) -> dict:
        """final(오프라인) 모드 시퀀스(vapasr/data/offline_seq): 오디오 K 청크 전체 → 텍스트 토큰(이벤트 제외) + <|im_end|>. 스트리밍과 같은 키로(이벤트·결정 통계 0)."""
        from types import SimpleNamespace
        it = self.items[i]; ev = self._form(i, fmt)["tokens"]
        text = [x[0] for x in ev if len(x) == 2 and x[0] not in (self.sem_id, self.turn_id)]
        s = self._offline.sequence(text, K, it["lang"])
        s.update(st=SimpleNamespace(tokens=len(text), overflow_tokens=0), n_flush=0, n_rounds=0, n_sem=0, n_turn=0, n_B=0, n_N=0, n_decision_on_event=0,
                 sem_in_flush=0, turn_in_flush=0)
        return s

def collate_semcommit(batch):
    """collate_streams + pos_weight (B, L) float(패딩 0 = 기본 가중) + 이벤트 수 합(+ 표기형 항목 수 n_written — 데이터셋이 낼 때만)."""
    out = collate_streams(batch); B, L = out["ids"].shape; pw = torch.zeros(B, L, dtype=torch.float32)
    for i, b in enumerate(batch): pw[i, : len(b["pos_weight"])] = b["pos_weight"]
    out["pos_weight"] = pw
    for k in ("n_sem", "n_turn", "n_B", "n_N", "n_decision_on_event", "sem_in_flush", "turn_in_flush"): out[k] = sum(int(b.get(k, 0)) for b in batch)
    if "n_written" in batch[0]: out["n_written"] = sum(int(b.get("n_written", 0)) for b in batch)
    if "mode" in batch[0]: out["n_offline"] = sum(int(b.get("mode") == "offline") for b in batch)
    if any(b.get("sem_free") for b in batch):                                                 # SEM 중립 행 → 모델이 그 행의 <SEM_END>·턴 종료 logit 을 뺀다(없으면 키도 없다)
        sm = torch.zeros(B, L, dtype=torch.bool)
        for i, b in enumerate(batch):
            if b.get("sem_free"): sm[i, : len(b["ids"])] = True
        out["sem_mask"] = sm; out["n_sem_free"] = sum(int(bool(b.get("sem_free"))) for b in batch)
    return out

def short_quota_order(main_counts: Dict[str, int], short_counts: Dict[str, int], fraction: float, schedule: str) -> List[str]:
    """One main-pool pass plus a capped, evenly spaced fraction of short batches.

    The cap is by *optimizer steps*, not utterance count or audio hours. Main
    corpora keep their normal epoch length; a large short pool cannot dominate
    merely because it has millions of brief utterances.
    """
    from ..hf.data import proportional_order
    if not 0 <= fraction < 0.5:
        raise ValueError("short batch fraction must be in [0, 0.5)")
    main_names = [k for k, n in main_counts.items() if n > 0]
    short_names = [k for k, n in short_counts.items() if n > 0]
    main_n = sum(main_counts.values())
    if not main_names or (fraction > 0 and not short_names):
        raise ValueError("main and requested short pools must be nonempty")
    if schedule not in ("balanced", "proportional"):
        raise ValueError(f"unknown schedule: {schedule}")
    main_order = (proportional_order(main_counts) if schedule == "proportional"
                  else [main_names[i % len(main_names)] for i in range(main_n)])
    if fraction == 0:
        return main_order
    short_n = max(1, round(main_n * fraction / (1 - fraction)))
    if schedule == "balanced":
        short_order = [short_names[i % len(short_names)] for i in range(short_n)]
    else:
        total = sum(short_counts.values())
        weight = {k: short_n * n / total for k, n in short_counts.items() if n > 0}
        quota = {k: int(v) for k, v in weight.items()}
        for k in sorted(weight, key=lambda x: (-(weight[x] - quota[x]), x))[:short_n - sum(quota.values())]:
            quota[k] += 1
        short_order = proportional_order(quota)
    total_n = main_n + short_n
    short_at = {int((i + 0.5) * total_n / short_n) for i in range(short_n)}
    result, im, ishort = [], 0, 0
    for pos in range(total_n):
        if pos in short_at:
            result.append(short_order[ishort]); ishort += 1
        else:
            result.append(main_order[im]); im += 1
    assert im == main_n and ishort == short_n
    return result


def semcommit_round_robin(datasets: Dict[str, "SemCommitDataset"], bs_of, seed: int = 0, num_workers: int = 4, schedule: str = "balanced", rank: int = 0, world: int = 1,
                          short_fraction: float = 0.0, sampler_factory=None):
    """vapasr.hf.data.RoundRobinLoader(언어별 버킷 배치·step 교대)와 같고 collate 만 collate_semcommit(pos_weight 포함). 버킷 길이 = 패딩된 K'.
    sampler_factory(vapasr.hf.packing.packing_sampler_factory) 를 주면 셋마다 토큰 예산 배치 + packed collate(한 줄로 이어 붙임)."""
    from ..hf.data import RoundRobinLoader
    class QuotaLoader(RoundRobinLoader):
        def __len__(self):
            return len(self._quota_order())

        def _quota_order(self):
            main = {k: len(v) for k, v in self.samplers.items() if not k.startswith("short_")}
            short = {k: len(v) for k, v in self.samplers.items() if k.startswith("short_")}
            return short_quota_order(main, short, short_fraction, self.schedule)

        def __iter__(self):
            its = {name: self._cycle(name, self.epoch) for name in self.names}
            for name in self._quota_order():
                yield next(its[name])

    return QuotaLoader(datasets, bs_of, seed=seed, rank=rank, world=world, num_workers=num_workers, schedule=schedule,
                       sampler_factory=sampler_factory, collate_fn=collate_semcommit)
