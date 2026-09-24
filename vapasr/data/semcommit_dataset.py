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
M 이월, flush)을 재현하면서 명시 청크 이벤트와 방출 출처(src)를 더한다 — 이벤트가 없으면 build_interleaved 와 출력·통계가 같다(tests/test_semcommit_dataset.py 가 고정)."""
import hashlib, json, math, os, random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np, torch
from torch.utils.data import Dataset
from .interleave import InterleaveStats, Specials
from .semcommit_tokens import SEM_SPECIALS, BASE_VOCAB, TURN_TOKEN, add_semantic_specials, assert_frozen_semantic
from ..uslm.interleave_data import specials_of, CHUNK_S
from ..uslm.mono_data import build_mono_sequence, collate_streams

SR = 16000

def check_sem_ids(ids: Dict[str, int]) -> None:
    """SEM 이 Phase 2 registry 를 밀어내지 않았는지(가짜 tokenizer 에도 적용되는 구조 검사): SEM 이 모든 Phase 1/2 id 보다 뒤, 턴 종료 <EOT> 는 Phase 2 registry 안.
    실제 Qwen tokenizer(<NEXT_AUDIO> = BASE_VOCAB)면 semcommit_tokens.assert_frozen_semantic 로 동결 id(<EOT> 151722, <SEM_END> 151723 등)까지 대조한다."""
    s = ids["<SEM_END>"]
    assert s > max(v for k, v in ids.items() if k not in SEM_SPECIALS) and TURN_TOKEN in ids and ids[TURN_TOKEN] < s, f"SEM/턴 종료 id 순서 위반: {ids}"
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
      다르면 ValueError(words.jsonl 의 토큰 id 가 이 tokenizer 에서 다른 단어가 된다). tokenizer 에 convert_ids_to_tokens 가 없으면(가짜) 건너뛰고 stats['tok_check_skipped']."""
    def __init__(self, words_jsonl, labels_jsonl, tok, sp_ids: Optional[Dict[str, int]] = None, delays=(2, 3, 4, 6), turn_end: bool = False,
                 hangover_s: float = 0.48, hardneg_weight: float = 1.0, max_items: Optional[int] = None, seed: int = 0, online: bool = True,
                 allow_unlabeled: bool = False, max_per_chunk: int = 0, langs: Optional[Sequence[str]] = None, path_map: Optional[Dict[str, str]] = None,
                 tail_margin: int = 2, pad_tail: bool = True, max_audio_retries: int = 8, tok_check: int = 32):
        self.tok = tok; self.sp_ids = dict(sp_ids) if sp_ids is not None else add_semcommit_specials(tok); check_sem_ids(self.sp_ids)
        self.sp = specials_of(self.sp_ids); self.sem_id, self.turn_id = self.sp_ids["<SEM_END>"], self.sp_ids[TURN_TOKEN]
        self.delays = tuple(int(d) for d in delays); self.M = max_per_chunk; self.online = online; self.turn_end = turn_end; self.hangover_s = hangover_s
        self.hardneg_weight = hardneg_weight; self.max_audio_retries = max_audio_retries
        self.audio_pad = tok.convert_tokens_to_ids("<|audio_pad|>"); self._pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        assert isinstance(self.audio_pad, int) and self.audio_pad >= 0, "tokenizer 에 <|audio_pad|> 없음(Qwen3-ASR tokenizer 필요)"
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
                lab = labels.get(r["id"]); self.stats["words_rows"] += 1; self.stats["labeled"] += lab is not None
                if lab is None and not allow_unlabeled: self.stats["skipped_unlabeled"] += 1; continue
                if lab is not None and lab.get("lang") not in (None, r["lang"]): self.bad["lang_mismatch"] += 1; continue
                if not r.get("words"): self.bad["no_words"] += 1; continue
                try: ev, marks = build_semcommit_tokens(r["words"], r["tokens"], lab, self.sem_id, self.turn_id, turn_end=turn_end, hangover_s=hangover_s)
                except ValueError as e: self.bad[str(e).split(":")[0]] += 1; continue          # 메시지 앞머리 = 사유 코드
                except KeyError as e: self.bad[f"missing_key:{e}"] += 1; continue
                if can_split and self.stats["tok_check"] < tok_check:                          # tokenizer 내용 관문(계약을 통과한 행만; 빌더와 같은 분할)
                    from .semcommit_words import split_words
                    self.stats["tok_check"] += 1
                    try: ws = split_words([int(t[0]) for t in r["tokens"]], [float(t[1]) for t in r["tokens"]], tok)
                    except Exception as e: ws = f"{type(e).__name__}: {e}"
                    want = [(w["text"], int(w["a"]), int(w["b"])) for w in r["words"]]
                    if not isinstance(ws, list) or [(w["text"], w["a"], w["b"]) for w in ws] != want:
                        tok_bad.append((r["id"], " ".join(w[0] for w in want[:8]), ws if not isinstance(ws, list) else " ".join(w["text"] for w in ws[:8])))
                K0 = int(r["K"]); has_turn = any(len(x) > 2 for x in ev)
                need = max(last_emit_chunk(ev, d, self.sp, max_per_chunk) for d in self.delays) + tail_margin if (has_turn or pad_tail) else 0
                K = max(K0, need); dur = float(r["duration_s"]) if K == K0 else K * CHUNK_S
                segs = [dict(s) for s in r["segments"]]
                for s in segs:
                    for old, new in (path_map or {}).items():
                        if s["path"].startswith(old): s["path"] = new + s["path"][len(old):]; break
                n_sem = sum(1 for x in ev if len(x) == 2 and x[0] == self.sem_id)
                self.items.append(dict(name=r.get("set", "semcommit"), id=r["id"], lang=r["lang"], K=K, K0=K0, duration_s=dur, duration0=float(r["duration_s"]), segments=segs,
                                       tokens=ev, marks=marks, n_tok=len(r["tokens"]), n_sem=n_sem, n_turn=int(has_turn), text=r.get("text", "")))
        if tok_bad: raise ValueError(f"tokenizer_mismatch: {len(tok_bad)}/{self.stats['tok_check']} 행의 토큰 id 가 이 tokenizer 에서 다른 단어가 된다(words.jsonl 을 만든 tokenizer 와 다름) 예: {tok_bad[:2]}")
        if not can_split: self.stats["tok_check_skipped"] = 1
        self.stats["labels_without_words"] = sum(1 for k in labels if k not in seen)
        random.Random(seed).shuffle(self.items)
        if max_items: self.items = self.items[:max_items]
        for i, it in enumerate(self.items):                                                   # 통계는 남은 항목 기준
            self.stats["padded"] += int(it["K"] > it["K0"]); self.stats["pad_chunks"] += it["K"] - it["K0"]; self.stats["sem"] += it["n_sem"]; self.stats["turn"] += it["n_turn"]
            self.stats["B"] += sum(g == "B" for _, g in it["marks"]); self.stats["N"] += sum(g == "N" for _, g in it["marks"]); self.stats["words_tokens"] += it["n_tok"]
            ev = it["tokens"]
            if any(j + 1 < len(ev) and len(ev[j + 1]) > 2 for j, _ in it["marks"]):          # B/N 원소 바로 뒤가 TURN → δ 에 따라 결정 위치 = TURN 타깃(정확한 수는 시퀀스로)
                n = sum(self.sequence(i, d)["n_decision_on_event"] for d in self.delays)
                self.stats["decision_on_event"] += n; self.stats["decision_on_event_items"] += int(n > 0)
        self.stats["items"] = len(self.items)

    def __len__(self): return len(self.items)
    def prefix(self, lang: str, delay: int) -> List[int]:
        """MonoStreamDataset.prefix 와 같다: system/assistant 머리 + 'language {Lang}<asr_text>' + <DELAY_d>. (lang, δ) 별로 한 번만 토큰화."""
        c = self.__dict__.setdefault("_prefix_cache", {})
        if (lang, delay) not in c: c[(lang, delay)] = self._pre + self.tok(f"language {lang}<asr_text>", add_special_tokens=False)["input_ids"] + [self.sp_ids[f"<DELAY_{delay}>"]]
        return list(c[(lang, delay)])
    def sequence(self, i: int, delay: int) -> dict:
        it = self.items[i]
        return build_semcommit_sequence(it["tokens"], it["marks"], it["K"], self.prefix(it["lang"], delay), self.audio_pad, self.sp, delay, self.M,
                                        hardneg_weight=self.hardneg_weight, sem_id=self.sem_id, turn_id=self.turn_id)
    def audio(self, it: dict) -> np.ndarray:
        """(T,) float32 @16 kHz, T = round(duration_s·SR) — 패딩된 길이까지 assemble_stream 이 끝 무음을 합성한다(마지막 발화 뒤 20 ms 배경을 늘림)."""
        if not self.online: return np.zeros(int(round(it["duration_s"] * SR)), np.float32)
        from .streams import assemble_stream
        return assemble_stream(dict(id=it["id"], duration_s=it["duration_s"], segments=it["segments"]))
    def target_problems(self, i: int, delay: int) -> List[str]:
        """학습 전 시퀀스 불변식 위반 목록(빈 목록 = 통과):
        text_order(라벨 위치 텍스트 ≠ 참조 토큰열) · event_count(SEM/TURN 수) · turn_in_flush · sem_in_flush(패딩한 항목) ·
        sem_detached(SEM 이 앞 원소 = 단어 마지막 토큰 바로 뒤가 아님 — 사이에 NEXT/오디오, 즉 다른 청크; M>0 에서 생긴다) ·
        event_target_altered(SEM/TURN 위치의 라벨이 가려졌거나 pos_weight 로 덮임) · decision(B 결정 위치가 안 가려짐 / N 가중 ≠ hardneg / 이벤트 위 결정이 '@' 로 안 남음)."""
        s = self.sequence(i, delay); it = self.items[i]; ev = it["tokens"]; ids, lab, pw, pos_of = s["ids"], s["labels"], s["pos_weight"], s["pos_of"]
        skip = {self.sp.next_audio, self.sp.empty_audio, self.sem_id, self.turn_id}; P = []
        got = [t for t, inp in zip(ids, s["is_input"]) if not inp and t not in skip]
        if got != [x[0] for x in ev if x[0] not in (self.sem_id, self.turn_id)]: P.append("text_order")
        if (s["n_sem"], s["n_turn"]) != (it["n_sem"], it["n_turn"]): P.append(f"event_count: sem {s['n_sem']}/{it['n_sem']} turn {s['n_turn']}/{it['n_turn']}")
        if s["turn_in_flush"]: P.append("turn_in_flush")
        if s["sem_in_flush"] and it["K"] > it["K0"]: P.append("sem_in_flush")
        P += [f"sem_detached@{j}" for j, x in enumerate(ev) if len(x) == 2 and x[0] == self.sem_id and not (j > 0 and pos_of[j] == pos_of[j - 1] + 1)]
        P += [f"event_target_altered@{p}" for p in s["sem_pos"] + s["turn_pos"] if lab[p] != ids[p] or pw[p]]
        for p, g in s["decision"]:
            on_ev = ids[p] in (self.sem_id, self.turn_id)
            if on_ev != ("@" in g) or (g == "B" and lab[p] != -100) or (g == "N" and pw[p] != float(self.hardneg_weight)): P.append(f"decision@{p}:{g}")
        return P
    def check_targets(self, i: int, delay: int) -> bool:
        """target_problems 가 비었는가(학습 스크립트의 시작 전 관문)."""
        return not self.target_problems(i, delay)
    def __getitem__(self, i):
        delay = random.choice(self.delays); err = None
        for r in range(self.max_audio_retries):                                          # 오디오 조립 실패 → 이웃 항목(재귀 대신 유한 반복)
            j = (i + r) % len(self.items); it = self.items[j]
            try: wav = self.audio(it); break
            except (ValueError, OSError, RuntimeError) as e:
                err = e; self._audio_fail = getattr(self, "_audio_fail", 0) + 1
                if self._audio_fail <= 5: print(f"  ! 항목 {it['id']} 오디오 조립 실패({type(e).__name__}: {str(e)[:120]}) → 이웃 항목 (누적 {self._audio_fail})", flush=True)
        else: raise RuntimeError(f"오디오 조립 {self.max_audio_retries} 회 연속 실패: {err}")
        s = self.sequence(j, delay)
        return dict(wav=torch.from_numpy(wav), ids=torch.tensor(s["ids"]), is_audio=torch.tensor([c >= 0 for c in s["chunk_of"]]), chunk_of=torch.tensor(s["chunk_of"]),
                    labels=torch.tensor(s["labels"]), pos_weight=torch.tensor(s["pos_weight"], dtype=torch.float32), lang=it["lang"], delay=delay, name=it["name"], id=it["id"],
                    n_text=s["st"].tokens - s["n_sem"] - s["n_turn"], overflow=s["st"].overflow_tokens, n_flush=s["n_flush"], K=it["K"], rounds=s["n_rounds"],
                    n_sem=s["n_sem"], n_turn=s["n_turn"], n_B=s["n_B"], n_N=s["n_N"], n_decision_on_event=s["n_decision_on_event"], sem_in_flush=s["sem_in_flush"], turn_in_flush=s["turn_in_flush"])

def collate_semcommit(batch):
    """collate_streams + pos_weight (B, L) float(패딩 0 = 기본 가중) + 이벤트 수 합."""
    out = collate_streams(batch); B, L = out["ids"].shape; pw = torch.zeros(B, L, dtype=torch.float32)
    for i, b in enumerate(batch): pw[i, : len(b["pos_weight"])] = b["pos_weight"]
    out["pos_weight"] = pw
    for k in ("n_sem", "n_turn", "n_B", "n_N", "n_decision_on_event", "sem_in_flush", "turn_in_flush"): out[k] = sum(int(b.get(k, 0)) for b in batch)
    return out

def semcommit_round_robin(datasets: Dict[str, "SemCommitDataset"], bs_of, seed: int = 0, num_workers: int = 4, schedule: str = "balanced", rank: int = 0, world: int = 1):
    """vapasr.hf.data.RoundRobinLoader(언어별 버킷 배치·step 교대)와 같고 collate 만 collate_semcommit(pos_weight 포함). 버킷 길이 = 패딩된 K'."""
    from torch.utils.data import DataLoader
    from ..hf.data import RoundRobinLoader
    rr = RoundRobinLoader(datasets, bs_of, seed=seed, rank=rank, world=world, num_workers=num_workers, schedule=schedule)
    rr.loaders = {m: DataLoader(ds, batch_sampler=rr.samplers[m], num_workers=num_workers, collate_fn=collate_semcommit, persistent_workers=False) for m, ds in datasets.items()}
    return rr
