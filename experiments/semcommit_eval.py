#!/usr/bin/env python3
"""Semantic commit(<SEM_END>/<TURN_END>) 스트리밍 디코드·평가 — 계획 §24 v0 (지표: vapasr/hf/commit_metrics.py).

words.jsonl 세그먼트로 스트림 오디오를 온라인 조립(vapasr.data.streams.assemble_stream)하되 학습(SemCommitDataset)과 같은 길이·청크 수로 끝 무음을 늘리고
(K' = max(K, max_δ∈학습 δ∪{평가 δ} 마지막 방출 청크 + tail_margin), TURN 이면 k_turn 포함 — commit_metrics.eval_audio_len; 안 늘린 행은 words K 를 그대로 인코더에),
Nemotron 특징은 스트림당 1 회 인코드한 뒤, 디코드 설정(행) = 스트림 × {sem-bias b | threshold θ} 를 dense KV 배치로 한꺼번에 greedy 디코드한다.
학습 설정 복원: --turn-end/--hangover-s/--tail-margin/--pad-delays 를 안 주면 체크포인트 config.json 의 semcommit(학습 fingerprint)·delays,
  없으면 run 디렉토리 run.json/results.json 의 semcommit_train 인자에서 가져온다(모르면 학습 기본: TURN 켬·0.48 s·2·2 3 4 6). 준 값이 학습과 다르면 멈춘다(--allow-train-mismatch).
  평가 δ 가 학습 δ 목록에 없으면 멈춘다(--allow-untrained-delay).
  bias 모드      : logits[<SEM_END>] += b 후 argmax (b 스윕 = PR 곡선)
  threshold 모드 : p(<SEM_END>) ≥ θ 이면 <SEM_END>, 아니면 <SEM_END> 를 뺀 argmax
  둘 다 매 결정 step 의 p(<SEM_END>)·p(<TURN_END>)(blocked 제외 softmax, 편향 전)를 기록(trace)한다.
  sem-guard(기본 켬): 직전 <SEM_END> 이후(또는 스트림 시작 이후) 텍스트 토큰이 없으면 <SEM_END> 금지 — 빈 commit·SEM 폭주 방지(학습 시퀀스에 없는 패턴).
    가드가 막은 발화(bias 모드: 가드 전 argmax=SEM, threshold 모드: p(SEM) ≥ θ)는 행마다 guard_blocked 로 세고, 보고서 text.pcr_raw 가 조기 commit 으로 센다.
디코드 규약(청크당 오디오 임베딩 1 개 → greedy → <NEXT_AUDIO> 로 청크 종료, runaway_cap, <EMPTY_AUDIO> flush)은 batch_decode/stream_decode 와 같다
(--verify: bias 0·guard 끔 행을 model.stream_decode 와 토큰 단위 대조).
출력: --out report.json(설정·언어·세트별 지표, PR 곡선) + 스트림별 jsonl(방출·이벤트·가설 단어·trace·지표; 이어하기 가능).
이어하기 지문(<streams>.config.json): 디코드 설정 + 체크포인트 가중 파일 크기·mtime·config.json sha256 + words 파일 sha256 + 코드 sha256 — 하나라도 다르면 멈춘다.
보고서는 항상 스트림 jsonl 의 방출에서 현재 labels 로 다시 채점한다(--score-only 는 GPU 없이 재채점만). 재채점 전에 행마다 words 행 지문(words_digest)과
  현재 labels 로 다시 계산한 K' 가 기록과 같은지 확인한다(다르면 다른 오디오로 디코드한 방출 — 멈춘다).
--oracle: 모델 대신 참조 직렬화를 가설로 — 지표 상한(정밀도·재현율 1, PCR 0) 점검.

  python experiments/semcommit_eval.py --model /data3/tskim/runs/semcommit-v0/final --words eval.words.jsonl --labels eval.labels.jsonl \\
      --delay 4 --sem-bias -2 -1 0 1 2 --theta 0.2 0.35 0.5 --gpu 0 --max-streams 300 --out /data3/tskim/eval/semcommit-v0/report.json
"""
import argparse, gzip, hashlib, itertools, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __name__ == "__main__" and "--gpu" in sys.argv[:-1]:                          # vapasr.hf 가 torch 를 import 하기 전에 GPU 를 고정(s3_train_hf 와 같은 방식)
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.hf.commit_metrics import (CHUNK_S, EARLY, HANGOVER_S, LATES, SEM_END_ID, SEM_SPECIALS, TURN_END_ID, aggregate, events_from_emits,
                                      eval_audio_len, oracle_hyp, pr_point, score_stream, split_hyp, turn_flag)

PROTOCOL = "semcommit-eval-v1"
ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ("experiments/semcommit_eval.py", "vapasr/hf/commit_metrics.py", "vapasr/hf/batch_decode.py", "vapasr/hf/modeling_vapasr.py", "vapasr/data/streams.py")
MXC_PREFIX = "/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/"
DEFAULT_PAD_DELAYS = (2, 3, 4, 6)                                                 # experiments/semcommit_train.py --delays 기본(모델이 있으면 config.delays 를 쓴다)
TRAIN_DEFAULTS = dict(turn_end=True, hangover_s=HANGOVER_S, tail_margin=2)          # semcommit_train 기본(--no-turn-end 없음·--hangover 0.48·--tail-margin 2)

# ───────────────────────────── I/O ─────────────────────────────
def read_jsonl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); tmp.replace(path)

def remap_path(p, remaps):
    for old, new in remaps:
        if p.startswith(old): return new + p[len(old):]
    return p

def parse_remaps(items):
    out = []
    for s in items or []:
        assert "=" in s, f"--path-remap OLD=NEW: {s}"; out.append(tuple(s.split("=", 1)))
    return out

def eval_len(wrow, lrow, dc):
    """평가 오디오 (길이 s, 청크 수 K') = 학습 값(commit_metrics.eval_audio_len; TURN 포함 = turn_end ∧ labels.turn_end(없으면 True), SemCommitDataset 과 같다).
    dc = decode_config(디코드 때 저장된 설정) — 디코드와 재채점 검사가 같은 식을 쓴다. no_pad 면 (원래 길이, words K)."""
    K0 = int(wrow.get("K") or round(float(wrow["duration_s"]) / CHUNK_S))
    if dc.get("no_pad"): return float(wrow["duration_s"]), K0
    has_turn = bool(dc["turn_end"] and turn_flag(lrow) and wrow.get("words"))
    return eval_audio_len(wrow, dc["pad_delays"], has_turn, dc["hangover_s"], dc["tail_margin"], dc.get("pad_tail", True))

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def row_digest(wrow):
    """디코드 입력을 정하는 words 행 내용(오디오 조립 segments·길이·K·토큰·단어 경계/끝 시각)의 지문 — 이어하기·재채점이 다른 words 로 만든 방출을 섞지 않게."""
    key = dict(K=wrow.get("K"), duration_s=wrow.get("duration_s"), segments=wrow.get("segments"), tokens=wrow.get("tokens"),
               words=[[w.get("i"), w.get("a"), w.get("b"), w.get("end_time")] for w in wrow.get("words") or []])
    return hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]

def checkpoint_fingerprint(path):
    """eval_single_turn_asr 와 같은 체크포인트 지문: *.safetensors 크기·mtime_ns + config.json sha256(없으면 None)."""
    if not path: return dict(checkpoint_weights=None, checkpoint_config_sha256=None)
    ck = Path(path); cfg = ck / "config.json"
    return dict(checkpoint_weights={q.name: dict(size=q.stat().st_size, mtime_ns=q.stat().st_mtime_ns) for q in sorted(ck.glob("*.safetensors"))},
                checkpoint_config_sha256=sha256_file(cfg) if cfg.exists() else None)

def _load_json(q):
    try: return json.loads(Path(q).read_text(encoding="utf-8"))
    except (OSError, ValueError): return None

def train_params(model_path):
    """학습이 쓴 패딩·TURN 설정 → dict(turn_end, hangover_s, tail_margin, pad_tail, M, delays 중 아는 것, src=[출처]).
    낮은 우선 → 높은 우선: run 디렉토리(model/ 또는 model/..)의 run.json·results.json args(semcommit_train 인자: no_turn_end·hangover·tail_margin·M·delays)
    → config.json delays → config.json semcommit(학습 fingerprint, semcommit_dataset.semcommit_fingerprint: turn_end·hangover_s·tail_margin·pad_tail·M·delays)."""
    if not model_path: return {}
    p = Path(model_path); out, src = {}, []
    for q in (p / "run.json", p / "results.json", p.parent / "run.json", p.parent / "results.json"):
        args = (_load_json(q) or {}).get("args") if q.exists() else None
        if not isinstance(args, dict): continue
        if "no_turn_end" in args: out["turn_end"] = not args["no_turn_end"]
        for k, v in (("hangover", "hangover_s"), ("tail_margin", "tail_margin"), ("M", "M")):
            if args.get(k) is not None: out[v] = args[k]
        if args.get("delays"): out["delays"] = [int(x) for x in (args["delays"].split(",") if isinstance(args["delays"], str) else args["delays"]) if str(x).strip()]
        src.append(str(q)); break
    cfg = _load_json(p / "config.json") or {}
    if cfg.get("delays"): out["delays"] = [int(d) for d in cfg["delays"]]; src.append(str(p / "config.json"))
    sc = cfg.get("semcommit")
    if isinstance(sc, dict) and sc:
        for k, v in (("turn_end", "turn_end"), ("hangover", "hangover_s"), ("hangover_s", "hangover_s"), ("tail_margin", "tail_margin"), ("pad_tail", "pad_tail"),
                     ("max_per_chunk", "M"), ("M", "M"), ("delays", "delays")):
            if sc.get(k) is not None: out[v] = sc[k]
        src.append(str(p / "config.json") + ":semcommit")
    for k, f in (("turn_end", bool), ("pad_tail", bool), ("hangover_s", float), ("tail_margin", int), ("M", int)):
        if k in out: out[k] = f(out[k])
    if "delays" in out: out["delays"] = sorted(int(d) for d in out["delays"])
    if out: out["src"] = list(dict.fromkeys(src))
    return out

def resolve_settings(a):
    """--turn-end/--hangover-s/--tail-margin/--pad-delays 미지정 → 학습 설정(train_params) → 없으면 TRAIN_DEFAULTS. 지정했는데 학습과 다르면 멈춘다(--allow-train-mismatch).
    패딩 δ = (--pad-delays | 학습 δ | 2 3 4 6) ∪ {평가 δ}. 평가 δ ∉ 학습 δ 면 멈춘다(--allow-untrained-delay; 학습 δ 를 모르면 경고만). → a.turn_end 등을 채우고 a.train 에 출처 기록."""
    tp = train_params(a.model) if (a.model and not a.oracle) else {}
    a.train = tp; bad = []
    if a.model and not a.oracle and not tp: print(f"주의: 학습 설정을 못 찾음({a.model}: config.json semcommit/delays·run.json 없음) — 학습 기본 {TRAIN_DEFAULTS} 로 패딩·TURN", flush=True)
    for k, default in TRAIN_DEFAULTS.items():
        v, t = getattr(a, k), tp.get(k)
        if v is None: setattr(a, k, default if t is None else t)
        elif t is not None and v != t: bad.append(f"{k}: 학습 {t} ≠ 평가 {v}")
    a.pad_tail = bool(tp.get("pad_tail", True))
    if int(tp.get("M") or 0): bad.append(f"M: 학습 {tp['M']} — M>0 패딩은 미구현(train_pad_K 는 M=0 규칙)")
    tr = tp.get("delays")
    if a.pad_delays and tr and sorted(a.pad_delays) != tr: bad.append(f"pad_delays: 학습 {tr} ≠ --pad-delays {sorted(a.pad_delays)}")
    if bad:
        msg = "평가 설정이 학습과 다름(패딩·TURN 이 학습 시퀀스와 달라진다): " + "; ".join(bad) + f" [출처 {tp.get('src')}]"
        assert a.allow_train_mismatch, msg + " — 의도했다면 --allow-train-mismatch"; print("주의: " + msg, flush=True)
    base = [int(d) for d in (a.pad_delays or tr or DEFAULT_PAD_DELAYS)]
    if not a.oracle and a.delay not in base:
        msg = f"평가 δ={a.delay} 가 학습 δ {base} 에 없음" + ("" if tr else " (학습 δ 를 모름 — 기본값 기준)")
        assert a.allow_untrained_delay or not tr, msg + " — 의도했다면 --allow-untrained-delay"; print("주의: " + msg, flush=True)
    a.pad = sorted(set(base) | {int(a.delay)})
    return a

def load_stream_audio(wrow, duration_s, remaps):
    from vapasr.data.streams import assemble_stream                              # 무음은 인접 20 ms 배경 연장(학습 조립과 같은 함수·같은 seed)
    segs = [dict(s, path=remap_path(s["path"], remaps)) for s in wrow["segments"]]
    return assemble_stream(dict(id=wrow["id"], duration_s=duration_s, segments=segs))

def select_streams(words, labels, langs=None, max_streams=0):
    """words ∩ labels (파일 순서), 언어 필터, --max-streams 는 언어별 상한(파일 순서에서 고르게 뽑음 — 결정적)."""
    ids = [i for i in words if i in labels and (not langs or words[i]["lang"] in langs)]
    info = dict(words=len(words), labels=len(labels), words_without_labels=sum(i not in labels for i in words), labels_without_words=sum(i not in words for i in labels))
    if max_streams:
        keep = set()
        for L in sorted({words[i]["lang"] for i in ids}):
            li = [i for i in ids if words[i]["lang"] == L]; n = min(len(li), max_streams)
            keep |= {li[int(round(q * (len(li) - 1) / max(1, n - 1)))] for q in range(n)} if n > 1 else set(li[:n])
        ids = [i for i in ids if i in keep]
    info["selected"] = len(ids); return ids, info

def make_configs(sem_bias, thetas):
    return [dict(name=f"bias={float(b):g}", mode="bias", value=float(b)) for b in (sem_bias or [])] + [dict(name=f"theta={float(t):g}", mode="threshold", value=float(t)) for t in (thetas or [])]

# ───────────────────────────── 디코드 ─────────────────────────────
class RowState:
    """batch_decode.DecodeState 와 같은 청크 진행 규약 + sem-guard 용 상태(guard_blocked = 가드가 막은 SEM 발화 수) + p(SEM)/p(TURN) trace."""
    def __init__(self, K, cap, max_flush, max_total):
        self.K, self.cap, self.max_flush, self.max_total = K, cap, max_flush, max_total
        self.k = self.n = self.forced = self.flush_rounds = self.guard_blocked = 0; self.advance = self.done = False; self.emitted = []; self.text_since_sem = False; self.trace = []
    @property
    def deciding(self): return not self.done and not self.advance
    @property
    def free(self):
        """이번 step 의 토큰이 실제로 방출될 수 있음(결정 step 이고 runaway_cap·max_total 강제 NEXT 가 아님)."""
        return self.deciding and self.n < self.cap and len(self.emitted) < self.max_total
    def consume(self, tid, next_id):
        """→ (다음 입력 종류, 값): 'audio' k · 'empty' · 'next' · 'text' tid (DecodeState.consume 과 동일)."""
        if self.done: return "next", 0
        if self.advance:
            self.k += 1; self.n = 0; self.advance = False
            return ("audio", self.k) if self.k < self.K else ("empty", 0)
        if tid == next_id or self.n >= self.cap or len(self.emitted) >= self.max_total:
            self.forced += int(tid != next_id)
            if self.k >= self.K:
                self.flush_rounds += 1; self.done = self.n == 0 or self.flush_rounds >= self.max_flush
            elif self.k == self.K - 1 and self.max_flush == 0: self.done = True
            self.advance = True; return "next", 0
        self.emitted.append((self.k, tid)); self.n += 1; return "text", tid

def decode_rows(model, feats, prefix, policies, *, sem_id=SEM_END_ID, turn_id=TURN_END_ID, next_bias=0.0, turn_bias=0.0, max_flush_rounds=None,
                sem_guard=True, trace=True, cache=None):
    """feats: 행별 (1,K,D) 인코더 출력(같은 스트림을 여러 행이 공유 가능), prefix: 공통 prefix id(한 언어·δ), policies: 행별 dict(mode='bias'|'threshold', value).
    → 행별 RowState(emitted [(k, tid)], forced, flush_rounds, guard_blocked, trace [(k, p_sem, p_turn)]).
    guard_blocked: 가드가 켜진 행에서 가드가 없었다면 SEM 을 냈을 step 수(bias: 편향 뒤 argmax=SEM, threshold: p(SEM) ≥ θ) — 방출은 다음 순위 토큰."""
    import torch
    with torch.inference_mode():
        emb = model.get_input_embeddings(); dev = emb.weight.device; B = len(feats)
        ce = [model.chunk_embed(f[None])[0].to(emb.weight.dtype) for f in feats]
        flush = model.config.max_flush_rounds if max_flush_rounds is None else max_flush_rounds
        states = [RowState(len(c), model.config.runaway_cap, flush, int(6 * len(c)) + 64) for c in ce]; assert all(s.K > 0 for s in states)
        if cache is None:
            from transformers import DynamicCache; cache = DynamicCache()
        bias = torch.tensor([p["value"] if p["mode"] == "bias" else 0.0 for p in policies], device=dev)
        theta = torch.tensor([p["value"] if p["mode"] == "threshold" else float("nan") for p in policies], device=dev); thr = ~torch.isnan(theta); any_thr = bool(thr.any())
        model.thinker.model(inputs_embeds=emb(torch.tensor([list(prefix)] * B, device=dev)), past_key_values=cache, use_cache=True)
        x = torch.stack([c[0] for c in ce])[:, None]
        while not all(s.done for s in states):
            h = model.thinker.model(inputs_embeds=x, past_key_values=cache, use_cache=True).last_hidden_state
            logits = model.thinker.lm_head(h[:, -1]).float()
            logits[:, model.blocked] = -torch.inf; logits[:, model.next_audio] -= next_bias
            lp = torch.log_softmax(logits, -1); p_sem = lp[:, sem_id].exp(); p_turn = lp[:, turn_id].exp()
            logits[:, sem_id] += bias; logits[:, turn_id] += turn_bias
            if sem_guard:
                g = torch.tensor([not s.text_since_sem for s in states], device=dev)
                gblk = (g & torch.where(thr, p_sem >= theta, logits.argmax(-1) == sem_id)).tolist(); logits[g, sem_id] = -torch.inf
            else: g = torch.zeros(B, dtype=torch.bool, device=dev); gblk = [False] * B
            tid = logits.argmax(-1)
            if any_thr:
                fire = thr & (p_sem >= theta) & ~g; l2 = logits.clone(); l2[:, sem_id] = -torch.inf
                tid = torch.where(thr, torch.where(fire, torch.full_like(tid, sem_id), l2.argmax(-1)), tid)
            tids = tid.tolist(); ps = p_sem.tolist(); pt = p_turn.tolist(); nxt = []
            for i, (s, t) in enumerate(zip(states, tids)):
                if trace and s.deciding: s.trace.append((s.k, ps[i], pt[i]))
                if gblk[i] and s.free: s.guard_blocked += 1
                kind, value = s.consume(t, model.next_audio)
                if kind == "text": s.text_since_sem = False if value == sem_id else (s.text_since_sem or value != turn_id)
                if kind == "audio": nxt.append(ce[i][value])
                else: nxt.append(emb.weight[{"next": model.next_audio, "empty": model.empty_audio}.get(kind, value)])
            x = torch.stack(nxt)[:, None]
    return states

def tokenizer_fns(tok, sem_id, turn_id):
    """split_hyp 에 넘길 (is_text, word_start, decode). 텍스트 = 특수/추가 토큰이 아닌 id, 단어 시작 = byte-BPE 조각이 공백(Ġ)으로 시작."""
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}) or {}) | {sem_id, turn_id}
    cache = {}
    def piece(t):
        if t not in cache: cache[t] = tok.convert_ids_to_tokens(int(t)) or ""
        return cache[t]
    return (lambda t: t not in special), (lambda t: piece(t)[:1] in ("Ġ", " ", "▁")), (lambda ids: tok.decode(ids, skip_special_tokens=True))

def hyp_from_emitted(emitted, tok, sem_id, turn_id):
    import re
    is_text, word_start, decode = tokenizer_fns(tok, sem_id, turn_id)
    h = split_hyp(emitted, is_text=is_text, word_start=word_start, decode=decode, event_ids=(sem_id, turn_id))
    h["text"] = re.sub(r"\s+", " ", decode([t for _, t in emitted if is_text(t)])).strip()        # 이벤트·특수 토큰은 디코드 전에 뺀다
    return h

# ───────────────────────────── 모델·토큰 ─────────────────────────────
def load_sem_model(path, encoder, dtype, device):
    """infer.load_model 과 같은 로드. tokenizer 는 load_tokenizer 가 SEM 토큰을 모르면(sp_ids 불일치) Phase1+Phase2+SEM 순서로 직접 붙인다."""
    from transformers import AutoTokenizer
    from vapasr.hf import VapAsrForStreamingASR, load_tokenizer
    from vapasr.hf.stage import stage_dir
    enc = encoder or os.environ.get("MXC_NEMOTRON_DIR", os.environ.get("NEMOTRON_DIR", ""))
    model = VapAsrForStreamingASR.from_pretrained(stage_dir(path), encoder_path=stage_dir(enc) if enc else None)
    try: tok = load_tokenizer(path)
    except AssertionError as e:
        print(f"load_tokenizer: {str(e)[:200]} → tokenizer 직접 구성", flush=True)
        cfg = json.load(open(os.path.join(path, "config.json")))
        tok = AutoTokenizer.from_pretrained(path if os.path.exists(os.path.join(path, "tokenizer_config.json")) else cfg.get("thinker_name_or_path", "Qwen/Qwen3-ASR-0.6B"))
    model = model.to(device).eval()
    if model.encoder is not None: model.encoder.set_trainable(False)
    if dtype is not None: model.thinker.to(dtype)
    return model, tok

def check_sem_ids(model, tok, turn_end=False, allow_other_ids=False):
    """<SEM_END>/<TURN_END> 를 Phase1+Phase2 뒤에 붙여(이미 있으면 그대로) id 를 전역 계약·config.sp_ids 와 대조, decode 차단 목록에 없는지 확인."""
    from vapasr.uslm.interleave_data import SPECIAL_TOKENS as P1
    from vapasr.data.dialogue_tokens import PHASE2_SPECIALS as P2, FROZEN_REGISTRY
    tok.add_tokens(P1 + P2 + SEM_SPECIALS, special_tokens=True)
    ids = {t: tok.convert_tokens_to_ids(t) for t in P1 + P2 + SEM_SPECIALS}
    bad = {t: (ids[t], v) for t, v in FROZEN_REGISTRY.items() if ids[t] != v}; assert not bad, f"Phase1/2 id 불일치: {bad}"
    bad = {t: (ids[t], v) for t, v in model.config.sp_ids.items() if t in ids and ids[t] != v}; assert not bad, f"tokenizer vs config.sp_ids 불일치: {bad}"
    sem, turn = ids["<SEM_END>"], ids["<TURN_END>"]
    reg = getattr(model.config, "sem_registry", None) or {}                        # add_semantic_tokens 로 만든 체크포인트는 config.sem_registry 를 가진다
    assert all(ids[t] == v for t, v in reg.items()), f"tokenizer vs config.sem_registry 불일치: {reg} vs {sem}/{turn}"
    if not reg: print("주의: config.sem_registry 없음 — SEM 토큰으로 학습한 체크포인트가 맞는지 확인", flush=True)
    if not allow_other_ids: assert (sem, turn) == (SEM_END_ID, TURN_END_ID), f"SEM/TURN id {sem}/{turn} ≠ 계약 {SEM_END_ID}/{TURN_END_ID}"
    assert max(sem, turn) < model.get_input_embeddings().weight.shape[0], "임베딩에 SEM 행 없음"
    blocked = set(model.blocked.tolist()); assert sem not in blocked, "<SEM_END> 가 decode 차단 목록에 있음"
    assert not (turn_end and turn in blocked), "<TURN_END> 가 decode 차단 목록에 있음(--turn-end)"
    assert model.config.lanes == 0, "mono 모델 전용"
    return sem, turn

# ───────────────────────────── 실행 ─────────────────────────────
def streams_path(a): return Path(a.streams_out) if a.streams_out else Path(str(Path(a.out).with_suffix("")) + ".streams.jsonl")

def decode_config(a):
    """이어하기 지문(= 방출을 정하는 모든 것): 디코드·패딩 설정(resolve_settings 뒤) + 체크포인트 가중·config + words 파일 + 코드 sha256."""
    model = "oracle" if a.oracle else (os.path.abspath(a.model) if a.model else None)
    return dict(protocol=PROTOCOL, model=model, **checkpoint_fingerprint(None if a.oracle else a.model), train=a.train, delay=a.delay, turn_end=a.turn_end,
                hangover_s=a.hangover_s, pad_delays=a.pad, tail_margin=a.tail_margin, pad_tail=a.pad_tail, no_pad=a.no_pad, dtype=a.dtype, next_bias=a.next_bias,
                turn_bias=a.turn_bias, sem_guard=not a.no_sem_guard, max_flush=a.max_flush, block_turn=a.block_turn, path_remap=list(a.path_remap or []),
                words=os.path.abspath(a.words), words_sha256=sha256_file(a.words), code={f: sha256_file(ROOT / f) for f in CODE_FILES if (ROOT / f).exists()})

def make_record(wrow, cfg, delta, K, dur, emitted, hyp, forced=0, flush_rounds=0, trace=None, trace_min_p=0.0, sem_id=SEM_END_ID, turn_id=TURN_END_ID, pad_delays=None,
                guard_blocked=0):
    rec = dict(id=wrow["id"], set=wrow.get("set"), lang=wrow["lang"], config=cfg, delta=delta, K=K, K0=wrow.get("K"), duration_s=round(dur, 3), pad_delays=pad_delays, event_ids=[sem_id, turn_id],
               words_digest=row_digest(wrow), emitted=[[int(k), int(t)] for k, t in emitted], sem_k=events_from_emits(emitted, sem_id), turn_k=events_from_emits(emitted, turn_id),
               hyp=dict(words=hyp["words"], events=hyp["events"], text=hyp["text"]), forced=forced, flush_rounds=flush_rounds, guard_blocked=int(guard_blocked))
    if trace is not None:
        tr = [(k, ps, pt) for k, ps, pt in trace if ps >= trace_min_p or pt >= trace_min_p]
        rec["trace"] = dict(k=[k for k, _, _ in tr], p_sem=[round(p, 4) for _, p, _ in tr], p_turn=[round(p, 4) for _, _, p in tr], steps=len(trace))
    return rec

def run(a):
    words = {r["id"]: r for r in read_jsonl(a.words)}; labels = {r["id"]: r for r in read_jsonl(a.labels)}
    ids, info = select_streams(words, labels, a.langs, a.max_streams); configs = [dict(name="oracle", mode="oracle", value=0.0)] if a.oracle else make_configs(a.sem_bias, a.theta)
    dest = streams_path(a); dest.parent.mkdir(parents=True, exist_ok=True); cfg_path = Path(str(dest) + ".config.json"); dcfg = decode_config(a)
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text()); diff = sorted(k for k in set(old) | set(dcfg) if old.get(k) != dcfg.get(k))
        assert not diff, f"이어하기 지문 불일치 {diff} (체크포인트·words·코드·설정이 바뀜) — 새 --out/--streams-out 을 쓰세요: {cfg_path}"
    else: write_json(cfg_path, dcfg)
    prev = read_jsonl(dest) if dest.exists() else []; done = {(r["id"], r["config"]["name"]) for r in prev}
    todo = [(sid, c) for sid in ids for c in configs if (sid, c["name"]) not in done]
    print(f"START streams={len(ids)} configs={[c['name'] for c in configs]} todo_rows={len(todo)} done_rows={len(done)} {info}", flush=True)
    L = {sid: eval_len(words[sid], labels.get(sid), dcfg) for sid in ids}                   # (길이 s, K') — K' 는 학습과 같은 정수(오디오 길이에서 다시 반올림하지 않는다)
    if a.oracle:
        with dest.open("a", encoding="utf-8") as out:
            for sid, c in todo:
                dur, K = L[sid]; h = oracle_hyp(words[sid], labels.get(sid), a.delay, K, eval_turn=a.turn_end, hangover_s=a.hangover_s)
                out.write(json.dumps(make_record(words[sid], c, a.delay, K, dur, h["emitted"], h, pad_delays=a.pad), ensure_ascii=False) + "\n")
        return
    if not todo: return
    import numpy as np, torch
    from vapasr.hf.infer import prefix_ids
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_sem_model(a.model, a.encoder, getattr(torch, a.dtype), device); sem_id, turn_id = check_sem_ids(model, tok, a.turn_end, a.allow_other_ids)
    assert f"<DELAY_{a.delay}>" in model.config.sp_ids, f"<DELAY_{a.delay}> 없음"
    trained = [int(d) for d in (getattr(model.config, "delays", None) or [])]
    assert not trained or a.delay in trained or a.allow_untrained_delay, f"평가 δ={a.delay} 가 model.config.delays {trained} 에 없음 — 의도했다면 --allow-untrained-delay"
    if a.block_turn: model.blocked = torch.unique(torch.cat([model.blocked, torch.tensor([turn_id], device=model.blocked.device)]))
    print(f"pad: δ{a.pad} tail_margin {a.tail_margin} pad_tail {a.pad_tail} turn {a.turn_end} hangover {a.hangover_s} → 늘린 스트림 "
          f"{sum(L[s][1] > int(words[s].get('K') or 0) for s in ids)}/{len(ids)} [학습 설정 {a.train or '모름'}]", flush=True)
    remaps = parse_remaps(a.path_remap); dev = next(model.parameters()).device
    rows = sorted(todo, key=lambda r: (words[r[0]]["lang"], L[r[0]][1], r[0], r[1]["name"]))
    order = list(dict.fromkeys(sid for sid, _ in rows)); last_use = {sid: i for i, (sid, _) in enumerate(rows)}
    feats, failed, verified, n_done, t0 = {}, {}, set(), 0, time.monotonic()
    window = a.prefetch or max(4 * a.batch_size, a.io_workers)                                  # 오디오 선읽기 상한(스트림 수) — 결과 wav 가 host RAM 에 쌓이지 않게
    with ThreadPoolExecutor(a.io_workers) as pool, dest.open("a", encoding="utf-8", buffering=1) as out:
        fut, pending, taken = {}, iter(order), set()
        def refill():
            for sid in itertools.islice(pending, max(0, window - len(fut))):
                if sid not in taken: fut[sid] = pool.submit(load_stream_audio, words[sid], L[sid][0], remaps)
        refill(); pos = 0
        while pos < len(rows):
            lang = words[rows[pos][0]]["lang"]; end = pos
            while end < len(rows) and end - pos < a.batch_size and words[rows[end][0]]["lang"] == lang: end += 1
            batch = rows[pos:end]
            for sid, _ in batch:
                if sid in feats or sid in failed: continue
                f = fut.pop(sid, None); taken.add(sid)
                if f is None: f = pool.submit(load_stream_audio, words[sid], L[sid][0], remaps)     # 방어(소비 순서 = order 라 보통 없음)
                refill()
                try: wav = f.result()
                except (OSError, ValueError, RuntimeError) as e:                   # 오디오 조립 실패(경로·길이): 그 스트림만 건너뛰고 보고서에서 complete=False 로 드러난다
                    failed[sid] = f"{type(e).__name__}: {str(e)[:200]}"; print(f"  ! audio {sid}: {failed[sid]}", flush=True); continue
                w = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).to(dev); K = L[sid][1]
                with torch.inference_mode(): feats[sid] = model.encode(w[None], torch.tensor([len(wav)], device=dev), torch.tensor([K], device=dev))[0]
                assert feats[sid].shape[-2] == K, f"인코더 청크 수 {tuple(feats[sid].shape)} ≠ K' {K} ({sid})"
                del wav, w
            batch = [r for r in batch if r[0] not in failed]
            if not batch: pos = end; continue
            prefix = prefix_ids(model, tok, lang, a.delay); bt = time.monotonic()
            states = decode_rows(model, [feats[sid] for sid, _ in batch], prefix, [c for _, c in batch], sem_id=sem_id, turn_id=turn_id, next_bias=a.next_bias,
                                 turn_bias=a.turn_bias, max_flush_rounds=a.max_flush, sem_guard=not a.no_sem_guard, trace=True)
            if a.verify and lang not in verified and device == "cuda":
                sid = batch[0][0]; ref, forced, _, rounds = model.stream_decode(feats[sid], prefix, max_flush_rounds=a.max_flush, next_bias=a.next_bias)
                mine = decode_rows(model, [feats[sid]], prefix, [dict(mode="bias", value=0.0)], sem_id=sem_id, turn_id=turn_id, next_bias=a.next_bias, max_flush_rounds=a.max_flush, sem_guard=False, trace=False)[0]
                ok = ref == mine.emitted and forced == mine.forced and rounds == mine.flush_rounds
                write_json(dest.parent / f"parity-{lang}.json", dict(id=sid, match=ok, stream_decode=ref, decode_rows=mine.emitted)); assert ok, f"stream_decode 대조 실패 ({lang}, {sid})"
                verified.add(lang); print(f"PARITY lang={lang} id={sid} PASS", flush=True)
            for (sid, c), s in zip(batch, states):
                h = hyp_from_emitted(s.emitted, tok, sem_id, turn_id)
                rec = make_record(words[sid], c, a.delay, s.K, L[sid][0], s.emitted, h, s.forced, s.flush_rounds, s.trace, a.trace_min_p, sem_id, turn_id, a.pad, s.guard_blocked)
                rec["metrics"] = score_stream(words[sid], labels.get(sid), dict(emitted=rec["emitted"], **rec["hyp"]), a.delay, s.K, early=a.early, lates=a.lates,
                                              eval_turn=a.turn_end, hangover_s=a.hangover_s, sem_id=sem_id, turn_id=turn_id, guard_blocked=s.guard_blocked)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_done += 1
            out.flush(); os.fsync(out.fileno())
            for sid in {sid for sid, _ in batch}:
                if last_use[sid] < end: feats.pop(sid, None)
            print(f"PROGRESS rows={n_done}/{len(rows)} lang={lang} batch={len(batch)} decode_s={time.monotonic() - bt:.1f} elapsed_s={time.monotonic() - t0:.0f}"
                  + (f" peak_gpu_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}" if device == "cuda" else ""), flush=True)
            pos = end
    if failed: write_json(Path(str(dest) + ".audio-failures.json"), failed)

def build_report(a):
    """스트림 jsonl → 현재 words/labels 로 재채점 → 설정·언어·세트별 요약 + PR 곡선(bias·threshold).
    재채점 전 검사: 행마다 words_digest(= 디코드 때 words 행) 와 현재 labels 로 다시 계산한 K' 가 기록과 같아야 한다 — 다르면 다른 오디오·패딩의 방출이라 멈춘다."""
    words = {r["id"]: r for r in read_jsonl(a.words)}; labels = {r["id"]: r for r in read_jsonl(a.labels)}
    ids, info = select_streams(words, labels, a.langs, a.max_streams); sel = set(ids); src = streams_path(a); cfg_path = Path(str(src) + ".config.json")
    if cfg_path.exists(): dc = json.loads(cfg_path.read_text())                                  # 오디오를 어떻게 디코드했는지(패딩·δ)는 저장된 설정이 정본
    else: resolve_settings(a); dc = decode_config(a)
    if any(getattr(a, k) is not None and getattr(a, k) != dc.get(k) for k in ("turn_end", "hangover_s", "tail_margin")):
        print(f"주의: --turn-end/--hangover-s/--tail-margin 대신 저장된 디코드 설정 {dc['turn_end']}/{dc['hangover_s']}/{dc['tail_margin']} 로 채점", flush=True)
    recs = [r for r in read_jsonl(src) if r["id"] in sel]
    stale = [r["id"] for r in recs if r.get("words_digest") != row_digest(words[r["id"]])]
    assert not stale, f"words 행이 디코드 때와 다름(words_digest 불일치) {len(stale)}/{len(recs)} 행, 예 {stale[:3]} — 다시 디코드하세요(새 --out)"
    k_bad = [(r["id"], r["K"], eval_len(words[r["id"]], labels.get(r["id"]), dc)[1]) for r in recs]; k_bad = [x for x in k_bad if x[1] != x[2]]
    assert not k_bad, f"기록 K' ≠ 현재 words/labels 로 계산한 K' {len(k_bad)}/{len(recs)} 행, 예 (id, 기록, 지금) {k_bad[:3]} — labels.turn_end 등이 바뀌었으면 다시 디코드하세요"
    by_cfg = {}
    for r in recs:
        sem_id, turn_id = r.get("event_ids") or (SEM_END_ID, TURN_END_ID)
        r["metrics"] = score_stream(words[r["id"]], labels.get(r["id"]), dict(emitted=r["emitted"], **r["hyp"]), r["delta"], r["K"], early=a.early, lates=a.lates,
                                    eval_turn=dc["turn_end"], hangover_s=dc["hangover_s"], sem_id=sem_id, turn_id=turn_id, guard_blocked=r.get("guard_blocked", 0))
        by_cfg.setdefault(r["config"]["name"], dict(config=r["config"], recs=[]))["recs"].append(r)
    configs = {}
    for name, g in by_cfg.items():
        configs[name] = dict(mode=g["config"]["mode"], value=g["config"]["value"], n_streams=len(g["recs"]), complete=len(g["recs"]) == len(ids), **aggregate(g["recs"], ("lang", "set")))
    pr = {}
    for name, c in sorted(configs.items(), key=lambda kv: (kv[1]["mode"], kv[1]["value"])):
        for grp, summ in [("overall", c["overall"])] + list(c["by_lang"].items()):
            pr.setdefault(c["mode"], []).append(dict(value=c["value"], group=grp, **pr_point(summ, a.primary_late)))
    return dict(protocol=PROTOCOL, created=time.strftime("%Y-%m-%dT%H:%M:%S"), selection=info, n_selected=len(ids), streams_file=str(streams_path(a)),
                decode=dc, scoring=dict(early=a.early, lates=list(a.lates), primary_late=a.primary_late, hangover_s=dc["hangover_s"], turn_end=dc["turn_end"],
                                                      chunk_s=CHUNK_S, ref_time="(a) ref_k=int(word_end/0.08)+delta for P/R/F1; (b) latency vs word_end, hyp_time=(k+1)*0.08",
                                                      pcr_raw="(premature + guard_blocked)/(n_hyp + guard_blocked): sem-guard 가 막은 SEM 발화를 조기 commit 으로 센 PCR"),
                configs=configs, pr_curve=pr)

def print_table(rep, late):
    print(f"{'config':>12} {'group':>8} {'n_sem':>6} {'tP':>6} {'tR':>6} {'txtP':>6} {'txtR':>6} {'PCR':>6} {'PCRraw':>6} {'gblk':>6} {'lat50':>6} {'err':>6}")
    f = lambda v: "   -  " if v is None else f"{v:6.3f}"
    for name, c in sorted(rep["configs"].items(), key=lambda kv: (kv[1]["mode"], kv[1]["value"])):
        for grp, s in [("all", c["overall"])] + list(c["by_lang"].items()):
            if not s: continue
            w = s["timing"].get(f"late{late}", {}); t = s["text"]; asr = s["asr"].get("cer_nospace") or s["asr"].get("wer") if grp == "Korean" else s["asr"].get("wer")
            print(f"{name:>12} {grp[:8]:>8} {t['n_hyp']:>6} {f(w.get('precision'))} {f(w.get('recall'))} {f(t['precision'])} {f(t['recall'])} {f(t['pcr'])} {f(t.get('pcr_raw'))} "
                  f"{t.get('guard_blocked', 0):>6} {f((w.get('latency_s') or {}).get('p50'))} {f(asr['rate'] if asr else None)}")

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", help="HF 산출물 디렉토리(final/ 또는 checkpoint-N/)"); p.add_argument("--encoder", default=None, help="Nemotron .nemo 디렉토리(기본 env MXC_NEMOTRON_DIR/NEMOTRON_DIR)")
    p.add_argument("--words", required=True); p.add_argument("--labels", required=True)
    p.add_argument("--delay", type=int, default=4); p.add_argument("--sem-bias", type=float, nargs="*", default=[0.0], help="bias 모드: logits[<SEM_END>] += b 스윕")
    p.add_argument("--theta", type=float, nargs="*", default=[], help="threshold 모드: p(<SEM_END>) ≥ θ 이면 방출 (스윕)")
    p.add_argument("--turn-end", dest="turn_end", action="store_true", help="TURN_END 평가(TURN 지표 + 패딩에 k_turn 포함). 기본: 학습 설정(없으면 켬)")
    p.add_argument("--no-turn-end", dest="turn_end", action="store_false", help="TURN_END 평가·패딩 끔(--no-turn-end 로 학습한 모델)")
    p.add_argument("--hangover-s", type=float, default=None, help=f"기본: 학습 설정(없으면 {HANGOVER_S})")
    p.add_argument("--pad-delays", type=int, nargs="+", default=None, help="패딩 기준 학습 δ 목록(기본: 체크포인트 config.delays/학습 인자, oracle·모름은 2 3 4 6) — 평가 δ 는 늘 포함")
    p.add_argument("--tail-margin", type=int, default=None, help="기본: 학습 --tail-margin(없으면 2)"); p.add_argument("--no-pad", action="store_true", help="원래 길이로(학습과 다름 — 진단용)")
    p.add_argument("--allow-train-mismatch", action="store_true", help="--turn-end/--hangover-s/--tail-margin/--pad-delays 가 학습 설정과 달라도 진행")
    p.add_argument("--allow-untrained-delay", action="store_true", help="평가 δ 가 학습 δ 목록에 없어도 진행")
    p.add_argument("--turn-bias", type=float, default=0.0); p.add_argument("--block-turn", action="store_true", help="<TURN_END> 방출 금지")
    p.add_argument("--next-bias", type=float, default=0.0); p.add_argument("--no-sem-guard", action="store_true", help="빈 commit(<SEM_END> 연속·텍스트 전) 허용")
    p.add_argument("--gpu", type=int, default=None); p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--batch-size", type=int, default=16, help="디코드 행(스트림×설정) 수"); p.add_argument("--io-workers", type=int, default=8)
    p.add_argument("--prefetch", type=int, default=0, help="오디오 선읽기 스트림 수 상한(0 = max(4·batch-size, io-workers))")
    p.add_argument("--max-flush", type=int, default=8); p.add_argument("--max-streams", type=int, default=0, help="언어별 상한(0=전체)")
    p.add_argument("--langs", nargs="*", default=None, help="English Korean 중 선택")
    p.add_argument("--path-remap", nargs="*", default=[], help="OLD=NEW 오디오 경로 접두 치환 (예: " + MXC_PREFIX + "LibriSpeech/=/data5/LibriSpeech/)")
    p.add_argument("--early", type=int, default=EARLY); p.add_argument("--lates", type=int, nargs="+", default=list(LATES)); p.add_argument("--primary-late", type=int, default=2)
    p.add_argument("--trace-min-p", type=float, default=0.0, help="trace 저장 필터(p_sem 또는 p_turn ≥ 값인 step 만; 0 = 모든 결정 step)")
    p.add_argument("--allow-other-ids", action="store_true", help="SEM/TURN id 가 151723/151724 가 아니어도 진행")
    p.add_argument("--verify", action="store_true", help="언어별 첫 배치 1 행을 model.stream_decode 와 대조(cuda)")
    p.add_argument("--out", required=True); p.add_argument("--streams-out", default=None, help="스트림별 jsonl (기본 <out 확장자 제거>.streams.jsonl)")
    p.add_argument("--score-only", action="store_true", help="디코드 없이 스트림 jsonl 을 재채점해 보고서만"); p.add_argument("--oracle", action="store_true", help="참조 직렬화를 가설로(지표 상한 점검, GPU 불필요)")
    p.set_defaults(turn_end=None)
    a = p.parse_args(argv)
    assert a.oracle or a.score_only or a.model, "--model 필요"
    assert a.primary_late in a.lates and a.batch_size > 0 and a.max_flush >= 0 and a.prefetch >= 0
    return a

def main(argv=None):
    a = parse_args(argv)
    if a.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    if not a.score_only: resolve_settings(a); run(a)
    elif not streams_path(a).exists(): raise SystemExit(f"--score-only: 스트림 jsonl 없음 {streams_path(a)} (다른 --out 이면 --streams-out 으로 원래 파일을 주세요)")
    rep = build_report(a); write_json(a.out, rep); print_table(rep, a.primary_late); print(f"REPORT {a.out}", flush=True)
    return rep

if __name__ == "__main__":
    main()
