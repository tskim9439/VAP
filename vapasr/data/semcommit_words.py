"""SEM_END v0 — word/eojeol index (words.jsonl) builder. 계획: raw/inbox/streaming_asr_semantic_commit_plan.md §6–7 (Word/Eojeol Indexing).

입력 한 스트림 = aligned-manifest 행(MonoStreamDataset `_items-online-v2` 항목: tokens=[[qwen_id, end_time_s]] 스트림 절대시각·시간순, text=정렬된 발화 lexical 의 join)
             + 같은 id 의 streams.jsonl 행(segments[].utt_id·lexical_text·raw_text).
출력(words.jsonl 한 줄):
  {id, set, lang, K, duration_s, text, tokens(= aligned tokens 그대로), segments:[{path(rack4), offset_s, silence_before_s, dur_s, utt_id, raw_text}],
   words:[{i, text, a, b, end_time, seg, tags}], pnc_text?}      tokens[a:b] = 단어의 BPE 토큰, end_time = 그 토큰 시각의 max.
- 단어 경계: Qwen byte-level BPE 에서 토큰 0 또는 바이트가 0x20 으로 시작하는 토큰이 새 단어(s1_align 은 idx>0 발화 앞에 ' ' 를 붙여 토큰화).
  단어 텍스트 = 토큰 바이트를 이어 UTF-8 디코드 후 strip — KO 토큰은 음절 바이트를 쪼개므로 토큰 단위 decode 금지. [w.text] != text.split() 이면 fail closed.
- segment: streams 의 segment 별 lexical 단어 수로 자르고(시각 창 검사), 안 맞으면 end_time ∈ [offset_s, offset_s+dur_s+0.2] 로, 그것도 실패하면 스트림 제외.
- tags(Stage A 힌트 전용, 라벨 아님): KO 는 raw_text 표지 — filler(끝 '/'), rep('+'), unclear('*'), punct_final(끝 . ? !), punct_comma(끝 ,).
  raw 공백 토큰 ↔ lexical 단어 1:1 (kspon.normalize_kspon_v1 의 DUAL 선택·표지 regex 를 토큰마다 적용; 단독 b/ l/ o/ n/ u/ 와 'b/.' 는 버린다).
  (철자)/(발음) 쌍 전체에 걸린 표지(쌍 뒤 꼬리의 / + *, 버려진 형태의 단어 끝 표지)는 선택된 모든 어절에 붙인다.
  EN 은 LibriSpeech-PC 구두점 텍스트를 lexical 단어에 정규화 토큰 편집거리 정렬로 옮긴다(일치율 < 0.9 인 발화는 태그 없음).
- 제외(None, 사유): 빈 tokens·시간 역행·단어 분할 불일치·segment 불일치/배정 실패·모르는 경로 접두사·오디오 없음(exists 를 줄 때)
  ·KO lexical 에 남은 단독 b/l/o/n/u 어절(stray_marker_letter)·KO raw 에서 어절에 붙은 표지('들어서.b/' → lexical '들어서b', glued_marker_letter).
  제외 근거 어절은 info['drop_detail'] 에 남긴다(감사용).
"""
import os, re, json, unicodedata
from typing import Dict, List, Optional, Tuple
from .kspon import DUAL, NOISE, FILLER, PUNCT

MXC_PREFIX = "/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/"
DEFAULT_ROOTS = {"LibriSpeech": "/data5/LibriSpeech", "KsponSpeech": "/data4/tskim/DBs/KsponSpeech/extracted"}   # rack4
DEFAULT_PNC_DIR = "/data5/LibriSpeech/librispeech_pnc"
LS_SPLITS = ("train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other", "test-clean", "test-other")
TAGS = ("filler", "rep", "unclear", "punct_final", "punct_comma")
# lexical 의 단독 b/l/o/n/u 어절. 원천(로컬 T5 streams.jsonl 실측): kspon-full 은 slash 없는 표지 글자('n 저희 엄마도', 'b.', 'b,')가 대부분이고
# 실제로 읽은 알파벳('a b c d까지', 'b, c+ c … b쁠')도 섞여 있다('b/.' 토큰은 kspon-full 에 없다). kspon-eval 은 구두점이 붙은 표지('b/.' 'l/?' —
# NOISE 가 '/' 뒤 공백을 요구)와 앞 slash 표지('/u'). 구별하지 않고 스트림을 fail closed 로 버린다(streams.jsonl 기준 kspon-full 117·dev 2·eval 5 스트림;
# 근거 어절은 info['drop_detail']).
STRAY_KO = set("blonu")
SEG_TAIL_S = 0.2

# ───────────────────────── 경로 ─────────────────────────
def remap_path(p: str, roots: Dict[str, str], exists=None) -> str:
    """mxc 절대경로 → rack4. roots = {"LibriSpeech": ls_root, "KsponSpeech": kspon_root}. 접미사(.flac/.pcm — streams.py 가 .pcm 으로 raw reader 선택)는 유지.
    이미 roots 아래 경로는 rel 을 그대로 두고 같은 규칙을 탄다(exists 검사 포함). 모르는 접두사·코퍼스는 ValueError.
    exists(callable) 를 주면 후보(rel, eval 은 KsponSpeech_eval/rel) 중 존재하는 첫 경로, 없으면 FileNotFoundError.
    주의: rack4 /data5/LibriSpeech 에는 dev-clean/dev-other 가 없다 — exists 없이는 그 경로도 그대로 돌려준다."""
    hit = next(((c, r.rstrip("/")) for c, r in roots.items() if p.startswith(r.rstrip("/") + "/")), None)
    if hit: corpus, root = hit; rel = p[len(root) + 1:]
    else:
        if not p.startswith(MXC_PREFIX): raise ValueError(f"unknown audio path prefix: {p}")
        corpus, _, rel = p[len(MXC_PREFIX):].partition("/")
        if corpus not in roots or not rel: raise ValueError(f"unknown corpus in audio path: {p}")
        root = roots[corpus].rstrip("/")
    cands = [f"{root}/{rel}"]
    if corpus == "KsponSpeech" and rel.startswith("eval_"): cands.append(f"{root}/KsponSpeech_eval/{rel}")   # 공식 trn 은 KsponSpeech_eval/eval_clean/…
    if exists is None: return cands[0]
    for c in cands:
        if exists(c): return c
    raise FileNotFoundError(f"audio not found under {root}: {rel}")

# ───────────────────────── byte-level BPE ─────────────────────────
def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2/Qwen byte-level 표: byte → 출력 가능한 unicode 문자(공백 0x20 → 'Ġ')."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
_B2U = _bytes_to_unicode(); _U2B = {v: k for k, v in _B2U.items()}

def piece_bytes(piece) -> Optional[bytes]:
    """convert_ids_to_tokens 의 byte-level piece → 원 바이트. 표 밖 문자(특수 토큰 등)면 None."""
    try: return bytes(_U2B[c] for c in piece)
    except (KeyError, TypeError): return None

def _load_json(path: str):
    with open(path, encoding="utf-8") as f: return json.load(f)

class PieceVocab:
    """id → byte-level piece 만 읽는 경량 tokenizer(transformers 불필요). path = tokenizer.json | vocab.json | 둘 중 하나를 담은 디렉터리.
    added_tokens(특수 토큰)는 special_ids 로 따로 두어 split_words 가 fail closed 한다."""
    def __init__(self, path: str):
        if os.path.isdir(path):
            path = next((os.path.join(path, f) for f in ("tokenizer.json", "vocab.json") if os.path.exists(os.path.join(path, f))), path)
        d = _load_json(path); self.path = path; self.special_ids = set()
        tj = isinstance(d.get("model"), dict) and "vocab" in d["model"]       # Qwen vocab.json 에는 'model' 이라는 토큰(id 2528)이 있다 — 키 존재로 판별 금지
        vocab = d["model"]["vocab"] if tj else d
        self.id2piece = {int(i): p for p, i in vocab.items()}
        added = d.get("added_tokens", []) if tj else []
        extra = os.path.join(os.path.dirname(path), "added_tokens.json")
        if not tj and os.path.exists(extra): added = [dict(id=i, content=c) for c, i in _load_json(extra).items()]
        for a in added: self.id2piece[int(a["id"])] = a["content"]; self.special_ids.add(int(a["id"]))
    def convert_ids_to_tokens(self, ids): return [self.id2piece.get(int(i)) for i in ids]

def _special_ids(tok) -> set:
    s = getattr(tok, "special_ids", None)
    if s is not None: return s
    try: return set(tok.get_added_vocab().values())                          # HF: 특수·추가 토큰
    except Exception: return set()

def _pieces(tok, ids: List[int]) -> List[Optional[str]]:
    if hasattr(tok, "convert_ids_to_tokens"): return list(tok.convert_ids_to_tokens(list(ids)))
    return [tok.id_to_token(int(i)) for i in ids]                              # tokenizers.Tokenizer

def split_words(token_ids, times, tok, text: Optional[str] = None) -> Optional[List[dict]]:
    """토큰 → 단어/어절 [{i, text, a, b, end_time}] (tokens[a:b], end_time = max 시각). 새 단어 = 토큰 0 또는 바이트가 0x20 으로 시작하는 토큰.
    특수/미지 토큰·UTF-8 오류·빈 단어·(text 가 있으면) [w.text] != text.split() 이면 None(fail closed)."""
    if len(token_ids) != len(times): raise ValueError(f"len(token_ids)={len(token_ids)} != len(times)={len(times)}")
    if not token_ids: return None
    sp = _special_ids(tok); bs = []
    for i, p in zip(token_ids, _pieces(tok, token_ids)):
        b = None if (int(i) in sp or p is None) else piece_bytes(p)
        if not b: return None
        bs.append(b)
    words, a = [], 0
    for j in range(1, len(bs) + 1):
        if j < len(bs) and bs[j][:1] != b" ": continue
        try: w = b"".join(bs[a:j]).decode("utf-8").strip()
        except UnicodeDecodeError: return None
        if not w or len(w.split()) != 1: return None
        words.append(dict(i=len(words), text=w, a=a, b=j, end_time=max(float(t) for t in times[a:j]))); a = j
    if text is not None and [w["text"] for w in words] != text.split(): return None
    return words

# ───────────────────────── segment 배정 ─────────────────────────
def assign_segments(words: List[dict], segs: List[dict], lex_counts: Optional[List[int]] = None, tail_s: float = SEG_TAIL_S) -> Tuple[Optional[List[int]], str]:
    """단어별 segment index. 1) segment 별 lexical 단어 수로 자르고 각 end_time 이 그 segment 창 [offset_s, offset_s+dur_s+tail_s] 안이면 'count'.
    2) 아니면 end_time 창으로(단조 증가) 'time'. 3) 실패 → (None, 'seg_assign_fail')."""
    def ok(w, j): g = segs[j]; return g["offset_s"] - 1e-3 <= w["end_time"] <= g["offset_s"] + g["dur_s"] + tail_s
    if lex_counts is not None and len(lex_counts) == len(segs) and sum(lex_counts) == len(words):
        idx = [j for j, c in enumerate(lex_counts) for _ in range(c)]
        if all(ok(w, j) for w, j in zip(words, idx)): return idx, "count"
    idx: List[int] = []
    for w in words:
        j = next((j for j in range(idx[-1] if idx else 0, len(segs)) if ok(w, j)), None)
        if j is None: return None, "seg_assign_fail"
        idx.append(j)
    return idx, "time"

def _sort_tags(tags) -> List[str]: return [t for t in TAGS if t in set(tags)]
def _punct_tags(tail: str) -> List[str]:
    return (["punct_final"] if any(c in tail for c in ".?!") else []) + (["punct_comma"] if any(c in tail for c in ",;:") else [])

# ───────────────────────── KO: raw_text 표지 → 어절 태그 ─────────────────────────
_KS_TAIL = re.compile(r"[.,?!/+*]*$")
_KS_NOISE_TOK = re.compile(r"^[blonu]/[.,?!]*$")                              # 단독 표지 + 'b/.' 류
_KS_FILLER = re.compile(r"/[.,?!+*]*$")
_KS_GLUED = re.compile(r"(?<=[^\sA-Za-z])[blonu]/[.,?!]*$")                  # 어절에 붙은 표지('들어서.b/') — NOISE 가 못 잡아 lexical 에 '들어서b' 로 남는다
_KS_DUAL_TAIL = re.compile(r"[^\s(]*")                                        # 쌍 뒤에 붙은 꼬리(다음 공백·다음 쌍 전까지)
_KS_WORD_MARK = re.compile(r"(?<![\d\s(])[/+*]+(?=[.,?!]*(?:\s|$))")         # 단어 끝 표지 — '(7*7)' 곱셈·'(/쓰리 디)' 머리 slash 는 아니다
_SENT = {"filler": "\ue000", "rep": "\ue001", "unclear": "\ue002"}; _SENT_TAG = {v: k for k, v in _SENT.items()}; _SENT_RX = re.compile("[\ue000-\ue002]")
_MARK_TAG = {"/": "filler", "+": "rep", "*": "unclear"}
def _dual_sides(a: str, b: str) -> Tuple[str, str]:                           # (선택, 버림) — kspon.normalize_kspon_v1 와 같은 선택: 철자형에 digit·Latin 이 있으면 발음형
    return (b, a) if re.search(r"[0-9A-Za-z]", a) else (a, b)
def _dual_pick(m): return _dual_sides(m.group(1), m.group(2))[0]
def _dual_marked(m) -> str:
    """_dual_pick + 쌍 전체에 걸린 표지를 선택된 모든 어절 끝에 sentinel(PUA 문자)로 붙인다 — sentinel 을 지우면 _dual_pick 치환과 같은 문자열.
    쌍 전체 표지 = 꼬리의 + * 와 꼬리 끝 filler '/'('(7명)/(일곱 명)+' → 일곱·명 rep), 버려진 형태의 단어 끝 / + *('(그래서)/(캐서*)' → 그래서 unclear).
    선택된 형태 안의 표지('(이십 일+ 일 학점)')는 그 어절에만 남는다(토큰 단위 처리)."""
    pick, dropped = _dual_sides(m.group(1), m.group(2)); tail = _KS_DUAL_TAIL.match(m.string, m.end()).group()
    tags = {_MARK_TAG[c] for c in "+*" if c in tail} | ({"filler"} if _KS_FILLER.search(tail) else set())
    tags |= {_MARK_TAG[c] for mk in _KS_WORD_MARK.findall(dropped) for c in mk}
    s = "".join(_SENT[t] for t in TAGS if t in tags)
    return re.sub(r"(?<=\S)(?=\s|$)", s, pick) if s else pick
_KO_TAG = re.compile(r"<[^>\s]{1,20}>"); _KO_PUNCT = re.compile(r"[^\w\s]")          # textnorm._TAG / _KO_PUNCT 와 같다(textnorm 은 num2words 를 요구해 import 하지 않는다)
def _ks_norm_token(t: str) -> List[str]:
    r"""lexical_text = textnorm.target_ko(raw, 'kspon') = normalize_kspon_v1(NOISE → PUNCT → FILLER → + * 제거) → NFKC·태그 제거 → [^\w\s] 를 공백.
    이 규칙들을 공백 토큰 하나에 적용한다 — regex 경계가 모두 공백이라 전체 문자열에 적용 후 split 한 것과 같다. 'K-POP' 처럼 한 토큰이 여러 어절이 될 수 있다."""
    s = NOISE.sub(" ", t); s = PUNCT.sub("", s); s = FILLER.sub(r"\1", s); s = s.replace("+", "").replace("*", "")
    return _KO_PUNCT.sub(" ", unicodedata.normalize("NFKC", _KO_TAG.sub(" ", s))).split()

def kspon_raw_tags(raw: str) -> List[Tuple[str, List[str]]]:
    """raw_text → [(정규화 어절, tags)]. DUAL 선택은 괄호 안 공백 때문에 전체 문자열에 먼저 적용한 뒤 공백으로 나눈다(쌍 전체 표지는 _dual_marked).
    단독 표지(b/ l/ o/ n/ u/, 'b/.')·정규화 후 빈 토큰은 버리고, 그 토큰의 끝 구두점은 직전 어절의 태그로 옮긴다('안 먹을래 b/.' → 먹을래 punct_final).
    한 토큰이 여러 어절이 되면('K-POP+' → K POP, '(7명)/(일곱 명)+' → 일곱 명) filler/rep/unclear 는 모든 어절에, 구두점 태그는 마지막 어절에 붙인다.
    어절에 붙은 표지('들어서.b/')는 태그 계산에서 떼어낸다(filler 아님, '.' 은 punct_final) — 어절 자체('들어서b')는 lexical 과 같게 두고 build_stream 이 스트림을 버린다."""
    out: List[Tuple[str, List[str]]] = []
    for t in DUAL.sub(_dual_marked, raw or "").split():
        marks = [_SENT_TAG[c] for c in t if c in _SENT_TAG]; t = _SENT_RX.sub("", t); tt = _KS_GLUED.sub("", t)
        p = _punct_tags(_KS_TAIL.search(tt).group())
        ws = [] if _KS_NOISE_TOK.match(t) else _ks_norm_token(t)
        if not ws:
            if out: out[-1] = (out[-1][0], _sort_tags(out[-1][1] + p))
            continue
        tags = marks + (["filler"] if _KS_FILLER.search(tt) else []) + (["rep"] if "+" in tt else []) + (["unclear"] if "*" in tt else [])
        out += [(w, _sort_tags(tags + (p if k == len(ws) - 1 else []))) for k, w in enumerate(ws)]
    return out

def kspon_glued_markers(raw: str) -> List[str]:
    """raw_text 에서 어절에 붙은 단독 표지 토큰('들어서.b/') — lexical 에 표지 글자 잔재를 남긴다(kspon-eval 1건, kspon-full·dev 0건)."""
    return [t for t in DUAL.sub(_dual_pick, raw or "").split() if _KS_GLUED.search(t)]

def kspon_tags_for(word_texts: List[str], raw: str) -> Tuple[Optional[List[List[str]]], str]:
    """segment 어절 목록과 raw_text 의 1:1 대응 → (어절별 tags, 'ok') 또는 (None, 'kspon_tag_map_mismatch')."""
    pairs = kspon_raw_tags(raw)
    if [w for w, _ in pairs] != list(word_texts): return None, "kspon_tag_map_mismatch"
    return [t for _, t in pairs], "ok"

# ───────────────────────── EN: LibriSpeech-PC 구두점 → 단어 태그 ─────────────────────────
_EN_DROP = re.compile(r"[^\w']+"); _EN_SPLIT = re.compile(r"[-–—/]+"); _EN_TAIL = re.compile(r"[^\w]*$")
def en_norm(w: str) -> str:
    """정렬용 정규화: NFKC·소문자·’→'·글자/숫자/아포스트로피 외 제거·양끝 아포스트로피 제거."""
    w = unicodedata.normalize("NFKC", w).lower().replace("’", "'").replace("‘", "'")
    return _EN_DROP.sub("", w).strip("'")

def pnc_words(pc_text: str) -> List[Tuple[str, List[str]]]:
    """구두점 텍스트 → [(정규화 단어, tags)]. 하이픈·대시·slash 는 단어 구분(LibriSpeech 는 'flour fattened'), 끝 구두점: . ? ! → punct_final, , ; : → punct_comma.
    정규화 후 빈 조각(단독 '—', '...')의 구두점은 직전 단어로."""
    out: List[Tuple[str, List[str]]] = []
    for tok in (pc_text or "").split():
        for piece in _EN_SPLIT.split(tok):
            p = _punct_tags(_EN_TAIL.search(piece).group()); w = en_norm(piece)
            if not w:
                if out and p: out[-1] = (out[-1][0], _sort_tags(out[-1][1] + p))
                continue
            out.append((w, p))
    return out

def _align_pairs(a: List[str], b: List[str]) -> List[Tuple[int, int]]:
    """Levenshtein 정렬(치환·삽입·삭제 비용 1)의 일치 쌍 [(i, j)] — backtrace 는 일치 대각선을 우선."""
    n, m = len(a), len(b); D = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        ai, Di, Dp = a[i - 1], D[i], D[i - 1]
        for j in range(1, m + 1): Di[j] = min(Dp[j - 1] + (ai != b[j - 1]), Dp[j] + 1, Di[j - 1] + 1)
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1] and D[i][j] == D[i - 1][j - 1]: pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif D[i][j] == D[i - 1][j - 1] + 1: i -= 1; j -= 1
        elif D[i][j] == D[i - 1][j] + 1: i -= 1
        else: j -= 1
    return pairs[::-1]

def en_pnc_tags(lex_words: List[str], pc_text: str, min_ratio: float = 0.9) -> Tuple[Optional[List[List[str]]], float]:
    """lexical 단어 ↔ PC 단어 정렬 → (단어별 tags, 일치율). 일치율 = 일치 쌍 / max(len) — min_ratio 미만이면 tags None. 일치한 단어에만 태그를 옮긴다."""
    pc = pnc_words(pc_text); a = [en_norm(w) for w in lex_words]; pairs = _align_pairs(a, [w for w, _ in pc])
    ratio = len(pairs) / max(len(a), len(pc), 1)
    if ratio < min_ratio: return None, ratio
    tags: List[List[str]] = [[] for _ in a]
    for i, j in pairs: tags[i] = list(pc[j][1])
    return tags, ratio

def load_pnc_file(path: str) -> Dict[str, str]:
    """LibriSpeech-PC manifest(<split>.json; JSON lines 또는 JSON list of {audio_filepath, text, duration}) → {utt_id(파일 stem): 구두점 텍스트}."""
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        while head and head.isspace(): head = f.read(1)
        f.seek(0)
        rows = json.load(f) if head == "[" else (json.loads(l) for l in f if l.strip())
        return {os.path.splitext(os.path.basename(r["audio_filepath"]))[0]: r["text"] for r in rows}

class PncIndex:
    """--pnc-dir 의 <split>.json 을 처음 필요할 때 읽는다(정확한 파일명만 열어 '._*' AppleDouble 과 섞이지 않음; 없는 split 은 빈 사전)."""
    def __init__(self, pnc_dir: str): self.dir = pnc_dir; self.cache: Dict[str, Dict[str, str]] = {}
    def lookup(self, utt_id: str, split: Optional[str]) -> Optional[str]:
        if split is None: return None
        if split not in self.cache:
            p = os.path.join(self.dir, f"{split}.json"); self.cache[split] = load_pnc_file(p) if os.path.isfile(p) else {}
        return self.cache[split].get(utt_id)

def ls_split(path: str) -> Optional[str]:
    return next((c for c in path.split("/") if c in LS_SPLITS), None)

def _pnc_get(pnc_lookup, utt_id, split):
    return pnc_lookup.lookup(utt_id, split) if hasattr(pnc_lookup, "lookup") else pnc_lookup.get(utt_id)

def apply_en_pnc(stream: dict, pnc_lookup, min_ratio: float = 0.9, info: Optional[dict] = None) -> dict:
    """words.jsonl dict 에 LibriSpeech-PC 태그(punct_final/punct_comma)와 pnc_text 를 더한다(in place). 스트림은 떨어뜨리지 않는다 — 사정은 info['tag_notes'].
    pnc_text 는 모든 segment 에 PC 텍스트가 있을 때만 둔다(부분 텍스트는 Stage A 입력을 오도)."""
    info = info if info is not None else {}; notes = info.setdefault("tag_notes", []); ratios = info.setdefault("pnc_ratios", []); pcs = []
    for j, s in enumerate(stream["segments"]):
        uid = s.get("utt_id") or os.path.splitext(os.path.basename(s["path"]))[0]
        pc = _pnc_get(pnc_lookup, uid, ls_split(s["path"])); pcs.append(pc)
        ws = [w for w in stream["words"] if w["seg"] == j]
        if pc is None: notes.append("pnc_missing"); continue
        if not ws: continue
        tg, r = en_pnc_tags([w["text"] for w in ws], pc, min_ratio); ratios.append(round(r, 4))
        if tg is None: notes.append("pnc_low_match"); continue
        for w, t in zip(ws, tg): w["tags"] = _sort_tags(w["tags"] + t)
    if pcs and all(pc is not None for pc in pcs): stream["pnc_text"] = " ".join(pcs)
    elif pcs: notes.append("pnc_text_partial")
    return stream

# ───────────────────────── 스트림 ─────────────────────────
def build_stream(aligned_row: dict, streams_row: dict, tok, roots: Optional[Dict[str, str]] = DEFAULT_ROOTS, pnc_lookup=None,
                 info: Optional[dict] = None, set_name: Optional[str] = None, pnc_min_ratio: float = 0.9, path_exists=None):
    """aligned 행 + streams 행 → words.jsonl dict, 또는 (None, 사유). roots=None 이면 경로를 바꾸지 않는다.
    info(dict) 에 seg_assign('count'|'time'), tag_notes(태그 대응 실패 사유), pnc_ratios, (KO 표지 잔재로 제외하면) drop_detail 을 남긴다.
    pnc_lookup: {utt_id: text} 또는 PncIndex. path_exists: remap_path 의 exists(없으면 존재 확인 안 함)."""
    info = info if info is not None else {}; notes = info.setdefault("tag_notes", [])
    rid = aligned_row.get("id")
    if not streams_row or streams_row.get("id") != rid: return None, "streams_row_mismatch"
    toks = aligned_row.get("tokens") or []
    if not toks: return None, "empty_tokens"
    ids = [int(t[0]) for t in toks]; times = [float(t[1]) for t in toks]
    if any(t1 < t0 for t0, t1 in zip(times, times[1:])): return None, "tokens_not_sorted"
    text = aligned_row.get("text") or ""
    words = split_words(ids, times, tok, text)
    if words is None: return None, "word_split_mismatch"
    lang = aligned_row.get("lang") or streams_row.get("lang"); ko = lang == "Korean"
    if ko:
        stray = [" ".join(w["text"] for w in words[max(0, k - 2):k + 3]) for k, w in enumerate(words) if w["text"] in STRAY_KO]
        if stray: info["drop_detail"] = stray[:3]; return None, "stray_marker_letter"
        glued = [t for s in streams_row.get("segments") or [] for t in kspon_glued_markers(s.get("raw_text") or "")]
        if glued: info["drop_detail"] = glued[:3]; return None, "glued_marker_letter"
    dur = float(aligned_row["duration_s"])
    if words[-1]["end_time"] > dur + 1e-6: return None, "word_after_stream_end"
    asegs, ssegs = aligned_row.get("segments") or [], streams_row.get("segments") or []
    if not asegs or len(asegs) != len(ssegs) or any(a["path"] != s["path"] for a, s in zip(asegs, ssegs)): return None, "segments_mismatch"
    lex = [len((s.get("lexical_text") or s.get("text") or "").split()) for s in ssegs]
    seg_idx, how = assign_segments(words, asegs, lex)
    if seg_idx is None: return None, how
    info["seg_assign"] = how
    try: paths = [remap_path(a["path"], roots, path_exists) if roots else a["path"] for a in asegs]
    except ValueError: return None, "unknown_path_prefix"
    except FileNotFoundError: return None, "audio_missing"
    segments = [dict(path=p, offset_s=a["offset_s"], silence_before_s=a["silence_before_s"], dur_s=a["dur_s"], utt_id=s.get("utt_id"), raw_text=s.get("raw_text") or "",
                     **({"src_offset_s": a["src_offset_s"]} if a.get("src_offset_s") is not None else {})) for p, a, s in zip(paths, asegs, ssegs)]
    for w, j in zip(words, seg_idx): w["seg"] = j; w["tags"] = []
    out = dict(id=rid, set=set_name or aligned_row.get("name"), lang=lang, K=int(aligned_row["K"]), duration_s=dur, text=text,
               tokens=[[i, t] for i, t in zip(ids, times)], segments=segments, words=words)
    if ko:
        for j, s in enumerate(ssegs):
            ws = [w for w in words if w["seg"] == j]
            if not ws: continue
            tg, why = kspon_tags_for([w["text"] for w in ws], s.get("raw_text") or "")
            if tg is None: notes.append(why); continue
            for w, t in zip(ws, tg): w["tags"] = t
    elif pnc_lookup is not None:
        apply_en_pnc(out, pnc_lookup, pnc_min_ratio, info)
    return out
