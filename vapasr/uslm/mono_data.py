"""Stage 1 mono 데이터셋 — streams.jsonl 스트림 하나 = 학습 샘플 하나 (plans/stage1-mono-pilot.md §4).

입력: 특징 캐시 index.jsonl(experiments/s1_extract_features.py, (1,T',D) @12.5 Hz, mode·subset 포함)
    + 정렬 align/<manifest>/<id>.jsonl(experiments/s1_align.py, speaker=0)
시퀀스 (화자 태그 없음):
  [prefix … <DELAY_d>] [AUDIO_0] tok… <NEXT_AUDIO> … [AUDIO_K-1] tok… <NEXT_AUDIO>
  flush 라운드: <EMPTY_AUDIO> tok… <NEXT_AUDIO>  ×(ceil(n_flush/M) + 1; M=0 이면 1 + 1)   ← 마지막 라운드는 빈 라운드(= "남은 것 없음" 학습)
  <EMPTY_AUDIO> 는 오디오 자리 대신 들어가는 **입력** 토큰(손실 없음). δ 때문에 스트림 끝을 넘긴 토큰은 여기서 방출된다.
chunk k = 특징 프레임 k (12.5 Hz = 80 ms). 스트림 길이가 제각각이라 길이 버킷 배치를 쓴다.
"""
import os, json, math, random
from typing import List, Dict, Tuple, Optional
import numpy as np, torch
from torch.utils.data import Dataset, Sampler
from ..data.interleave import build_interleaved, Specials
from ..data.streams import read_streams, assemble_stream
from ..features.online import frames_for
from ..probe.data import FeatureIndex
from .interleave_data import add_specials, specials_of, bad_utterance, _read_jsonl, CHUNK_S

MAN = os.environ.get("MXC_DATA_MANIFEST_DIR", os.environ.get("DATA_MANIFEST_DIR", "/tmp")); FEAT = os.environ.get("MXC_DATA_FEATURE_CACHE_DIR", os.environ.get("DATA_FEATURE_CACHE_DIR", "/tmp"))
LANG_OF = {"librispeech": "English", "ls": "English", "kspon": "Korean", "ks": "Korean"}
def lang_of(name: str) -> str: return LANG_OF[name.split("-")[0]]

def build_mono_sequence(chunks, K: int, prefix: List[int], audio_pad: int, sp: Specials, M: int):
    """build_interleaved 출력 → (ids, is_input, chunk_of, n_flush_rounds).
    is_input: 손실을 걸지 않는 위치(prefix·오디오 자리·<EMPTY_AUDIO>). chunk_of: 오디오 자리의 프레임 번호, 그 외 -1."""
    ids = list(prefix); is_input = [True] * len(prefix); chunk_of = [-1] * len(prefix); flush: List[int] = []
    for k, emits in chunks:
        if k < K:
            ids.append(audio_pad); is_input.append(True); chunk_of.append(k)
            ids += emits; is_input += [False] * len(emits); chunk_of += [-1] * len(emits)
        else:
            flush += [t for t in emits if t != sp.empty_audio]
    step = M if M else max(1, len(flush)); rounds = [flush[i: i + step] for i in range(0, len(flush), step)] + [[]]   # M=0: flush 토큰 전부 한 라운드
    for r in rounds:
        ids.append(sp.empty_audio); is_input.append(True); chunk_of.append(-1)
        ids += r + [sp.next_audio]; is_input += [False] * (len(r) + 1); chunk_of += [-1] * (len(r) + 1)
    return ids, is_input, chunk_of, len(rounds), len(flush)

class MonoStreamDataset(Dataset):
    def __init__(self, manifests: List[str], tok, encoder: str = "nemotron-c0", mode: str = "stream", subsets: Optional[List[str]] = None,
                 delays=(2, 3, 4, 6), max_per_chunk: int = 0, max_items: Optional[int] = None, seed: int = 0, qc: bool = True, align_root: Optional[str] = None, online: bool = True):
        """online=True(기본): 특징 캐시 없이 manifest 의 오디오를 조립해 돌려준다(K = round(duration·12.5)). False: 특징 캐시(index.jsonl) 경로."""
        self.tok = tok; self.sp_ids = add_specials(tok); self.sp = specials_of(self.sp_ids); self.delays = tuple(delays); self.M = max_per_chunk; self.online = online
        self.audio_pad = tok.convert_tokens_to_ids("<|audio_pad|>"); self._pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        self.items: List[dict] = []; self.dropped = 0; self.no_align = 0; self.align_dirs: Dict[str, str] = {}
        class _Rows:                                        # online: manifest 행이 특징 index 역할(id → frames·subset·mode·segments)
            def __init__(self, name):
                self.rows = {r["id"]: dict(frames=frames_for(r["duration_s"]), subset=r["subset"], mode=r["mode"], duration_s=r["duration_s"],
                                           segments=[dict(path=s["path"], offset_s=s["offset_s"], silence_before_s=s["silence_before_s"], dur_s=s["dur_s"]) for s in r["segments"]])
                             for r in read_streams(os.path.join(MAN, name))}; self.frame_hz = 1 / CHUNK_S
        for name in manifests:
            fi = _Rows(name) if online else FeatureIndex(FEAT, encoder, name); assert abs(fi.frame_hz - 1 / CHUNK_S) < 1e-6, f"{encoder} frame_hz {fi.frame_hz} != 12.5"
            # 정렬 루트: 명시 > align2(선행 공백 규약, 2026-09-05) > align. 서버 산출물은 지우지 않으므로 새 규약은 새 루트에 쌓인다.
            cands = [align_root] if align_root else ([os.environ["VAPASR_ALIGN_ROOT"]] if os.environ.get("VAPASR_ALIGN_ROOT") else []) + [os.path.join(MAN, "align-asr-tn-v1"), os.path.join(MAN, "align2"), os.path.join(MAN, "align")]
            adir = next((os.path.join(c, name) for c in cands if os.path.isdir(os.path.join(c, name))), os.path.join(cands[-1], name)); self.align_dirs[name] = adir
            # asr-tn-v1.0.0 관문: 정렬 산출물의 textnorm fingerprint 가 없거나(동결 전 align2/align) 코드와 다르면 실패. 재현은 VAPASR_ALLOW_LEGACY_TN=1
            from ..data.textnorm import check_fingerprint
            fpp = os.path.join(adir, "fingerprint.json"); check_fingerprint(json.load(open(fpp)) if os.path.exists(fpp) else None, f"dataset {name} ← {adir}")
            # 항목 캐시: 정렬 jsonl 수만 개를 매 프로세스가 읽으면 Lustre 에서 수십 분 걸린다(8-rank DDP 면 ×8). 한 번 만들어 _items.json.gz 에 저장하고
            # (정렬 파일 수가 같으면) 재사용한다. 모든 mode·subset 을 담고 메모리에서 거른다.
            import gzip
            parts: Dict[str, list] = {}; pdir = os.path.join(adir, "parts")                 # 청크별 part 파일(NFS 친화) — 있으면 우선, 없으면 스트림별 파일
            if os.path.isdir(pdir):
                for pf in sorted(os.listdir(pdir)):
                    if not pf.endswith(".jsonl"): continue
                    for line in open(os.path.join(pdir, pf), encoding="utf-8"):
                        try: r = json.loads(line); parts.setdefault(r["id"], r["utts"])
                        except Exception: pass
            n_files = (sum(1 for f in os.listdir(adir) if f.endswith(".jsonl")) if os.path.isdir(adir) else 0) + len(parts); cache = os.path.join(adir, "_items-online.json.gz" if online else "_items.json.gz"); allitems = None
            if os.path.exists(cache):
                try:
                    c = json.load(gzip.open(cache, "rt", encoding="utf-8"))
                    if c.get("n_files") == n_files and c.get("n_rows") == len(fi.rows): allitems = c["items"]; self.dropped += c["dropped"]; self.no_align += c["no_align"]
                except Exception: allitems = None
            if allitems is None:
                allitems, dropped, no_align = [], 0, 0
                legacy = {f[:-6] for f in os.listdir(adir) if f.endswith(".jsonl")} if os.path.isdir(adir) else set()
                for sid, row in fi.rows.items():
                    if sid in parts: utts = parts[sid]
                    elif sid in legacy: utts = _read_jsonl(os.path.join(adir, sid + ".jsonl"))
                    else: no_align += 1; continue
                    bad = [u for u in utts if qc and bad_utterance(u)]
                    if bad: dropped += 1; continue                       # 불량 발화(동일 종료시각 뭉침)가 있는 스트림은 통째로 제외
                    toks = sorted(((t["id"], t["end_time"]) for u in utts for t in u["tokens"]), key=lambda x: x[1])
                    allitems.append(dict(name=name, id=sid, K=int(row["frames"]), npy=row.get("npy"), lang=lang_of(name), subset=row.get("subset"), mode=row.get("mode", "stream"), tokens=toks, text=" ".join(u["text"] for u in utts),
                                         **({"duration_s": row["duration_s"], "segments": row["segments"]} if online else {})))
                self.dropped += dropped; self.no_align += no_align
                try:
                    with gzip.open(cache + f".tmp{os.getpid()}", "wt", encoding="utf-8") as f: json.dump(dict(n_files=n_files, n_rows=len(fi.rows), dropped=dropped, no_align=no_align, items=allitems), f, ensure_ascii=False)
                    os.replace(cache + f".tmp{os.getpid()}", cache)
                except Exception: pass
            for it in allitems:
                if it["mode"] != mode or (subsets and it["subset"] not in subsets): continue
                self.items.append(dict(it, tokens=[tuple(t) for t in it["tokens"]]))
        random.Random(seed).shuffle(self.items)
        if max_items: self.items = self.items[:max_items]

    def __len__(self): return len(self.items)
    def prefix(self, lang: str, delay: int) -> List[int]:
        return self._pre + self.tok(f"language {lang}<asr_text>", add_special_tokens=False)["input_ids"] + [self.sp_ids[f"<DELAY_{delay}>"]]

    def audio(self, it: dict) -> np.ndarray:
        """online: manifest 세그먼트를 조립한 float32 (T,) @16 kHz (특징 캐시 추출과 같은 assemble_stream)."""
        return assemble_stream(dict(id=it["id"], duration_s=it["duration_s"], segments=it["segments"]))
    def stream(self, i: int, delay: int, audio: bool = True):
        """(x, ref tokens [(id,end)], chunks, stats, item) — 평가용 공개 API. x = online 이면 wav (T,) float32, 아니면 feats (1,K,D) float32; audio=False 면 None."""
        it = self.items[i]
        f = None if not audio else (self.audio(it) if self.online else np.asarray(np.load(it["npy"], mmap_mode="r")[:, : it["K"]]).astype(np.float32))
        chunks, st = build_interleaved([it["tokens"]], (it["K"] - 0.5) * CHUNK_S, self.sp, chunk_s=CHUNK_S, delay_frames=delay, max_per_chunk=self.M, add_spk_tags=False)
        return f, it["tokens"], chunks, st, it

    def sequence(self, i: int, delay: int, audio: bool = True):
        f, toks, chunks, st, it = self.stream(i, delay, audio); pre = self.prefix(it["lang"], delay)
        ids, is_input, chunk_of, n_rounds, n_flush = build_mono_sequence(chunks, it["K"], pre, self.audio_pad, self.sp, self.M)
        return f, ids, is_input, chunk_of, st, it, n_rounds, n_flush

    def check_targets(self, i: int, delay: int) -> bool:
        """라벨 위치의 텍스트 토큰을 이어 붙이면 참조 토큰열과 정확히 같은가 (overfit 검사용 assert)."""
        _, ids, is_input, _, _, it, _, _ = self.sequence(i, delay, audio=False); skip = {self.sp.next_audio, self.sp.empty_audio}
        got = [t for t, inp in zip(ids, is_input) if not inp and t not in skip]
        return got == [t for t, _ in it["tokens"]]

    def __getitem__(self, i):
        delay = random.choice(self.delays); f, ids, is_input, chunk_of, st, it, n_rounds, n_flush = self.sequence(i, delay)
        lab = [(-100 if inp else t) for t, inp in zip(ids, is_input)]
        return dict(**({"wav": torch.from_numpy(f)} if self.online else {"feats": torch.from_numpy(f)}), ids=torch.tensor(ids), is_audio=torch.tensor([c >= 0 for c in chunk_of]), chunk_of=torch.tensor(chunk_of), labels=torch.tensor(lab),
                    lang=it["lang"], delay=delay, name=it["name"], id=it["id"], n_text=st.tokens, overflow=st.overflow_tokens, n_flush=n_flush, K=it["K"], rounds=n_rounds)

class BucketBatchSampler(Sampler):
    """길이(K) 순으로 정렬해 비슷한 길이끼리 배치 → padding 낭비를 줄인다. 배치 순서는 매 epoch 셔플(seed+epoch 로 결정적).
    DDP: rank/world 를 주면 셔플된 배치 목록을 rank 별로 나눠 갖는다(모든 rank 가 같은 epoch 에 같은 순열을 만들고 자기 몫만 취함)."""
    def __init__(self, ds: MonoStreamDataset, bs: int, seed: int = 0, drop_last: bool = True, rank: int = 0, world: int = 1):
        order = sorted(range(len(ds)), key=lambda i: ds.items[i]["K"]); self.batches = [order[i: i + bs] for i in range(0, len(order), bs)]
        if drop_last and self.batches and len(self.batches[-1]) < bs: self.batches = self.batches[:-1]
        self.seed, self.rank, self.world, self.epoch = seed, rank, world, 0
        self.n = len(self.batches) // world                      # rank 당 배치 수(균등, 나머지 버림)
    def set_epoch(self, e: int): self.epoch = e
    def __iter__(self):
        b = list(self.batches); random.Random(f"{self.seed}:{self.epoch}").shuffle(b); return iter(b[self.rank::self.world][: self.n])
    def __len__(self): return self.n

def collate_streams(batch):
    B = len(batch); L = max(len(b["ids"]) for b in batch); online = "wav" in batch[0]
    ids = torch.zeros(B, L, dtype=torch.long); is_audio = torch.zeros(B, L, dtype=torch.bool)
    chunk_of = torch.full((B, L), -1, dtype=torch.long); labels = torch.full((B, L), -100, dtype=torch.long); mask = torch.zeros(B, L, dtype=torch.long)
    if online:                                      # 오디오(가변 길이) 패딩 + 길이 + 목표 프레임 수 K → 모델이 인코더를 돌린다
        Lw = max(b["wav"].shape[0] for b in batch); wav = torch.zeros(B, Lw); wav_len = torch.zeros(B, dtype=torch.long); Kt = torch.zeros(B, dtype=torch.long)
        for i, b in enumerate(batch): n = b["wav"].shape[0]; wav[i, :n] = b["wav"]; wav_len[i] = n; Kt[i] = b["K"]
        x = dict(wav=wav, wav_len=wav_len, K=Kt)
    else:
        K = max(b["feats"].shape[1] for b in batch); D = batch[0]["feats"].shape[2]; feats = torch.zeros(B, 1, K, D)
        for i, b in enumerate(batch): feats[i, :, : b["feats"].shape[1]] = b["feats"]
        x = dict(feats=feats)
    for i, b in enumerate(batch):
        n = len(b["ids"]); ids[i, :n] = b["ids"]; is_audio[i, :n] = b["is_audio"]; chunk_of[i, :n] = b["chunk_of"]; labels[i, :n] = b["labels"]; mask[i, :n] = 1
    return dict(**x, ids=ids, is_audio=is_audio, chunk_of=chunk_of, labels=labels, mask=mask, lang=[b["lang"] for b in batch], delay=[b["delay"] for b in batch],
                n_text=sum(b["n_text"] for b in batch), n_flush=sum(b["n_flush"] for b in batch), frames=sum(b["K"] for b in batch))
