"""Stage 1 mono 데이터셋 — streams.jsonl 스트림 하나 = 학습 샘플 하나 (plans/stage1-mono-pilot.md §4).

입력: 특징 캐시 index.jsonl(experiments/s1_extract_features.py, (1,T',D) @12.5 Hz, mode·subset 포함)
    + 정렬 align/<manifest>/<id>.jsonl(experiments/s1_align.py, speaker=0)
시퀀스 (화자 태그 없음):
  [prefix … <DELAY_d>] [AUDIO_0] tok… <NEXT_AUDIO> … [AUDIO_K-1] tok… <NEXT_AUDIO>
  flush 라운드: <EMPTY_AUDIO> tok… <NEXT_AUDIO>  ×(ceil(n_flush/M) + 1)   ← 마지막 라운드는 빈 라운드(= "남은 것 없음" 학습)
  <EMPTY_AUDIO> 는 오디오 자리 대신 들어가는 **입력** 토큰(손실 없음). δ 때문에 스트림 끝을 넘긴 토큰은 여기서 방출된다.
chunk k = 특징 프레임 k (12.5 Hz = 80 ms). 스트림 길이가 제각각이라 길이 버킷 배치를 쓴다.
"""
import os, json, math, random
from typing import List, Dict, Tuple, Optional
import numpy as np, torch
from torch.utils.data import Dataset, Sampler
from ..data.interleave import build_interleaved, Specials
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
    rounds = [flush[i: i + M] for i in range(0, len(flush), M)] + [[]]
    for r in rounds:
        ids.append(sp.empty_audio); is_input.append(True); chunk_of.append(-1)
        ids += r + [sp.next_audio]; is_input += [False] * (len(r) + 1); chunk_of += [-1] * (len(r) + 1)
    return ids, is_input, chunk_of, len(rounds), len(flush)

class MonoStreamDataset(Dataset):
    def __init__(self, manifests: List[str], tok, encoder: str = "nemotron-c0", mode: str = "stream", subsets: Optional[List[str]] = None,
                 delays=(2, 3, 4, 6), max_per_chunk: int = 4, max_items: Optional[int] = None, seed: int = 0, qc: bool = True, align_root: Optional[str] = None):
        self.tok = tok; self.sp_ids = add_specials(tok); self.sp = specials_of(self.sp_ids); self.delays = tuple(delays); self.M = max_per_chunk
        self.audio_pad = tok.convert_tokens_to_ids("<|audio_pad|>"); self._pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        self.items: List[dict] = []; self.dropped = 0; self.no_align = 0; self.align_dirs: Dict[str, str] = {}
        for name in manifests:
            fi = FeatureIndex(FEAT, encoder, name); assert abs(fi.frame_hz - 1 / CHUNK_S) < 1e-6, f"{encoder} frame_hz {fi.frame_hz} != 12.5"
            # 정렬 루트: 명시 > align2(선행 공백 규약, 2026-09-05) > align. 서버 산출물은 지우지 않으므로 새 규약은 새 루트에 쌓인다.
            cands = [align_root] if align_root else [os.path.join(MAN, "align2"), os.path.join(MAN, "align")]
            adir = next((os.path.join(c, name) for c in cands if os.path.isdir(os.path.join(c, name))), os.path.join(cands[-1], name)); self.align_dirs[name] = adir
            # 항목 캐시: 정렬 jsonl 수만 개를 매 프로세스가 읽으면 Lustre 에서 수십 분 걸린다(8-rank DDP 면 ×8). 한 번 만들어 _items.json.gz 에 저장하고
            # (정렬 파일 수가 같으면) 재사용한다. 모든 mode·subset 을 담고 메모리에서 거른다.
            import gzip
            n_files = sum(1 for f in os.listdir(adir) if f.endswith(".jsonl")) if os.path.isdir(adir) else 0; cache = os.path.join(adir, "_items.json.gz"); allitems = None
            if os.path.exists(cache):
                try:
                    c = json.load(gzip.open(cache, "rt", encoding="utf-8"))
                    if c.get("n_files") == n_files and c.get("n_rows") == len(fi.rows): allitems = c["items"]; self.dropped += c["dropped"]; self.no_align += c["no_align"]
                except Exception: allitems = None
            if allitems is None:
                allitems, dropped, no_align = [], 0, 0
                for sid, row in fi.rows.items():
                    p = os.path.join(adir, sid + ".jsonl")
                    if not os.path.exists(p): no_align += 1; continue
                    utts = _read_jsonl(p); bad = [u for u in utts if qc and bad_utterance(u)]
                    if bad: dropped += 1; continue                       # 불량 발화(동일 종료시각 뭉침)가 있는 스트림은 통째로 제외
                    toks = sorted(((t["id"], t["end_time"]) for u in utts for t in u["tokens"]), key=lambda x: x[1])
                    allitems.append(dict(name=name, id=sid, K=int(row["frames"]), npy=row["npy"], lang=lang_of(name), subset=row.get("subset"), mode=row.get("mode", "stream"), tokens=toks, text=" ".join(u["text"] for u in utts)))
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

    def stream(self, i: int, delay: int):
        """(feats (1,K,D) float32, ref tokens [(id,end)], chunks, stats, item) — 평가용 공개 API."""
        it = self.items[i]; f = np.asarray(np.load(it["npy"], mmap_mode="r")[:, : it["K"]]).astype(np.float32)
        chunks, st = build_interleaved([it["tokens"]], (it["K"] - 0.5) * CHUNK_S, self.sp, chunk_s=CHUNK_S, delay_frames=delay, max_per_chunk=self.M, add_spk_tags=False)
        return f, it["tokens"], chunks, st, it

    def sequence(self, i: int, delay: int):
        f, toks, chunks, st, it = self.stream(i, delay); pre = self.prefix(it["lang"], delay)
        ids, is_input, chunk_of, n_rounds, n_flush = build_mono_sequence(chunks, it["K"], pre, self.audio_pad, self.sp, self.M)
        return f, ids, is_input, chunk_of, st, it, n_rounds, n_flush

    def check_targets(self, i: int, delay: int) -> bool:
        """라벨 위치의 텍스트 토큰을 이어 붙이면 참조 토큰열과 정확히 같은가 (overfit 검사용 assert)."""
        _, ids, is_input, _, _, it, _, _ = self.sequence(i, delay); skip = {self.sp.next_audio, self.sp.empty_audio}
        got = [t for t, inp in zip(ids, is_input) if not inp and t not in skip]
        return got == [t for t, _ in it["tokens"]]

    def __getitem__(self, i):
        delay = random.choice(self.delays); f, ids, is_input, chunk_of, st, it, n_rounds, n_flush = self.sequence(i, delay)
        lab = [(-100 if inp else t) for t, inp in zip(ids, is_input)]
        return dict(feats=torch.from_numpy(f), ids=torch.tensor(ids), is_audio=torch.tensor([c >= 0 for c in chunk_of]), chunk_of=torch.tensor(chunk_of), labels=torch.tensor(lab),
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
    B = len(batch); K = max(b["feats"].shape[1] for b in batch); D = batch[0]["feats"].shape[2]; L = max(len(b["ids"]) for b in batch)
    feats = torch.zeros(B, 1, K, D); ids = torch.zeros(B, L, dtype=torch.long); is_audio = torch.zeros(B, L, dtype=torch.bool)
    chunk_of = torch.full((B, L), -1, dtype=torch.long); labels = torch.full((B, L), -100, dtype=torch.long); mask = torch.zeros(B, L, dtype=torch.long)
    for i, b in enumerate(batch):
        k = b["feats"].shape[1]; n = len(b["ids"]); feats[i, :, :k] = b["feats"]
        ids[i, :n] = b["ids"]; is_audio[i, :n] = b["is_audio"]; chunk_of[i, :n] = b["chunk_of"]; labels[i, :n] = b["labels"]; mask[i, :n] = 1
    return dict(feats=feats, ids=ids, is_audio=is_audio, chunk_of=chunk_of, labels=labels, mask=mask, lang=[b["lang"] for b in batch], delay=[b["delay"] for b in batch],
                n_text=sum(b["n_text"] for b in batch), n_flush=sum(b["n_flush"] for b in batch), frames=sum(b["K"] for b in batch))
