"""VoxPopuli EN(유럽의회 본회의) 리더 — asr-tn-v1.2.0 corpus parser.
서버 배치: EN/TRAIN/OPEN/voxpopuli/train/train_part_{0..35}.tar.gz (HF facebook/voxpopuli 오디오 아카이브, 멤버 train_part_k/<session>-<id>.wav, 16 kHz float32 mono).
  → 비압축 tar 로 재포장한 사본(VAPKT-data/data/audio/voxpopuli/train_part_k.tar)을 archive.py 의 "tar::member" 로 무작위 읽기.
라벨: 공식 annotation asr_en.tsv (dl.fbaipublicfiles.com/voxpopuli/annotations/asr/asr_en.tsv.gz, '|' 구분):
  id_|paragraph_id|session_id|speaker_id|original_text|normed_text|decoded|start_time|end_time|cer|wer|vad|split|gender
  wav 이름 = f"{session_id}-{id_}"; normed_text 는 소문자·구두점 제거·숫자 단어화가 이미 된 텍스트(asr-tn 표면 규칙만 다시 적용). split ∈ train/dev/test/invalid."""
import os, csv, json, glob
from typing import Dict, Iterator, List, Optional
csv.field_size_limit(1 << 30)

def read_annotations(tsv: str, splits=("train",)) -> Iterator[dict]:
    with open(tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="|"):
            if r["split"] not in splits: continue
            try: st, en = float(r["start_time"]), float(r["end_time"])
            except Exception: continue
            yield dict(utt_id=f"{r['session_id']}-{r['id_']}", session=r["session_id"], speaker=(None if r["speaker_id"] in ("None", "") else r["speaker_id"]), gender=r["gender"],
                       raw=r["original_text"], normed=r["normed_text"], dur_s=round(en - st, 3), split=r["split"])

def member_map(audio_dir: str, cache: Optional[str] = None) -> Dict[str, str]:
    """wav 이름(확장자 없음) → "<tar>::<member>". 36 개 tar 의 인덱스(archive.tar_index) 를 합쳐 한 번 캐시한다."""
    from .archive import tar_index, SEP
    cache = cache or os.path.join(audio_dir, "_members.json")
    if os.path.exists(cache): return json.load(open(cache))
    m: Dict[str, str] = {}
    for tar in sorted(glob.glob(os.path.join(audio_dir, "*.tar"))):
        for member in tar_index(tar):
            if member.endswith(".wav"): m[os.path.splitext(os.path.basename(member))[0]] = f"{tar}{SEP}{member}"
    json.dump(m, open(cache + ".tmp", "w")); os.replace(cache + ".tmp", cache); return m
