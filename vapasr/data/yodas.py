"""YODAS-Granary EN(en129 샤드) 리더 — asr-tn-v1.2.0 corpus parser.
서버 배치: EN/TRAIN/OPEN/yodas-granary/en129/asr_only/<utt_id>.wav (16 kHz PCM16 mono, 유튜브 세그먼트 6–40 s, 138,589 개).
라벨: nvidia/Granary en/yodas/en_asr.jsonl(22 GB, CC-BY-4.0) 에서 en129 줄만 뽑은 granary_yodas_en129.jsonl — {utt_id, text, duration, video}.
  text 는 Whisper/Canary 계열 의사 라벨(대소문자·구두점·숫자 표기 있음) → target_en(text, "yodas") 로 소문자·구두점 제거·숫자 단어화(NUMERIC_CORPORA_EN 에 yodas 등록).
  표본 추출은 영상(video) 단위(같은 영상의 세그먼트가 train/표본에 갈리지 않도록)."""
import os, json
from typing import Iterator

def read_labels(jsonl: str) -> Iterator[dict]:
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line); yield dict(utt_id=r["utt_id"], video=r.get("video") or r["utt_id"].split("_")[2], raw=r["text"], dur_s=round(float(r["duration"]), 3))

def wav_path(root: str, utt_id: str) -> str:
    shard = utt_id.split("_")[0]; return os.path.join(root, shard, "asr_only", utt_id + ".wav")
