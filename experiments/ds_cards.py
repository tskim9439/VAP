#!/usr/bin/env python
"""데이터 카드 생성·검증 (vapasr/data/schema.py). 기존 산출물은 행을 고치지 않고 카드(dataset.json / align.json)만 추가한다.
  python experiments/ds_cards.py --manifests librispeech-960,kspon-full,... --align-root align-asr-tn-v1 [--sample 20000] [--dry]
  python experiments/ds_cards.py --all                       # $MXC_DATA_MANIFEST_DIR 아래 streams.jsonl 이 있는 모든 디렉토리 + align-asr-tn-v1/*
출력: 디렉토리별 행 수·시간·오류 요약. --dry 는 쓰지 않고 검증만."""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data.schema import build_dataset_card, build_align_card, write_card, CARD, ALIGN_CARD, export_json_schema
ap = argparse.ArgumentParser(); ap.add_argument("--manifests", default=""); ap.add_argument("--all", action="store_true"); ap.add_argument("--align-root", default="align-asr-tn-v1")
ap.add_argument("--sample", type=int, default=None, help="검증할 행 수(기본 전체)"); ap.add_argument("--dry", action="store_true"); ap.add_argument("--export-schema", default=None, help="JSON Schema 를 이 디렉토리에 기록")
a = ap.parse_args()
MAN = os.environ.get("MXC_DATA_MANIFEST_DIR", os.environ.get("DATA_MANIFEST_DIR", "/tmp"))
if a.export_schema:
    os.makedirs(a.export_schema, exist_ok=True)
    for k, v in export_json_schema().items(): json.dump(v, open(os.path.join(a.export_schema, f"{v['$id'].replace('/', '.')}.json"), "w"), indent=1, ensure_ascii=False); print("schema →", os.path.join(a.export_schema, f"{v['$id'].replace('/', '.')}.json"))
names = [m for m in a.manifests.replace(":", ",").split(",") if m] if not a.all else sorted(d for d in os.listdir(MAN) if os.path.exists(os.path.join(MAN, d, "streams.jsonl")))
for m in names:
    d = os.path.join(MAN, m); t = time.time(); card, errs = build_dataset_card(d, sample=a.sample)
    print(f"[{m}] rows {card['rows']} · {card['hours']} h · subsets {len(card['subsets'])} · modes {card['modes']} · textnorm {card['textnorm_version']} · 오류 {len(errs)} ({time.time()-t:.0f}s)", flush=True)
    for e in errs[:5]: print("   !", e)
    if not a.dry:
        try: write_card(d, card, CARD)
        except OSError as e: print(f"   !! dataset.json 기록 실패({e}) — 계속 진행(정렬 카드는 별도 디렉토리)", flush=True)    # 컨테이너(root)가 만든 디렉토리에 SLURM 사용자가 못 쓰는 경우(67077)
    ad = os.path.join(MAN, a.align_root, m)
    if os.path.isdir(ad):
        t = time.time(); acard, aerrs = build_align_card(ad, manifest_name=m, sample=a.sample); rec = acard["records"]
        print(f"   align/{a.align_root}: streams {rec['streams']} (empty {rec['empty']}, dup lines {rec['duplicate_lines']}) · utts {rec['utts']} · tokens {rec['tokens']} · parts {rec['part_files']} · 오류 {len(aerrs)} ({time.time()-t:.0f}s)", flush=True)
        for e in aerrs[:5]: print("   !", e)
        if not a.dry: write_card(ad, acard, ALIGN_CARD)
