"""Stage 2 준비 — 동결 전 산출물(특징 캐시·정렬)을 새 manifest 로 이어받는다. 오디오가 같으면 특징은 재사용(spec 버전 정책), 정렬은 lexical_text 가
세그먼트 단위로 완전히 같을 때만 재사용한다(token ID 도 같은 tokenizer 라 동일). 하드링크(Lustre 같은 FS) → 실패 시 복사. 기존 파일은 건드리지 않는다.
python experiments/s2_seed_cache.py --new librispeech-960 --old librispeech-100 [--encoder nemotron-c0] [--old-align align2]"""
import os, sys, json, argparse, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--old", required=True); ap.add_argument("--encoder", default="nemotron-c0")
ap.add_argument("--old-align", default="align2"); ap.add_argument("--new-align", default="align-asr-tn-v1"); ap.add_argument("--feat", default="0", help="1 이면 특징 캐시도 이어받음(기본은 정렬만 — 온라인 특징)"); ap.add_argument("--new-align-root", default=None, help="새 정렬 루트 전체 경로(기본 $MAN/<new-align>)"); a = ap.parse_args()
MAN = os.environ["MXC_DATA_MANIFEST_DIR"]; FEAT = os.environ["MXC_DATA_FEATURE_CACHE_DIR"]
def rows(name):
    p = os.path.join(MAN, name, "streams.jsonl"); return {r["id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8"))} if os.path.exists(p) else {}
def link(src, dst):
    if os.path.exists(dst): return False
    try: os.link(src, dst)
    except OSError:
        try: shutil.copy2(src, dst)                      # 다른 FS → 복사. 그사이 다른 run 이 만든 파일(권한 등)이면 건너뜀
        except OSError: return os.path.exists(dst)
    return True
new, old = rows(a.new), rows(a.old); assert new, f"새 manifest 없음: {a.new}"
same_audio = [i for i in new if i in old and [s["path"] for s in new[i]["segments"]] == [s["path"] for s in old[i]["segments"]]
             and abs(new[i]["duration_s"] - old[i]["duration_s"]) < 1e-6 and [s["offset_s"] for s in new[i]["segments"]] == [s["offset_s"] for s in old[i]["segments"]]]
same_text = [i for i in same_audio if [s["text"] for s in new[i]["segments"]] == [s["text"] for s in old[i]["segments"]]]
print(f"{a.new} ← {a.old}: 공통 id {len(same_audio)} (오디오 동일), 그중 lexical_text 동일 {len(same_text)}", flush=True)
# 특징 캐시(선택)
if a.feat == "1":
  od, nd = os.path.join(FEAT, a.encoder, a.old), os.path.join(FEAT, a.encoder, a.new); os.makedirs(nd, exist_ok=True)
  oi = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(od, "index.jsonl"))} if os.path.exists(os.path.join(od, "index.jsonl")) else {}
  ni = os.path.join(nd, "index.jsonl"); have = {json.loads(l)["id"] for l in open(ni)} if os.path.exists(ni) and os.path.getsize(ni) else set(); n = 0
  with open(ni, "a") as f:
      for i in same_audio:
          if i in have or i not in oi: continue
          e = dict(oi[i]); dst = os.path.join(nd, os.path.basename(e["npy"]))
          if not os.path.exists(e["npy"]): continue
          link(e["npy"], dst); e["npy"] = dst; e["subset"] = new[i]["subset"]; f.write(json.dumps(e) + "\n"); n += 1
  print(f"  특징 {a.encoder}: {n} 개 이어받음 → {nd}", flush=True)
# 정렬
oa, na = os.path.join(MAN, a.old_align, a.old), os.path.join(a.new_align_root or os.path.join(MAN, a.new_align), a.new); os.makedirs(na, exist_ok=True); m = 0
from concurrent.futures import ThreadPoolExecutor                       # 다른 파일시스템(Lustre → Blob NFS)이면 복사가 되므로 스레드 32 개로 병렬
have = set(os.listdir(na))
def _one(i):
    src = os.path.join(oa, i + ".jsonl")
    return os.path.exists(src) and (i + ".jsonl") not in have and link(src, os.path.join(na, i + ".jsonl"))
with ThreadPoolExecutor(32) as ex: m = sum(1 for ok in ex.map(_one, same_text) if ok)
fp = json.load(open(os.path.join(MAN, a.new, "stats.json")))["fingerprint"]; json.dump(fp, open(os.path.join(na, "fingerprint.json"), "w"), indent=1)
print(f"  정렬: {m} 개 이어받음 → {na} (fingerprint.json 기록)", flush=True)
