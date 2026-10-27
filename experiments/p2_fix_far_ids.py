#!/usr/bin/env python
"""원거리 대화 conv_id 고치기(2026-10-27, 사용자 "ami-far conv_id 고쳐줘"): p2_build_far.py 의 변수 덮어쓰기(버린 변형 출력 반복문이 출력 키 k 를 덮었다)로
<key>-far.refined.dialogues.jsonl 의 conv_id 앞에 마지막으로 버린 기록(dict 문자열)이 붙고 build.json 도 그 이름으로 저장됐다 — 버린 변형이 있었던 ami-far 만.
  python experiments/p2_fix_far_ids.py --data <phase2> --key ami [--dry-run]
- conv_id 를 "<key>-far:<회의>@<장치>"(meta.far_of 의 회의 · meta.mono_src.device)로 바꾼다. 옛 conv_id 끝이 "-far:<회의>@<장치>" 인지 확인한다.
- mono 캐시(<data>/mono/<corpus>/<conv>.npy · .stems.npy · .stems.json — dialogue_dataset.mono_cache_path)는 새 이름으로 링크한다(복사 없음, 옛 이름도 남는다).
  --link auto(기본)는 하드 링크를 먼저 해 보고 저장소가 막으면(mxc VAPKT-data = Azure Blob NFS, Errno 524) 같은 폴더 상대 심볼릭 링크로 한다.
- jsonl 은 옛 파일을 <파일>.bad-conv-id 로 복사해 남긴 뒤 새 파일로 원자적으로 바꾼다. 옛 이름의 build.json 은 <key>-far.build.json 으로 복사한다(없을 때만).
아무것도 지우지 않는다. 다시 돌려도 된다(이미 고친 행·있는 링크는 건너뛴다)."""
import argparse, glob, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--key", required=True); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--link", choices=["auto", "hard", "sym"], default="auto")
    a = ap.parse_args()
    from vapasr.data.dialogue import Dialogue
    from vapasr.data.dialogue_dataset import mono_cache_path
    f = os.path.join(a.data, f"{a.key}-far.refined.dialogues.jsonl"); bak = f + ".bad-conv-id"; mono = os.path.join(a.data, "mono")
    rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    out, links, have, miss, done = [], [], 0, [], 0
    for r in rows:
        m = r["meta"]["far_of"].split(":", 1)[1]; dev = r["meta"]["mono_src"]["device"]; new = f"{a.key}-far:{m}@{dev}"
        if r["conv_id"] == new: out.append(r); done += 1; continue
        assert r["conv_id"].endswith(f"-far:{m}@{dev}"), f"꼴이 다르다: {r['conv_id'][:200]}"
        x = dict(r, conv_id=new)
        p0 = mono_cache_path(mono, Dialogue.from_json(json.dumps(r)))[:-4]; p1 = mono_cache_path(mono, Dialogue.from_json(json.dumps(x)))[:-4]
        for suf in (".npy", ".stems.npy", ".stems.json"):
            if not os.path.exists(p0 + suf): continue
            if os.path.exists(p1 + suf):
                assert os.path.samefile(p0 + suf, p1 + suf), f"새 이름에 다른 파일이 있다: {p1 + suf}"; have += 1
            else: links.append((p0 + suf, p1 + suf))
        if not os.path.exists(p0 + ".npy") and not os.path.exists(p1 + ".npy"): miss.append(new)
        out.append(x)
    ids = [r["conv_id"] for r in out]; assert len(set(ids)) == len(ids), "새 conv_id 가 겹친다"
    bj = [p for p in glob.glob(os.path.join(a.data, "*-far.build.json")) if os.path.basename(p) != f"{a.key}-far.build.json"
          and json.load(open(p, encoding="utf-8")).get("corpus") == a.key]
    print(f"{a.key}-far: 행 {len(rows)} · 고칠 행 {len(rows) - done} · 이미 고침 {done} · 캐시 링크 {len(links)} · 링크 있음 {have} · 캐시 없음 {len(miss)} · 옛 build.json {len(bj)}", flush=True)
    for p0, p1 in links[:2]: print(f"  링크 예 {os.path.basename(p0)[:80]}… → {os.path.basename(p1)}", flush=True)
    if a.dry_run or done == len(rows) and not links: print("DRY OK" if a.dry_run else "FIX_NOOP", flush=True); return
    mode = a.link
    for p0, p1 in links:
        if mode in ("auto", "hard"):
            try: os.link(p0, p1); mode = "hard"; continue
            except OSError as e:
                if mode == "hard": raise
                print(f"  하드 링크 안 됨({e.errno} {e.strerror}) → 심볼릭 링크", flush=True); mode = "sym"
        os.symlink(os.path.basename(p0), p1)                                                    # 같은 폴더 상대 경로
        assert os.path.samefile(p0, p1)
    tmp = f + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fo:
        for x in out: fo.write(json.dumps(x, ensure_ascii=False) + "\n")
    if not os.path.exists(bak): shutil.copyfile(f, bak)
    os.replace(tmp, f)
    good = os.path.join(a.data, f"{a.key}-far.build.json")
    if bj and not os.path.exists(good): shutil.copyfile(bj[0], good)
    print(f"FIX_OK {f}(옛 파일 {bak}) · 캐시 링크 {len(links)}({mode}){' · build.json → ' + good if bj else ''}", flush=True)


if __name__ == "__main__":
    main()
