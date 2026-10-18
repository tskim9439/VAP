# 되살리기 v2 정렬 입력(2026-10-19): 기존 정렬 결과에 없는(정렬 대상이 아니었던) REVIVABLE 격리 발화만 담은 대화 파일을 만든다.
#   한국어(134-1·134-2): 정제 전 <corpus>.dialogues.jsonl 의 정규화 text 를 그대로 쓴다(정렬이 그 발화가 생기기 전 판으로 돌았다).
#   NSC: 정제 전 text 는 비어 있다(격리라 비움, nsc_pseudo_label build) → raw(교사 Qwen 전사)를 target_de 로 정규화한다(TN 사유가 생기면 뺀다).
#   python revive_extra_input.py <phase2> <out dir> <corpus,…> → <out>/<corpus>.dialogues.jsonl (p2_align.py --dialogues 입력). 원본은 읽기만 한다.
import collections, glob, json, os, sys
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.cst_dialogue import REVIVABLE
from vapasr.data.dialogue import Dialogue
from vapasr.data.textnorm_de import target_de, target_flags_de

P, OUT, corpora = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
ALIGN = {"nsc": "align-asr-tn-de-v1", "kec": "align-asr-tn-de-v1", "ztc": "align-asr-tn-de-v1"}
os.makedirs(OUT, exist_ok=True)
for c in corpora:
    done = {}
    for f in glob.glob(f"{P}/{ALIGN.get(c, 'align-asr-tn-v1')}/{c}/parts/*.jsonl"):
        for l in open(f, encoding="utf-8"):
            try: r = json.loads(l)
            except Exception: continue
            done.setdefault(r["conv_id"], set()).update(r.get("utts", {}))
    tgt = {}
    for l in open(f"{P}/{c}.refined.dialogues.jsonl", encoding="utf-8"):
        d = json.loads(l); miss = set((d.get("meta") or {}).get("missing", []))
        for i, u in enumerate(d["utterances"], 1):
            if u.get("flags") and not u.get("tokens") and i not in miss and set(u["flags"]) <= REVIVABLE and u["utt_id"] not in done.get(d["conv_id"], ()):
                tgt.setdefault(d["conv_id"], set()).add(u["utt_id"])
    st = collections.Counter(); dur = 0.0
    with open(f"{OUT}/{c}.dialogues.jsonl.tmp", "w", encoding="utf-8") as f:
        for l in open(f"{P}/{c}.dialogues.jsonl", encoding="utf-8"):
            d = Dialogue.from_json(l); want = tgt.get(d.conv_id)
            if not want: continue
            keep = []
            for u in d.utterances:
                if u.utt_id not in want: continue
                if c == "nsc":
                    if not (u.raw or "").strip() or target_flags_de(u.raw): st["tn_flag"] += 1; continue
                    u.text = target_de(u.raw)
                if not (u.text or "").strip(): st["no_text"] += 1; continue
                u.flags = None; u.tokens = None; keep.append(u); dur += u.end - u.start
            if keep: d.utterances = keep; f.write(d.to_json() + "\n"); st["dialogues"] += 1; st["utts"] += len(keep)
    os.replace(f"{OUT}/{c}.dialogues.jsonl.tmp", f"{OUT}/{c}.dialogues.jsonl")
    print(json.dumps(dict(corpus=c, targets=sum(map(len, tgt.values())), **st, hours=round(dur / 3600, 1)), ensure_ascii=False), flush=True)
