"""Stage 4 행 단위 점검: 원 utt_id(조각 #n 묶음)가 episode 여럿에 걸치는가, 일반 조각과 되살린 조각이 섞인 행이 있는가."""
import sys, os, json
from collections import Counter
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.dialogue import Dialogue, build_episodes
from vapasr.data.cst_dialogue import REVIVABLE, load_revive
D = "/soundai/users/tskim/VAPKT-data/data/phase2"; RV = D + "/revive-asr-v2"
for c in sys.argv[1].split(","):
    st = Counter(); f = os.path.join(RV, f"{c}.revive.jsonl"); rv = load_revive([f]) if os.path.exists(f) else {}
    for line in open(os.path.join(D, f"{c}.refined.dialogues.jsonl"), encoding="utf-8"):
        d = Dialogue.from_json(line); m = rv.get(d.conv_id, {}); miss = set(d.meta.get("missing", []))
        kind = {}
        for i, u in enumerate(d.utterances, 1):
            if u.tokens and not u.flags: kind[u.utt_id] = "n"
            elif not u.tokens and u.flags and set(u.flags) <= REVIVABLE and i not in miss and u.utt_id in m: kind[u.utt_id] = "r"; u.tokens = list(m[u.utt_id])
        eps = build_episodes(d); ep_of = {}
        for j, e in enumerate(eps):
            for uid in e.utt_ids: ep_of[uid] = j
        rows = {}
        for u in d.utterances:
            if u.utt_id not in kind: continue
            rows.setdefault((u.speaker, u.utt_id.split("#")[0]), []).append(u.utt_id)
        for k, us in rows.items():
            ks = {kind[x] for x in us}; es = {ep_of.get(x) for x in us}
            st["rows"] += 1; st["rows_split"] += len(us) > 1; st["rows_multi_ep"] += len(es) > 1; st["rows_mixed"] += len(ks) > 1
            st["rows_r"] += ks == {"r"}
        # episode 안에 같은 행이 다른 행 사이에 끼어 있나(행 순서가 시각순이 아닌 경우)
        for e in eps:
            b = [x.split("#")[0] for x in e.utt_ids if x in kind]; seen = []
            for x in b:
                if not seen or seen[-1] != x: seen.append(x)
            st["ep_interleaved_rows"] += len(seen) != len(set(seen))
    print(c, dict(st), flush=True)
