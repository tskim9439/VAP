# 정체 회복 학습 실데이터 점검: 학습 대화 셋(split v2 held-out 제외, 맞장구 1.0,3)에서 창 표본 × δ4·10 에 p=1 로 뒤집어 성공률·종류·가린 출력 비율·시퀀스 점검. 읽기만 한다.
import sys, json, random, collections, statistics
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from transformers import AutoTokenizer
from vapasr.data.cst_dialogue import CstDialogueDataset, Recover
D = "/soundai/users/tskim/VAPKT-data/data/phase2"; split = json.load(open(f"{D}/splits/v2.json"))
tok = AutoTokenizer.from_pretrained("/soundai/Model/VAPASR/stage3-q06-st3-r1/checkpoint-2000"); rc = Recover(p=1.0)
for key, cs in (("dlg_ko", ["aihub71631", "aihub134-1", "aihub134-2"]), ("dlg_en", ["otoSpeech"]), ("dlg_de", ["kec", "ztc", "nsc"])):
    held = {c for x in cs for c in split["corpora"].get(x, {}).get("heldout", [])}
    ds = CstDialogueDataset([f"{D}/{c}.refined.dialogues.jsonl" for c in cs], tok, delays=(4, 10), exclude=held, window_s=(20.0, 40.0), hop_s=10.0, seed=0,
                            mono_cache_dir=f"{D}/mono", decision_weight=2.0, backchannel=(1.0, 3), recover=rc)
    idx = random.Random(0).sample(range(len(ds)), min(800, len(ds))); kinds = collections.Counter(); bad = []; frac = []; span = []; n = ok = 0
    for i in idx:
        for dl in (4, 10):
            s = ds.cst_sequence(i, dl, rng=random.Random(i * 13 + dl)); n += 1
            P = ds.problems_of(s)
            if P: bad.append((ds.items[i], dl, P[:3]))
            f = s["flip"]
            if not f: continue
            ok += 1; kinds[f["kind"]] += 1; span.append(len(f["keys"]))
            for key_ in ("n0", "n1"):
                tid = s["streams"][f[key_][0]][f[key_][1]][0]
                if not tok.decode([tid]).startswith(" "): bad.append((ds.items[i], dl, "not_word_start", key_, tok.decode([tid])))
            outs = [p for p in range(len(s["ids"])) if not s["is_input"][p]]; base = ds.cst_sequence(i, dl)
            lab0 = sum(1 for p in range(len(base["ids"])) if base["labels"][p] != -100); lab1 = sum(1 for p in outs if s["labels"][p] != -100)
            frac.append(1 - lab1 / max(1, lab0))
    print(f"{key}: 창 {len(ds)} · 표본 {len(idx)}×δ2 = {n} · 뒤집기 성공 {ok / n:.3f} · 점검 불일치 {len(bad)} {bad[:2]}")
    print(f"   종류 {dict((k, round(v / max(1, ok), 3)) for k, v in sorted(kinds.items()))} · 구간 원소 중앙값 {statistics.median(span)} p90 {sorted(span)[int(.9 * (len(span) - 1))]} · "
          f"가려진 label 비율(창 대비) 중앙값 {statistics.median(frac):.3f} 평균 {statistics.mean(frac):.3f}", flush=True)
