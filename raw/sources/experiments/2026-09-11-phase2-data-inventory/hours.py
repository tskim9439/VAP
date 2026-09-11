import os, json, glob, unicodedata as ud, collections
def nfc(s): return ud.normalize("NFC", s)
L="/soundai/users/tskim/VAPKT-data/data/labels/aihub71631"
K="/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/asr_db/korean_16kHz/NIA24"
def fl(j):
    try: return float(str(j["File"]["FileLength"]).replace(",",""))
    except Exception: return 0.0
# adult: hours per label subset, and crop coverage at dialog level (outdoor complete known; indoor per-stem counts from earlier: 8270 stems present)
tot=collections.defaultdict(float); n=collections.Counter(); utts=collections.Counter()
for sub in os.listdir(L):
    for root,_,files in os.walk(os.path.join(L,sub)):
        for f in files:
            if not f.endswith(".json"): continue
            j=json.load(open(os.path.join(root,f),encoding="utf-8")); s=nfc(sub); tot[s]+=fl(j); n[s]+=1; utts[s]+=len(j.get("Conversation",[]))
print("## adult (71631) label hours by subset")
for s in sorted(tot): print(f"  {s}: dialogs={n[s]} hours={tot[s]/3600:.1f} utts={utts[s]}")
print(f"  TOTAL hours={sum(tot.values())/3600:.1f}")
# youth labels at /soundai/DB/raw/aihub/71632
Y="/soundai/DB/raw/aihub/71632"
ytot=collections.defaultdict(float); yn=collections.Counter(); yutt=collections.Counter(); ystems=collections.defaultdict(set)
for root,_,files in os.walk(Y):
    for f in files:
        if not f.endswith(".json"): continue
        p=os.path.join(root,f)
        try: j=json.load(open(p,encoding="utf-8"))
        except Exception: continue
        key=nfc(os.path.relpath(root,Y).split(os.sep)[0]) if os.path.relpath(root,Y)!="." else "."
        # find subset name TL_01.실내 etc in path
        sub=next((nfc(x) for x in os.path.relpath(root,Y).split(os.sep) if x.startswith(("TL_","VL_"))), key)
        ytot[sub]+=fl(j); yn[sub]+=1; yutt[sub]+=len(j.get("Conversation",[])); ystems[sub].add(os.path.splitext(f)[0])
print("## youth (71632) label hours by subset")
for s in sorted(ytot): print(f"  {s}: dialogs={yn[s]} hours={ytot[s]/3600:.1f} utts={yutt[s]}")
print(f"  TOTAL hours={sum(ytot.values())/3600:.1f}")
# youth crop coverage at dialog level
for sub, d in (("TL_01.실내","TS_01.실내"),("TL_02.실외","TS_02.실외")):
    dd=None
    for x in os.listdir(f"{K}/134-2_Emotion_Conv_Youth/Training"):
        if nfc(x)==d: dd=f"{K}/134-2_Emotion_Conv_Youth/Training/{x}"
    if dd is None or sub not in ystems: print("skip", sub, d); continue
    have=set(); tot_files=0
    for f in os.listdir(dd):
        tot_files+=1; have.add(f.rsplit("_",1)[0])
    inter=ystems[sub]&have
    print(f"## youth {d}: crop_files={tot_files} crop_stems={len(have)} label_stems={len(ystems[sub])} overlap={len(inter)} label_utts={yutt[sub]} → file/utt ratio={tot_files/max(1,yutt[sub]):.2f}")
print("DONE")
