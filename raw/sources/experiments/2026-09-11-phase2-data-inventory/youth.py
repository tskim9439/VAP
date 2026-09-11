import os, json, zipfile, unicodedata as ud, collections, glob
def nfc(s): return ud.normalize("NFC", s)
Y="/soundai/DB/raw/aihub/71632"
print("## aihub/71632 tree (depth 3)")
for root,dirs,files in os.walk(Y):
    dep=os.path.relpath(root,Y).count(os.sep)
    if dep<=2: print("  ", os.path.relpath(root,Y), "dirs=",len(dirs), "files=",len(files), [f for f in files[:3]])
    if dep>=2: dirs[:]=[]
K="/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/asr_db/korean_16kHz/NIA24/134-2_Emotion_Conv_Youth"
zips=sorted(glob.glob(f"{K}/Json/*.zip"))+sorted(glob.glob(f"{Y}/**/*.zip", recursive=True))
print("## zips", [ (z, os.path.getsize(z)) for z in zips][:8])
crops={}
for x in os.listdir(f"{K}/Training"):
    d=f"{K}/Training/{x}"; st=collections.Counter()
    for f in os.listdir(d): st[f.rsplit("_",1)[0]]+=1
    crops[nfc(x)]=st; print(f"## crops {nfc(x)}: files={sum(st.values())} stems={len(st)}")
seen=set()
for z in zips:
    name=nfc(os.path.basename(z)); sub=name.replace(".zip","")
    if sub in seen: continue
    seen.add(sub)
    try: zf=zipfile.ZipFile(z)
    except Exception as e: print("bad zip", z, e); continue
    mem=[m for m in zf.namelist() if m.endswith(".json")]
    hours=0.0; utts=0; stems=set(); n=0
    for m in mem:
        try: j=json.loads(zf.read(m).decode("utf-8"))
        except Exception: continue
        n+=1; stems.add(os.path.splitext(os.path.basename(m))[0]); utts+=len(j.get("Conversation",[]))
        try: hours+=float(str(j["File"]["FileLength"]).replace(",",""))/3600
        except Exception: pass
    cd = {"TL_01.실내":"TS_01.실내","TL_02.실외":"TS_02.실외"}.get(sub)
    cov=""
    if cd and cd in crops:
        st=crops[cd]; inter=[s for s in stems if s in st]; files=sum(st[s] for s in inter)
        cov=f" | crops: stems_with_audio={len(inter)}/{len(stems)} files={files} file/utt={files/max(1,utts):.2f}"
    print(f"## youth {sub}: dialogs={n} hours={hours:.1f} utts={utts}{cov}")
print("DONE")
