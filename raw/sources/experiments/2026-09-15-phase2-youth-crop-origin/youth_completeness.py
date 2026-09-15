import os, glob, zipfile, json, collections, unicodedata as ud
def nfc(s): return ud.normalize("NFC", s)
NIA="/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/asr_db/korean_16kHz/NIA24"
YT=f"{NIA}/134-2_Emotion_Conv_Youth/Training"
crops={}
for x in os.listdir(YT):
    st=collections.defaultdict(set)
    for f in os.listdir(f"{YT}/{x}"):
        if f.endswith(".wav"):
            s,n=f[:-4].rsplit("_",1)
            try: st[s].add(int(n))
            except: pass
    crops[nfc(x)]=st
idx={}
for z in sorted(glob.glob(f"{NIA}/134-2_Emotion_Conv_Youth/Json/*.zip"))+sorted(glob.glob("/soundai/DB/raw/aihub/71632/**/*.zip",recursive=True)):
    try: zf=zipfile.ZipFile(z)
    except Exception: continue
    sub=nfc(os.path.basename(z)).replace(".zip","")
    for m in zf.namelist():
        if m.endswith(".json"): idx.setdefault(os.path.splitext(os.path.basename(m))[0],(z,m,sub))
zcache={}
res=collections.defaultdict(lambda: collections.defaultdict(float)); cnt=collections.defaultdict(collections.Counter)
for s,(z,m,sub) in idx.items():
    cd={"TL_01.실내":"TS_01.실내","TL_02.실외":"TS_02.실외"}.get(sub)
    if cd is None or cd not in crops: continue
    zf=zcache.setdefault(z,zipfile.ZipFile(z))
    try: j=json.loads(zf.read(m).decode("utf-8"))
    except Exception: continue
    n=len(j.get("Conversation",[]))
    try: fl=float(str(j["File"]["FileLength"]).replace(",",""))
    except Exception: fl=0.0
    have=len([i for i in crops[cd].get(s,()) if 1<=i<=n])
    frac=have/n if n else 0
    b="complete" if have==n and n>0 else ("ge95" if frac>=0.95 else ("ge80" if frac>=0.8 else ("some" if have>0 else "none")))
    cnt[sub][b]+=1; res[sub][b]+=fl/3600; res[sub]["label_total"]+=fl/3600
for sub in sorted(cnt):
    print("== youth %s: dialogs=%d label_hours=%.1f" % (sub, sum(cnt[sub].values()), res[sub]["label_total"]))
    for b in ("complete","ge95","ge80","some","none"):
        print("   %s: dialogs=%d hours=%.1f" % (b, cnt[sub][b], res[sub][b]))
