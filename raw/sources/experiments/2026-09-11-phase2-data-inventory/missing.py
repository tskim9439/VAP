import os, json, unicodedata as ud, collections, random
def find_dir(parent, name):
    for d in os.listdir(parent):
        if ud.normalize("NFC", d) == ud.normalize("NFC", name): return os.path.join(parent, d)
K=find_dir("/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/asr_db/korean_16kHz/NIA24/134-1_Emotion_Conv_Adult/Training","TS_01.실내")
L=find_dir("/soundai/users/tskim/VAPKT-data/data/labels/aihub71631","TL_01.실내")
js=[]
for root,_,files in os.walk(L):
    js+= [os.path.join(root,f) for f in files if f.endswith(".json")]
random.seed(7); random.shuffle(js)
tot=collections.Counter(); pres=collections.Counter(); dl=collections.Counter(); dur_p=[]; dur_m=[]; emo_p=collections.Counter(); emo_m=collections.Counter(); n=0
for jp in js[:12]:
    s=os.path.splitext(os.path.basename(jp))[0]; j=json.load(open(jp,encoding="utf-8")); u=j.get("Conversation") or []
    if not u: continue
    n+=1
    for i,x in enumerate(u,1):
        sp=x.get("SpeakerNo"); f=os.path.exists(f"{K}/{s}_{i}.wav"); tot[sp]+=1
        d=float(str(x["EndTime"]).replace(","," ").replace(" ",""))-float(str(x["StartTime"]).replace(",",""))
        e=x.get("SpeakerEmotionLevel") or x.get("VerifyEmotionObject")
        if f: pres[sp]+=1; dur_p.append(d); emo_p[str(e)[:12]]+=1
        else: dur_m.append(d); emo_m[str(e)[:12]]+=1
import statistics as st
print(f"dialogs={n} total_by_spk={dict(tot)} present_by_spk={dict(pres)}")
print(f"dur present med={st.median(dur_p):.2f} mean={st.mean(dur_p):.2f} | missing med={st.median(dur_m):.2f} mean={st.mean(dur_m):.2f} | short(<0.5s) present={sum(d<0.5 for d in dur_p)} missing={sum(d<0.5 for d in dur_m)}")
print("emo present", emo_p.most_common(4)); print("emo missing", emo_m.most_common(4))
