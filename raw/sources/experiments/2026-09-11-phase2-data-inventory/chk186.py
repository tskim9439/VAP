import os, json, glob, collections, itertools
B="/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f"
C=f"{B}/asr_db/korean_16kHz/NIA23/186_WelfareCallCenter/Training/Jsons"
cnt=collections.Counter(); n=0; ex=None
for cat in os.listdir(C):
    d=os.path.join(C,cat)
    for f in itertools.islice((x for x in os.listdir(d) if x.endswith(".json") and not x.startswith(".")), 300):
        try: j=json.load(open(os.path.join(d,f),encoding="utf-8"))
        except Exception: continue
        m=j["info"][0]["metadata"]; cnt[(m.get("speaker_type"), m.get("rec_device"), m.get("rec_place"))]+=1; n+=1
        if ex is None: ex=(f, j.get("dialogs"), m)
print("186 sample", n, cnt.most_common(8)); print("example", ex)
# call-level grouping: files share prefix HOS0003122xx? check stem patterns
d=os.path.join(C,"Hospital"); names=[x for x in os.listdir(d) if x.endswith(".json") and not x.startswith(".")][:2000]
pref=collections.Counter(x[:10] for x in names); print("prefix groups (first 10 chars):", len(pref), pref.most_common(3))
suf=collections.Counter(x[-9:-5] for x in names); print("suffix A/B pattern:", collections.Counter(x[-5] for x in names).most_common(4))
# 132 info
I=f"{B}/raw/132.연령대별_특징적_발화(은어·속어_등)_음성_데이터/132_400K_info"
fs=os.listdir(I)[:5]; print("132 info files", len(os.listdir(I)), fs)
for f in fs[:2]:
    p=os.path.join(I,f)
    if os.path.isfile(p): print(f, open(p,errors="ignore").read(500).replace("\n"," | "))
