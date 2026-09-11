import os, sys, glob, json, unicodedata as ud, numpy as np, soundfile as sf, re, collections
def find_dir(parent, name):
    for d in os.listdir(parent):
        if ud.normalize("NFC", d) == ud.normalize("NFC", name): return os.path.join(parent, d)
    raise SystemExit(f"dir not found: {parent}/{name}")
K=find_dir("/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/asr_db/korean_16kHz/NIA24/134-1_Emotion_Conv_Adult/Training","TS_01.실내")
W=find_dir(find_dir("/soundai/DB/raw/aihub/71631_audio/Training","01.원천데이터"),"TS_01.실내_5")
L=find_dir("/soundai/users/tskim/VAPKT-data/data/labels/aihub71631","TL_01.실내")
stems=[os.path.splitext(f)[0] for f in os.listdir(W) if f.endswith(".wav")]
have=[s for s in stems if os.path.exists(f"{K}/{s}_1.wav") or os.path.exists(f"{K}/{s}_2.wav")]
print(f"stereo_stems={len(stems)} with_crops={len(have)}")
jidx={}
for root,_,files in os.walk(L):
    for f in files:
        if f.endswith(".json"): jidx[os.path.splitext(f)[0]]=os.path.join(root,f)
print("json_indexed", len(jidx))
def utt_list(j):
    for k,v in j.items():
        if isinstance(v,list) and v and isinstance(v[0],dict): return k, v
    return None, None
def fnum(x): return float(str(x).replace(",",""))
# B) missing pattern for 2 stems (any stem with crops)
cands=[s for s in have if s in jidx][:2] or [s for s in list(jidx)[:50] if os.path.exists(f"{K}/{s}_1.wav")][:2]
for s in cands:
    j=json.load(open(jidx[s],encoding="utf-8")); key,u=utt_list(j)
    if u is None: print("no utt list", s, list(j.keys())); continue
    idx=set()
    for n in range(1,len(u)+1):
        if os.path.exists(f"{K}/{s}_{n}.wav"): idx.add(n)
    spk=collections.Counter(); spk_have=collections.Counter()
    for i,x in enumerate(u,1):
        sp=x.get("Speaker", x.get("SpeakerID", x.get("speaker","?"))); spk[sp]+=1
        if i in idx: spk_have[sp]+=1
    print(f"## {s} utts={len(u)} key={key} fields={list(u[0].keys())[:8]} crops_present={len(idx)} by_speaker total={dict(spk)} present={dict(spk_have)} first_missing={sorted(set(range(1,len(u)+1))-idx)[:8]}")
    # A) xcorr if stereo exists
    if s in stems:
        x,sr=sf.read(f"{W}/{s}.wav", dtype="float32"); tested=0
        for i,ut in enumerate(u,1):
            if i not in idx: continue
            c,csr=sf.read(f"{K}/{s}_{i}.wav", dtype="float32"); st=fnum(ut.get("StartTime",ut.get("Start",0))); et=fnum(ut.get("EndTime",ut.get("End",0)))
            a=int(st*sr); b=min(len(x), a+len(c)); 
            if b-a < sr//2: continue
            seg=x[a:b]; cc=c[:b-a]
            r=[float(np.corrcoef(cc, seg[:,ch])[0,1]) for ch in (0,1)]; rm=float(np.corrcoef(cc, seg[:,0]+seg[:,1])[0,1])
            print(f"  utt{i} spk={ut.get('Speaker',ut.get('SpeakerID','?'))} t={st:.2f}-{et:.2f} crop={len(c)/csr:.2f}s lab={et-st:.2f}s r_ch0={r[0]:.3f} r_ch1={r[1]:.3f} r_mix={rm:.3f}")
            tested+=1
            if tested>=6: break
print("DONE")
