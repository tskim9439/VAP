# 조각(crop)의 채널 기원 검증: 서로 다른 화자의 시간 겹침 조각 쌍을 시각으로 정렬해 겹침 구간 상호상관을 잰다.
# 같은 혼합 마이크에서 잘렸으면 |r|→1, 각자 채널이면 |r|→0. 양성 대조군: 성인 조각을 인위적으로 섞어 만든 mixed crop.
import os, sys, glob, json, zipfile, re, random, unicodedata as ud, collections
import numpy as np, soundfile as sf
random.seed(0)
def nfc(s): return ud.normalize("NFC", s)
def find_dir(parent, name):
    for d in os.listdir(parent):
        if nfc(d)==nfc(name): return os.path.join(parent,d)
    raise SystemExit(f"dir not found {parent}/{name}")
NIA="/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/asr_db/korean_16kHz/NIA24"
def key(d,pat):
    for k in d:
        if re.search(pat,k,re.I): return k
def tf(x):
    x=str(x).replace(",","")
    if ":" in x:
        p=x.split(":"); return int(p[0])*3600+int(p[1])*60+float(p[2])
    return float(x)
def utts(j):
    c=j.get("Conversation") or next((v for v in j.values() if isinstance(v,list) and v and isinstance(v[0],dict)),[])
    if not c: return []
    ks=key(c[0],"start"); ke=key(c[0],"end"); kp=key(c[0],"speaker")
    out=[]
    for i,u in enumerate(c,1):
        try: out.append((i,str(u[kp]),tf(u[ks]),tf(u[ke])))
        except Exception: pass
    return out
def label_index_youth():
    idx={}
    for z in sorted(glob.glob(f"{NIA}/134-2_Emotion_Conv_Youth/Json/*.zip"))+sorted(glob.glob("/soundai/DB/raw/aihub/71632/**/*.zip",recursive=True)):
        try: zf=zipfile.ZipFile(z)
        except Exception: continue
        sub=nfc(os.path.basename(z)).replace(".zip","")
        for m in zf.namelist():
            if m.endswith(".json"): idx.setdefault(os.path.splitext(os.path.basename(m))[0],(z,m,sub))
    return idx
def load_json_zip(z,m):
    return json.loads(zipfile.ZipFile(z).read(m).decode("utf-8"))
def label_index_adult():
    idx={}
    for root,_,files in os.walk("/soundai/users/tskim/VAPKT-data/data/labels/aihub71631"):
        for f in files:
            if f.endswith(".json"): idx.setdefault(os.path.splitext(f)[0],(None,os.path.join(root,f),nfc(os.path.relpath(root,"/soundai/users/tskim/VAPKT-data/data/labels/aihub71631")).split(os.sep)[0]))
    return idx
def xcorr_max(a,b,maxlag):
    a=a-a.mean(); b=b-b.mean(); na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na<1e-6 or nb<1e-6: return None,0
    best=0.0; bl=0
    for lag in range(-maxlag,maxlag+1,8):
        if lag>=0: x=a[lag:]; y=b[:len(b)-lag]
        else: x=a[:len(a)+lag]; y=b[-lag:]
        n=min(len(x),len(y))
        if n<1600: continue
        r=float(np.dot(x[:n],y[:n])/(np.linalg.norm(x[:n])*np.linalg.norm(y[:n])+1e-9))
        if abs(r)>abs(best): best=r; bl=lag
    return best,bl
def run(name, cropdir, lidx, max_dialogs=150, max_pairs=400, want_sub=None, positive_control=False):
    stems=collections.defaultdict(set)
    for f in os.listdir(cropdir):
        if f.endswith(".wav"):
            s,n=f[:-4].rsplit("_",1)
            try: stems[s].add(int(n))
            except: pass
    cand=[s for s in stems if s in lidx and (want_sub is None or lidx[s][2]==want_sub)]
    random.shuffle(cand); rs=[]; lags=[]; npairs=0; nd=0; pos=[]
    for s in cand:
        if npairs>=max_pairs or nd>=max_dialogs: break
        z,m,sub=lidx[s]
        try: j=load_json_zip(z,m) if z else json.load(open(m,encoding="utf-8"))
        except Exception: continue
        U=[u for u in utts(j) if u[0] in stems[s]]
        pairs=[(a,b) for a in U for b in U if a[0]<b[0] and a[1]!=b[1] and min(a[3],b[3])-max(a[2],b[2])>=0.3]
        if not pairs: continue
        nd+=1; random.shuffle(pairs)
        for a,b in pairs[:3]:
            try:
                xa,sr=sf.read(f"{cropdir}/{s}_{a[0]}.wav",dtype="float32"); xb,_=sf.read(f"{cropdir}/{s}_{b[0]}.wav",dtype="float32")
            except Exception: continue
            if xa.ndim>1: xa=xa.mean(1)
            if xb.ndim>1: xb=xb.mean(1)
            o0=max(a[2],b[2]); o1=min(a[3],b[3])
            ia=int((o0-a[2])*sr); ib=int((o0-b[2])*sr); n=int((o1-o0)*sr)
            sa=xa[ia:ia+n]; sb=xb[ib:ib+n]
            if len(sa)<1600 or len(sb)<1600: continue
            r,l=xcorr_max(sa,sb,640)
            if r is None: continue
            rs.append(abs(r)); lags.append(l); npairs+=1
            if positive_control:
                # 인위 혼합: a 조각의 겹침 구간에 b 조각을 더한 뒤 b 와 상관 → 혼합 기원이면 이 값이 나온다
                mixed=sa[:min(len(sa),len(sb))]+sb[:min(len(sa),len(sb))]
                rp,_=xcorr_max(mixed,sb[:len(mixed)],640)
                if rp is not None: pos.append(abs(rp))
    rs=np.array(rs)
    def q(v): return f"n={len(v)} median={np.median(v):.3f} mean={v.mean():.3f} p90={np.percentile(v,90):.3f} frac>0.3={np.mean(v>0.3):.3f} frac>0.6={np.mean(v>0.6):.3f}" if len(v) else "n=0"
    print(f"== {name}: dialogs={nd} pairs={npairs} |r| {q(rs)}  lag_ms median={np.median(np.abs(lags))/16 if lags else 0:.1f}")
    if positive_control and pos: print(f"   positive control (artificial mix) |r| {q(np.array(pos))}")
    sys.stdout.flush()
YT=f"{NIA}/134-2_Emotion_Conv_Youth/Training"; AT=f"{NIA}/134-1_Emotion_Conv_Adult/Training"
li=label_index_youth(); print("youth label index", len(li)); sys.stdout.flush()
la=label_index_adult(); print("adult label index", len(la)); sys.stdout.flush()
run("adult 134-1 실외 (대조군, 채널 기원 확인됨)", find_dir(AT,"TS_02.실외"), la, want_sub="TL_02.실외", positive_control=True)
run("youth 134-2 실외", find_dir(YT,"TS_02.실외"), li, want_sub="TL_02.실외")
run("youth 134-2 실내", find_dir(YT,"TS_01.실내"), li, want_sub="TL_01.실내")
print("DONE")
