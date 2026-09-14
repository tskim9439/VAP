import zipfile, io, collections, re, sys, json, glob, os
import xml.etree.ElementTree as ET
HOP=0.08; GAP=0.25; DTXT=0.32; HOR=3.0
def ami():
    z=zipfile.ZipFile("/soundai/DB/raw/ami/annotations/ami_public_manual_1.6.2.zip")
    segs=collections.defaultdict(lambda: collections.defaultdict(list))
    for m in z.namelist():
        mm=re.match(r"segments/([A-Z]{2}\d{4}[a-z]?)\.([A-E])\.segments\.xml$", m)
        if not mm: continue
        meet, agent = mm.groups()
        for el in ET.fromstring(z.read(m)):
            a,b=el.get("transcriber_start"),el.get("transcriber_end")
            if a and b: segs[meet][agent].append((float(a),float(b)))
    return segs
def icsi():
    segs=collections.defaultdict(lambda: collections.defaultdict(list))
    for zp in ["/soundai/DB/raw/icsi/annotations/ICSI_core_NXT.zip"]:
        z=zipfile.ZipFile(zp)
        for m in z.namelist():
            mm=re.match(r"ICSI/Segments/(B[a-z]{2}\d{3})\.([A-Za-z0-9]+)\.segs\.xml$", m)
            if not mm: continue
            meet, agent = mm.groups()
            for el in ET.fromstring(z.read(m)).iter():
                a,b=el.get("starttime"),el.get("endtime")
                if a and b and el.tag.endswith("segment"):
                    segs[meet][agent].append((float(a),float(b)))
    return segs
def notsofar():
    segs=collections.defaultdict(lambda: collections.defaultdict(list))
    for gt in glob.glob("/soundai/DB/raw/notsofar/*/benchmark-datasets/*/*/MTG/MTG_*/gt_transcription.json"):
        meet=gt.split("/")[-2]+"@"+gt.split("/")[4]
        for u in json.load(open(gt)):
            segs[meet][u["speaker_id"]].append((float(u["start_time"]),float(u["end_time"])))
    return segs
def run(name, segs, R=4):
    nspk=collections.Counter(); simul=collections.Counter(); need=collections.Counter()
    ep_per_min=[]; exhausted=collections.Counter(); ambig=collections.Counter(); reassign=collections.Counter(); nev=0
    stats=collections.Counter()
    for meet, agents in segs.items():
        merged={}
        for ag,L in agents.items():
            L=sorted(L); out=[]
            for s,e in L:
                if out and s-out[-1][1]<GAP: out[-1][1]=max(out[-1][1],e)
                else: out.append([s,e])
            if out: merged[ag]=out
        N=len(merged); nspk[N]+=1
        T=max(e for L in merged.values() for _,e in L); n=int(T/HOP)+1; cnt=[0]*n
        for ag,L in merged.items():
            for s,e in L:
                for k in range(int(s/HOP), min(n,int(e/HOP)+1)): cnt[k]+=1
        for c in cnt: simul[c]+=1
        eps=sorted((s,e,ag) for ag,L in merged.items() for s,e in L)
        ep_per_min.append(len(eps)/(T/60))
        # required lanes (eager free at e+DTXT)
        ends=[]; mx=0
        for s,e,ag in eps:
            ends=[x for x in ends if x> s]; ends.append(e+DTXT); mx=max(mx,len(ends))
        need[mx]+=1
        # eager allocator with R lanes: count episodes that find no free lane
        lanes=[None]*R  # (free_at, owner, closed_at)
        for s,e,ag in eps:
            free=[i for i,l in enumerate(lanes) if l is None or l[0]<=s]
            if not free: exhausted[name]+=1
            else: lanes[free[0]]=(e+DTXT,ag,e)
            stats["eps"]+=1
        # lazy allocator: lane freed only on demand (LRU of closed lanes); same speaker reuses own lane if still held
        lanes=[None]*R  # dict: owner, open, closed_at(last end)
        for s,e,ag in eps:
            nev+=1
            own=[i for i,l in enumerate(lanes) if l and l["owner"]==ag]
            if own: i=own[0]
            else:
                free=[i for i,l in enumerate(lanes) if l is None]
                if free: i=free[0]
                else:
                    cand=[(l["closed"],i) for i,l in enumerate(lanes) if l["closed"] is not None and l["closed"]+DTXT<=s]
                    if not cand: exhausted[name+"_lazy"]+=1; continue
                    i=min(cand)[1]; reassign[name]+=1
                    # ambiguity: previous owner's last episode still within EOT horizon at reassignment
                    if s - lanes[i]["closed"] < HOR: ambig[name]+=1
            lanes[i]={"owner":ag,"closed":None}
            lanes[i]["closed"]=e
    tot=sum(simul.values()); sp=sum(v for k,v in simul.items() if k>0)
    print(f"== {name}: meetings {len(segs)}  N/meeting {dict(sorted(nspk.items()))}")
    print("  simultaneous among speech frames:", {k: f"{v/sp*100:.1f}%" for k,v in sorted(simul.items()) if k>0})
    print("  lanes needed (eager free, max concurrent open):", dict(sorted(need.items())))
    print(f"  episodes/min mean {sum(ep_per_min)/len(ep_per_min):.1f}  total eps {stats['eps']}")
    print(f"  R={R} eager exhausted {exhausted[name]} ({exhausted[name]/stats['eps']*100:.2f}%)  lazy exhausted {exhausted[name+'_lazy']}  lazy reassign {reassign[name]} ({reassign[name]/stats['eps']*100:.2f}%)  reassign within {HOR}s of prev close {ambig[name]}")
run("AMI", ami(), 4)
s=icsi(); run("ICSI", s, 4); run("ICSI_R6", s, 6); run("ICSI_R8", s, 8)
n=notsofar(); run("NOTSOFAR", n, 4); run("NOTSOFAR_R6", n, 6)
