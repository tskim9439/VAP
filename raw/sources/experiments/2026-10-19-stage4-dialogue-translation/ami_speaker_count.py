# AMI 창별 화자 수 분포(2026-10-19): 공식 NXT 단어 라벨(ami_public_manual_1.6.2.zip)로 20·30·40 s 창(겹치지 않게 회의 처음 단어부터)마다
# 말한 화자 수를 센다 — 화자 슬롯 2 개로 회의를 얼마나 다룰 수 있는지 판단 근거. 읽기만 한다(zip 을 풀지 않는다).
# 분할: Kaldi/ESPnet full-corpus-ASR(dev: ES2011·IS1008·TS3004·IB4001–4004·IB4010–4011, eval: ES2004·IS1009·TS3003·EN2002).
import collections, glob, json, os, re, sys, zipfile
import xml.etree.ElementTree as ET
ROOT = "/soundai/DB/raw/ami"
z = zipfile.ZipFile(f"{ROOT}/annotations/ami_public_manual_1.6.2.zip")
have = {os.path.basename(p) for p in glob.glob(f"{ROOT}/*") if os.path.isdir(p) and re.match(r"[A-Z]{2}\d{4}[a-z]?$", os.path.basename(p))}
DEV = {f"{m}{s}" for m in ("ES2011", "IS1008", "TS3004") for s in "abcd"} | {"IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011"}
EVAL = {f"{m}{s}" for m in ("ES2004", "IS1009", "TS3003", "EN2002") for s in "abcd"}
words = collections.defaultdict(list)                     # 회의 → [(중앙 시각, 화자)]
for n in z.namelist():
    m = re.match(r"words/([A-Z]{2}\d{4}[a-z]?)\.([A-Z])\.words\.xml$", n)
    if not m or m.group(1) not in have: continue
    for w in ET.fromstring(z.read(n)).iter("w"):
        if w.get("punc") == "true" or w.get("starttime") is None or w.get("endtime") is None: continue
        words[m.group(1)].append(((float(w.get("starttime")) + float(w.get("endtime"))) / 2, m.group(2)))
print(f"회의 {len(words)} (오디오 있음 {len(have)}) · eval {len(EVAL & set(words))} · dev {len(DEV & set(words))} · 단어 {sum(len(v) for v in words.values()):,}")
res = {}
for split, meet in (("eval", EVAL), ("dev", DEV), ("all", set(words))):
    for L in (20, 30, 40):
        hist, hist3, wtot, w_le2, w_out2 = collections.Counter(), collections.Counter(), 0, 0, 0
        for mt in sorted(meet & set(words)):
            ws = sorted(words[mt]); t0 = ws[0][0]
            win = collections.defaultdict(collections.Counter)
            for t, s in ws: win[int((t - t0) // L)][s] += 1
            for c in win.values():
                k = len(c); k3 = sum(v >= 3 for v in c.values()); n = sum(c.values()); top2 = sum(v for _, v in c.most_common(2))
                hist[min(k, 5)] += 1; hist3[min(k3, 5)] += 1; wtot += n; w_le2 += n if k <= 2 else 0; w_out2 += n - top2
        nwin = sum(hist.values())
        res[f"{split}/{L}s"] = dict(windows=nwin, active=dict((str(k), round(v / nwin, 3)) for k, v in sorted(hist.items())),
                                    active_ge3words=dict((str(k), round(v / nwin, 3)) for k, v in sorted(hist3.items())),
                                    words_in_le2_windows=round(w_le2 / wtot, 3), words_outside_top2=round(w_out2 / wtot, 3), windows_exactly2=hist[2], hours_exactly2=round(hist[2] * L / 3600, 2))
        r = res[f"{split}/{L}s"]
        print(f"{split:4s} {L}s 창 {nwin:5d} | 말한 화자 수 비율 {r['active']} | 3 단어 이상 화자 {r['active_ge3words']} | "
              f"화자 ≤ 2 창의 단어 몫 {r['words_in_le2_windows']:.3f} | 상위 2 명 밖 단어 몫 {r['words_outside_top2']:.3f} | 정확히 2 명 창 {r['windows_exactly2']}({r['hours_exactly2']} h)")
json.dump(res, open(sys.argv[1], "w"), ensure_ascii=False, indent=1) if len(sys.argv) > 1 else None
