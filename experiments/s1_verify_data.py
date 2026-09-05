#!/usr/bin/env python
"""Stage 1 학습 전 데이터 검증 (plans/stage1-mono-pilot.md §3.5 의 1·2 항목).

A. KsponSpeech PCM 형식 — 16 kHz·16-bit LE·mono·headerless 가정을 실측으로 확인한다.
   길이(바이트÷32000) 대비 글자 속도, 음성 대역(300–3400 Hz) 에너지 비율, 클리핑, 홀수 바이트 파일 수.
B. KsponSpeech .trn 파서 — (철자)/(발음) 왼쪽 철자형 채택, 표지(b/ l/ o/ n/ u/)·기호(+ * /)·구두점 제거.
   전체 train.trn 통계, 동봉 jsonl(eval) 과의 차이율(기록용), 무작위 100 개 검토 파일(tsv).
C. LibriSpeech — trans.txt 집계(화자·챕터·발화·시간), 소문자 변환 샘플.

mxc 컨테이너에서:  conda activate vapasr && python experiments/s1_verify_data.py [--out DIR] [--n-pcm 10] [--n-review 100]
출력: <out>/verify.json, <out>/kspon_trn_review.tsv, <out>/kspon_jsonl_diff.tsv
"""
import os, re, sys, json, glob, random, argparse, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--kspon-root", default=os.environ.get("MXC_KSPONSPEECH_DIR", os.environ.get("KSPONSPEECH_DIR")))
ap.add_argument("--libri-root", default=os.environ.get("MXC_LIBRISPEECH_DIR", os.environ.get("LIBRISPEECH_DIR")))
ap.add_argument("--out", default=os.path.join(os.environ.get("MXC_DATA_MANIFEST_DIR", os.environ.get("DATA_MANIFEST_DIR", "/tmp")), "s1-verify"))
ap.add_argument("--n-pcm", type=int, default=10); ap.add_argument("--n-review", type=int, default=100)
ap.add_argument("--libri-limit", type=int, default=None, help="flac 헤더 읽기 개수 제한(속도)"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip", default="", help="건너뛸 절: A,B,C 쉼표 구분")
a = ap.parse_args(); random.seed(a.seed); os.makedirs(a.out, exist_ok=True); skip = set(a.skip.split(",")) if a.skip else set()
assert a.kspon_root and a.libri_root, "코퍼스 경로 없음: .env 의 MXC_KSPONSPEECH_DIR / MXC_LIBRISPEECH_DIR 을 로드하거나 --kspon-root/--libri-root"
SR, BPS = 16000, 2
report = {}
def hdr(t): print(f"\n=== {t} ===", flush=True)

# ───────────────────────────── .trn 파서 ─────────────────────────────
DUAL = re.compile(r"\(([^()]*)\)/\(([^()]*)\)")           # (철자)/(발음)
NOISE = re.compile(r"(?<!\S)[blonu]/(?!\S)")               # b/ l/ o/ n/ u/ 단독 표지
FILLER = re.compile(r"(\S+?)/(?=\s|$)")                     # 아/ 그/ → 단어 유지
PUNCT = re.compile(r"[.,?!]")

def normalize_kspon(raw: str, form: str = "spelling") -> str:
    """KsponSpeech 전사 → 학습·채점용 텍스트. form: spelling(왼쪽) | pron(오른쪽)."""
    s = DUAL.sub(lambda m: m.group(1) if form == "spelling" else m.group(2), raw)
    s = NOISE.sub(" ", s); s = FILLER.sub(r"\1", s)
    s = s.replace("+", "").replace("*", ""); s = PUNCT.sub("", s)
    return re.sub(r"\s+", " ", s).strip()

def read_text(path):
    for enc in ("utf-8", "cp949"):
        try: return open(path, encoding=enc).read()
        except UnicodeDecodeError: continue
    raise RuntimeError(f"인코딩 판별 실패: {path}")

def read_trn(path):
    rows = []
    for line in read_text(path).splitlines():
        if " :: " not in line: continue
        p, t = line.split(" :: ", 1); rows.append((p.strip(), t.strip()))
    return rows

# ───────────────────────────── PCM 리더 ─────────────────────────────
def read_pcm(path):
    """raw int16 LE mono. 홀수 바이트면 마지막 1 바이트 제외. → (float32 [-1,1], odd:bool)"""
    b = open(path, "rb").read(); odd = len(b) % 2 == 1
    if odd: b = b[:-1]
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0, odd

def band_ratio(x, lo=300, hi=3400):
    """가운데 1 s 구간의 300–3400 Hz 에너지 비율. 음성이면 보통 0.6 이상, 잡음/형식 오류면 낮다."""
    n = min(len(x), SR); c = len(x) // 2; seg = x[max(0, c - n // 2): max(0, c - n // 2) + n]
    if len(seg) < SR // 4: return float("nan")
    f = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2; fr = np.fft.rfftfreq(len(seg), 1 / SR)
    tot = f[fr > 50].sum(); return float(f[(fr >= lo) & (fr <= hi)].sum() / tot) if tot > 0 else float("nan")

# ═════════════════════════════ A. PCM ═════════════════════════════
if "A" not in skip:
    hdr("A. KsponSpeech PCM 형식 검증 (16 kHz · 16-bit LE · mono · headerless 가정)")
    trn = read_trn(os.path.join(a.kspon_root, "train.trn")); report["kspon_train_rows"] = len(trn)
    sub = [r for r in trn if r[0].startswith("KsponSpeech_01/")]
    pick = random.sample(sub, a.n_pcm); rows = []
    print(f"  train.trn {len(trn):,} 행, KsponSpeech_01 {len(sub):,} 행, 표본 {a.n_pcm}")
    print(f"  {'파일':38s} {'초':>6s} {'글자/s':>6s} {'대역비':>6s} {'RMS':>6s} {'clip%':>6s} 홀수")
    for rel, raw in pick:
        p = os.path.join(a.kspon_root, rel); x, odd = read_pcm(p); dur = len(x) / SR; txt = normalize_kspon(raw)
        cps = len(txt.replace(" ", "")) / dur if dur > 0 else float("nan"); br = band_ratio(x)
        rms = float(np.sqrt(np.mean(x ** 2))); clip = float((np.abs(x) > 0.99).mean() * 100)
        rows.append(dict(file=rel, sec=round(dur, 2), chars_per_s=round(cps, 2), band_ratio=round(br, 3), rms=round(rms, 4), clip_pct=round(clip, 3), odd=odd, text=txt[:60]))
        print(f"  {rel[-38:]:38s} {dur:6.2f} {cps:6.2f} {br:6.3f} {rms:6.4f} {clip:6.3f} {'O' if odd else '-'}")
    cps_med = float(np.median([r["chars_per_s"] for r in rows])); br_med = float(np.median([r["band_ratio"] for r in rows]))
    # 홀수 바이트 파일 수 (부분집합 0001~0062 전체, 크기만 본다)
    odd_n = 0; total_bytes = 0; n_files = 0
    for rel, _ in sub:
        d = rel.split("/")[1]
        if int(d.split("_")[1]) > 62: continue
        n = os.path.getsize(os.path.join(a.kspon_root, rel)); n_files += 1; total_bytes += n; odd_n += n % 2
    hours = total_bytes / (SR * BPS) / 3600
    verdict = "통과" if (3.0 <= cps_med <= 9.0 and br_med >= 0.5) else "**실패 — 형식 가정 재검토**"
    print(f"\n  글자/s 중앙값 {cps_med:.2f} (한국어 자발 발화 기대 3–9), 대역비 중앙값 {br_med:.3f} (기대 ≥ 0.5) → {verdict}")
    print(f"  부분집합 0001~0062: {n_files:,} 파일, {hours:.1f} h (바이트 실측), 홀수 바이트 {odd_n} 개")
    # eval_clean 의 .pcm 과 .wav 가 같은 바이트인지
    ep = os.path.join(a.kspon_root, "eval_clean", "KsponSpeech_E00001")
    same = os.path.getsize(ep + ".pcm") == os.path.getsize(ep + ".wav") and open(ep + ".pcm", "rb").read(4096) == open(ep + ".wav", "rb").read(4096)
    print(f"  eval_clean .pcm vs .wav: {'동일 바이트(둘 다 raw PCM)' if same else '다름 — .wav 별도 확인 필요'}")
    report["pcm"] = dict(samples=rows, chars_per_s_median=cps_med, band_ratio_median=br_med, verdict=verdict,
                         subset_files=n_files, subset_hours=round(hours, 2), subset_odd_byte_files=odd_n, eval_pcm_wav_same=bool(same))

# ═════════════════════════════ B. .trn 파서 ═════════════════════════════
if "B" not in skip:
    hdr("B. KsponSpeech .trn 파서 — 철자형 · 표지/기호 제거")
    trn = trn if "A" not in skip else read_trn(os.path.join(a.kspon_root, "train.trn"))
    st = collections.Counter()
    for _, raw in trn:
        st["dual"] += len(DUAL.findall(raw)); st["noise"] += len(NOISE.findall(raw)); st["filler"] += len(FILLER.findall(DUAL.sub("X", NOISE.sub(" ", raw))))
        st["plus"] += raw.count("+"); st["star"] += raw.count("*"); st["punct"] += len(PUNCT.findall(raw))
        if not normalize_kspon(raw): st["empty_after_norm"] += 1
    print("  train.trn 표기 빈도:", dict(st))
    review = random.sample(trn, a.n_review)
    with open(os.path.join(a.out, "kspon_trn_review.tsv"), "w", encoding="utf-8") as f:
        f.write("id\traw\tspelling\tpron\n")
        for rel, raw in review: f.write(f"{rel}\t{raw}\t{normalize_kspon(raw)}\t{normalize_kspon(raw, 'pron')}\n")
    print(f"  검토 파일: {os.path.join(a.out, 'kspon_trn_review.tsv')} ({a.n_review} 행) — 사람이 읽는다")
    print("  예시 5 개:")
    for rel, raw in review[:5]: print(f"    원문  {raw[:70]}\n    철자  {normalize_kspon(raw)[:70]}")
    # 동봉 jsonl 과의 차이 (기록용 — 기준 아님)
    diff = {}
    for split in ("eval_clean", "eval_other"):
        jp = os.path.join(a.kspon_root, f"{split}.jsonl"); tp = os.path.join(a.kspon_root, f"{split}.trn")
        if not (os.path.exists(jp) and os.path.exists(tp)): continue
        lab = {}
        for l in open(jp, encoding="utf-8"):
            d = json.loads(l); lab[os.path.basename(d["id"]).split(".")[0]] = d["label"]
        n = m = 0; ex = []
        for rel, raw in read_trn(tp):
            k = os.path.basename(rel).split(".")[0]
            if k not in lab: continue
            n += 1; ours = normalize_kspon(raw); theirs = re.sub(r"\s+", " ", lab[k]).strip()
            if ours != theirs:
                m += 1
                if len(ex) < 30: ex.append((k, raw, ours, theirs))
        diff[split] = dict(n=n, differ=m, rate=round(m / max(1, n), 4)); print(f"  {split}: jsonl 과 다른 행 {m}/{n} ({100*m/max(1,n):.1f} %)")
        with open(os.path.join(a.out, "kspon_jsonl_diff.tsv"), "a", encoding="utf-8") as f:
            for k, raw, ours, theirs in ex: f.write(f"{split}\t{k}\t{raw}\t{ours}\t{theirs}\n")
    report["trn"] = dict(stats=dict(st), jsonl_diff=diff)

# ═════════════════════════════ C. LibriSpeech ═════════════════════════════
if "C" not in skip:
    hdr("C. LibriSpeech train-clean-100 / dev / test 집계")
    import soundfile as sf
    out = {}
    for split in ("train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other"):
        root = os.path.join(a.libri_root, split); utts = []
        for tp in glob.glob(os.path.join(root, "*", "*", "*.trans.txt")):
            spk, chap = tp.split(os.sep)[-3:-1]
            for line in open(tp):
                uid, txt = line.strip().split(" ", 1); utts.append((spk, chap, uid, txt))
        n_read = utts if a.libri_limit is None else utts[: a.libri_limit]; secs = 0.0
        for spk, chap, uid, _ in n_read: secs += sf.info(os.path.join(root, spk, chap, uid + ".flac")).duration
        hours = secs / 3600 * (len(utts) / max(1, len(n_read)))
        chaps = len({(s, c) for s, c, _, _ in utts}); spks = len({s for s, _, _, _ in utts})
        per_chap = collections.Counter((s, c) for s, c, _, _ in utts); med = float(np.median(list(per_chap.values())))
        print(f"  {split:16s} 발화 {len(utts):6,} · 화자 {spks:4d} · 챕터 {chaps:5d} · {hours:6.1f} h{' (추정)' if a.libri_limit else ''} · 챕터당 발화 중앙값 {med:.0f}")
        out[split] = dict(utts=len(utts), speakers=spks, chapters=chaps, hours=round(hours, 2), utts_per_chapter_median=med)
    s0 = utts[0][3] if utts else ""; print(f"  소문자 변환 예: '{s0[:50]}' → '{s0.lower()[:50]}'")
    report["libri"] = out

json.dump(report, open(os.path.join(a.out, "verify.json"), "w"), ensure_ascii=False, indent=1)
print(f"\n기록: {os.path.join(a.out, 'verify.json')}")
