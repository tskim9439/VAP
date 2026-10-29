#!/usr/bin/env python
"""E1 화자 분리 가지 평가(2026-10-25): AMI 창 평가(cst_stage3_eval 의 ami-bench-v1 창 — 20–90 s, draw 1, 화자 4, seed 0)와 같은 창에서 80 ms 프레임 DER(collar 0)을 잰다.
  python experiments/diar_eval.py run --diar <out/final 또는 checkpoint-N> --nemotron-dir <dir> --out <dir> [--data ami-bench-v1] [--corpora ami-ihmmix,ami-sdm]
      [--threshold 0.5] [--batch 8] [--rank r --world-size n] [--limit-windows 0] [--save-probs]
  python experiments/diar_eval.py run --session [--session-modes aosc,reset90] … — 회의 통째(E7, 2026-10-27): 회의 하나 = 행 하나. 인코더를 겹침 조각으로
      (long_session.segment_plan) 돌려 가지 임베딩을 한 번 만들고, 방식마다 <out>/<방식>/ 에 행을 쓴다. aosc = AOSC 스트리밍(diar.aosc_stream — diar.pt 의 aosc 설정,
      없으면 기본 · --aosc-chunk·--aosc-cache·--aosc-fifo·--aosc-period 로 덮는다 — 회의 내내 같은 슬롯), resetS = S 초 조각마다 가지를 따로(슬롯이 조각마다 새로 — 비교 기준).
      참조 = 회의 전체 도착 순서 slot_activity, conf_best = 회의 전체에 한 순열(조각마다 바뀐 슬롯은 혼동으로 센다). 요약은 summarize --out <out>/<방식>.
  python experiments/diar_eval.py summarize --out <dir>
  python experiments/diar_eval.py sweep --out <dir> [--thresholds 0.2,0.3,0.4,0.5,0.6]   — --save-probs 로 남긴 확률에서 임계별 DER(dev 에서 고른다)
--save-probs(2026-10-25): 창마다 out/probs/<id>.npz(prob float16 (K, 화자), ref uint8, mask) — 임계 훑기와 E3(예측 활동을 E2 대상 화자 디코드에 주입)의 입력.
참조 = slot_activity(창 안 먼저 말한 순서), 셈에서 빼는 프레임 = 전사 없는 음성 구간(_bad_spans). DER = (miss + fa + conf)/참조 화자 프레임.
  conf_arrival = 슬롯 순서(도착 순서) 그대로 짝, conf_best = 창마다 혼동이 가장 적은 순열. 공식 분할(manifest dev·eval)·조건·창 화자 수별로 합친다.
  주의: Nemotron 3 Diarization 의 AMI DER(SDM 12.8 @1.04 s)은 세션 단위·강제 정렬 참조라 이 창 단위 값과 바로 비교되지 않는다.
E8(2026-10-27): diar.pt 의 shift R(위치 p 의 출력 = 프레임 p − R 의 활동)이면 출력 [R:] 을 참조 [:K − R] 과 맞춘다 — 창·회의 끝 R 프레임은 뒤를 볼 오디오가 없어 셈에서 뺀다.
E11(2026-10-29): run --n3d <.nemo> [--session] [--n3d-chunk 4 --n3d-rc 0] — 고정 화자 분리 전문가 Nemotron 3 Diarization(vapasr/hf/n3d)을 같은 창(창마다 새로)·
  같은 회의(회의 전체 스트리밍 한 번)·같은 참조로 잰다(--diar 대신, 창 목록용 tokenizer 는 --tok-from). 행의 DER 은 모델이 쓰는 앞 4 채널(슬롯) 기준이고,
  extra = 채널 4–7 에만 있는 활성 프레임(우리 슬롯 밖으로 간 말 — DER 에서는 놓침), n_hyp_spk8 = 8 채널 중 활성 채널 수. 회의 통째는 <out>/n3d/ 에 쓴다.
G1(2026-10-29): run --graft <graft.pt> [--graft-offline] [--session] — 이식 학생(우리 인코더 탭 → 어댑터 → N3D 위층, vapasr/hf/n3d_graft.GraftExpert)을 같은
  창·회의·참조로 잰다(기본 스트리밍 청크 --n3d-chunk 4, --graft-offline 이면 창 통째). --n3d 는 graft.pt 의 .nemo 를 덮을 때만. 회의 통째는 <out>/graft/ 에 쓴다."""
import argparse, collections, glob, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNK_S, SR = 0.08, 16000


def unshift(prob, ref, m, R: int):
    """E8: (K, N) 출력의 위치 p = 프레임 p − R → (출력 [R:], 참조 [:K − R], 마스크 [:K − R]). R 0 이면 그대로."""
    if not R: return prob, ref, m
    K = prob.shape[0]; return prob[R:], ref[: K - R], m[: K - R]


def run(a):
    from experiments.cst_stage3_eval import windows_of
    from vapasr.data.cst_dialogue import slot_activity, window_speakers
    from vapasr.hf import load_tokenizer
    from vapasr.hf.diar import DiarBranch, der_counts, load_encoder
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    x = torch.load(os.path.join(a.diar, "diar.pt"), map_location="cuda"); n_spk = x["cfg"]["n_spk"]; R = int(x.get("shift", 0) or 0)
    model = DiarBranch(**x["cfg"]).cuda().eval(); model.load_state_dict(x["state"])
    enc, _ = load_encoder(x["init"], a.nemotron_dir, "cuda"); taps = tuple(x["taps"]); tok = load_tokenizer(x["init"])
    man = json.load(open(os.path.join(a.data, "manifest.json"))) if os.path.exists(os.path.join(a.data, "manifest.json")) else {}
    sub = {**{m: "dev" for m in man.get("dev", [])}, **{m: "eval" for m in man.get("eval", [])}}
    ns = argparse.Namespace(corpora=a.corpora, data=a.data, split=None, deltas=[10], limit_windows=a.limit_windows, backchannel=(1.0, 3), max_speakers=n_spk,
                            window=tuple(float(v) for v in a.window.split(",")), hop=10.0, window_draws=a.window_draws, declare=n_spk)
    sets, items = windows_of(ns, tok); items = [it for j, it in enumerate(items) if j % a.world_size == a.rank]
    os.makedirs(a.out, exist_ok=True); dest = os.path.join(a.out, f"windows-rank{a.rank}.jsonl"); pdir = os.path.join(a.out, "probs")
    if a.save_probs: os.makedirs(pdir, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(dest)} if os.path.exists(dest) else set(); t0 = time.time(); n = 0
    with open(dest, "a", buffering=1) as fo, torch.inference_mode():
        for b0 in range(0, len(items), a.batch):
            batch = [(c, i) for c, i in items[b0: b0 + a.batch] if (wid := f"{sets[c].items[i][0]}@{sets[c].items[i][1]}") not in done
                     or (a.save_probs and not os.path.exists(os.path.join(pdir, f"{wid}.npz")))]      # 행이 있어도 확률이 없으면 다시(행은 다시 쓰지 않는다)
            if not batch: continue
            wavs, Ks, meta = [], [], []
            for c, i in batch:
                ds = sets[c]; cid, t0w, L = ds.items[i]; d = ds.dlgs[cid]; K = int(round(L / CHUNK_S))
                wavs.append(np.asarray(ds.mono(cid)[int(round(t0w * SR)): int(round((t0w + L) * SR))], dtype=np.float32)); Ks.append(K)
                sp = window_speakers(ds.eps[cid], t0w, L)[:n_spk]
                ref = torch.tensor(slot_activity(ds.eps[cid], sp, t0w, L, K, n_slots=n_spk), dtype=torch.float32).reshape(K, n_spk)
                m = torch.ones(K, dtype=torch.bool)
                for s, e in ds._bad_spans(d):
                    x0, x1 = s - t0w, e - t0w
                    if x1 > 0 and x0 < L: m[max(0, int(x0 / CHUNK_S)): min(K, int(math.ceil(x1 / CHUNK_S)))] = False
                meta.append((c, cid, t0w, L, K, len(sp), ref, m))
            Nw = max(len(w) for w in wavs); Km = max(Ks); W = torch.zeros(len(wavs), Nw)
            for j, w in enumerate(wavs): W[j, : len(w)] = torch.from_numpy(w)
            Kt = torch.tensor(Ks); pad = torch.arange(Km)[None] >= Kt[:, None]
            feats = enc(W.cuda(), torch.tensor([len(w) for w in wavs]).cuda(), Kt.cuda(), taps=taps)
            prob = torch.sigmoid(model(feats, pad.cuda()).float()).cpu()
            for j, (c, cid, t0w, L, K, nsp, ref, m) in enumerate(meta):
                pj, ref, m = unshift(prob[j, :K], ref, m, R); pr = (pj > a.threshold).float(); dc = der_counts(pr, ref, m)
                row = dict(id=f"{cid}@{t0w}", corpus=c, sub=sub.get(cid.split(":", 1)[1], "?"), L=L, K=K, n_spk=nsp, n_hyp_spk=int((pr.sum(0) > 0).sum()), **dc)
                if a.save_probs:
                    np.savez_compressed(os.path.join(pdir, f"{row['id']}.npz"), prob=pj.numpy().astype(np.float16), ref=ref.numpy().astype(np.uint8), mask=m.numpy())
                if row["id"] not in done: fo.write(json.dumps(row) + "\n"); n += 1
            print(f"  {min(b0 + a.batch, len(items))}/{len(items)} 창 · {n} 행 · {time.time() - t0:.0f}s", flush=True)
    print(f"DONE rank={a.rank} rows={n}", flush=True)


def run_n3d(a):
    """E11: 고정 화자 분리 전문가(Nemotron 3 Diarization)의 창·회의 DER — run·run_session 과 같은 창·회의·참조·마스크·행 꼴(summarize 그대로)."""
    from experiments.cst_stage3_eval import windows_of
    from vapasr.data.cst_dialogue import slot_activity, window_speakers
    from vapasr.hf.diar import der_counts
    from vapasr.hf.n3d import N3DExpert
    from transformers import AutoTokenizer
    n_spk = 4; tok = AutoTokenizer.from_pretrained(a.tok_from)
    if a.graft:                                                                                       # G1: 이식 학생(우리 인코더 → 어댑터 → N3D 위층)
        from vapasr.hf.n3d_graft import GraftExpert
        gx = GraftExpert(a.graft, n3d_path=os.path.abspath(a.n3d) if a.n3d else None, nemotron_dir=a.nemotron_dir, chunk=a.n3d_chunk or 4, rc=a.n3d_rc or 0)
        class _Graft:                                                                                 # N3DExpert 와 같은 probs·cfg 꼴
            cfg = {**gx.cfg, "offline": bool(a.graft_offline)}

            @staticmethod
            def probs(w, wl_, k): return gx.probs(w, wl_, k, offline=a.graft_offline)
        ex = _Graft()
    else:
        ex = N3DExpert({"path": os.path.abspath(a.n3d), **({"chunk": a.n3d_chunk} if a.n3d_chunk else {}), **({"rc": a.n3d_rc} if a.n3d_rc is not None else {})})
    man = json.load(open(os.path.join(a.data, "manifest.json"))) if os.path.exists(os.path.join(a.data, "manifest.json")) else {}
    sub = {**{m: "dev" for m in man.get("dev", [])}, **{m: "eval" for m in man.get("eval", [])}}
    full = bool(a.session)
    ns = argparse.Namespace(corpora=a.corpora, data=a.data, split=None, deltas=[10], limit_windows=a.limit_windows, backchannel=(1.0, 3), max_speakers=n_spk,
                            window="full" if full else tuple(float(v) for v in a.window.split(",")), hop=10.0, window_draws=0 if full else a.window_draws, declare=n_spk)
    sets, items = windows_of(ns, tok); items = [it for j, it in enumerate(items) if j % a.world_size == a.rank]
    od = os.path.join(a.out, "graft" if a.graft else "n3d") if full else a.out; os.makedirs(od, exist_ok=True); dest = os.path.join(od, f"windows-rank{a.rank}.jsonl"); pdir = os.path.join(od, "probs")
    if a.save_probs: os.makedirs(pdir, exist_ok=True)
    json.dump(dict(n3d=ex.cfg, threshold=a.threshold, corpora=a.corpora, session=full, window=a.window, window_draws=a.window_draws),
              open(os.path.join(od, f"config-rank{a.rank}.json"), "w"), indent=1)
    done = {json.loads(l)["id"] for l in open(dest)} if os.path.exists(dest) else set(); t0 = time.time(); n = 0
    bs = 1 if full else a.batch
    with open(dest, "a", buffering=1) as fo, torch.inference_mode():
        for b0 in range(0, len(items), bs):
            batch = [(c, i) for c, i in items[b0: b0 + bs] if f"{sets[c].items[i][0]}@{'full' if full else sets[c].items[i][1]}" not in done]
            if not batch: continue
            wavs, Ks, meta = [], [], []
            for c, i in batch:
                ds = sets[c]; cid, t0w, L = ds.items[i]; d = ds.dlgs[cid]; t0w = 0.0 if full else t0w; K = int(round(L / CHUNK_S))
                wavs.append(np.asarray(ds.mono(cid)[int(round(t0w * SR)): int(round((t0w + L) * SR))], dtype=np.float32)); Ks.append(K)
                allsp = window_speakers(ds.eps[cid], t0w, L); sp = allsp[:n_spk]
                ref = torch.tensor(slot_activity(ds.eps[cid], sp, t0w, L, K, n_slots=n_spk), dtype=torch.float32).reshape(K, n_spk)
                m = torch.ones(K, dtype=torch.bool)
                for s_, e_ in ds._bad_spans(d):
                    x0, x1 = s_ - t0w, e_ - t0w
                    if x1 > 0 and x0 < L: m[max(0, int(x0 / CHUNK_S)): min(K, int(math.ceil(x1 / CHUNK_S)))] = False
                if len(allsp) > n_spk:                                                             # 참조 밖 화자가 말하는 프레임은 셈에서 뺀다(run_session 과 같다)
                    extra_ref = torch.tensor(slot_activity(ds.eps[cid], allsp[n_spk:], t0w, L, K, n_slots=len(allsp) - n_spk), dtype=torch.float32).reshape(K, -1)
                    m &= ~(extra_ref.sum(-1) > 0)
                meta.append((c, cid, t0w, L, K, len(sp), len(allsp), ref, m))
            W = torch.zeros(len(wavs), max(len(w) for w in wavs))
            for j, w in enumerate(wavs): W[j, : len(w)] = torch.from_numpy(w)
            te = time.time(); prob = ex.probs(W.cuda(), torch.tensor([len(w) for w in wavs]).cuda(), torch.tensor(Ks).cuda()).cpu(); dec_s = time.time() - te
            for j, (c, cid, t0w, L, K, nsp, nall, ref, m) in enumerate(meta):
                p8 = (prob[j, :K] > a.threshold).float(); pr = p8[:, :n_spk]; dc = der_counts(pr, ref, m)
                extra = float(((p8[:, n_spk:].sum(-1) > 0) & (pr.sum(-1) == 0) & m).sum())
                wid = f"{cid}@{'full' if full else t0w}"
                row = dict(id=wid, corpus=c, sub=sub.get(cid.split(":", 1)[1], "?"), L=L, K=K, n_spk=nsp, n_spk_all=nall, n_hyp_spk=int((pr.sum(0) > 0).sum()),
                           n_hyp_spk8=int((p8.sum(0) > 0).sum()), extra=extra, dec_s=round(dec_s, 1), **dc)
                if a.save_probs: np.savez_compressed(os.path.join(pdir, f"{wid}.npz"), prob=prob[j, :K].numpy().astype(np.float16), ref=ref.numpy().astype(np.uint8), mask=m.numpy())
                fo.write(json.dumps(row) + "\n"); n += 1
                if full: print(f"  {cid} {c}: K {K} · DER 최적 짝 {100 * (dc['miss'] + dc['fa'] + dc['conf_best']) / max(1, dc['ref']):.2f} · 도착 순서 "
                               f"{100 * (dc['miss'] + dc['fa'] + dc['conf_arrival']) / max(1, dc['ref']):.2f} · 채널 {row['n_hyp_spk8']} · {dec_s:.0f}s", flush=True)
            if not full: print(f"  {min(b0 + bs, len(items))}/{len(items)} 창 · {n} 행 · {time.time() - t0:.0f}s", flush=True)
    print(f"DONE rank={a.rank} rows={n} {time.time() - t0:.0f}s", flush=True)


def run_session(a):
    """회의 통째 DER(--session). 행 = 회의(ami-bench-v1 의 대화 하나 × 조건). 화자가 n_spk 보다 많으면 먼저 말한 n_spk 명만 참조에 두고 나머지 화자 프레임은 셈에서 뺀다."""
    from experiments.cst_stage3_eval import windows_of
    from vapasr.data.cst_dialogue import slot_activity, window_speakers
    from vapasr.hf import load_tokenizer
    from vapasr.hf.diar import AoscConfig, DiarBranch, aosc_stream, der_counts, load_encoder
    from vapasr.hf.long_session import segment_plan
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    x = torch.load(os.path.join(a.diar, "diar.pt"), map_location="cuda"); n_spk = x["cfg"]["n_spk"]; R = int(x.get("shift", 0) or 0)
    model = DiarBranch(**x["cfg"]).cuda().eval(); model.load_state_dict(x["state"])
    enc, _ = load_encoder(x["init"], a.nemotron_dir, "cuda"); taps = tuple(x["taps"]); tok = load_tokenizer(x["init"])
    ch = int(json.load(open(os.path.join(x["init"], "config.json"))).get("encoder_right_context", 0) or 0) + 1
    cfg = AoscConfig.from_dict(x.get("aosc"))
    for k in ("chunk", "cache", "fifo", "period"):
        v = getattr(a, f"aosc_{k}", None)
        if v is not None: setattr(cfg, k, int(v))
    modes = [m_ for m_ in a.session_modes.split(",") if m_]
    assert modes and all(m_ == "aosc" or (m_.startswith("reset") and float(m_[5:]) > 0) for m_ in modes), f"--session-modes aosc,resetS — {a.session_modes}"
    assert not (R and "aosc" in modes), "E8 shift 헤드는 아직 AOSC 와 함께 재지 않는다(캐시 점수가 p − R 시점과 어긋난다)"
    man = json.load(open(os.path.join(a.data, "manifest.json"))) if os.path.exists(os.path.join(a.data, "manifest.json")) else {}
    sub = {**{m: "dev" for m in man.get("dev", [])}, **{m: "eval" for m in man.get("eval", [])}}
    ns = argparse.Namespace(corpora=a.corpora, data=a.data, split=None, deltas=[10], limit_windows=a.limit_windows, backchannel=(1.0, 3), max_speakers=n_spk,
                            window="full", hop=10.0, window_draws=0, declare=n_spk)                    # 화자 n_spk 넘는 회의는 windows_of 가 뺀다(회의 통째 cpWER 평가와 같은 회의)
    sets, items = windows_of(ns, tok); items = [it for j, it in enumerate(items) if j % a.world_size == a.rank]
    dests, done, fos = {}, {}, {}
    for m_ in modes:
        od = os.path.join(a.out, m_); os.makedirs(od, exist_ok=True); dests[m_] = os.path.join(od, f"windows-rank{a.rank}.jsonl")
        done[m_] = {json.loads(l)["id"] for l in open(dests[m_])} if os.path.exists(dests[m_]) else set()
        if a.save_probs: os.makedirs(os.path.join(od, "probs"), exist_ok=True)
        json.dump(dict(diar=os.path.abspath(a.diar), mode=m_, aosc=cfg.to_dict() if m_ == "aosc" else None, threshold=a.threshold, corpora=a.corpora),
                  open(os.path.join(od, f"config-rank{a.rank}.json"), "w"), indent=1)
    t0 = time.time(); n = 0
    fos = {m_: open(dests[m_], "a", buffering=1) for m_ in modes}
    with torch.inference_mode():
        for c, i in items:
            ds = sets[c]; cid, _, L = ds.items[i]; d = ds.dlgs[cid]; wid = f"{cid}@full"
            todo = [m_ for m_ in modes if wid not in done[m_]]
            if not todo: continue
            K = int(round(L / CHUNK_S)); wav = np.asarray(ds.mono(cid)[: int(round(L * SR))], dtype=np.float32); te = time.time(); embs = []
            for seg, Ks, s0, c0, nn in segment_plan(wav, K, ch):
                w = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32)).cuda()
                f = enc(w[None], torch.tensor([len(seg)]).cuda(), torch.tensor([Ks]).cuda(), taps=taps)
                embs.append(model.embed(f)[0, c0 - s0: c0 - s0 + nn]); del f
            e = torch.cat(embs); enc_s = time.time() - te
            allsp = window_speakers(ds.eps[cid], 0.0, L); sp = allsp[:n_spk]
            ref = torch.tensor(slot_activity(ds.eps[cid], sp, 0.0, L, K, n_slots=n_spk), dtype=torch.float32).reshape(K, n_spk)
            m = torch.ones(K, dtype=torch.bool)
            for s_, e_ in ds._bad_spans(d):
                m[max(0, int(s_ / CHUNK_S)): min(K, int(math.ceil(e_ / CHUNK_S)))] = False
            if len(allsp) > n_spk:                                                                 # 참조 밖 화자가 말하는 프레임은 셈에서 뺀다
                extra = torch.tensor(slot_activity(ds.eps[cid], allsp[n_spk:], 0.0, L, K, n_slots=len(allsp) - n_spk), dtype=torch.float32).reshape(K, -1)
                m &= ~(extra.sum(-1) > 0)
            for m_ in todo:
                te = time.time()
                if m_ == "aosc": lg = aosc_stream(model, e, cfg)
                else: S = int(round(float(m_[5:]) / CHUNK_S)); lg = torch.cat([model.head(e[k0: k0 + S][None])[0] for k0 in range(0, K, S)]).float()
                prob = torch.sigmoid(lg).cpu(); dec_s = time.time() - te
                prob, rf, mk = unshift(prob, ref, m, R); pr = (prob > a.threshold).float(); dc = der_counts(pr, rf, mk)
                row = dict(id=wid, corpus=c, sub=sub.get(cid.split(":", 1)[1], "?"), L=L, K=K, n_spk=len(sp), n_spk_all=len(allsp), n_hyp_spk=int((pr.sum(0) > 0).sum()),
                           mode=m_, enc_s=round(enc_s, 1), dec_s=round(dec_s, 1), **dc)
                if a.save_probs: np.savez_compressed(os.path.join(a.out, m_, "probs", f"{wid}.npz"), prob=prob.numpy().astype(np.float16), ref=rf.numpy().astype(np.uint8), mask=mk.numpy())
                fos[m_].write(json.dumps(row) + "\n"); n += 1
                print(f"  {cid} {c}: K {K} · {m_} DER 최적 짝 {100 * (dc['miss'] + dc['fa'] + dc['conf_best']) / max(1, dc['ref']):.2f} · 인코딩 {enc_s:.0f}s · 가지 {dec_s:.0f}s", flush=True)
    for fo in fos.values(): fo.close()
    print(f"DONE rank={a.rank} rows={n} {time.time() - t0:.0f}s", flush=True)


def summarize(a):
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.out, "windows-rank*.jsonl"))) for l in open(f)]
    res = {}; der = lambda g, k: round(100 * (g["miss"] + g["fa"] + g[k]) / max(1, g["ref"]), 2)
    keys = sorted({(r["corpus"], r["sub"]) for r in rows})
    for c, s in keys:
        rr = [r for r in rows if r["corpus"] == c and r["sub"] == s]; g = collections.Counter()
        for r in rr: g.update({k: r[k] for k in ("ref", "miss", "fa", "conf_arrival", "conf_best")})
        by = {}
        for k in sorted({r["n_spk"] for r in rr}):
            h = collections.Counter()
            for r in rr:
                if r["n_spk"] == k: h.update({q: r[q] for q in ("ref", "miss", "fa", "conf_arrival", "conf_best")})
            by[k] = dict(der_arrival=der(h, "conf_arrival"), der_best=der(h, "conf_best"), ref_share=round(100 * h["ref"] / max(1, g["ref"]), 1))
        spk_ok = sum(r["n_hyp_spk"] == r["n_spk"] for r in rr) / max(1, len(rr))
        res[f"{c}/{s}"] = dict(windows=len(rr), der_arrival=der(g, "conf_arrival"), der_best=der(g, "conf_best"), miss=round(100 * g["miss"] / max(1, g["ref"]), 2),
                               fa=round(100 * g["fa"] / max(1, g["ref"]), 2), conf_arrival=round(100 * g["conf_arrival"] / max(1, g["ref"]), 2),
                               conf_best=round(100 * g["conf_best"] / max(1, g["ref"]), 2), spk_count_acc=round(spk_ok, 3), by_nspk=by)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for k, v in res.items():
        print(f"{k:16s} 창 {v['windows']:4d} · DER 도착 순서 {v['der_arrival']:6.2f} · 최적 짝 {v['der_best']:6.2f}(miss {v['miss']} · fa {v['fa']} · conf {v['conf_arrival']}/{v['conf_best']}) · "
              f"화자 수 맞춤 {v['spk_count_acc']} · " + " · ".join(f"{s}인 {b['der_best']}" for s, b in v["by_nspk"].items()))


def sweep(a):
    from vapasr.hf.diar import der_counts
    meta = {r["id"]: r for r in (json.loads(l) for f in glob.glob(os.path.join(a.out, "windows-rank*.jsonl")) for l in open(f))}
    ths = [float(x) for x in a.thresholds.split(",")]; acc = collections.defaultdict(collections.Counter); n = 0
    for p in sorted(glob.glob(os.path.join(a.out, "probs", "*.npz"))):
        r = meta[os.path.basename(p)[: -len(".npz")]]; z = np.load(p); n += 1
        prob, ref, m = torch.from_numpy(z["prob"].astype(np.float32)), torch.from_numpy(z["ref"].astype(np.float32)), torch.from_numpy(z["mask"])
        for t in ths: acc[(r["corpus"], r["sub"], t)].update(der_counts((prob > t).float(), ref, m))
    assert n, f"확률이 없다 — run --save-probs: {a.out}/probs"
    pct = lambda g, k: round(100 * g[k] / max(1, g["ref"]), 2); res = {}
    for (c, s, t), g in sorted(acc.items()):
        res.setdefault(f"{c}/{s}", {})[f"{t:g}"] = dict(der_arrival=round(pct(g, "miss") + pct(g, "fa") + pct(g, "conf_arrival"), 2),
                                                       der_best=round(pct(g, "miss") + pct(g, "fa") + pct(g, "conf_best"), 2), miss=pct(g, "miss"), fa=pct(g, "fa"),
                                                       conf_arrival=pct(g, "conf_arrival"), conf_best=pct(g, "conf_best"))
    json.dump(res, open(os.path.join(a.out, "sweep.json"), "w"), indent=1); print(f"창 {n} · 임계 {ths}")
    for k, v in res.items():
        print(f"{k:16s} " + " · ".join(f"{t}: {x['der_best']:5.2f}/{x['der_arrival']:5.2f}(m {x['miss']:.1f} f {x['fa']:.1f})" for t, x in v.items()))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run"); q.add_argument("--diar", default=None); q.add_argument("--nemotron-dir", default=None); q.add_argument("--out", required=True)
    q.add_argument("--n3d", default=None, help="E11: --diar 대신 고정 화자 분리 전문가 Nemotron 3 Diarization(.nemo)을 잰다")
    q.add_argument("--n3d-chunk", type=int, default=None, help="--n3d 스트리밍 청크(80 ms 프레임, 기본 4)"); q.add_argument("--n3d-rc", type=int, default=None, help="--n3d 오른쪽 문맥(기본 0)")
    q.add_argument("--tok-from", default="/soundai/Model/VAPASR/stage3-q17-e9/final", help="--n3d·--graft: 창 목록(windows_of)용 tokenizer 체크포인트")
    q.add_argument("--graft", default=None, help="G1: 이식 학생 graft.pt(n3d_graft_train 산출)를 잰다 — 기본 스트리밍 청크 --n3d-chunk(4)")
    q.add_argument("--graft-offline", action="store_true", help="--graft 를 창 통째(오프라인)로")
    q.add_argument("--data", default="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1"); q.add_argument("--corpora", default="ami-ihmmix,ami-sdm")
    q.add_argument("--window", default="20,90"); q.add_argument("--window-draws", type=int, default=1); q.add_argument("--threshold", type=float, default=0.5)
    q.add_argument("--batch", type=int, default=8); q.add_argument("--rank", type=int, default=0); q.add_argument("--world-size", type=int, default=1); q.add_argument("--limit-windows", type=int, default=0)
    q.add_argument("--save-probs", action="store_true")
    q.add_argument("--session", action="store_true", help="회의 통째 — 회의 하나 = 행 하나, 방식마다 <out>/<방식>/")
    q.add_argument("--session-modes", default="aosc", help="쉼표: aosc(회의 전체 AOSC) · resetS(S 초 조각마다 슬롯 새로 — 예 reset90)")
    for k in ("chunk", "cache", "fifo", "period"): q.add_argument(f"--aosc-{k}", type=int, default=None)
    q = sub.add_parser("summarize"); q.add_argument("--out", required=True)
    q = sub.add_parser("sweep"); q.add_argument("--out", required=True); q.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6")
    a = ap.parse_args()
    if a.cmd == "run": assert bool(a.diar) + bool(a.n3d and not a.graft) + bool(a.graft) == 1, "run 은 --diar · --n3d · --graft 중 하나(--graft 에 --n3d 는 .nemo 덮기)"
    if a.cmd == "run" and (a.n3d or a.graft): run_n3d(a); return
    if a.cmd == "run" and a.session: run_session(a); return
    {"run": run, "summarize": summarize, "sweep": sweep}[a.cmd](a)


if __name__ == "__main__":
    main()
