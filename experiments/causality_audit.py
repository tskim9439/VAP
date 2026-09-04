#!/usr/bin/env python
"""Encoder causality / lookahead 감사 (task-audit-encoder-causality-lookahead).

방법: 전체 오디오의 프론트엔드 특징을 한 번 계산한 뒤 **특징 단위로 절단**해 인코더에 넣고,
전체 입력에서 얻은 h_full 과 절단 입력의 h_cut 을 프레임별로 비교한다.
절단점 이전 프레임 중 값이 바뀐 프레임 수 × frame_ms = 인코더의 실효 lookahead.
(특징 단위 절단이므로 프론트엔드의 utterance-level 정규화는 결과에서 배제된다. 프론트엔드 causality 는 별도 보고.)

실행(컨테이너):  source scripts/activate-env.sh && CUDA_VISIBLE_DEVICES=3 python experiments/causality_audit.py [nemotron qwen cpc]
결과: stdout 표 + $DATA_LOG_DIR/causality-audit-<ts>.json
"""
import os, sys, json, time, math, traceback
import numpy as np, torch, soundfile as sf

torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
def _undeterministic():
    torch.use_deterministic_algorithms(False)
DEV = "cuda"
AUDIO = os.path.join(os.environ.get("DATA_ROOT", "/data3/tskim"), "smoke", "libri1.wav")
CUTS_S = [3.0, 6.0, 9.0, 12.0]
REL_TOL = 1e-3          # |diff| / std(h_full) 가 이보다 크면 "바뀐 프레임"
results = []

def lookahead_frames(h_full, h_cut):
    """h_*: (T, D). 절단 출력의 프레임 중 전체 출력과 달라진 프레임 수(뒤에서부터 연속)."""
    n = min(h_cut.shape[0], h_full.shape[0])
    scale = h_full.float().std().item() + 1e-8
    rel = (h_cut[:n].float() - h_full[:n].float()).abs().amax(dim=1) / scale   # (n,)
    changed = (rel > REL_TOL).nonzero().flatten()
    first = int(changed[0]) if len(changed) else n
    return n, n - first, int(len(changed)), float(rel.max())

def audit(name, frame_ms, feats_full, encode, extra=None, front_ms=None, cuts=None):
    """feats_full: 프론트엔드 특징 (전체). encode(feats_slice) -> (T', D) 텐서."""
    with torch.inference_mode():
        h_full = encode(feats_full)
    T_full = h_full.shape[0]
    rows = []
    for cut_s in (cuts or CUTS_S):
        n_in = int(round(cut_s * 1000 / (front_ms or FRONT_MS[name])))
        with torch.inference_mode():
            h_cut = encode(feats_full[..., :n_in])   # 시간축은 항상 마지막
        n, la, nchg, mx = lookahead_frames(h_full, h_cut)
        rows.append(dict(cut_s=cut_s, out_frames=n, lookahead_frames=la, lookahead_ms=la * frame_ms, changed_frames=nchg, max_rel=mx))
        print(f"  cut {cut_s:5.1f}s | out {n:4d} fr | changed {nchg:4d} | lookahead {la:4d} fr = {la*frame_ms:7.1f} ms | max_rel {mx:.2e}")
    la_ms = [r["lookahead_ms"] for r in rows]
    summary = dict(encoder=name, frame_ms=frame_ms, out_frames_full=T_full, lookahead_ms_min=min(la_ms), lookahead_ms_max=max(la_ms),
                   lookahead_ms_mean=sum(la_ms) / len(la_ms), cuts=rows, **(extra or {}))
    results.append(summary); return summary

FRONT_MS = {}
y, sr = sf.read(AUDIO); y = y.astype(np.float32); dur = len(y) / sr
print(f"audio {AUDIO}  {dur:.2f}s @ {sr}")

# ───────────────────────────── Nemotron ─────────────────────────────
def run_nemotron():
    _undeterministic()
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu").to(DEV).eval().float()
    pre = m.preprocessor; enc = m.encoder
    pcfg = m.cfg.preprocessor
    print(f"[nemotron] preprocessor: window {pcfg.get('window_size')}s stride {pcfg.get('window_stride')}s normalize={pcfg.get('normalize')} "
          f"| conv_context_size={getattr(enc,'conv_context_size',None)} | streaming_cfg={getattr(enc,'streaming_cfg',None) is not None}")
    x = torch.tensor(y)[None].to(DEV)
    with torch.inference_mode():
        feats, flen = pre(input_signal=x, length=torch.tensor([x.shape[1]]).to(DEV))   # (1, 128, T)
    FRONT_MS["nemotron"] = pcfg.get("window_stride", 0.01) * 1000
    def make_encode():
        def encode(f):
            f = f[..., :]; L = torch.tensor([f.shape[-1]]).to(DEV)
            h, hl = enc(audio_signal=f.contiguous(), length=L)
            return h[0, :, :int(hl[0])].T.contiguous()   # (T', D)
        return encode
    for ctx in ([56, 0], [56, 1], [56, 3], [56, 6], [56, 13]):
        enc.set_default_att_context_size(ctx)
        print(f"\n[nemotron] att_context_size={ctx}  (문서상 우측 {ctx[1]*80} ms)")
        audit(f"nemotron[{ctx[0]},{ctx[1]}]", 80.0, feats, make_encode(), front_ms=FRONT_MS["nemotron"],
              extra=dict(documented_right_ms=ctx[1] * 80, frontend_normalize=str(pcfg.get("normalize")), frontend_window_ms=pcfg.get("window_size", 0.025) * 1000,
                         conv_context_size=str(getattr(enc, "conv_context_size", None))))

# ───────────────────────────── Qwen3 AuT ─────────────────────────────
def run_qwen():
    _undeterministic()
    import torch.nn as nn
    from qwen_asr import Qwen3ASRModel
    m = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=torch.float32, device_map=DEV, max_new_tokens=8)
    root = next(v for v in vars(m).values() if isinstance(v, nn.Module))
    enc = root.thinker.audio_tower.float().eval()
    fe = m.processor.feature_extractor
    print(f"[qwen] FE: {type(fe).__name__} hop {fe.hop_length} | n_window={enc.n_window} (conv chunk {enc.n_window*2*10} ms) "
          f"| n_window_infer={enc.n_window_infer} (attn block {enc.n_window_infer*10} ms) | conv_chunksize={enc.conv_chunksize}")
    out = fe(y, sampling_rate=sr, return_tensors="pt", padding="longest", truncation=False, return_attention_mask=True)
    feats = out.input_features[0].to(DEV)               # (128, T)
    T = int(out.attention_mask[0].sum()); feats = feats[:, :T]
    FRONT_MS["qwen"] = 10.0
    mod = sys.modules[type(enc).__module__]
    get_out_len = getattr(enc, "_get_feat_extract_output_lengths", None) or getattr(mod, "_get_feat_extract_output_lengths")
    def encode(f):
        L = torch.tensor([f.shape[-1]], device=DEV)
        after = get_out_len(L)
        r = enc(f.contiguous(), feature_lens=L, aftercnn_lens=after)
        h = r.last_hidden_state if hasattr(r, "last_hidden_state") else (r[0] if isinstance(r, tuple) else r)
        return h.reshape(-1, h.shape[-1])   # (T', D)
    # (a) transformers/sdpa 경로 그대로: forward 가 _prepare_attention_mask 를 호출하지 않아 마스크 None →
    #     전체 발화 양방향. n_window_infer 는 이 경로에서 무효 (800 vs 100 출력 동일 확인, 2026-09-03).
    name = "qwen-aut[sdpa full-utterance]"; FRONT_MS[name] = 10.0
    print(f"\n[qwen] {name}  (n_window_infer={enc.n_window_infer} 이지만 sdpa 경로에서 마스크 미적용)")
    audit(name, 80.0, feats, encode, front_ms=10.0, extra=dict(mode="sdpa, attention_mask=None", conv_chunk_ms=enc.n_window * 20,
          frontend="WhisperFeatureExtractor(log-mel, utterance max 정규화)"))
    # (b) 의도된 streaming 동작 재현: cu_seqlens 블록은 서로 attention 하지 않으므로(FA2 varlen) 블록별 독립 forward 와 동등.
    #     conv chunk(1 s) 경계와 정렬된 블록 크기만 유효. 블록 경계와 어긋난 절단점으로 블록 내 lookahead 를 측정.
    CUTS_BLOCK = [3.5, 6.25, 9.75, 12.0]
    for blk in (100, 200):   # fbank 프레임 = 1 s, 2 s
        def encode_block(f, blk=blk):
            outs = []
            for i in range(0, f.shape[-1], blk):
                outs.append(encode(f[..., i:i + blk]))
            return torch.cat(outs, 0)
        name = f"qwen-aut[per-block {blk*10} ms]"; FRONT_MS[name] = 10.0
        print(f"\n[qwen] {name}  (블록 독립 forward = FA2 varlen 의미; 예상 lookahead = 블록 끝까지 거리)")
        audit(name, 80.0, feats, encode_block, front_ms=10.0, cuts=CUTS_BLOCK,
              extra=dict(mode="per-block independent forward", block_ms=blk * 10, expected_lookahead="0 ~ block_ms (평균 block_ms/2)", cuts_note="블록 경계 비정렬"))

# ───────────────────────────── CPC / 원 VAP ─────────────────────────────
def run_cpc():
    _undeterministic()
    import glob
    repo = os.path.join(os.environ.get("DATA_ROOT", "/data3/tskim"), "third_party", "VoiceActivityProjection")
    os.chdir(repo)
    from vap.model import VapConfig, VapGPT, load_older_state_dict
    ck = sorted(glob.glob("example/VAP_*.ckpt"))[-1]
    m = VapGPT(VapConfig()); m.load_state_dict(load_older_state_dict(ck), strict=False); m = m.to(DEV).eval().float(); _undeterministic()
    FRONT_MS["cpc"] = 1000.0 / sr   # 파형 자체가 입력
    wav = torch.tensor(y)[None].to(DEV)   # (1, T)
    def enc_only(w):
        return m.encoder(w[None] if w.dim() == 1 else w)[0]           # (T', D)
    def full_model(w):
        stereo = torch.stack([w, torch.zeros_like(w)], dim=1) if w.dim() == 2 else torch.stack([w, torch.zeros_like(w)])[None]
        out = m.probs(stereo); return out["probs"][0]                   # (T', 256)
    for name, fn in (("cpc-encoder", enc_only), ("vap-full(probs)", full_model)):
        FRONT_MS[name] = FRONT_MS["cpc"]
        print(f"\n[{name}]")
        audit(name, 20.0, wav, lambda w: fn(w), extra=dict(note="파형 절단 (프론트엔드 없음)"))

steps = sys.argv[1:] or ["nemotron", "qwen", "cpc"]
for s in steps:
    try: globals()[f"run_{s}"]()
    except Exception: print(f"!! {s} 실패"); traceback.print_exc()

print("\n" + "=" * 96)
print(f"{'encoder':36s} {'frame':>6s} {'lookahead ms (min/mean/max over cuts)':>40s}")
for r in results:
    print(f"{r['encoder']:36s} {r['frame_ms']:6.0f} {r['lookahead_ms_min']:12.1f} {r['lookahead_ms_mean']:12.1f} {r['lookahead_ms_max']:12.1f}")
out = os.path.join(os.environ.get("DATA_LOG_DIR", "/tmp"), f"causality-audit-{time.strftime('%Y%m%d-%H%M')}.json")
json.dump(dict(audio=AUDIO, duration_s=dur, cuts_s=CUTS_S, rel_tol=REL_TOL, results=results), open(out, "w"), indent=1, ensure_ascii=False)
print("saved", out)
