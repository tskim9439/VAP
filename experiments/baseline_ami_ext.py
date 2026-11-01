#!/usr/bin/env python
"""외부 기준을 우리 AMI 조건으로 직접 잰다(2026-10-30, 사용자 "이 둘을 준비해줘" — wiki output-ami-external-comparison-20261027):
같은 회의(ami-bench-v1 — 공식 dev 18·eval 16 × IHM-mix·SDM), 같은 오디오(우리 평가가 받는 mono 혼합 16 kHz), 같은 참조(cst_stage3_eval.window_ref δ —
전사 없는 구간 규칙까지), 같은 채점(cst_eval.score_window 단어마다 · experiments/meeteval_cpwer.py 화자 통째)으로.

  python experiments/baseline_ami_ext.py run --system vibevoice --model <VibeVoice-ASR-Streaming 폴더> --src <VibeVoice 소스> --checkpoint <우리 체크포인트> \\
      --data <ami-bench-v1> --out <dir> [--segment-s 480] [--rank r --world-size n]
  python experiments/baseline_ami_ext.py run --system parakeet --asr <multitalker-parakeet .nemo> --diar <.nemo> --diar-kind sortformer|n3d \\
      --checkpoint … --data … --out <dir> [--att-context 70,13] [--wav-dir <노드 로컬 폴더>]
그다음 cst_stage3_eval.py summarize · ami_report.py · meeteval_cpwer.py 를 그대로 쓴다 — 행 꼴이 cst_stage3_eval 과 같다(emitted 는 비고 EOT 는 셈 0,
원출력 단어는 hyp_words [화자, 단어, 회의 기준 시각 s], 시스템 정보는 ext).

시스템
  vibevoice: VibeVoice-ASR-Streaming(7B·1.5B — experiments/baseline_vibevoice.load_vibevoice·hyp_words, 체크포인트의 청크·lookahead, 탐욕, bf16). 입력은 24 kHz 로
    올린다(resample_poly 3/2). 단어 시각은 그 단어가 나온 청크 안에 고르게 놓는다(근사 — 마스크에만 쓴다).
  parakeet: Multitalker Parakeet(NeMo EncDecMultiTalkerRNNTBPEModel — 화자 분리기 활동으로 화자마다 인스턴스, 병렬 스트리밍). NeMo 예제
    examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py(설치본과 같은 커밋 6989ae9fc, Apache-2.0)의 manifest 경로를 옮겼다 —
    설정 기본값(MultitalkerTranscriptionConfig)·병렬 화자 전략·bf16 autocast 가 같다. 화자 분리기: sortformer(.nemo restore_from — 카드 평가의 Streaming Sortformer v2)
    또는 n3d(Nemotron 3 Diarization — vapasr.hf.n3d.load_n3d: NeMo 3.1.0 에 없는 손실 설정을 고쳐 읽고 SDPA, 스트리밍 캐시는 우리 전문가와 같은 264·264·222).
    회의 오디오는 --wav-dir 에 16 kHz wav 로 써서 manifest 로 넣는다(파일 이름 = 세션 id). 출력 SegLST 문장 구간 안에 단어 시각을 고르게 놓는다.
조각(--segment-s S, 기본 0 = 회의 통째): 회의를 S 초 이하 조각으로 나눠(조각 끝은 S 초 전 60 초 안의 가장 늦은 무음 지점, 없으면 S 초) 조각마다 따로 디코드하고
  조각마다 채점한다(행 id = <코퍼스>:<회의>@<조각 시작> — 창 평가처럼 조각마다 화자 짝). VibeVoice-Streaming 논문 AMI 수치(480 s 이하 조각)를 다시 잴 때.
마스크: 가설 단어도 참조와 같은 규칙으로 버린다 — 시각 t 의 단어는 청크 int(t / 80 ms) + δ 가 전사 없는 구간 청크(mask_chunks)이면 뺀다.
점수: 가설 화자가 7 명 이상이면 헝가리안으로 최적 짝을 찾는다(score_window 의 순열과 같은 최소). 아무것도 지우지 않는다(이어 하기 — 끝난 행은 건너뛴다).
--audio headset-sum(2026-10-30, 사용자 "논문의 AMI-IHM 이 우리가 쓰는 헤드셋 혼합과 다른 입력일 가능성 … 검토"): 공식 Mix-Headset 대신 대화의 화자별 헤드셋
  채널(Headset-0..3.wav)을 그대로 더해(넘치면 최대 1 로 줄임) 넣는다 — 혼합을 만드는 방식이 결과를 바꾸는지 본다. 참조·마스크는 그대로(행 ext.audio).
--diar-probs <폴더>(parakeet, 2026-11-01, 사용자 "둘 다 구현해놓자" — Parakeet + G2): 화자 분리기를 돌리지 않고, 저장한 회의 통째 화자 확률
  (diar_eval --session --save-probs 의 <코퍼스>:<회의>@full.npz — prob (K, 8), 80 ms)을 NeMo 의 미리 준 활동 경로(diar_model.rttms_mask_mats — 정답 RTTM 용,
  청크마다 블록 끝까지 꺼내 쓴다)에 배치마다 넣는다. 확률은 스트리밍(인과)으로 뽑은 값이라 그대로 되돌려 주어도 공정하다. --diar 는 틀(설정·프레임 검사)로만 읽는다.
  같은 경로로 N3D 자신의 저장 확률을 되돌려 주면 직접 돌린 mtp-n3d 와 견줘 경로를 확인할 수 있다. 회의 통째(--segment-s 0)만.
vapasr-spk(2026-11-01, 사용자 "G2 + N3D 조합으로 추론하는거, 그런데 화자마다 세션을 부여하고"): 우리 모델(--checkpoint, --encoder-state 면 그 graft.pt 의 enc —
  G2 인코더)을 Parakeet 처럼 화자마다 따로 돌린다. 학습 없는 Diarization Masked Input(arXiv 2609.10265 의 2 번째 길)이다.
  회의 오디오로 모델의 고정 전문가(N3D) 활동을 한 번(스트리밍, 인과) 뽑고, 화자 s 마다 그 활동 프레임(± --spk-margin)만 남긴 오디오를 회의 통째로 디코드한다
  (encode_long·long_block_decode, LLM 문맥 --context-s). 슬롯 활동 입력은 슬롯 A = 화자 s 활동, 나머지 0 이다. 나온 단어는 모두 화자 s 다(--spk-keep slotA 면 슬롯 A 만).
  단어 시각 = 방출 청크 − δ(오디오 시각 — 마스크 규칙이 우리 평가와 같다). δ 는 --deltas 하나.
  끝난 화자 세션은 <out>/sessions-rank{r}.jsonl 에 바로 남겨, 선점 뒤 다시 돌면 회의 중간(세션 단위)부터 이어 한다(2026-11-02)."""
import argparse, json, os, sys, time
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNK_S = 0.08
NO_EOT = dict(n_hyp=0, n_pos=0, n_hold=0, n_both=0, hit_pos=0, hit_hold=0, hit_both=0, unmatched=0, lat=[])


def ami_items(a, tok):
    """우리 AMI 회의 통째 평가(v3_eval_task.sh amifull)와 같은 회의 목록·데이터셋."""
    from cst_stage3_eval import windows_of
    ns = argparse.Namespace(data=a.data, split=None, corpora=a.corpora, deltas=a.deltas, limit_windows=a.limit_windows, backchannel=(1.0, 3), window="full",
                            hop=10.0, max_speakers=4, window_draws=0, declare=4, xl_order="lang")
    return windows_of(ns, tok)


def segment_bounds(d, seg_s: float, back_s: float = 60.0) -> List[tuple]:
    """회의를 seg_s 초 이하 조각으로: 끝은 (시작 + seg_s − back_s, 시작 + seg_s] 안의 가장 늦은 무음 지점(발화 구간 합집합 사이), 없으면 시작 + seg_s."""
    sp = sorted((u.start, u.end) for u in d.utterances); merged = []
    for s, e in sp:
        if merged and s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([s, e])
    gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]
    out, t, T = [], 0.0, float(d.duration_s)
    while t < T - 1e-6:
        hi = t + seg_s
        if hi >= T: out.append((t, T)); break
        c = [min(b, hi) for a, b in gaps if a <= hi and b > hi - back_s and min(b, hi) > t + 1.0]
        e = max(c) if c else hi; out.append((t, e)); t = e
    return out


def score_ext(ref_w, hyp_w, lang):
    """score_window 와 같은 dict — 가설 화자가 7 명 이상이면 순열 대신 헝가리안(같은 최소, 같은 단위)."""
    from vapasr.data.cst_eval import score_window, units, ed
    if len(hyp_w) <= 6: return score_window(ref_w, hyp_w, lang)
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    U = lambda ws: [u for w, _ in ws for u in units(w, lang)]
    n = max(2, len(ref_w), len(hyp_w)); Rs = [U(ref_w[i]) if i < len(ref_w) else [] for i in range(n)]; Hs = [U(hyp_w[i]) if i < len(hyp_w) else [] for i in range(n)]
    C = np.array([[ed(Rs[r], Hs[h]) for h in range(n)] for r in range(n)]); ri, hi = linear_sum_assignment(C); perm = [0] * n
    for r, h in zip(ri, hi): perm[int(h)] = int(r)
    best = int(C[ri, hi].sum()); same = int(sum(C[i, i] for i in range(n)))
    ra = [u for w, _ in sorted([x for sl in ref_w for x in sl], key=lambda x: x[1]) for u in units(w, lang)]
    ha = [u for w, _ in sorted([x for sl in hyp_w for x in sl], key=lambda x: x[1]) for u in units(w, lang)]
    return dict(n_ref=sum(map(len, Rs)), n_hyp=sum(map(len, Hs)), slot=same, cp=best, agn=ed(ra, ha), swap=bool(best < same), perm=perm)


def headset_sum(d, s0: float, s1: float):
    """대화의 화자별 채널 wav(ChannelRef.path — `wav#chN` 은 쓰지 않는 AMI 헤드셋 파일)를 [s0, s1) 구간만 더한다 → float32 16 kHz(넘치면 최대 1 로 줄인다)."""
    import numpy as np, soundfile as sf
    a, b = int(round(s0 * 16000)), int(round(s1 * 16000)); out = np.zeros(b - a, np.float32)
    for ch in d.channels.values():
        assert "#" not in ch.path and not ch.pieces, f"헤드셋 합은 화자별 단일 wav 만: {ch.path}"
        x, sr = sf.read(ch.path, start=a, stop=b, dtype="float32", always_2d=True); assert sr == 16000, (ch.path, sr)
        out[: len(x)] += x[:, 0]
    m = float(np.abs(out).max()) if len(out) else 0.0
    return out / m if m > 1.0 else out


def hyp_by_speaker(hw):
    """[(화자, 단어, 시각)] → 화자별 [(단어, 시각)](처음 나온 순서, 시각순)."""
    order, by = [], {}
    for sp, w, t in sorted(hw, key=lambda x: x[2]):
        if sp not in by: by[sp] = []; order.append(sp)
        by[sp].append((w, t))
    return [by[s] for s in order]


# ── VibeVoice-ASR-Streaming
class VibeVoiceStream:
    def __init__(self, a):
        from baseline_vibevoice import load_vibevoice
        import torch
        self.a = a; self.model, self.proc, self.frame = load_vibevoice(argparse.Namespace(mode="stream", model=a.model, src=a.src, lm_tokenizer=None, attn=a.attn))
        self.info = dict(system="vibevoice-stream", model=os.path.abspath(a.model), chunk_s=round(self.frame["chunk"], 4), lookahead_s=round(self.frame["lookahead"], 4),
                         **({"attn": a.attn, "torch": torch.__version__} if a.attn != "sdpa" or not torch.__version__.startswith("2.9") else {}))

    def __call__(self, units):
        """units [(키, wav16)] → {키: ([(화자, 단어, 시각 s)], 원출력)}."""
        import numpy as np, torch
        from scipy.signal import resample_poly
        from baseline_vibevoice import hyp_words, degenerate
        out = {}
        for key, x16 in units:
            w = resample_poly(x16, 3, 2).astype(np.float32); st = time.time()
            with torch.inference_mode():
                chunks = [t for _, _, t in self.model.streaming_generate(audio_tensor=torch.from_numpy(w), tokenizer=self.proc.tokenizer, chunk_duration=self.frame["chunk"],
                                                                       text_audio_delay=self.frame["lookahead"], sample_rate=self.frame["sample_rate"],
                                                                       max_new_tokens_per_chunk=self.a.max_new_tokens, temperature=0.0)]
            o = dict(chunks=chunks, chunk_s=self.frame["chunk"], lookahead_s=self.frame["lookahead"])
            out[key] = (hyp_words(dict(out=o)), dict(raw=o, decode_s=round(time.time() - st, 1), degenerate=degenerate("".join(chunks))))
        return out


# ── Multitalker Parakeet(NeMo)
@dataclass
class MultitalkerTranscriptionConfig:
    """NeMo examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py(커밋 6989ae9fc) 의 설정 — 기본값 그대로 옮겼다(설치본에 없다)."""
    diar_model: Optional[str] = None
    diar_pretrained_name: Optional[str] = None
    max_num_of_spks: Optional[int] = 4
    parallel_speaker_strategy: bool = True
    masked_asr: bool = False
    mask_preencode: bool = False
    cache_gating: bool = True
    cache_gating_buffer_size: int = 2
    single_speaker_mode: bool = False
    feat_len_sec: float = 0.01
    session_len_sec: float = -1
    num_workers: int = 8
    random_seed: Optional[int] = None
    log: bool = True
    precision: str = "bf16"
    streaming_mode: bool = True
    spkcache_len: Optional[int] = None
    spkcache_update_period: int = 144
    fifo_len: int = 188
    diar_right_context: int = 0
    cuda: Optional[int] = None
    allow_mps: bool = False
    matmul_precision: str = "highest"
    asr_model: Optional[str] = None
    device: str = "cuda"
    audio_file: Optional[str] = None
    manifest_file: Optional[str] = None
    att_context_size: Optional[List[int]] = field(default_factory=lambda: [70, 13])
    use_amp: bool = True
    debug_mode: bool = False
    deploy_mode: bool = False
    batch_size: int = 32
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 5
    online_normalization: bool = False
    output_path: Optional[str] = None
    diar_output_rttm_dir: Optional[str] = None
    diar_collar: float = 0.0
    diar_ignore_overlap: bool = False
    pad_and_drop_preencoded: bool = False
    generate_realtime_scripts: bool = False
    spk_supervision: str = "diar"
    binary_diar_preds: bool = True
    verbose: bool = False
    word_window: int = 50
    sent_break_sec: float = 1.0
    fix_prev_words_count: int = 5
    update_prev_words_sentence: int = 5
    left_frame_shift: int = -1
    right_frame_shift: int = 0
    min_sigmoid_val: float = 1e-2
    discarded_frames: int = 8
    print_time: bool = True
    print_sample_indices: List[int] = field(default_factory=lambda: [0])
    colored_text: bool = True
    real_time_mode: bool = False
    print_path: Optional[str] = None
    ignored_initial_frame_steps: int = 5
    finetune_realtime_ratio: float = 0.01


class MTParakeet:
    def __init__(self, a):
        import torch
        from omegaconf import OmegaConf
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
        from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import configure_diar_streaming, validate_feature_frame_strides
        self.a = a; os.makedirs(a.wav_dir, exist_ok=True)
        kw = dict(spkcache_len=264, fifo_len=264, spkcache_update_period=222) if a.diar_kind == "n3d" else {}   # n3d: 우리 전문가(vapasr.hf.n3d.DEFAULTS)와 같은 캐시
        self.cfg = OmegaConf.structured(MultitalkerTranscriptionConfig(asr_model=a.asr, diar_model=a.diar, att_context_size=list(a.att_context), batch_size=a.batch_size,
                                                                       device=a.device, log=False, print_time=False, colored_text=False, **kw))
        torch.set_float32_matmul_precision(self.cfg.matmul_precision)
        if a.diar_kind == "n3d":
            from vapasr.hf.n3d import load_n3d
            diar, _ = load_n3d(a.diar, a.device)
        else: diar = SortformerEncLabelModel.restore_from(restore_path=a.diar, map_location=torch.device(a.device))
        self.use_bf16 = str(self.cfg.precision).lower().startswith("bf16") and self.cfg.use_amp and a.device.startswith("cuda") and torch.cuda.is_bf16_supported()
        self.diar = diar.to(dtype=torch.bfloat16 if self.use_bf16 else torch.float32).eval()
        asr = nemo_asr.models.ASRModel.restore_from(restore_path=a.asr)
        asr.encoder.set_default_att_context_size(att_context_size=list(a.att_context))
        self.asr = asr.to(self.cfg.device).eval()
        validate_feature_frame_strides(asr_model=self.asr, diar_model=self.diar)
        sc = self.asr.encoder.streaming_cfg
        configure_diar_streaming(diar_model=self.diar, cfg=self.cfg, output_subsampling_factor=self.asr.encoder.subsampling_factor, diar_chunk_len=sc.valid_out_len + sc.cache_drop_size)
        if isinstance(self.diar.encoder.pre_encode, FeatureStacking) and not self.cfg.pad_and_drop_preencoded: self.cfg.pad_and_drop_preencoded = True   # 예제와 같다(N3D)
        self.info = dict(system="multitalker-parakeet", asr=os.path.abspath(a.asr), diar=os.path.abspath(a.diar), diar_kind=a.diar_kind, att_context=list(a.att_context),
                         spkcache=self.diar.sortformer_modules.spkcache_len, fifo=self.diar.sortformer_modules.fifo_len, period=self.diar.sortformer_modules.spkcache_update_period,
                         chunk_len=self.diar.sortformer_modules.chunk_len, bf16=self.use_bf16, **({"diar_probs": os.path.abspath(a.diar_probs)} if a.diar_probs else {}))

    def _replay(self, part):
        """--diar-probs: 배치(회의 순서 = manifest 순서)의 저장 확률 → diar.rttms_mask_mats (B, T, 8). 길이가 다른 줄과 끝 블록은 0 으로 채운다."""
        import numpy as np, torch
        ps = []
        for key, x16 in part:
            p = os.path.join(self.a.diar_probs, key.rsplit("@", 1)[0] + "@full.npz"); assert os.path.exists(p), f"저장 확률이 없다: {p}"
            ps.append(np.load(p)["prob"].astype(np.float32))
        T = max(len(p) for p in ps) + 256
        M = np.zeros((len(ps), T, ps[0].shape[1]), dtype=np.float32)
        for j, p in enumerate(ps): M[j, : len(p)] = p
        self.diar.rttms_mask_mats = torch.from_numpy(M).to(self.cfg.device)

    def _stream(self, buf):
        """예제 launch_parallel_streaming 과 같다."""
        import torch
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
        cfg, asr = self.cfg, self.asr
        it = buf.iter_with_right_context(0 if cfg.spk_supervision == "rttm" else cfg.diar_right_context * self.diar.encoder.subsampling_factor)
        mt = SpeakerTaggedASR(cfg, asr, self.diar)
        for step, (ca, cl, dca, dcl) in enumerate(it):
            drop = 0 if step == 0 and not cfg.pad_and_drop_preencoded else asr.encoder.streaming_cfg.drop_extra_pre_encoded
            with torch.inference_mode(), torch.amp.autocast(device_type=asr.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
                mt.perform_parallel_streaming_stt_spk(step_num=step, chunk_audio=ca, chunk_lengths=cl, diar_chunk_audio=dca, diar_chunk_lengths=dcl,
                                                      is_buffer_empty=buf.is_buffer_empty(), drop_extra_pre_encoded=drop)
        return mt

    def __call__(self, units):
        import soundfile as sf
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import get_multi_talker_samples_from_manifest
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
        out = {}
        for b0 in range(0, len(units), self.a.batch_size):
            part = units[b0: b0 + self.a.batch_size]; names = {}; man = os.path.join(self.a.wav_dir, f"manifest-r{self.a.rank}-{b0}.jsonl")
            with open(man, "w") as fo:
                for key, x16 in part:
                    nm = key.replace(":", "__").replace("@", "_at_").replace(".", "p"); p = os.path.join(self.a.wav_dir, nm + ".wav"); names[nm] = key
                    if not os.path.exists(p): sf.write(p, x16, 16000, subtype="PCM_16")
                    fo.write(json.dumps(dict(audio_filepath=p, offset=0, duration=round(len(x16) / 16000, 3), text="-")) + "\n")
            self.cfg.manifest_file = man
            fps = round(self.asr.cfg.preprocessor.window_stride * self.asr.cfg.encoder.subsampling_factor, 2)
            samples, _ = get_multi_talker_samples_from_manifest(self.cfg, manifest_file=man, feat_per_sec=fps, max_spks=self.cfg.max_num_of_spks)
            buf = CacheAwareStreamingAudioBuffer(model=self.asr, online_normalization=self.cfg.online_normalization, pad_and_drop_preencoded=self.cfg.pad_and_drop_preencoded)
            for s in samples: buf.append_audio_file(s["audio_filepath"], stream_id=-1)
            if self.a.diar_probs: self._replay(part)                                                # 화자 분리기 대신 저장 확률(Parakeet + G2)
            st = time.time(); mt = self._stream(buf); segs = mt.generate_seglst_dicts_from_parallel_streaming(samples=samples); el = time.time() - st
            by = {nm: [] for nm in names}
            for g in segs: by.setdefault(g["session_id"], []).append(g)
            for nm, key in names.items():
                hw = []
                for g in by.get(nm, []):
                    ws = str(g["words"]).split(); s, e = float(g["start_time"]), float(g["end_time"]); e = max(e, s)
                    hw += [(str(g["speaker"]), w, s + (e - s) * (k + 1) / len(ws)) for k, w in enumerate(ws)]
                out[key] = (hw, dict(raw=by.get(nm, []), decode_s=round(el / len(part), 1)))
            buf.reset_buffer()
        return out


def done_sessions(path):
    """vapasr-spk 이어 하기: sessions-rank{r}.jsonl → {회의 키: {화자: 기록}}. 선점으로 끊긴 줄은 건너뛴다."""
    out = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8", errors="replace"):
            try: r = json.loads(l)
            except ValueError: continue
            out.setdefault(r["key"], {})[int(r["spk"])] = r
    return out


def append_jsonl(path, obj):
    """한 줄 덧붙이기 — 앞 줄이 쓰는 중 선점으로 끊겼으면 줄을 바꾼 뒤 쓴다(끊긴 줄만 버려지게)."""
    with open(path, "a+b") as fo:
        if fo.seek(0, os.SEEK_END):
            fo.seek(-1, os.SEEK_END)
            if fo.read(1) != b"\n": fo.write(b"\n")
        fo.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


class OursSpk:
    """vapasr-spk: 우리 모델 + 고정 전문가(N3D) 활동, 화자마다 세션(모듈 설명). where(키 → (데이터셋, 창 index))는 run 이 채운다(prefix·선언 슬롯 수).
    끝난 세션은 sessions-rank{r}.jsonl 에 바로 남긴다(2026-11-02 — 회의 하나가 4 세션 × 회의 통째라 1 시간을 넘어 low_p_hpc 선점 사이에 못 끝났다, job 99349).
    다시 돌면 그 회의의 끝난 세션은 디코드하지 않고 기록을 쓴다."""

    def __init__(self, a):
        import torch
        from vapasr.hf.infer import load_model
        from vapasr.data.cst_dialogue import SLOTS
        self.a = a; self.where = {}
        self.model, self.tok = load_model(a.checkpoint, encoder_path=a.encoder, dtype=getattr(torch, a.dtype))
        m = self.model
        if a.encoder_state:                                                                         # G2 인코더(화자 분리와 함께 학습 — ASR 지킴으로 받아쓰기는 그대로)
            xs = torch.load(a.encoder_state, map_location="cpu"); assert xs.get("enc") is not None, f"인코더 state(enc)가 없다: {a.encoder_state}"
            m.encoder.enc.load_state_dict(xs["enc"]); m.encoder.eval()
        assert getattr(m, "n3d", None) is not None and getattr(m.config, "slot_blocks", False), "vapasr-spk 는 고정 전문가가 있는 슬롯 블록 체크포인트(E11)"
        sp = m.config.sp_ids; self.spk = tuple(sp[t] for t in SLOTS[: len(m.config.slot_tokens)]); self.EOT = sp["<EOT>"]; self.BEGIN = sp.get("<ONSET>")
        self.N = int(m.config.slot_input_slots); self.delta = int(a.deltas[0])
        self.info = dict(system="vapasr-spk", checkpoint=os.path.abspath(a.checkpoint), encoder_state=os.path.abspath(a.encoder_state) if a.encoder_state else None,
                         n3d=dict(m.config.n3d_cfg), spk_margin=a.spk_margin, spk_keep=a.spk_keep, context_s=a.context_s, dtype=a.dtype, delta=self.delta)
        self.side = os.path.join(a.out, f"sessions-rank{a.rank}.jsonl"); self.done = done_sessions(self.side)

    def __call__(self, units):
        import numpy as np, torch
        from cst_stage3_eval import words_of
        from vapasr.data.cst_eval import parse_emitted
        from vapasr.hf.long_session import encode_long, long_block_decode
        m, a, out = self.model, self.a, {}; dev = next(m.parameters()).device; HOP = 1280
        for key, wav in units:
            ds, i = self.where[key]; K = int(round(len(wav) / HOP)); st = time.time(); hw = []; ses = []; prev = self.done.get(key, {})
            langs = ds.slots(i)[1][:4]; prefix = ds.cst_prefix(langs, self.delta); D = len(langs)
            with torch.inference_mode():
                A = m.n3d_act(torch.from_numpy(wav).to(dev)[None], torch.tensor([len(wav)], device=dev), torch.tensor([K], device=dev))[0].float().cpu().numpy()   # (K, N) 원 오디오
            for s in range(A.shape[1]):
                on = A[:, s] > 0.5
                if not on.any(): continue
                if s in prev:                                                                        # 앞 실행(선점 전)에 끝난 세션
                    hw += [tuple(x) for x in prev[s]["words"]]; ses.append(dict(prev[s]["session"], resumed=True)); continue
                t1 = time.time()
                fm = np.convolve(on.astype(np.int32), np.ones(2 * a.spk_margin + 1, np.int32), mode="same") > 0      # 활동 프레임 ± margin
                sm = np.repeat(fm, HOP)[: len(wav)]; ws = wav * np.pad(sm, (0, len(wav) - len(sm))).astype(np.float32)
                act = np.zeros((K, self.N), np.float32); act[:, 0] = on                              # 슬롯 A = 화자 s, 나머지 0
                with torch.inference_mode():
                    feats = encode_long(m, ws, K, act=act)
                    em, fc, rd, used, nf, tk = long_block_decode(m, feats, prefix, D, context_s=a.context_s)
                hs, _ = parse_emitted(em, self.spk, self.EOT, begin_id=self.BEGIN); sw = []
                for j, toks in enumerate(hs):
                    if a.spk_keep == "slotA" and j: continue
                    sw += [(f"spk{s}", w, round(te - (self.delta + 1) * 0.08 + 0.04, 3)) for w, te in words_of(self.tok, toks)]   # 오디오 시각 = 방출 청크 − δ
                se = dict(spk=s, frames=int(on.sum()), masked_frames=int(fm.sum()), words=len(sw), tokens=len(em), forced=fc); hw += sw; ses.append(se)
                append_jsonl(self.side, dict(key=key, spk=s, words=sw, session=se))
                print(f"    {key} 화자 {s}: 단어 {len(sw)} · {time.time() - t1:.0f}s", flush=True)
            out[key] = (hw, dict(sessions=ses, decode_s=round(time.time() - st, 1), resumed=sum(x.get("resumed", False) for x in ses)))
            print(f"  {key}: 세션 {len(ses)} · 단어 {len(hw)} · {time.time() - st:.0f}s", flush=True)
        return out


def run(a):
    import numpy as np
    from transformers import AutoTokenizer
    from cst_stage3_eval import window_ref
    from vapasr.data.cst_eval import set_en_norm
    set_en_norm("ours")
    tok = AutoTokenizer.from_pretrained(a.checkpoint); EOT = tok.convert_tokens_to_ids("<EOT>")
    sets, items = ami_items(a, tok); items = [x for j, x in enumerate(items) if j % a.world_size == a.rank]
    os.makedirs(a.out, exist_ok=True); dest = os.path.join(a.out, f"predictions-rank{a.rank}.jsonl")
    done = {r["id"] for r in map(json.loads, open(dest, encoding="utf-8"))} if os.path.exists(dest) else set()
    work = []                                                                                    # (코퍼스, 데이터셋 창, 회의 키, 조각 시작, 조각 끝)
    for c, i in items:
        ds = sets[c]; cid, t0w, L = ds.items[i]; d = ds.dlgs[cid]
        for s0, s1 in (segment_bounds(d, a.segment_s) if a.segment_s else [(0.0, float(L))]):
            if f"{cid}@{round(s0, 3)}" not in done: work.append((c, i, cid, round(s0, 3), round(s1, 3)))
    sysm = VibeVoiceStream(a) if a.system == "vibevoice" else MTParakeet(a) if a.system == "parakeet" else OursSpk(a)
    if a.system == "vapasr-spk": sysm.where = {f"{cid}@{s0}": (sets[c], i) for c, i, cid, s0, s1 in work}
    conf = dict(sysm.info, checkpoint=os.path.abspath(a.checkpoint), data=os.path.abspath(a.data), corpora=a.corpora, deltas=a.deltas, segment_s=a.segment_s,
                world_size=a.world_size, limit_windows=a.limit_windows, **({"audio": a.audio} if a.audio != "mono" else {}))
    cp = os.path.join(a.out, f"config-rank{a.rank}.json")
    if os.path.exists(cp): assert json.load(open(cp)) == conf, "재개 설정 불일치 — 새 --out 을 쓴다"
    else: json.dump(conf, open(cp, "w"), indent=1)
    print(f"{conf['system']} · 단위 {len(work)}(끝 {len(done)}) · {json.dumps(sysm.info, ensure_ascii=False)}", flush=True)
    t0 = time.time(); n = 0; step = max(1, a.batch_size) if a.system == "parakeet" else 1                      # vapasr-spk: 회의마다 행을 바로 쓴다(선점 대비)
    with open(dest, "a", buffering=1, encoding="utf-8") as fo:
        for b0 in range(0, len(work), step):
            part = work[b0: b0 + step]
            res = sysm([(f"{cid}@{s0}", headset_sum(sets[c].dlgs[cid], s0, s1) if a.audio == "headset-sum" else
                         np.asarray(sets[c].mono(cid)[int(round(s0 * 16000)): int(round(s1 * 16000))], dtype=np.float32)) for c, i, cid, s0, s1 in part])
            for c, i, cid, s0, s1 in part:
                ds = sets[c]; d = ds.dlgs[cid]; hw_raw, info = res[f"{cid}@{s0}"]
                for delta in a.deltas:
                    ref_w, _, mask, sp_order = window_ref(ds, i, delta, EOT, tok)
                    ref_s = [[(w, t) for w, t in r if s0 <= t < s1] for r in ref_w] if a.segment_s else ref_w
                    hw = [(sp, w, t + s0) for sp, w, t in hw_raw]; keep = [x for x in hw if int(x[2] / CHUNK_S) + delta not in mask]
                    hyp = hyp_by_speaker(keep); sc = score_ext(ref_s, hyp, d.lang)
                    k0, k1 = int(s0 / CHUNK_S), int(round(s1 / CHUNK_S))
                    row = dict(id=f"{cid}@{s0}", corpus=c, lang=d.lang, delta=delta, L=round(s1 - s0, 3), K=k1 - k0, speakers=sp_order, n_spk=len(sp_order),
                               masked_chunks=sum(k0 <= k < k1 for k in mask), forced=0, flush_rounds=0, emitted=[],
                               ext=dict(sysm.info, n_hyp_spk=len(hyp), dropped_masked=len(hw) - len(keep), audio=a.audio, **{k: v for k, v in info.items() if k != "raw"}),
                               hyp_words=[[sp, w, round(t, 3)] for sp, w, t in hw], raw=info.get("raw"),
                               ref_text=[" ".join(w for w, _ in r) for r in ref_s], hyp_text=[" ".join(w for w, _ in h) for h in hyp], **sc, eot=dict(NO_EOT))
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
            print(f"  {min(b0 + step, len(work))}/{len(work)} 단위 · {n} 행 · {time.time() - t0:.0f}s", flush=True)
    print(f"DONE rank={a.rank} rows={n}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run"); q.add_argument("--system", choices=["vibevoice", "parakeet", "vapasr-spk"], required=True)
    q.add_argument("--checkpoint", required=True, help="참조 단어를 만들 우리 tokenizer(우리 AMI 평가와 같은 체크포인트)")
    q.add_argument("--data", required=True); q.add_argument("--out", required=True); q.add_argument("--corpora", default="ami-ihmmix,ami-sdm")
    q.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[10], help="참조 δ(우리 평가와 같은 마스크) — 가설은 δ 와 무관")
    q.add_argument("--limit-windows", type=int, default=0); q.add_argument("--segment-s", type=float, default=0.0, help="0 = 회의 통째, 480 = VibeVoice-Streaming 논문 조각")
    q.add_argument("--audio", choices=["mono", "headset-sum"], default="mono", help="mono = 우리 평가 오디오(IHM-mix = 공식 Mix-Headset), headset-sum = 화자별 헤드셋 채널을 그대로 더한 것")
    q.add_argument("--rank", type=int, default=0); q.add_argument("--world-size", type=int, default=1)
    q.add_argument("--model", default=None, help="vibevoice: VibeVoice-ASR-Streaming 폴더"); q.add_argument("--src", default=None, help="vibevoice: 소스 폴더")
    q.add_argument("--max-new-tokens", type=int, default=256, help="vibevoice: 청크당 토큰 상한(데모 기본)")
    q.add_argument("--attn", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa", help="vibevoice: 어텐션 구현(2026-10-30 — 저자 측이 Flash Attention 으로 쟀다고 함)")
    q.add_argument("--asr", default=None, help="parakeet: multitalker .nemo"); q.add_argument("--diar", default=None, help="parakeet: 화자 분리기 .nemo")
    q.add_argument("--diar-kind", choices=["sortformer", "n3d"], default="sortformer"); q.add_argument("--att-context", type=lambda s: [int(x) for x in s.split(",")], default=[70, 13])
    q.add_argument("--batch-size", type=int, default=8, help="parakeet: 한 번에 스트리밍할 회의·조각 수"); q.add_argument("--wav-dir", default=None, help="parakeet: 회의 wav 를 쓸 노드 로컬 폴더")
    q.add_argument("--device", default="cuda", help="parakeet: cuda | cpu(맥 배선 점검)")
    q.add_argument("--diar-probs", default=None, help="parakeet: 화자 분리기 대신 저장한 회의 통째 화자 확률 폴더(diar_eval --session --save-probs) — Parakeet + G2")
    q.add_argument("--encoder", default=None, help="vapasr-spk: Nemotron 인코더 폴더"); q.add_argument("--encoder-state", default=None, help="vapasr-spk: 이 graft.pt 의 enc(G2 인코더)로")
    q.add_argument("--spk-margin", type=int, default=2, help="vapasr-spk: 화자 활동 프레임 앞뒤로 남길 프레임(80 ms)")
    q.add_argument("--spk-keep", choices=["all", "slotA"], default="all", help="vapasr-spk: 세션 출력 중 화자에게 줄 슬롯(all = 모든 슬롯)")
    q.add_argument("--context-s", type=float, default=30.0, help="vapasr-spk: LLM 문맥 창(초 — 우리 회의 통째 평가와 같은 30)"); q.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    if a.system == "vibevoice": assert a.model and a.src, "--model·--src(VibeVoice) 가 필요하다"
    elif a.system == "vapasr-spk": assert a.encoder and len(a.deltas) == 1 and not a.segment_s, "vapasr-spk: --encoder, δ 하나, 회의 통째"
    else:
        assert a.asr and a.diar, "--asr·--diar(.nemo) 가 필요하다"
        assert not a.diar_probs or (os.path.isdir(a.diar_probs) and not a.segment_s), "--diar-probs 는 있는 폴더, 회의 통째(--segment-s 0)만"
        a.wav_dir = a.wav_dir or os.path.join(os.environ.get("TMPDIR", "/tmp"), "ami-ext-wav")
    {"run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
