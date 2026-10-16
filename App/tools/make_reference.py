"""s4(q06-s4-ast) torch fp32 CPU 기준 디코드 → 골든 데이터 (App MLX 엔진 검증용).

연구 환경 전용 — torch·NeMo·vapasr 를 쓴다(App 런타임 코드는 이 파일을 import 하지 않는다).

  PY=~/anaconda3/envs/vapasr-local/bin/python
  $PY App/tools/make_reference.py --list                     # 케이스 목록
  $PY App/tools/make_reference.py                            # 전체 (끝난 케이스는 건너뜀 → 재개 가능)
  $PY App/tools/make_reference.py --cases tts-en1-en-asr-d6  # 일부만 (쉼표 구분)

무거운 작업(4.8 GB fp32 가중치; 맥 mini M4 실측 최대 RSS 6.5 GB, 39 케이스 806 s, torch 스레드 4) — heavy.lock 아래에서 단독으로, 배경 실행:
  mkdir -p $GOLD && /usr/bin/lockf -k -t 14400 <scratchpad>/heavy.lock env PYTHONPYCACHEPREFIX=<scratch>/pycache \
      $PY App/tools/make_reference.py >> $GOLD/make_reference.log 2>&1

기준 = decode-contract.md §g 레시피:
  - 인코더 캐시(.pt) = 체크포인트 자신의 encoder.pre.* / encoder.enc.* 텐서 + NeMo yaml 의 preprocessor/encoder 설정
    → --encoder-cache (T5). VAPASR_ENCODER_CACHE 로 가리키고 MXC_NEMOTRON_DIR/NEMOTRON_DIR 는 지우고, 빈 디렉토리를 encoder_path 로 준다.
  - VAPASR_STAGE_DIR="", HF 오프라인, torch.cuda.synchronize 는 no-op 으로 패치(stream_decode 가 무조건 부른다).
  - load_model(CK, device="cpu", dtype=float32) → att_context [56,3]·ast-v1·caps(32/8)·blocked_ids·prefix_ids 를 assert.
  - 케이스마다 s4 eval 과 같게: prefix = config.prefix_ids + tok(ast_head_text(src, "<SPK_A>", tgt)) + <DELAY_δ>,
    wav + 16000 zeros, K = int(round(n/16000*12.5)), feats = model.encode(...)[0] (오프라인 masked forward),
    model.stream_decode(feats, prefix, max_flush_rounds=8). final(오프라인) 모드는 EV:480-501 그대로(offline_decode, 상한 K+64).

출력 <GOLD>/<case>/:
  meta.json          입력·prefix·방출 [[k,id]]·forced·flush·텍스트·결정별 top-2 margin·teacher-forced 수열·층별 max|활성|(fp16 여유)·final 모드
  wav.npy            꼬리(16000 zeros) 포함 float32 샘플 그대로 (n_total,)
  mel.npy            (T,128) fp32 log-mel, T = n_total//160 (유효 프레임만)
  feats.npy          (K,1024) fp32 인코더 특징 — 기준(eval 경로, 오프라인 masked forward)
  feats_stream.npy   (K,1024) fp32 — torch cache-aware 스트리밍 경로(vapasr/hf/realtime.StreamingEncoder), 참고용
  ce.npy             (K,1024) fp32 adapter 출력
<GOLD>/index.json (케이스 요약), 모두 성공하면 <GOLD>/DONE.

teacher-forced 수열 tf_seq: 토큰 id(≥0) 또는 오디오 프레임 k 를 -(k+1) 로 적은 thinker 입력 순서(위치 = 인덱스, prefix 부터 0).
decisions.pos 는 logits 가 결정에 쓰인 위치(라운드 입력·방출 토큰 위치). <NEXT_AUDIO> 위치는 결정이 아니다.

오디오 출처(내용은 wav.npy 에 그대로 저장):
  tts-*    scratchpad/testaudio/*.wav (16 kHz PCM_16, soundfile float32 = int16/32768)
  kspon*   KsponSpeech_01 .pcm (16 kHz s16le, int16/32768 — eval read_audio 와 같다)
  epst-*   Europarl-ST v1.1 test (CoVoST 2 디렉토리가 T5 에서 비어 있어 대체). m4a(AAC 44.1 kHz 스테레오)를 PyAV 로 float32 디코드,
           채널 평균 → segments.lst 구간 [round(t0·sr), round(t1·sr)) 자르기 → scipy resample_poly(160, 441) → 16-bit wav 왕복
           (soundfile 쓰기·읽기 = eval 의 _mp3_to_wav + read_audio 와 같은 양자화).
  edge-*   tts-en1 앞부분 자르기: (n_total−4096) mod 5120 < 384 (K = 4c+3 경계) 두 길이, n_total = 69760 (round(n/1280) ≠ eval K).
"""
import argparse, gc, glob, io, json, os, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path("/private/tmp/claude-501/-Users-taesookim-Desktop-VAPKT/e6f0e8db-a78d-4321-bce1-f9fa25adbcf1/scratchpad")
APPWORK = Path("/Volumes/Samsung_T5/vapkt-models/app-work")
DB = Path("/Volumes/Samsung_T5/VAPKT-DB")
CK_DEFAULT = Path.home() / "Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync/final"

LANG = {"en": "English", "ko": "Korean", "de": "German"}
EXPECTED_BLOCKED = [151644, 151645, 151669, 151670, 151676, 151704, 151706, 151709, 151710, 151711, 151712, 151713, 151714,
                    151715, 151716, 151717, 151718, 151719, 151720, 151721, 151722, 151724, 151725]
EXPECTED_PREFIX = [151644, 8948, 198, 151645, 198, 151644, 77091, 198]
EXPECTED_HEAD = {  # decode-contract §b.2 (언어 줄)
    ("English", None): [11528, 220, 151707, 22574, 151704],
    ("Korean", None): [11528, 220, 151707, 42, 45195, 151704],
    ("German", None): [11528, 220, 151707, 32079, 151704],
    ("English", "German"): [11528, 220, 151707, 22574, 198, 14045, 220, 151707, 32079, 151704],
    ("English", "Korean"): [11528, 220, 151707, 22574, 198, 14045, 220, 151707, 42, 45195, 151704],
    ("German", "English"): [11528, 220, 151707, 32079, 198, 14045, 220, 151707, 22574, 151704],
    ("Korean", "English"): [11528, 220, 151707, 42, 45195, 198, 14045, 220, 151707, 22574, 151704],
}
EXPECTED_OBLK = sorted({151644, 151669, 151670, 151676, 151704, 151705, 151706} | set(range(151709, 151726)))
FP16_MAX = 65504.0

# ───────────────────────────── 케이스 ─────────────────────────────
KSP = DB / "KsponSpeech/KsponSpeech_01/KsponSpeech_0001"
EPST = DB / "Europarl-ST"


def audio_table(testaudio: Path) -> dict:
    """오디오 키 → 읽는 방법. edge 는 tts-en1 의 앞 n_real 샘플."""
    t = {f"tts-{c}": dict(kind="wav", path=str(testaudio / f"{c}.wav"), lang=c[:2]) for c in ("en1", "en2", "ko1", "ko2", "de1", "de2")}
    for key, f in (("kspon1", "KsponSpeech_000022"), ("kspon2", "KsponSpeech_000056"), ("kspon3", "KsponSpeech_000058")):
        t[key] = dict(kind="pcm", path=str(KSP / f"{f}.pcm"), lang="ko")
    # Europarl-ST test: (방향 디렉토리, segments 행 번호 1-based, 원문 언어)
    for key, d, line in (("epst-en1", "en/de", 14), ("epst-en2", "en/de", 18), ("epst-de1", "de/en", 6), ("epst-de2", "de/en", 21)):
        t[key] = dict(kind="m4a", split=d, line=line, lang=d[:2])
    for key, n_total in (("edge-k4c3a", 4096 + 5120 * 19 + 200), ("edge-k4c3b", 4096 + 5120 * 10 + 383), ("edge-n69760", 69760)):
        t[key] = dict(kind="crop", base="tts-en1", n_real=n_total - 16000, lang="en")
    return t


def case_table() -> list:
    """(이름, 오디오 키, src 코드, tgt 코드|None, δ)."""
    plan = []
    def add(audio, src, modes):
        for tgt, d in modes:
            plan.append((f"{audio}-{src + '-asr' if tgt is None else src + '2' + tgt}-d{d}", audio, src, tgt, d))
    add("tts-en1", "en", [(None, 6), ("de", 10), ("ko", 10), ("de", 4), (None, 4)])
    add("tts-en2", "en", [(None, 6), ("de", 10), ("ko", 10)])
    add("tts-ko1", "ko", [(None, 6), ("en", 8), ("en", 4), (None, 10)])
    add("tts-ko2", "ko", [(None, 6), ("en", 8)])
    add("tts-de1", "de", [(None, 6), ("en", 10), ("en", 4), (None, 10)])
    add("tts-de2", "de", [(None, 6), ("en", 10)])
    for a in ("kspon1", "kspon2", "kspon3"):
        add(a, "ko", [(None, 6), ("en", 8)])
    for a in ("epst-en1", "epst-en2"):
        add(a, "en", [(None, 6), ("de", 10), ("ko", 10)])
    for a in ("epst-de1", "epst-de2"):
        add(a, "de", [(None, 6), ("en", 10)])
    for a in ("edge-k4c3a", "edge-k4c3b", "edge-n69760"):
        add(a, "en", [(None, 6)])
    return plan


def load_audio(key: str, table: dict, cache: dict):
    """→ (float32 16 kHz 샘플(꼬리 없음), 출처 dict)."""
    import numpy as np, soundfile as sf
    if key in cache: return cache[key]
    a = table[key]
    if a["kind"] == "wav":
        x, sr = sf.read(a["path"], dtype="float32"); assert sr == 16000 and x.ndim == 1, (a, sr)
        src = dict(kind="wav", path=a["path"], method="soundfile float32 (PCM_16/32768)")
    elif a["kind"] == "pcm":
        raw = open(a["path"], "rb").read()
        x = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        txt = open(a["path"][:-4] + ".txt", "rb").read()
        src = dict(kind="kspon-pcm", path=a["path"], method="s16le/32768 (eval read_audio)", reference=txt.decode("cp949", "replace").strip())
    elif a["kind"] == "m4a":
        import av
        from math import gcd
        from scipy.signal import resample_poly
        sl, tl = a["split"].split("/")
        d = EPST / a["split"] / "test"
        seg = (d / "segments.lst").read_text().splitlines()[a["line"] - 1].split()
        ref = (d / f"segments.{sl}").read_text(encoding="utf-8").splitlines()[a["line"] - 1]
        ref_tr = (d / f"segments.{tl}").read_text(encoding="utf-8").splitlines()[a["line"] - 1]
        path = EPST / sl / "audios" / f"{seg[0]}.m4a"; t0, t1 = float(seg[1]), float(seg[2])
        with av.open(str(path)) as c:
            st = c.streams.audio[0]; sr = st.codec_context.sample_rate
            ys = []
            for f in c.decode(st):
                arr = f.to_ndarray()                                                      # planar: (ch, n), packed: (1, n·ch)
                ys.append(arr.reshape(-1, len(f.layout.channels)).T if f.format.is_packed else arr)
            y = np.concatenate(ys, axis=1)
        y = y.astype(np.float32).mean(0)
        y = y[int(round(t0 * sr)): int(round(t1 * sr))]
        g = gcd(int(sr), 16000); y = resample_poly(y, 16000 // g, int(sr) // g).astype(np.float32)
        bio = io.BytesIO(); sf.write(bio, np.clip(y, -1.0, 1.0), 16000, subtype="PCM_16", format="WAV"); bio.seek(0)
        x, _ = sf.read(bio, dtype="float32")
        src = dict(kind="europarl-st-v1.1", path=str(path), split=f"{a['split']}/test", line=a["line"], speech=seg[0], t0=t0, t1=t1, native_sr=sr,
                   method="PyAV float32 decode, channel mean, crop [round(t0*sr), round(t1*sr)), scipy resample_poly(16000/g, sr/g), "
                          "soundfile PCM_16 wav round trip (= eval _mp3_to_wav + read_audio)",
                   reference=ref, reference_translation=ref_tr)
    elif a["kind"] == "crop":
        base, bsrc = load_audio(a["base"], table, cache)
        assert a["n_real"] <= len(base); x = base[: a["n_real"]].copy()
        src = dict(kind="crop", base=a["base"], base_path=bsrc["path"], n_real=a["n_real"], method=f"first {a['n_real']} samples of {a['base']}")
    else:
        raise ValueError(a)
    x = np.ascontiguousarray(x, dtype=np.float32)
    cache[key] = (x, src)
    return cache[key]


# ───────────────────────────── 도우미 ─────────────────────────────
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_encoder_cache(ck: Path, cfg_path: Path, asset: Path, out: Path):
    """체크포인트 encoder.* 텐서 + yaml 설정 → NemotronOnline 캐시 파일(online.py _load_encoder_cache_file 형식)."""
    import torch
    from safetensors import safe_open
    from omegaconf import OmegaConf
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        pre_cfg, enc_cfg = (OmegaConf.to_container(cfg[s], resolve=True) for s in ("preprocessor", "encoder"))
        if asset.exists():
            js = json.loads(asset.read_text())
            assert js["preprocessor"] == pre_cfg and js["encoder"] == enc_cfg, "assets/nemotron_encoder.json 이 yaml 과 다르다"
        cfg_src = str(cfg_path)
    else:
        js = json.loads(asset.read_text()); pre_cfg, enc_cfg = js["preprocessor"], js["encoder"]; cfg_src = str(asset)
    pre_state, enc_state = {}, {}
    for f in sorted(glob.glob(str(ck / "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k.startswith("encoder.pre."): pre_state[k[len("encoder.pre."):]] = s.get_tensor(k)
                elif k.startswith("encoder.enc."): enc_state[k[len("encoder.enc."):]] = s.get_tensor(k)
    assert sorted(pre_state) == ["featurizer.fb", "featurizer.window"], sorted(pre_state)
    assert len(enc_state) == 636, len(enc_state)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    torch.save(dict(pre_cfg=pre_cfg, enc_cfg=enc_cfg, pre_state=pre_state, enc_state=enc_state), tmp)
    os.replace(tmp, out)
    log(f"encoder cache 저장 {out} ({out.stat().st_size / 2**30:.2f} GiB), cfg {cfg_src}")


class LogitTap:
    """thinker forward 마다 마지막 위치 logits(차단 마스크 후)의 argmax·top-2 차를 기록. 출력은 건드리지 않는다."""
    def __init__(self, thinker, blocked):
        self.calls, self.blocked = [], blocked
        self.h = thinker.register_forward_hook(self._hook)

    def _hook(self, mod, args, out):
        import torch
        raw = out.logits[0, -1].float()
        lg = raw.clone(); lg[self.blocked] = float("-inf")
        a = int(lg.argmax()); v, i = torch.topk(lg, 2)
        second = int(i[1]) if int(i[0]) == a else int(i[0])
        self.calls.append((a, float(v[0] - v[1]), second, bool(torch.isfinite(raw).all())))

    def close(self):
        self.h.remove()


class ActStats:
    """teacher-forced 한 번의 forward 에서 층별 max|활성| (fp16 여유 = 65504 / max)."""
    NAMES = ("h", "mid", "ln1", "ln2", "q", "k", "v", "o", "gate", "up", "act", "down")

    def __init__(self, text_model):
        self.st = {n: [0.0] * len(text_model.layers) for n in self.NAMES}
        self.ex = {n: [0.0] * len(text_model.layers) for n in self.NAMES}                # 위치 0(<|im_start|>) 제외
        self.pos = {n: [0] * len(text_model.layers) for n in ("h", "act")}
        self.hs = []
        for i, L in enumerate(text_model.layers):
            at, mlp = L.self_attn, L.mlp
            self._out(L, "h", i, pos=True); self._out(L.input_layernorm, "ln1", i); self._out(L.post_attention_layernorm, "ln2", i)
            self._pre(L.post_attention_layernorm, "mid", i)
            for n in ("q", "k", "v", "o"): self._out(getattr(at, f"{n}_proj"), n, i)
            self._out(mlp.gate_proj, "gate", i); self._out(mlp.up_proj, "up", i); self._out(mlp.down_proj, "down", i)
            self._pre(mlp.down_proj, "act", i, pos=True)

    def _upd(self, name, i, t, pos):
        m = t.detach().float().abs()
        v = float(m.max())
        if m.dim() == 3 and m.shape[1] > 1: self.ex[name][i] = max(self.ex[name][i], float(m[:, 1:].max()))
        if v > self.st[name][i]:
            self.st[name][i] = v
            if pos: self.pos[name][i] = int(m.reshape(-1, m.shape[-1]).amax(-1).argmax())

    def _out(self, mod, name, i, pos=False):                                              # 훅은 None 을 돌려준다(출력 불변)
        def f(m, a, o): self._upd(name, i, o, pos)
        self.hs.append(mod.register_forward_hook(f))

    def _pre(self, mod, name, i, pos=False):
        def f(m, a): self._upd(name, i, a[0], pos)
        self.hs.append(mod.register_forward_pre_hook(f))

    def close(self):
        for h in self.hs: h.remove()


def rnd(x, s=5):
    return float(f"{x:.{s}g}")


# ───────────────────────────── 메인 ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=str(CK_DEFAULT))
    ap.add_argument("--out", default=str(APPWORK / "golden-s4"))
    ap.add_argument("--encoder-cache", default=str(APPWORK / "s4-encoder-cache.pt"))
    ap.add_argument("--yaml", default=str(SCRATCH / "hf-nemotron/model_config.yaml"))
    ap.add_argument("--testaudio", default=str(SCRATCH / "testaudio"))
    ap.add_argument("--cases", default="", help="쉼표 구분 케이스 이름(기본: 전체)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU 스레드(0 = 기본값)")
    a = ap.parse_args()
    plan = case_table()
    if a.list:
        for c in plan: print(*c)
        print(len(plan), "cases"); return
    want = [c for c in plan if not a.cases or c[0] in a.cases.split(",")]
    assert want, f"케이스 없음: {a.cases}"
    gold = Path(a.out); gold.mkdir(parents=True, exist_ok=True)
    todo = [c for c in want if not (gold / c[0] / "meta.json").exists()]
    log(f"START {len(want)} cases, {len(want) - len(todo)} already done, {len(todo)} to run")
    t_start = time.time()

    # 환경 — vapasr/transformers import 전에 (decode-contract §g pitfalls 1–4)
    ck = Path(a.checkpoint)
    cache_path = Path(a.encoder_cache)
    empty = APPWORK / "empty-nemo-dir"; empty.mkdir(parents=True, exist_ok=True)
    assert not list(empty.glob("*.nemo"))
    os.environ.update(VAPASR_ENCODER_CACHE=str(cache_path), VAPASR_STAGE_DIR="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for k in ("MXC_NEMOTRON_DIR", "NEMOTRON_DIR"): os.environ.pop(k, None)
    sys.path.insert(0, str(REPO))
    import numpy as np, torch
    torch.cuda.synchronize = lambda *x, **k: None
    if a.threads: torch.set_num_threads(a.threads)
    if not cache_path.exists():
        t = time.time(); build_encoder_cache(ck, Path(a.yaml), REPO / "App/assets/nemotron_encoder.json", cache_path); gc.collect()
        log(f"encoder cache build {time.time() - t:.0f}s")
    if todo:
        run_cases(a, ck, gold, todo, empty)
    write_index(gold, plan, ck, t_start)


def run_cases(a, ck, gold, todo, empty):
    import numpy as np, torch
    from safetensors import safe_open
    from vapasr.hf.infer import load_model
    from vapasr.data.ast_dataset import ast_head_text
    from vapasr.data.ast_eval import split_runs, word_emission_times
    from vapasr.hf.offline_decode import offline_decode, offline_blocked_ids
    from vapasr.data.offline_seq import OfflineFormat
    from vapasr.hf.realtime import StreamingEncoder

    t = time.time()
    model, tok = load_model(str(ck), device="cpu", encoder_path=str(empty), dtype=torch.float32)
    load_s = time.time() - t
    cfg = model.config; sp = cfg.sp_ids; E = model.get_input_embeddings().weight
    # ── 로드 검증(decode-contract checklist)
    assert list(model.encoder.enc.att_context_size) == [56, 3], model.encoder.enc.att_context_size
    assert cfg.ast_format == "ast-v1" and cfg.runaway_cap == 32 and cfg.max_flush_rounds == 8 and cfg.lanes == 0
    assert cfg.encoder_right_context == 3 and cfg.frame_hz == 12.5 and (cfg.encoder_trainable or getattr(cfg, "encoder_saved", False))
    assert list(cfg.blocked_ids) == EXPECTED_BLOCKED, cfg.blocked_ids
    assert list(cfg.prefix_ids) == EXPECTED_PREFIX, cfg.prefix_ids
    assert sorted(int(x) for x in model.blocked.tolist()) == EXPECTED_BLOCKED
    want_sp = {"<NEXT_AUDIO>": 151705, "<EMPTY_AUDIO>": 151706, "<SPK_A>": 151707, "<SPK_B>": 151708, "<SEM_END>": 151723, "<TRANS>": 151726,
               **{f"<DELAY_{d}>": i for d, i in zip(range(4, 11), (151712, 151713, 151714, 151715, 151716, 151724, 151725))}}
    assert all(sp[k] == v for k, v in want_sp.items()), {k: sp.get(k) for k in want_sp}
    assert model.thinker.lm_head.weight.data_ptr() == E.data_ptr(), "lm_head 가 embed 와 tied 가 아님"
    assert E.dtype == torch.float32 and E.shape == (151936, 1024)
    assert not model.encoder.pre.training and not model.encoder.enc.training and not model.thinker.training
    with safe_open(str(ck / "model.safetensors"), "pt") as f:
        k0 = "layers.0.feed_forward1.linear1.weight"
        assert torch.equal(f.get_tensor("encoder.enc." + k0), model.encoder.enc.state_dict()[k0]), "인코더가 체크포인트 가중치가 아님"
        for k in ("featurizer.window", "featurizer.fb"):
            assert torch.equal(f.get_tensor("encoder.pre." + k), model.encoder.pre.state_dict()[k]), k
        keys = list(f.keys())
    assert not any("lm_head" in k for k in keys)
    log(f"model loaded {load_s:.0f}s; torch {torch.__version__} threads {torch.get_num_threads()}; att_context {model.encoder.enc.att_context_size}; "
        f"blocked {len(cfg.blocked_ids)}; runaway_cap {cfg.runaway_cap}; tied lm_head; dither {model.encoder.pre.featurizer.dither} (eval → off)")

    NEXT, EMPTY, SPK, TRANS, SEM = sp["<NEXT_AUDIO>"], sp["<EMPTY_AUDIO>"], [sp["<SPK_A>"], sp["<SPK_B>"]], sp["<TRANS>"], sp["<SEM_END>"]
    blocked = model.blocked

    class AstOfflineFormat(OfflineFormat):                       # EV:484-491 그대로
        def lang_ids(self, key):
            if key not in self._lang:
                lang, tgt = key.split("|")
                self._lang[key] = list(self.tok(ast_head_text(lang, "<SPK_A>", tgt or None), add_special_tokens=False)["input_ids"])
            return self._lang[key]
    ofmt = AstOfflineFormat(tok)
    oblk = torch.tensor(sorted(set(offline_blocked_ids(model, ofmt)) - set(SPK) - {TRANS}))
    assert oblk.tolist() == EXPECTED_OBLK, oblk.tolist()
    assert ofmt.pre == [151644, 8948, 198, 151645, 198, 151644, 872, 198] and ofmt.mid == [151645, 198, 151644, 77091, 198]

    table = audio_table(Path(a.testaudio)); acache = {}; feat_cache = {}; final_cache = {}
    stream_enc = []                                                                       # 첫 오프라인 인코드 뒤에 만든다(setup_streaming_params)

    def features(key):
        """오디오 키 → wav(+꼬리)·K·mel·feats(eval 경로)·feats_stream·ce. 같은 오디오의 케이스끼리 재사용."""
        if key in feat_cache: return feat_cache[key]
        x, src = load_audio(key, table, acache)
        t0 = time.time()
        w = np.pad(x, (0, 16000)).astype(np.float32); n = len(w); K = int(round(n / 16000 * cfg.frame_hz))
        wt = torch.from_numpy(w)
        cap = {}
        def tenc(m, i, o): cap.setdefault("T_enc", int(o[1][0]))                         # None 반환(출력은 그대로)
        h = model.encoder.enc.register_forward_hook(tenc)
        try:
            mel, ml = model.encoder.pre(input_signal=wt[None], length=torch.tensor([n]))
            feats = model.encode(wt[None], torch.tensor([n]), torch.tensor([K]))[0]      # (1,K,1024) — eval 경로
        finally: h.remove()
        ml = int(ml[0]); assert ml == n // 160, (ml, n)
        assert cap["T_enc"] >= K, (cap, K)
        first = not stream_enc
        if first: stream_enc.append(StreamingEncoder(model.encoder, cuda_graphs=False))
        fs, _, _ = stream_enc[0].encode(wt, K)                                            # (1,K,1024) — 스트리밍 경로
        if first:                                                                         # 스트리밍 설정·실행이 오프라인(기준) 경로를 바꾸지 않는지
            again = model.encode(wt[None], torch.tensor([n]), torch.tensor([K]))[0]
            assert torch.equal(again, feats), "스트리밍 설정 뒤 오프라인 특징이 달라졌다"
            log("check: offline features unchanged after StreamingEncoder setup/run")
        ce = model.chunk_embed(feats[None])[0].to(E.dtype)
        r = dict(x=x, src=src, w=w, n=n, K=K, mel=mel[0, :, :ml].T.contiguous().numpy(), ml=ml, mel_cols=int(mel.shape[-1]),
                 feats=feats, ce=ce, feats_stream=fs[0].numpy(), stream_maxdiff=float((fs[0] - feats[0]).abs().max()),
                 T_enc=cap["T_enc"], enc_s=time.time() - t0)
        feat_cache.clear(); feat_cache[key] = r                                           # 오디오 하나만 들고 있는다
        return r

    def final_mode(key, F, src, tgt):
        """final(오프라인) 모드 — EV:480-501. (audio, src, tgt) 마다 한 번."""
        fk = (key, src, tgt)
        if fk in final_cache: return final_cache[fk]
        K = F["K"]; lk = f"{src}|{tgt or ''}"
        tap = LogitTap(model.thinker, oblk); t0 = time.time()
        try: ids = offline_decode(model, tok, F["feats"][0], lk, fmt=ofmt, blocked=oblk, max_new_tokens=K + 64)
        finally: tap.close()
        dt = time.time() - t0; calls = tap.calls; cap_hit = len(ids) >= K + 64
        nd = len(ids) if cap_hit else len(ids) + 1
        assert [c[0] for c in calls[: len(ids)]] == ids and (cap_hit or calls[len(ids)][0] == ofmt.im_end), "final 재구성 불일치"
        s_t, r_t = split_runs(ids, SPK, TRANS)
        m = [c[1] for c in calls[:nd]]
        r = dict(final_ids=ids, final_src=tok.decode([t for _, t in s_t], skip_special_tokens=True).strip(),
                 final_tr=tok.decode([t for _, t in r_t], skip_special_tokens=True).strip(), final_max_new=K + 64, final_cap_hit=cap_hit,
                 final_header=dict(pre=ofmt.pre, audio_start=ofmt.audio_start, audio_pad=ofmt.audio_pad, audio_end=ofmt.audio_end, mid=ofmt.mid,
                                   head=ofmt.lang_ids(lk), a0=len(ofmt.pre) + 1, header_len=len(ofmt.header(K, lk)), lang_key=lk),
                 final_blocked=oblk.tolist(), final_margins=[rnd(x, 6) for x in m], final_min_margin=min(m) if m else None,
                 final_second_ids=[c[2] for c in calls[:nd]], final_nonfinite=sum(not c[3] for c in calls), final_s=dt)
        final_cache[fk] = r
        return r

    failures = []
    torch.set_grad_enabled(False)                                                         # 기록 없음(teacher-forced 활성 메모리)
    for name, akey, s_code, t_code, delta in todo:
        try:
            t_case = time.time()
            src, tgt = LANG[s_code], (LANG[t_code] if t_code else None)
            F = features(akey); K, n = F["K"], F["n"]
            head = tok(ast_head_text(src, "<SPK_A>", tgt), add_special_tokens=False)["input_ids"]
            assert head == EXPECTED_HEAD[(src, tgt)], (name, head)
            pfx = list(cfg.prefix_ids) + head + [sp[f"<DELAY_{delta}>"]]
            # ── 기준 디코드 (s4 eval 과 같은 stream_decode) + 결정마다 margin 기록
            tap = LogitTap(model.thinker, blocked); t0 = time.time()
            try: out, forced, ticks, rounds = model.stream_decode(F["feats"], pfx, max_flush_rounds=8)
            finally: tap.close()
            dec_s = time.time() - t0
            out = [(int(k), int(x)) for k, x in out]
            # ── 호출 열 → 결정 행 재구성 (stream_decode 규칙 그대로) + teacher-forced 수열
            calls = tap.calls; cap, max_total = cfg.runaway_cap, int(6.0 * K) + 64
            seq = [int(x) for x in pfx]; D = dict(pos=[], k=[], id=[], margin=[], id2=[]); emitted = []; nforced = 0; i = 1; nonfinite = sum(not c[3] for c in calls)

            def one_round(k, inp):
                nonlocal i, nforced
                seq.append(inp); nr = 0
                while True:
                    a_, m_, s_, _ = calls[i]; i += 1
                    D["pos"].append(len(seq) - 1); D["k"].append(k); D["id"].append(a_); D["margin"].append(m_); D["id2"].append(s_)
                    if a_ == NEXT or nr >= cap or len(emitted) >= max_total:
                        nforced += int(a_ != NEXT); i += 1; seq.append(NEXT); return nr          # 다음 호출은 <NEXT_AUDIO> 입력(결정 아님)
                    emitted.append((k, a_)); nr += 1; seq.append(a_)
            for k in range(K): one_round(k, -(k + 1))
            fl = 0
            for r in range(8):
                fl += 1
                if one_round(K + r, EMPTY) == 0: break
            assert i == len(calls), (i, len(calls))
            assert emitted == out and nforced == forced and fl == rounds, f"{name}: 재구성 불일치"
            # ── teacher-forced 재생(한 번의 forward): 층별 max|활성| + 결정 행 argmax 재확인
            t0 = time.time()
            x = torch.stack([F["ce"][-t - 1] if t < 0 else E[t] for t in seq])
            st = ActStats(model.thinker.model)
            try:
                hN = model.thinker.model(inputs_embeds=x[None], use_cache=False).last_hidden_state[0]
            finally: st.close()
            tf_id, tf_margin, logit_max = [], [], 0.0
            for j in range(0, len(D["pos"]), 128):
                lg = model.thinker.lm_head(hN[D["pos"][j: j + 128]]).float()
                logit_max = max(logit_max, float(lg.abs().max())); lg[:, blocked] = float("-inf")
                v, ix = torch.topk(lg, 2, dim=-1); tf_id += lg.argmax(-1).tolist(); tf_margin += (v[:, 0] - v[:, 1]).tolist()
            tf_s = time.time() - t0
            dis = [j for j, (p, q) in enumerate(zip(tf_id, D["id"])) if p != q]
            emb_audio = float(F["ce"].abs().max()); emb_tok = float(E[[t for t in seq if t >= 0]].abs().max())
            S = {kk: [rnd(v) for v in vv] for kk, vv in st.st.items()}
            mx = {kk: max(vv) for kk, vv in st.st.items()}
            mx_ex = {kk: max(vv) for kk, vv in st.ex.items()}
            fp16_mats = max(mx[kk] for kk in ("q", "k", "v", "o", "gate", "up", "act", "down"))
            stats = dict(per_layer=S, h_argpos=st.pos["h"], act_argpos=st.pos["act"], emb_audio_max=rnd(emb_audio), emb_token_max=rnd(emb_tok),
                         final_norm_max=rnd(float(hN.abs().max())), logits_absmax=rnd(logit_max),
                         max=dict({kk: rnd(v) for kk, v in mx.items()}),
                         headroom_resid32=rnd(FP16_MAX / max(fp16_mats, logit_max, mx["ln1"], mx["ln2"], emb_audio, emb_tok)),
                         headroom_plain_fp16=rnd(FP16_MAX / max(max(mx.values()), logit_max, emb_audio, emb_tok)),
                         per_layer_excl_pos0={kk: [rnd(v) for v in st.ex[kk]] for kk in ("h", "mid", "act", "down")},
                         max_excl_pos0={kk: rnd(v) for kk, v in mx_ex.items()},
                         headroom_plain_fp16_excl_pos0=rnd(FP16_MAX / max(max(mx_ex.values()), logit_max, emb_audio, emb_tok)),
                         note="max|x| over all teacher-forced positions incl. the prefix; h = residual after layer, mid = residual after attention, "
                              "ln1/ln2 = RMSNorm outputs, act = SwiGLU intermediate (down_proj input); resid32 = fp32 residual mode (matmul I/O in fp16); "
                              "*_excl_pos0 = same without position 0 (<|im_start|>, where the massive activation lives)")
            # ── 텍스트
            s_t, r_t = split_runs(out, SPK, TRANS)
            src_text = tok.decode([t for _, t in s_t], skip_special_tokens=True).strip()
            tr_text = tok.decode([t for _, t in r_t], skip_special_tokens=True).strip()
            tr_stream, tr_times = word_emission_times(tok, r_t) if r_t else ("", [])
            fin = final_mode(akey, F, src, tgt)
            M = D["margin"]; jmin = int(np.argmin(M))
            meta = dict(
                case=name, audio_key=akey, audio=F["src"], src=src, tgt=tgt, delta=delta, mode="asr" if tgt is None else "ast",
                n_real=len(F["x"]), tail_samples=16000, n_total=n, K=K, K_round_n_over_1280=int(round(n / 1280)), audio_s=len(F["x"]) / 16000,
                k4c3=bool((n - 4096) % 5120 < 384), mel_len=F["ml"], mel_cols_incl_pad=F["mel_cols"], T_enc_offline=F["T_enc"],
                feats_stream_maxdiff=F["stream_maxdiff"], prefix_ids=pfx, prefix_text=tok.decode(pfx),
                emitted=[list(e) for e in out], n_tokens=len(out), forced=forced, flush_rounds=rounds, max_total=max_total,
                total_cap_hit=len(out) >= max_total, runaway_cap=cap, max_flush_rounds=8,
                src_ids=[t for _, t in s_t], tr_ids=[t for _, t in r_t], src_text=src_text, tr_text=tr_text,
                tr_text_stream=tr_stream, tr_word_times=tr_times,
                tag_counts=dict(spk_a=sum(t == SPK[0] for _, t in out), spk_b=sum(t == SPK[1] for _, t in out), trans=sum(t == TRANS for _, t in out),
                                sem_end=sum(t == SEM for _, t in out)),
                post_audio_tokens=sum((k + 1) * 0.08 > len(F["x"]) / 16000 for k, _ in out),
                decisions=dict(D, margin=[rnd(v, 6) for v in M]), n_decisions=len(M),
                min_margin=M[jmin], min_margin_at=dict(idx=jmin, pos=D["pos"][jmin], k=D["k"][jmin], id=D["id"][jmin], id2=D["id2"][jmin]),
                n_margin_lt=dict({"1e-3": sum(v < 1e-3 for v in M), "1e-2": sum(v < 1e-2 for v in M), "0.1": sum(v < 0.1 for v in M), "1": sum(v < 1 for v in M)}),
                nonfinite_logit_calls=nonfinite,
                tf_seq=seq, tf_seq_encoding="id >= 0: token id (embedding row); id < 0: audio frame k = -id-1 (ce[k]); position = index",
                tf_agree=len(M) - len(dis), tf_disagree=[dict(idx=j, pos=D["pos"][j], ref=D["id"][j], tf=tf_id[j], margin=M[j]) for j in dis],
                tf_margin_maxdiff=rnd(max((abs(p - q) for p, q in zip(tf_margin, M)), default=0.0)),
                act_stats=stats, **fin,
                reference=dict(kind="torch fp32 CPU VapAsrForStreamingASR.stream_decode (decode-contract §g)", checkpoint=str(ck),
                               torch=torch.__version__, threads=torch.get_num_threads()),
                runtime_s=dict(features=rnd(F["enc_s"], 4), decode=rnd(dec_s, 4), teacher_forced=rnd(tf_s, 4), final=rnd(fin["final_s"], 4),
                               total=rnd(time.time() - t_case, 4)))
            meta.pop("final_s")
            od = gold / name; tmp = gold / f".{name}.tmp"
            if tmp.exists():
                for p in tmp.iterdir(): p.unlink()
            tmp.mkdir(exist_ok=True)
            np.save(tmp / "wav.npy", F["w"]); np.save(tmp / "mel.npy", F["mel"]); np.save(tmp / "feats.npy", F["feats"][0].numpy())
            np.save(tmp / "feats_stream.npy", F["feats_stream"]); np.save(tmp / "ce.npy", F["ce"].numpy())
            (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            os.replace(tmp, od)
            log(f"CASE {name}: n {n} K {K} tokens {len(out)} forced {forced} flush {rounds} cap_hit {meta['total_cap_hit']} | min margin {M[jmin]:.4g} "
                f"(<1e-2: {meta['n_margin_lt']['1e-2']}) | tf agree {meta['tf_agree']}/{len(M)} | headroom resid32 {stats['headroom_resid32']} "
                f"plain {stats['headroom_plain_fp16']} (excl pos0 {stats['headroom_plain_fp16_excl_pos0']}) | stream Δ {F['stream_maxdiff']:.2e} | dec {dec_s:.1f}s tf {tf_s:.1f}s total {time.time() - t_case:.1f}s")
            log(f"   SRC: {src_text}")
            if tgt: log(f"   TR : {tr_text}")
            log(f"   FINAL SRC: {fin['final_src']}")
            if tgt: log(f"   FINAL TR : {fin['final_tr']}")
        except Exception as e:
            failures.append(name); log(f"FAIL {name}: {type(e).__name__}: {e}"); traceback.print_exc(); sys.stdout.flush()
    if failures: log(f"FAILURES: {failures}")


def write_index(gold, plan, ck, t_start):
    rows, missing = [], []
    for name, akey, s, t, d in plan:
        p = gold / name / "meta.json"
        if not p.exists(): missing.append(name); continue
        m = json.loads(p.read_text())
        rows.append(dict(case=name, audio_key=akey, src=m["src"], tgt=m["tgt"], delta=d, n_real=m["n_real"], n_total=m["n_total"], K=m["K"],
                         n_tokens=m["n_tokens"], forced=m["forced"], flush_rounds=m["flush_rounds"], total_cap_hit=m["total_cap_hit"],
                         min_margin=m["min_margin"], n_margin_lt_1e2=m["n_margin_lt"]["1e-2"], tf_agree=f"{m['tf_agree']}/{m['n_decisions']}",
                         headroom_resid32=m["act_stats"]["headroom_resid32"], headroom_plain_fp16=m["act_stats"]["headroom_plain_fp16"],
                         headroom_plain_fp16_excl_pos0=m["act_stats"]["headroom_plain_fp16_excl_pos0"],
                         src_text=m["src_text"], tr_text=m["tr_text"], final_src=m["final_src"], final_tr=m["final_tr"],
                         final_min_margin=m["final_min_margin"], runtime_s=m["runtime_s"]["total"]))
    cfgp = ck / "config.json"
    import hashlib
    idx = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), checkpoint=str(ck),
               checkpoint_config_sha256=hashlib.sha256(cfgp.read_bytes()).hexdigest(),
               checkpoint_weights={p.name: dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sorted(ck.glob("*.safetensors"))},
               files=["meta.json", "wav.npy", "mel.npy", "feats.npy", "feats_stream.npy", "ce.npy"], cases=rows, missing=missing,
               summary=dict(n=len(rows), min_margin=min((r["min_margin"] for r in rows), default=None),
                            min_headroom_resid32=min((r["headroom_resid32"] for r in rows), default=None),
                            min_headroom_plain_fp16=min((r["headroom_plain_fp16"] for r in rows), default=None),
                            min_headroom_plain_fp16_excl_pos0=min((r["headroom_plain_fp16_excl_pos0"] for r in rows), default=None),
                            cap_hits=[r["case"] for r in rows if r["total_cap_hit"]]))
    (gold / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1))
    log(f"index.json: {len(rows)} cases, missing {missing}; run {time.time() - t_start:.0f}s")
    if not missing:
        (gold / "DONE").write_text(json.dumps(dict(finished=idx["created"], cases=len(rows), run_s=round(time.time() - t_start)), indent=1) + "\n")
        log("DONE written")


if __name__ == "__main__":
    main()
