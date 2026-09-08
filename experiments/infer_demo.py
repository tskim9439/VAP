# %% [markdown]
# # VAP-ASR 스트리밍 추론 데모 (셀 단위)
# VS Code 의 "Run Cell" 또는 Jupyter(ipykernel) 에서 한 셀씩 실행. 컨테이너에서: `source scripts/activate-env.sh` 후 커널 = `/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python`.
# 보여주는 것: (1) 후처리 텍스트, (2) 후처리 없는 원시 토큰 출력(청크 경계·<NEXT_AUDIO>·바이트 조각 그대로), (3) 청크별 표와 지연·속도 통계.
# %%
import os, sys, json
os.environ.setdefault("VAPKT_PROFILE", "mxc")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__")) if "__file__" not in globals() else os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT); os.chdir(ROOT)
for line in open(os.path.join(ROOT, ".env")):                                           # .env 의 MXC_* 경로(Qwen·Nemotron·manifest) 를 환경에
    line = line.strip()
    if line and not line.startswith("#") and "=" in line: k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
import torch; print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
from vapasr.hf.infer import load_model, load_audio, transcribe, iter_stream

# %% 모델 로드 (final/ · checkpoint-N/ · eval-ckpts/checkpoint-N/ · 기존 ckpt-last.pt 모두 가능) — 약 2 분
CKPT = "/soundai/Model/VAPASR/hf-C2/final"
if not os.path.exists(CKPT): CKPT = sorted(__import__("glob").glob("/soundai/Model/VAPASR/hf-C2/eval-ckpts/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
model, tok = load_model(CKPT); print("loaded", CKPT, "| sp_ids", model.config.sp_ids["<NEXT_AUDIO>"], model.config.sp_ids["<EMPTY_AUDIO>"])

# %% 오디오 선택 — 직접 경로를 주거나, dev manifest 에서 하나 뽑기
AUDIO, LANG = None, "Korean"                                                            # 예: AUDIO = "/path/to/sample.wav"; LANG = "English"
if AUDIO is None:
    from vapasr.data.streams import read_streams
    rows = read_streams(os.path.join(os.environ["MXC_DATA_MANIFEST_DIR"], "kspon-dev" if LANG == "Korean" else "librispeech-dev"), mode="utt" if LANG == "Korean" else "stream")
    row = rows[3]; AUDIO = row["segments"][0]["path"]; REF = " ".join(s["lexical_text"] for s in row["segments"]); print("ref:", REF)
wav = load_audio(AUDIO); print(AUDIO, f"{len(wav)/16000:.2f} s")

# %% 한 번에 디코드 → 후처리 텍스트 / 원시 출력 / 통계
res = transcribe(model, tok, wav, lang=LANG, delay=2, next_bias=0.0)
print("후처리 텍스트 :", res.text(tok))                                                     # asr-tn 채점 정규화까지
print("디코드(원문)   :", res.text(tok, normalize=False))                                    # 특수 토큰만 제거
print("원시 시퀀스    :", res.raw(tok))                                                      # [AUDIO_k] … <NEXT_AUDIO> 그대로
print("통계          :", json.dumps(res.stats(), ensure_ascii=False))

# %% 청크별 표(방출이 있는 청크만). pieces 는 tokenizer 조각(바이트 조각 'ê·¸' 류 포함 = 후처리 없음)
import pandas as pd; pd.set_option("display.max_colwidth", 80); pd.set_option("display.width", 200)
res.table(tok)

# %% 모든 청크(빈 청크 포함) 를 원시 토큰 id 로
[(c.k, c.ids, c.pieces) for c in res.chunks][:40]

# %% 한 청크씩 단계적으로 보기 — 실행하면 청크마다 즉시 출력(스트리밍 흐름 확인)
acc = []
for c in iter_stream(model, tok, wav, lang=LANG, delay=2):
    if c.ids or c.k >= res.K:
        acc += c.ids; print(f"[{c.k:4d}] {c.t1*1000:6.0f} ms  +{tok.decode(c.ids)!r:24s}  tick {c.tick_ms:5.1f} ms  → {tok.decode(acc)}")

# %% 지연 편향 실험: next_bias > 0 이면 <NEXT_AUDIO> 를 억제해 더 빨리 방출(정확도와 트레이드오프)
for b in (0.0, 1.0, 2.0):
    r = transcribe(model, tok, wav, lang=LANG, delay=2, next_bias=b); st = r.stats()
    print(f"bias {b}: tok/chunk {st['tok_per_chunk']} forced {st['forced']} | {r.text(tok)}")

# %% (참조가 있을 때) 정규화 WER/CER
if "REF" in globals():
    import jiwer; from vapasr.data.textnorm import score_en, score_ko
    hyp = res.text(tok); ref = score_ko(REF, True) if LANG == "Korean" else score_en(REF)
    print("ref:", ref); print("hyp:", hyp); print("CER" if LANG == "Korean" else "WER", round(jiwer.cer(ref, hyp) if LANG == "Korean" else jiwer.wer(ref, hyp), 4))
