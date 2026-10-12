"""SimulStreaming(CUNI, IWSLT 2025 동시통역 1 위) 연결형 베이스라인 실행기: Whisper large-v3 스트리밍 전사(AlignAtt) → EuroLLM-9B-Instruct
(CTranslate2) local agreement 번역. hibiki env(vapasr env 위 overlay + ctranslate2; PYTHONPATH 에 code/SimulStreaming). 공통 규약: simulst_common.py.

공식 두 단계(simulstreaming_whisper.py --task transcribe --comp_unaware → simulstreaming_translate.py --comp_unaware --input-jsonl)를 문장마다
한 프로세스 안에서 잇는다: Whisper 의 process_iter 출력(새로 확정된 전사, 방출 시각 = 그때까지 넣은 오디오 길이)을 같은 jsonl 행 형식으로
simulation_update 에 넣고, 끝에서 is_final 행(finish)으로 마무리·초기화한다. 번역 쪽은 공식 기본값(시스템 프롬프트·init 문장, min_chunk_size 1 단어,
min_len 1, sentences 버퍼, 길이 임계 없음)이다. 확정 번역의 방출 시각 = 그것을 확정시킨 입력 행의 방출 시각(계산 시간 제외).
Whisper 직접 번역(--task translate)은 large-v3 에서 '.' 만 내 쓰지 않는다(2026-10-11 확인: 같은 파일의 transcribe 는 정상).
transformers 4.57 차이 하나를 맞춘다: apply_chat_template(tokenize=True) 를 dict 로 받게 감싼다(_dict_chat_template).
바꾼 것: 입력은 원 발화 + 꼬리 무음.

usage: python experiments/simulst_simulstreaming.py --manifest M [M ...] --out DIR --ct2-dir CT2 --chunk 1.0 [--rank r --world n]
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulst_common as C  # noqa: E402

SYSTEM = "simulstreaming-cascade"
BL = "/soundai/users/tskim/VAPKT-data/vapkt-baselines"
WHISPER = f"{BL}/models/openai__whisper-large-v3/large-v3.pt"
EUROLLM = f"{BL}/models/utter-project__EuroLLM-9B-Instruct"
LAN = {"German": "de", "Korean": "ko", "English": "en"}


def _dict_chat_template(tok):
    """SimulStreaming 의 LLMTranslator.build_prompt 는 transformers 5 처럼 apply_chat_template(tokenize=True) 가 dict 를 돌려준다고 보고
    ["input_ids"] 로 읽는다. 이 env(transformers 4.57)는 기본이 리스트라 return_dict=True 로 같은 값을 dict 로 받게 한다(토큰은 같다)."""
    orig = tok.apply_chat_template

    def wrapped(*args, **kw):
        if kw.get("tokenize") and "return_dict" not in kw:
            kw["return_dict"] = True
        return orig(*args, **kw)
    tok.apply_chat_template = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ct2-dir", required=True, help="ct2-transformers-converter 로 바꾼 EuroLLM-9B-Instruct")
    ap.add_argument("--chunk", type=float, default=1.0, help="Whisper min_chunk_size(s)")
    ap.add_argument("--frame-threshold", type=int, default=25, help="AlignAtt frame_threshold(프레임 = 20 ms)")
    ap.add_argument("--tail-s", type=float, default=C.TAIL_S)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    setting = f"c{int(round(a.chunk * 1000))}-ft{a.frame_threshold}"
    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(a.out, f"predictions-{setting}-r{a.rank}of{a.world}.jsonl")
    rows = C.shard(C.load_rows(a.manifest, a.datasets), a.rank, a.world)
    done = C.done_ids(dest)
    todo = sorted((r for r in rows if r["id"] not in done), key=lambda r: (r["lang"], r["tgt_lang"]))   # 방향별로 모아 모델을 방향마다 한 번 만든다
    print(f"START {SYSTEM} {setting} rank={a.rank}/{a.world} rows={len(rows)} todo={len(todo)}", flush=True)
    if not todo: return
    import argparse as _ap
    from simulstreaming_whisper import simul_asr_factory
    import simulstreaming_translate as T
    asr = {}; mt = {}
    n = int(round(a.chunk * C.SR))
    with open(dest, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo):
            src, tgt = LAN[r["lang"]], LAN[r["tgt_lang"]]
            if src not in asr:
                asr.clear()
                asr[src] = simul_asr_factory(SimpleNamespace(
                    log_level="WARNING", decoder=None, beams=1, model_path=WHISPER, cif_ckpt_path=None, frame_threshold=a.frame_threshold,
                    audio_min_len=0.0, audio_max_len=30.0, task="transcribe", never_fire=False, init_prompt=None, static_init_prompt=None,
                    max_context_tokens=None, logdir=None, lan=src, min_chunk_size=a.chunk))[1]
            if (src, tgt) not in mt:
                p = _ap.ArgumentParser(); T.translate_args(p)
                targs = p.parse_args(["--src-lan", src, "--tgt-lan", tgt, "--model-dir", a.ct2_dir, "--tokenizer-dir", EUROLLM, "-l", "WARNING"])
                if mt:                                                                  # 이전 방향의 CT2 생성기를 내려 GPU 메모리를 비운다
                    mt.clear(); import gc; gc.collect()
                mt[(src, tgt)] = T.simul_translator_factory(targs)
                _dict_chat_template(mt[(src, tgt)].llmtranslator.tokenizer)
            online, simul = asr[src], mt[(src, tgt)]
            online.init(); simul.init()
            timer = T.SimulationTimer(comp_aware=False); incs = []

            def keep(out_seq, in_row, tm, is_final=False):
                for x in T.format_outputs(out_seq, in_row, tm, is_final=is_final):
                    if x["status"] == "COMPLETE" and x["text"]:
                        incs.append((" " + x["text"].strip(), x["emission_time"]))

            wav = C.read_wav_with_tail(r["path"], a.tail_s); err = None
            try:
                for s in range(0, len(wav), n):
                    online.insert_audio_chunk(wav[s:s + n])
                    o = online.process_iter()
                    if o and o.get("text"):
                        T.simulation_update(simul, [dict(text=o["text"], emission_time=min(s + n, len(wav)) / C.SR, end=o.get("end"), is_final=False)],
                                            timer, out_handler=keep)
                o = online.finish()
                T.simulation_update(simul, [dict(text=(o or {}).get("text", ""), emission_time=len(wav) / C.SR, end=None, is_final=True)],
                                    timer, out_handler=keep)
            except Exception as exc:  # noqa: BLE001  (예: Whisper 토크나이저 split_tokens_on_unicode 의 IndexError — 그 문장만 실패로 남긴다)
                err = f"{type(exc).__name__}: {str(exc)[:300]}"; incs = []
                online.init(); online.model.refresh_segment(complete=True); simul.init()
            text, times = C.words_with_times(incs)
            f.write(json.dumps(C.pred_row(SYSTEM, setting, r, text, times, a.tail_s, **({"error": err} if err else {})), ensure_ascii=False) + "\n"); f.flush()
            if i % 100 == 0: print(f"PROGRESS {i + 1}/{len(todo)} {r['dataset']} | {text[:80]}", flush=True)
    print(f"DONE {SYSTEM} {setting} rank={a.rank}/{a.world} → {dest}", flush=True)


if __name__ == "__main__":
    main()
