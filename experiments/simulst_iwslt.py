"""IWSLT 2026 공식 연결형 베이스라인(owaski/iwslt-2026-baselines, CascadeSpeechProcessor) 실행기 — iwslt env
(scripts/setup-baseline-envs.sh iwslt). 공통 규약: experiments/simulst_common.py.

공식 코드를 그대로 import 해 쓴다: Qwen3-ASR-1.7B(+ Qwen3-ForcedAligner-0.6B) → Qwen3-4B-Instruct-2507, local agreement, temperature 0,
repetition_penalty 1.05, max_new_tokens 100, 문맥(NER) 없음 = 공식 'baseline'. 바꾼 것:
  - 모델 경로(HF 이름 → mxc 로컬), vLLM 메모리 비율(ASR 0.2 → --asr-mem, LLM 0.75 → --llm-mem; max_num_seqs 1 이라 출력은 같다).
  - LLM max_model_len 1024 → --llm-max-len(기본 2048): 긴 문장에서 프롬프트(전사 + 이전 번역)가 1024 를 넘으면 공식 설정은 예외로 멈춘다.
    한도만 늘리므로 1024 안에 드는 입력의 출력은 같다. 그래도 실패한 문장은 빈 번역 + raw.error 로 남기고 다음 문장으로 간다.
  - min_start_seconds: 공식 장문 설정은 5 s — 문장 단위 평가라 --min-start(기본 1 s)로 둔다.
  - 원문·목표 언어: 행마다(set_source_language / set_target_language). 공식은 영어 원문만 썼다.
  - 입력: 원 발화 + 꼬리 무음을 speech_chunk_size 씩 넣고(process_chunk), 끝에 end_of_stream. 방출 시각 = 그때까지 넣은 오디오 길이.

usage: python experiments/simulst_iwslt.py --manifest M [M ...] --out DIR --chunk 0.96 [--rank r --world n]
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulst_common as C  # noqa: E402

SYSTEM = "iwslt26-cascade"
BL = "/soundai/users/tskim/VAPKT-data/vapkt-baselines"
ASR = "/soundai/Model/Qwen3-ASR-1.7B"
ALIGNER = "/soundai/Model/Qwen3-ForcedAligner-0.6B"
LLM_PATH = f"{BL}/models/Qwen__Qwen3-4B-Instruct-2507"


def load_agent_module():
    """공식 agent_simulstream.py 를 그대로 읽되, Python 3.12 전용 문법 한 곳(f-string 식 안의 '\\n')만 같은 값으로 미리 계산해 3.11 에서 import 한다.
    문맥(NER)이 없는 공식 'baseline' 에서 그 식은 늘 "" 라 프롬프트가 같다. 저장소 파일은 바꾸지 않는다(읽기 전용)."""
    import importlib.util
    path = f"{BL}/code/iwslt-2026-baselines/agent_simulstream.py"
    src = open(path, encoding="utf-8").read()
    expr = '{"\\nUse the provided context only to correctly resolve named entities." if context != "" else ""}'
    head = '        instruction = f"""You are a professional translator.'
    assert src.count(expr) == 1 and src.count(head) == 1, "agent_simulstream.py 가 예상과 다르다(커밋 7e2974b 기준)"
    src = src.replace(expr, "{_ctx_hint}").replace(
        head, '        _ctx_hint = "\\nUse the provided context only to correctly resolve named entities." if context != "" else ""\n' + head)
    spec = importlib.util.spec_from_loader("agent_simulstream", loader=None, origin=path)
    mod = importlib.util.module_from_spec(spec); mod.__file__ = path
    sys.modules["agent_simulstream"] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def build_processor(chunk: float, min_start: float, asr_mem: float, llm_mem: float, llm_max_len: int = 2048):
    import torch
    CascadeSpeechProcessor = load_agent_module().CascadeSpeechProcessor
    from qwen_asr import Qwen3ASRModel
    from vllm import LLM

    class LocalCascade(CascadeSpeechProcessor):
        @classmethod
        def load_model(cls, config):                                                   # 공식 load_model 과 같고 경로·메모리 비율만 다르다
            if getattr(cls, "asr", None) is None:
                cls.asr = Qwen3ASRModel.LLM(model=config.asr_model_name, gpu_memory_utilization=asr_mem, max_inference_batch_size=1,
                                            max_model_len=1024, max_new_tokens=1024, forced_aligner=ALIGNER,
                                            forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda"))
            cls.llm_client = None
            if getattr(cls, "llm", None) is None:
                cls.llm = LLM(model=config.llm_model_name, trust_remote_code=True, gpu_memory_utilization=llm_mem, tensor_parallel_size=1,
                              max_num_seqs=1, max_model_len=llm_max_len, enable_prefix_caching=True)
                cls.tokenizer = cls.llm.get_tokenizer()

    cfg = SimpleNamespace(type="agent_simulstream.CascadeSpeechProcessor", speech_chunk_size=chunk, latency_unit="word", detokenizer_type="simuleval",
                          asr_model_name=ASR, llm_model_name=LLM_PATH, source_lang="English", target_lang="German", min_start_seconds=min_start,
                          max_history_utterances=0, max_new_tokens=100, temperature=0.0, repetition_penalty=1.05, ner_results_path=None)
    return LocalCascade(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=float, default=0.96, help="speech_chunk_size(s) — 공식 0.64 / 0.96 / 1.28")
    ap.add_argument("--min-start", type=float, default=1.0)
    ap.add_argument("--asr-mem", type=float, default=0.12)
    ap.add_argument("--llm-mem", type=float, default=0.30)
    ap.add_argument("--llm-max-len", type=int, default=2048)
    ap.add_argument("--tail-s", type=float, default=C.TAIL_S)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    setting = f"c{int(round(a.chunk * 1000))}-ms{a.min_start:g}"
    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(a.out, f"predictions-{setting}-r{a.rank}of{a.world}.jsonl")
    rows = C.shard(C.load_rows(a.manifest, a.datasets), a.rank, a.world)
    done = C.done_ids(dest)
    todo = [r for r in rows if r["id"] not in done]
    print(f"START {SYSTEM} {setting} rank={a.rank}/{a.world} rows={len(rows)} todo={len(todo)}", flush=True)
    if not todo: return
    proc = build_processor(a.chunk, a.min_start, a.asr_mem, a.llm_mem, a.llm_max_len)
    n = int(round(a.chunk * C.SR))
    with open(dest, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo):
            proc.clear(); proc.set_source_language(r["lang"]); proc.set_target_language(r["tgt_lang"])
            wav = C.read_wav_with_tail(r["path"], a.tail_s); incs = []; err = None
            try:
                for s in range(0, len(wav), n):
                    out = proc.process_chunk(wav[s:s + n])
                    incs.append((out.new_string, min(s + n, len(wav)) / C.SR))
                out = proc.end_of_stream(); incs.append((out.new_string, len(wav) / C.SR))
            except Exception as exc:  # noqa: BLE001  (한 문장의 실패로 shard 를 멈추지 않는다)
                err = f"{type(exc).__name__}: {str(exc)[:300]}"; incs = []
            text, times = C.words_with_times(incs)
            extra = dict(asr=proc._state.asr_hypotheses[-1] if proc._state.asr_hypotheses else "", **({"error": err} if err else {}))
            f.write(json.dumps(C.pred_row(SYSTEM, setting, r, text, times, a.tail_s, **extra), ensure_ascii=False) + "\n"); f.flush()
            if i % 50 == 0: print(f"PROGRESS {i + 1}/{len(todo)} {r['dataset']} | {text[:80]}", flush=True)
    print(f"DONE {SYSTEM} {setting} rank={a.rank}/{a.world} → {dest}", flush=True)


if __name__ == "__main__":
    main()
