"""SeamlessStreaming(S2TT) 베이스라인 실행기 — seamless env(scripts/setup-baseline-envs.sh seamless). 공통 규약: experiments/simulst_common.py.

공식 평가(seamless_communication.cli.streaming.evaluate, --task s2tt)와 같은 에이전트·설정(source_segment_size 320 ms,
min_starting_wait_w2vbert 192, no_early_stop, max_len_a 0 / b 100)을 SimulEval 로 돌린다. 바꾼 것:
  - 데이터 로더: 원 발화 + 꼬리 무음(우리 모델과 같은 입력), 앞뒤 무음 제거(Silero VAD) 끔 — 입력·시간 기준을 같게.
  - 목표 언어: 행마다(tgt_lang 열). 점수는 내지 않고(--no-scoring) instances.log 의 SPM 조각별 지연(읽은 원문 ms)을 단어 시각으로 바꾼다.
  - dtype: 논문 수치와 같은 fp32 가 기본.
체크포인트는 오프라인 fairseq2 카드($FAIRSEQ2_USER_ASSET_DIR, file:// 경로)로 읽는다.

usage: python experiments/simulst_seamless.py --manifest M [M ...] --out DIR --threshold 0.5 [--rank r --world n] [--datasets ...]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulst_common as C  # noqa: E402

SYSTEM = "seamless-streaming"


def run_simuleval(tsv: str, out: str, threshold: float, dtype: str, tail_s: float) -> str:
    from fairseq2.assets import asset_store, download_manager
    from seamless_communication.streaming.agents.seamless_streaming_s2t import SeamlessStreamingS2TAgent
    from seamless_communication.streaming.dataloaders.s2tt import SimulEvalSpeechToTextDataloader
    from simuleval.cli import evaluate
    from simuleval.data.dataloader import register_dataloader

    @register_dataloader("vapkt_tail_s2tt")
    class TailS2TT(SimulEvalSpeechToTextDataloader):
        def get_source(self, index=None):
            src = super().get_source(index)
            sr = int(self.item["audio"]["data"]["sample_rate"][0])
            return src + [0.0] * int(round(tail_s * sr))

        def get_tgt_lang(self, index=None):
            return str(self.item["tgt_lang"][0])

    card = asset_store.retrieve_card(name="seamless_streaming_unity")
    tok = download_manager.download_tokenizer(card.field("tokenizer").as_uri(), card.name, force=False, progress=False)
    config = dict(dataloader="vapkt_tail_s2tt", source_segment_size=320, device="cuda:0", dtype=dtype, min_starting_wait_w2vbert=192,
                  decision_threshold=threshold, no_early_stop=True, max_len_a=0, max_len_b=100,
                  quality_metrics="BLEU", latency_metrics="AL LAAL", eval_latency_unit="spm", eval_latency_spm_model=tok)   # 점수는 내지 않는다(--no-scoring)
    parser = argparse.ArgumentParser(add_help=False, conflict_handler="resolve")
    parser.add_argument("--unity-model-name", type=str, default="seamless_streaming_unity")
    # --task 는 에이전트(UnitYAgentPipeline.add_args)가 기본값 없이 정의한다 — 공식 CLI 처럼 명령줄로 준다
    sys.argv = [sys.argv[0], "--task", "s2tt", "--data-file", tsv, "--audio-root-dir", "/", "--output", out, "--no-strip-silence", "--no-scoring"]
    evaluate(SeamlessStreamingS2TAgent, config, parser)
    return os.path.join(out, "instances.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--tail-s", type=float, default=C.TAIL_S)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    setting = f"t{a.threshold:g}-{a.dtype}"
    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(a.out, f"predictions-{setting}-r{a.rank}of{a.world}.jsonl")
    if os.path.exists(dest):
        print(f"이미 있음: {dest}"); return
    rows = C.shard(C.load_rows(a.manifest, a.datasets), a.rank, a.world)
    work = os.path.join(a.out, f"simuleval-{setting}-r{a.rank}of{a.world}"); os.makedirs(work, exist_ok=True)
    tsv = os.path.join(work, "data.tsv")
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("id\taudio\ttgt_text\ttgt_lang\n")
        for r in rows:
            f.write(f"{r['id']}\t{r['path'].lstrip('/')}\t{' '.join(r['translation_reference'].split())}\t{C.LANG3[r['tgt_lang']]}\n")
    print(f"START {SYSTEM} {setting} rank={a.rank}/{a.world} rows={len(rows)}", flush=True)
    log = run_simuleval(tsv, os.path.join(work, "out"), a.threshold, a.dtype, a.tail_s)
    inst = {}
    for line in open(log, encoding="utf-8"):
        x = json.loads(line); inst[int(x["index"])] = x
    assert len(inst) == len(rows), (len(inst), len(rows))
    with open(dest + ".partial", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            x = inst[i]; pieces = x.get("prediction_spm") or x["prediction"].split()
            assert len(pieces) == len(x["delays"]), (r["id"], len(pieces), len(x["delays"]))
            sep = "" if "prediction_spm" in x else " "
            text, times = C.words_with_times((p.replace("▁", " ") + sep, d / 1000.0) for p, d in zip(pieces, x["delays"]))
            f.write(json.dumps(C.pred_row(SYSTEM, setting, r, text, times, a.tail_s, source_length_ms=x["source_length"]), ensure_ascii=False) + "\n")
    os.replace(dest + ".partial", dest)
    print(f"DONE {SYSTEM} {setting} rank={a.rank}/{a.world} → {dest}", flush=True)


if __name__ == "__main__":
    main()
