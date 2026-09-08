"""VapAsrTrainer — transformers.Trainer 서브클래스.
덮어쓰는 곳: get_train_dataloader(라운드로빈·언어별 버킷 배치, rank 분할 → accelerate 로 다시 나누지 않는다), compute_loss(언어별 next_weight),
create_optimizer(adapter/thinker/임베딩(wd 0)/encoder 그룹), evaluate(전 rank 분산 스트리밍 디코드 → gloo all_gather), log(손실 분해 평균).
PreemptCallback: PREEMPT 파일·SIGUSR1/SIGTERM → 모든 rank 합의(gloo all_reduce) → 저장 후 종료(sbatch 가 requeue, 다음 시작은 resume_from_checkpoint)."""
import os, json, math, time, signal, difflib
from typing import Dict, List, Optional
import numpy as np, torch, torch.distributed as dist
from transformers import Trainer, TrainerCallback, TrainingArguments
from ..uslm.mono_data import lang_of, CHUNK_S
from ..data.textnorm import score_en, score_ko

def pct(x, p): return float(np.percentile(x, p)) if len(x) else None
def latency_stats(hyp, ref):
    h_ids = [t for _, t in hyp]; r_ids = [t for t, _ in ref]; lat = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, h_ids, r_ids, autojunk=False).get_opcodes():
        if tag == "equal":
            for di in range(i2 - i1): lat.append((hyp[i1 + di][0] + 1) * CHUNK_S - ref[j1 + di][1])
    return lat

class PreemptCallback(TrainerCallback):
    """out_dir/PREEMPT 파일 또는 SIGUSR1/SIGTERM → 다음 step 경계에서 저장·종료. 판정은 gloo all_reduce 로 rank 간 일치시킨다."""
    def __init__(self, out_dir: str, gloo_pg=None):
        self.out_dir, self.pg, self.flag = out_dir, gloo_pg, False; self.fired = False
        for s in (signal.SIGUSR1, signal.SIGTERM): signal.signal(s, self._sig)
    def _sig(self, *_): self.flag = True
    def on_step_end(self, args, state, control, **kw):
        f = self.flag or os.path.exists(os.path.join(self.out_dir, "PREEMPT"))
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([1 if f else 0]); dist.all_reduce(t, group=self.pg); f = bool(t.item())
        if f and not self.fired:
            self.fired = True; control.should_save = True; control.should_training_stop = True
            if state.is_world_process_zero: print(f"[{time.strftime('%T')}] 선점/종료 신호 → step {state.global_step} 저장 후 종료 (재시작하면 이어서 학습)", flush=True)
        return control

class VapAsrTrainer(Trainer):
    def __init__(self, *a, train_sets: Dict[str, object] = None, dev_sets: Dict[str, object] = None, tokenizer=None, bs_en: int = 12, bs_ko: int = 48, lr_adapter: float = 1e-3,
                 lr_encoder: float = 1e-5, eval_delay: int = 2, eval_biases=(0.0,), max_per_chunk: int = 0, gloo_pg=None, num_workers: int = 4, **kw):
        self.train_sets, self.dev_sets, self.tok = train_sets or {}, dev_sets or {}, tokenizer
        self.bs_en, self.bs_ko, self.lr_adapter, self.lr_encoder = bs_en, bs_ko, lr_adapter, lr_encoder
        self.eval_delay, self.eval_biases, self.M, self.gloo_pg, self.num_workers = eval_delay, list(eval_biases), max_per_chunk, gloo_pg, num_workers
        self._parts = {}; self._n_parts = 0; self.eval_hist: List[dict] = []
        super().__init__(*a, train_dataset=next(iter(self.train_sets.values())) if self.train_sets else None, eval_dataset=next(iter(self.dev_sets.values())) if self.dev_sets else None, **kw)   # dict 를 주면 Trainer 가 셋마다 evaluate 를 부른다 → 자리표시자 하나만

    # ── 데이터
    def bs_of(self, name): return self.bs_ko if lang_of(name) == "Korean" else self.bs_en
    def get_train_dataloader(self):
        from .data import RoundRobinLoader
        return RoundRobinLoader(self.train_sets, self.bs_of, seed=self.args.seed, rank=self.args.process_index, world=self.args.world_size, num_workers=self.num_workers)
    def num_examples(self, dataloader): return sum(len(ds) for ds in self.train_sets.values())

    # ── 손실
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        lang = inputs.get("lang", ["English"])[0]
        nw = self.model.config.next_weight_ko if (lang == "Korean" and self.model.config.next_weight_ko is not None) else self.model.config.next_weight
        x = {k: inputs[k] for k in ("wav", "wav_len", "K", "feats", "ids", "is_audio", "chunk_of", "labels", "mask") if k in inputs}
        out = model(**x, next_weight=nw)
        for k in ("loss_next", "loss_text", "top1_text"): self._parts[k] = self._parts.get(k, 0.0) + float(getattr(out, k))
        self._parts["n_labels"] = self._parts.get("n_labels", 0) + int(out.n_labels); self._n_parts += 1
        return (out.loss, out) if return_outputs else out.loss
    def log(self, logs: dict, start_time=None):
        if self._n_parts and "loss" in logs:
            for k in ("loss_next", "loss_text", "top1_text"): logs[k] = round(self._parts[k] / self._n_parts, 4)
            logs["labels_per_step"] = round(self._parts["n_labels"] / self._n_parts); self._parts = {}; self._n_parts = 0
        return super().log(logs, start_time) if start_time is not None else super().log(logs)

    # ── 옵티마이저 그룹 (adapter lr↑, 임베딩 wd 0, encoder 별도 lr)
    def create_optimizer(self):
        if self.optimizer is not None: return self.optimizer
        m = self.model; emb_w = m.get_input_embeddings().weight; wd = self.args.weight_decay
        groups = [{"params": [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("adapter")], "lr": self.lr_adapter, "weight_decay": wd},
                  {"params": [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith(("adapter", "encoder")) and p is not emb_w], "lr": self.args.learning_rate, "weight_decay": wd},
                  {"params": [emb_w] if emb_w.requires_grad else [], "lr": self.args.learning_rate, "weight_decay": 0.0},
                  {"params": [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("encoder")], "lr": self.lr_encoder, "weight_decay": wd}]
        groups = [g for g in groups if g["params"]]
        assert sum(len(g["params"]) for g in groups) == sum(1 for p in m.parameters() if p.requires_grad)
        cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, m); kwargs.pop("lr", None); kwargs.pop("weight_decay", None)
        self.optimizer = cls(groups, **kwargs); return self.optimizer

    # ── 분산 스트리밍 평가
    @torch.no_grad()
    def _eval_set(self, ds, bias, delay):
        m = self.model; m.eval(); rank, world = self.args.process_index, self.args.world_size; dev = self.args.device
        lang = ds.items[0]["lang"] if ds.items else "English"; R, H, lat, forced, chunks, n_tok, n_ref, backlog, ticks, rounds = [], [], [], 0, 0, 0, 0, [], [], []
        for i in range(rank, len(ds), world):
            f, ref, _, st, it = ds.stream(i, delay)
            with torch.autocast("cuda", dtype=torch.bfloat16): emitted, fc, tk, rd = m.stream_decode(torch.from_numpy(f).to(dev), ds.prefix(lang, delay), self.M, next_bias=bias)
            ticks += list(tk); rounds.append(rd); forced += fc; chunks += it["K"]; n_tok += len(emitted); n_ref += len(ref); backlog.append(st.max_backlog)
            R.append(self.tok.decode([t for t, _ in ref])); H.append(self.tok.decode([t for _, t in emitted])); lat += latency_stats(emitted, ref)
        if world > 1:
            parts = [None] * world; dist.all_gather_object(parts, dict(R=R, H=H, lat=lat, forced=forced, chunks=chunks, n_tok=n_tok, n_ref=n_ref, backlog=backlog, ticks=ticks, rounds=rounds), group=self.gloo_pg)
            R = sum((q["R"] for q in parts), []); H = sum((q["H"] for q in parts), []); lat = sum((q["lat"] for q in parts), []); backlog = sum((q["backlog"] for q in parts), [])
            ticks = sum((q["ticks"] for q in parts), []); rounds = sum((q["rounds"] for q in parts), []); forced = sum(q["forced"] for q in parts); chunks = sum(q["chunks"] for q in parts)
            n_tok = sum(q["n_tok"] for q in parts); n_ref = sum(q["n_ref"] for q in parts)
        lat = np.array(lat); ticks = np.array(ticks); mm = int(len(lat))
        r = dict(bias=bias, delay=delay, n=len(ds), matched=mm, lat_p50=pct(lat, 50), lat_p90=pct(lat, 90), lat_p99=pct(lat, 99), viol=(float((lat < 0).mean()) if mm else None),
                 viol_80ms=(float((lat < -0.08).mean()) if mm else None), tok_per_chunk=n_tok / max(1, chunks), ref_per_chunk=n_ref / max(1, chunks), forced_frac=forced / max(1, chunks),
                 backlog_p99=pct(np.array(backlog), 99), flush_rounds_mean=float(np.mean(rounds)) if rounds else None, tick_ms_p50=pct(ticks, 50), tick_ms_p99=pct(ticks, 99))
        if lang == "Korean":
            import jiwer; r["cer_official"] = jiwer.cer([score_ko(x, True) for x in R], [score_ko(x, True) for x in H]); r["cer_nospace"] = jiwer.cer([score_ko(x, False) for x in R], [score_ko(x, False) for x in H]); r["err"] = r["cer_official"]
        else:
            import jiwer; r["wer"] = jiwer.wer([score_en(x) for x in R], [score_en(x) for x in H]); r["err"] = r["wer"]
        return r
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """모든 rank 가 함께 호출(분산 디코드). 반환: {eval_<set>_err, …, eval_score}. results.json 의 hist 에도 누적."""
        t0 = time.time(); sets = eval_dataset if isinstance(eval_dataset, dict) else self.dev_sets; res = {}
        was_training = self.model.training
        for name, ds in sets.items():
            t_set = time.time(); runs = {b: self._eval_set(ds, b, self.eval_delay) for b in self.eval_biases}; best_b = min(runs, key=lambda b: runs[b]["err"]); b0 = runs.get(0.0, runs[min(runs)]); b0["sec"] = round(time.time() - t_set, 1)
            res[name] = dict(bias0=b0, best=runs[best_b], best_bias=best_b)
        if was_training: self.model.train()
        torch.cuda.empty_cache()
        score = float(np.mean([v["best"]["err"] for v in res.values()])) if res else float("nan")
        metrics = {f"{metric_key_prefix}_score": score, f"{metric_key_prefix}_sec": round(time.time() - t0, 1)}
        for name, v in res.items():
            b0 = v["bias0"]; metrics[f"{metric_key_prefix}_{name}_err"] = round(b0["err"], 4); metrics[f"{metric_key_prefix}_{name}_tok_per_chunk"] = round(b0["tok_per_chunk"], 3)
            if b0["viol_80ms"] is not None: metrics[f"{metric_key_prefix}_{name}_viol80"] = round(b0["viol_80ms"], 4); metrics[f"{metric_key_prefix}_{name}_lat_p50_ms"] = round(b0["lat_p50"] * 1000)
            metrics[f"{metric_key_prefix}_{name}_tick_p99_ms"] = round(b0["tick_ms_p99"], 1); metrics[f"{metric_key_prefix}_{name}_sec"] = b0["sec"]
        if self.args.process_index == 0:
            step = self.state.global_step; self.eval_hist.append(dict(step=step, score=score, **{k: dict(bias0_err=v["bias0"]["err"], best_err=v["best"]["err"], best_bias=v["best_bias"]) for k, v in res.items()}))
            os.makedirs(os.path.join(self.args.output_dir, "eval"), exist_ok=True)
            json.dump(res, open(os.path.join(self.args.output_dir, "eval", f"{metric_key_prefix}-{step}.json"), "w"), indent=1, ensure_ascii=False)
            json.dump(dict(hist=self.eval_hist), open(os.path.join(self.args.output_dir, "eval", "hist.json"), "w"), indent=1, ensure_ascii=False)
        self.log(metrics); self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
