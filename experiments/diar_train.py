#!/usr/bin/env python
"""E1 화자 분리 가지 학습(2026-10-25, 사용자 "계획대로 진행하는 데 1.7b 모델로 진행해" — wiki output-diarization-integration-20261025):
VapAsr 체크포인트(1.7B)의 인코더를 동결(fp32, 평가 경로와 같다)해 여러 층 출력을 뽑고 vapasr.hf.diar.DiarBranch(슬롯 = 도착 순서 활동)를 학습한다.

  torchrun --nproc_per_node 8 experiments/diar_train.py --init <체크포인트> --nemotron-dir <dir> --out-dir <out> \\
      [--data <phase2>] [--split <phase2>/splits/v4.json] [--corpora aihub71631,…,ami,icsi,notsofar] [--sim-data <dir>,<dir>] [--weights '패턴:가중,…'] \\
      [--window 20,90] [--window-draws 4] [--n-spk 4] [--taps odd] [--d 512 --layers 12 --heads 8 --ffn 2048] [--alpha 0.5] \\
      [--batch 8] [--lr 3e-4] [--warmup 1000] [--max-steps 20000] [--save-every 2000] [--noise-bank <dir> --noise-p 0.4 --rir-p 0.5] [--dry-run] [--smoke N]
데이터: 셋마다 CstDialogueDataset 의 창(길이 U(window), draw window-draws 개, 화자 상한 n-spk 을 넘으면 줄임)을 쓴다. 대화 셋은 split 의 held-out 을 뺀다(v4 는 AMI 공식
  dev·eval·NOTSOFAR dev·eval 도 뺀다). 합성 셋(--sim-data)은 폴더의 *.refined.dialogues.jsonl 전부 — 이름은 '<폴더>/<셋>'. 가중 키는 셋 이름, 폴더 이름(그 폴더 셋 전부),
  이름 패턴(fnmatch) 순으로 찾고 없으면 1(제출 래퍼는 '*' 없이 폴더 이름을 쓴다 — sbatch 의 따옴표 없는 확장에서 glob 이 되지 않게).
  라벨 = slot_activity(창 안 먼저 말한 순서, 80 ms), 마스크 = 패딩·전사 없는 음성 구간(_bad_spans). 증강 = 이득 ±6 dB, 잔향(실제 RIR)·잡음(noise_aug).
산출: out/checkpoint-N/{diar.pt, trainer.pt}, out/final/{diar.pt, diar.json}, out/parity.json, out/DONE. out/PREEMPT 가 생기면 저장하고 끝낸다(sbatch 가 requeue).
저장·재로드 대조(2026-10-25, 기본 — 메모리 규칙 checkpoint-save-load-parity): 학습 끝에 final 을 새 가지로, 인코더는 init 에서 새로 읽어 마지막 배치의 손실을 메모리 모델과
  대조하고(가지·인코더 가중치 차이 0, 손실 차이 ≤ 1e-4 — 어긋나면 DONE 을 쓰지 않는다) out/parity.json 에 남긴다. 끝난 run 은 --check-final N 으로 학습 설정(out/config.json)
  분포의 배치 N 개에서 재로드 모델의 손실·frame_acc 만 잰다(학습 로그 끝 값과 비교).
재개: 마지막 완전한 checkpoint(trainer.pt 있음)에서. 설정 지문(out/config.json)이 다르면 멈춘다.
--aosc(E7, 2026-10-27 — 회의 통째·실시간 화자 유지): 창 대신 긴 클립(--window 예 90,120)을 스트리밍으로 흉내 낸다(vapasr.hf.diar.aosc_clip_loss — Streaming Sortformer AOSC).
  청크(--aosc-chunk)마다 [화자 캐시 · FIFO · 청크] 를 가지에 넣고 청크 몫에 Sort·PIT 손실, 캐시는 정답으로 고르고(같은 점수는 --aosc-noise 로 흩뜨림) 확률 --aosc-perm 으로
  캐시 화자 순서를 섞는다(정답 슬롯도 같이). 클립마다 확률 --aosc-self 로 캐시를 모델 예측으로 고른다(NeMo 의 스트리밍 학습처럼 — 추론과 같은 규칙, 묶음의 정답 화자는
  겹침 최대 짝). 추론 설정(캐시·FIFO·갱신 주기, 추론 청크 6)은 diar.pt 의 aosc 에 남는다 — aosc_stream·회의 통째 평가가 쓴다. 저장·재로드 대조는 정답 캐시로만
  (모델 예측 캐시는 문턱 근처 예측이 조금만 달라도 캐시가 바뀐다).
  --init 은 가지가 붙을 모델(E6 final — stage 3 는 인코더도 학습한다), --init-branch 는 experiments/diar_extract.py 로 꺼낸 E6 헤드.
--shift R(E8, 2026-10-27): 위치 p 의 출력이 프레임 p − R 의 활동을 맞히게 정답·마스크를 R 만큼 민다(vapasr.hf.diar.shift_frames) — 인과 가지가 R 프레임 뒤를 보고 판단한다.
  LLM 은 받아쓰기 지연 δ ≥ R 이면 지연이 늘지 않는다(stage3_train --act-shift 와 같은 규약). diar.pt 의 shift 로 남고 diar_eval 이 맞춰 잰다. --aosc 와는 아직 함께 쓰지 않는다."""
import argparse, fnmatch, glob, json, math, os, random, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNK_S, SR = 0.08, 16000

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, help="인코더를 읽을 VapAsr 체크포인트(토크나이저도)"); ap.add_argument("--nemotron-dir", default=None); ap.add_argument("--out-dir", required=True)
ap.add_argument("--data", default="/soundai/users/tskim/VAPKT-data/data/phase2"); ap.add_argument("--split", default="/soundai/users/tskim/VAPKT-data/data/phase2/splits/v4.json")
ap.add_argument("--corpora", default="aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar")
ap.add_argument("--sim-data", default="", help="합성 대화 폴더(쉼표) — 폴더의 *.refined.dialogues.jsonl 전부"); ap.add_argument("--weights", default="", help="'이름패턴:가중,…'")
ap.add_argument("--window", default="20,90"); ap.add_argument("--window-draws", type=int, default=4); ap.add_argument("--n-spk", type=int, default=4)
ap.add_argument("--taps", default="odd", help="인코더 탭 층('odd'·'all'·'1,3,5') — 마지막 층은 늘 들어간다"); ap.add_argument("--d", type=int, default=512)
ap.add_argument("--layers", type=int, default=12); ap.add_argument("--heads", type=int, default=8); ap.add_argument("--ffn", type=int, default=2048); ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--alpha", type=float, default=0.5, help="손실 = α·도착 순서 BCE + (1 − α)·PIT"); ap.add_argument("--batch", type=int, default=8, help="GPU 당 창 수")
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--max-steps", type=int, default=20000)
ap.add_argument("--weight-decay", type=float, default=0.01); ap.add_argument("--save-every", type=int, default=2000); ap.add_argument("--log-every", type=int, default=20)
ap.add_argument("--noise-bank", default=None); ap.add_argument("--noise-p", type=float, default=0.4); ap.add_argument("--rir-p", type=float, default=0.5); ap.add_argument("--noise-snr", default="5,30")
ap.add_argument("--gain-db", type=float, default=6.0); ap.add_argument("--num-workers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dry-run", action="store_true", help="데이터만 만들고 표본 몇 개를 점검(GPU 없음)"); ap.add_argument("--smoke", type=int, default=0, help="이 step 만 돌고 저장·재로드 대조")
ap.add_argument("--check-final", type=int, default=0, help="학습 없이 끝난 run(out/final)의 저장·재로드 대조만 — out/config.json 설정 분포의 배치 N 개")
ap.add_argument("--lookahead", type=int, default=None, help="인과 가지(2026-10-26, E1c·E6): 프레임 t 의 출력이 t + R 프레임까지만 본다(DiarBranch lookahead, 기본 창 전체)")
ap.add_argument("--init-branch", default=None, help="가지 가중치 시작점(diar.pt — 예 E1 final, 양방향 가중치를 인과 가지에 그대로). 재개 checkpoint 가 있으면 그것이 우선")
ap.add_argument("--aosc", action="store_true", help="E7 AOSC 스트리밍 학습(긴 클립 — --window 90,120 등)"); ap.add_argument("--aosc-cache", type=int, default=188)
ap.add_argument("--aosc-fifo", type=int, default=188); ap.add_argument("--aosc-period", type=int, default=94); ap.add_argument("--aosc-chunk", type=int, default=47, help="학습 한 step 의 새 프레임(추론은 6)")
ap.add_argument("--aosc-perm", type=float, default=0.5); ap.add_argument("--aosc-noise", type=float, default=0.5)
ap.add_argument("--aosc-self", type=float, default=0.5, help="클립마다 이 확률로 캐시를 모델 예측으로(나머지는 정답으로)")
ap.add_argument("--shift", type=int, default=0, help="E8: 위치 p 의 출력 = 프레임 p − R 의 활동(정답을 R 만큼 민다 — R 프레임 뒤를 보고 판단)")
a = ap.parse_args()
assert not a.aosc or a.lookahead is not None, "--aosc 는 인과 가지(--lookahead R, 예 0)와 함께 — 캐시 뒤 청크가 앞만 본다"
assert a.shift >= 0 and not (a.shift and a.aosc), "--shift R(≥ 0)은 아직 --aosc 와 함께 쓰지 않는다"
assert not a.shift or a.lookahead is not None, "--shift 는 인과 가지(--lookahead, 예 0)와 함께 — 앞을 이미 보는 가지에 미룰 까닭이 없다"


def log(*x):
    if int(os.environ.get("RANK", "0")) == 0: print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)


# ── 데이터
class Source:
    def __init__(self, name, path, mono, tok, exclude):
        from vapasr.data.cst_dialogue import CstDialogueDataset
        lo, hi = (float(x) for x in a.window.split(","))
        self.name = name
        self.ds = CstDialogueDataset([path], tok, delays=(4,), mono_cache_dir=mono, seed=a.seed, window_s=(lo, hi), window_draws=a.window_draws, max_speakers=a.n_spk,
                                     shrink_speakers=True, exclude=exclude)
        self.hours = sum(L for _, _, L in self.ds.items) / 3600 / max(1, a.window_draws)


def build_sources(tok):
    split = json.load(open(a.split)) if a.split else {"corpora": {}}; out = []
    for c in [x for x in a.corpora.split(",") if x]:
        held = set(split["corpora"].get(c, {}).get("heldout", []))
        out.append(Source(c, os.path.join(a.data, f"{c}.refined.dialogues.jsonl"), os.path.join(a.data, "mono"), tok, held))
    for d in [x for x in a.sim_data.split(",") if x]:
        for p in sorted(glob.glob(os.path.join(d, "*.refined.dialogues.jsonl"))):
            out.append(Source(f"{os.path.basename(d.rstrip('/'))}/{os.path.basename(p).split('.')[0]}", p, os.path.join(d, "mono"), tok, None))
    W = dict((k, float(v)) for k, v in (x.rsplit(":", 1) for x in a.weights.split(",") if x))
    wts = [W.get(s.name, W.get(s.name.split("/", 1)[0], next((v for k, v in W.items() if fnmatch.fnmatch(s.name, k)), 1.0))) for s in out]   # 이름 > 폴더 이름 > 패턴
    return out, wts


def make_sample(src, i, rng, aug):
    from vapasr.data.cst_dialogue import slot_activity, window_speakers
    ds = src.ds; cid, t0, L = ds.items[i]; d = ds.dlgs[cid]; K = int(round(L / CHUNK_S))
    wav = np.asarray(ds.mono(cid)[int(round(t0 * SR)): int(round((t0 + L) * SR))], dtype=np.float32)
    if aug is not None: wav = aug(wav, rng)
    if a.gain_db > 0: wav = wav * float(10 ** (rng.uniform(-a.gain_db, a.gain_db) / 20))
    sp = window_speakers(ds.eps[cid], t0, L)[: a.n_spk]
    act = np.asarray(slot_activity(ds.eps[cid], sp, t0, L, K, n_slots=a.n_spk), dtype=np.float32).reshape(K, a.n_spk)
    m = np.ones(K, dtype=bool)
    for s, e in ds._bad_spans(d):
        x0, x1 = s - t0, e - t0
        if x1 > 0 and x0 < L: m[max(0, int(x0 / CHUNK_S)): min(K, int(math.ceil(x1 / CHUNK_S)))] = False
    return dict(wav=np.clip(wav, -1, 1), K=K, act=act, mask=m, src=src.name, n_spk=len(sp))


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, sources, weights, aug, tag):
        self.sources, self.weights, self.aug, self.tag = sources, weights, aug, tag

    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rng = random.Random(f"{self.tag}:{wi.id if wi else 0}")
        while True:
            s = rng.choices(self.sources, weights=self.weights)[0]
            yield make_sample(s, rng.randrange(len(s.ds.items)), rng, self.aug)


def collate(b):
    B, n, Km = len(b), max(len(x["wav"]) for x in b), max(x["K"] for x in b)
    wav = torch.zeros(B, n); act = torch.zeros(B, Km, a.n_spk); mask = torch.zeros(B, Km, dtype=torch.bool)
    for j, x in enumerate(b):
        wav[j, : len(x["wav"])] = torch.from_numpy(x["wav"]); act[j, : x["K"]] = torch.from_numpy(x["act"]); mask[j, : x["K"]] = torch.from_numpy(x["mask"])
    K = torch.tensor([x["K"] for x in b]); wl = torch.tensor([len(x["wav"]) for x in b])
    return dict(wav=wav, wav_len=wl, K=K, act=act, mask=mask, pad=torch.arange(Km)[None] >= K[:, None], src=[x["src"] for x in b], n_spk=[x["n_spk"] for x in b])


def fingerprint():
    keys = ("init", "data", "split", "corpora", "sim_data", "weights", "window", "window_draws", "n_spk", "taps", "d", "layers", "heads", "ffn", "dropout", "alpha", "batch",
            "lr", "warmup", "max_steps", "weight_decay", "noise_bank", "noise_p", "rir_p", "noise_snr", "gain_db", "seed")
    return {**{k: getattr(a, k) for k in keys}, **{k: getattr(a, k) for k in ("lookahead", "init_branch") if getattr(a, k) is not None},   # 켰을 때만(E1 지문 그대로)
            **({"shift": a.shift} if a.shift else {}),
            **({k: getattr(a, k) for k in ("aosc", "aosc_cache", "aosc_fifo", "aosc_period", "aosc_chunk", "aosc_perm", "aosc_noise", "aosc_self")} if a.aosc else {})}


def aosc_cfg():
    from vapasr.hf.diar import AoscConfig
    return AoscConfig(cache=a.aosc_cache, fifo=a.aosc_fifo, period=a.aosc_period, chunk=6)


def aosc_kw(b, dev, tag, self_p=None):
    """forward(aosc=…) 인자 — 표본마다 고정 씨앗(tag·표본 번호)이라 재로드 대조가 같은 스트리밍을 다시 돈다."""
    act, mask, pad, K = b["act"].to(dev), b["mask"].to(dev), b["pad"].to(dev), b["K"]
    return dict(act=act, mask=mask & ~pad, K=K, cfg=aosc_cfg(), seeds=[f"{tag}:{j}" for j in range(len(K))], alpha=a.alpha, perm_p=a.aosc_perm, chunk=a.aosc_chunk,
                noise=a.aosc_noise, self_p=a.aosc_self if self_p is None else self_p)


def reload_check(final_dir, batches, model=None, enc=None):
    """final/diar.pt 를 새 DiarBranch 로, 인코더는 init 에서 새로 읽어 batches 의 손실·frame_acc 를 학습과 같은 정밀도(인코더 fp32, 가지 bf16 autocast)로 잰다.
    model·enc(학습 끝 메모리 모델)가 있으면 가중치 차이와 손실을 대조해 어긋나면 멈추고, 없으면(--check-final) 값만 남긴다. out/parity.json 에 쓴다."""
    from vapasr.hf.diar import DiarBranch, load_encoder, sort_pit_loss
    dev = torch.device("cuda"); x = torch.load(os.path.join(final_dir, "diar.pt"), map_location=dev); taps = tuple(x["taps"])
    m2 = DiarBranch(**x["cfg"]).to(dev).eval(); m2.load_state_dict(x["state"]); enc2, _ = load_encoder(x["init"], a.nemotron_dir, "cuda")

    def score(m, e):
        loss_sum, hit, tot = 0.0, 0.0, 0.0
        with torch.no_grad():
            for b in batches:
                wav, wl, K, act, mask, pad = (b[k].to(dev) for k in ("wav", "wav_len", "K", "act", "mask", "pad"))
                feats = e(wav, wl, K, taps=taps)
                if a.aosc:                                                                         # E7: 같은 씨앗의 스트리밍 흉내 손실(정답 캐시 — 이산 선택이 예측에 매이지 않게)
                    with torch.autocast("cuda", dtype=torch.bfloat16): loss, st = m(feats, pad, aosc=aosc_kw(b, dev, f"parity:{len(K)}", self_p=0.0))
                    loss_sum += float(loss); hit += st["hit"]; tot += st["frames"]; continue
                with torch.autocast("cuda", dtype=torch.bfloat16): lg = m(feats, pad)
                tg, mk = shifted(act, mask, pad); loss, _, _ = sort_pit_loss(lg.float(), tg, mk, a.alpha); mm = mk.float()[..., None]
                loss_sum += loss.item(); hit += float((((lg.float() > 0).float() == tg).float() * mm).sum()); tot += float(mm.sum()) * act.shape[-1]
        return dict(loss=round(loss_sum / len(batches), 6), frame_acc=round(hit / max(1.0, tot), 4))

    def max_diff(s1, s2):
        assert s1.keys() == s2.keys(), "state_dict 키가 다르다"
        return max((float((s1[k].float() - s2[k].float()).abs().max()) for k in s1 if s1[k].numel()), default=0.0)

    out = dict(mode="offline" if model is None else "train-end", final=final_dir, step=x["step"], batches=len(batches), windows=sum(len(b["K"]) for b in batches),
               reload=score(m2, enc2))
    if model is not None:
        model.eval(); out.update(memory=score(model, enc), branch_max_diff=max_diff(model.state_dict(), m2.state_dict()),
                                 encoder_max_diff=max_diff(enc.state_dict(), enc2.state_dict()))
    json.dump(out, open(os.path.join(a.out_dir, "parity.json"), "w"), indent=1); log(f"저장·재로드 대조: {out}")
    if model is not None:
        assert out["branch_max_diff"] == 0.0 and out["encoder_max_diff"] == 0.0 and abs(out["memory"]["loss"] - out["reload"]["loss"]) <= 1e-4, f"재로드 불일치: {out}"
    return out


def shifted(act, mask, pad):
    """→ (정답, 손실 마스크) — --shift R 이면 둘 다 R 만큼 민다(끝 패딩 위치는 그대로 뺀다). E8."""
    from vapasr.hf.diar import shift_frames
    m = mask & ~pad
    return (shift_frames(act, a.shift), shift_frames(m, a.shift, False) & ~pad) if a.shift else (act, m)


def last_checkpoint(out):
    cks = sorted((int(p.rsplit("-", 1)[1]), p) for p in glob.glob(os.path.join(out, "checkpoint-*")) if p.rsplit("-", 1)[1].isdigit() and os.path.exists(os.path.join(p, "trainer.pt")))
    return cks[-1] if cks else (0, None)


def main():
    from vapasr.data.noise_aug import from_args as noise_from_args
    from vapasr.hf import load_tokenizer
    if a.check_final:                                                                              # 끝난 run: 학습 설정으로 데이터 인자를 맞춘다
        for k, v in json.load(open(os.path.join(a.out_dir, "config.json"))).items(): setattr(a, k, v)
    tok = load_tokenizer(a.init)
    sources, wts = build_sources(tok)
    for s, w in zip(sources, wts): log(f"  셋 {s.name}: 창 {len(s.ds.items)}(draw {a.window_draws}) · 약 {s.hours:.0f} h · 가중 {w:g}")
    assert sources and all(len(s.ds.items) for s in sources), "빈 셋이 있다"
    aug = noise_from_args(a.noise_bank, a.noise_p, a.noise_snr, a.rir_p) if a.noise_bank else None
    if a.dry_run:
        rng = random.Random(a.seed)
        for s in sources:
            x = make_sample(s, rng.randrange(len(s.ds.items)), rng, aug)
            log(f"  표본 {s.name}: K {x['K']} · 화자 {x['n_spk']} · 활동 몫 {x['act'].mean(0).round(3).tolist()} · 마스크 몫 {1 - x['mask'].mean():.3f} · 음량 {float(np.abs(x['wav']).max()):.3f}")
        b = collate([make_sample(sources[0], i, rng, aug) for i in range(3)]); log(f"  배치 wav {tuple(b['wav'].shape)} act {tuple(b['act'].shape)} pad {int(b['pad'].sum())}")
        print("DRY OK", flush=True); return
    if a.check_final:
        it = iter(Stream(sources, wts, aug, f"{a.seed}:check")); batches = [collate([next(it) for _ in range(a.batch)]) for _ in range(a.check_final)]
        reload_check(os.path.join(a.out_dir, "final"), batches); return
    import torch.distributed as dist
    from vapasr.hf.diar import DiarBranch, load_encoder, sort_pit_loss, tap_layers
    ddp = "RANK" in os.environ
    if ddp: dist.init_process_group("nccl"); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1); dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    os.makedirs(a.out_dir, exist_ok=True); fp = os.path.join(a.out_dir, "config.json")
    if rank == 0:
        if os.path.exists(fp): assert json.load(open(fp)) == fingerprint(), f"재개 설정 불일치 — 새 --out-dir 을 쓴다 ({fp})"
        else: json.dump(fingerprint(), open(fp, "w"), indent=1)
    enc, n_keys = load_encoder(a.init, a.nemotron_dir, "cuda")
    taps = tap_layers(a.taps, len(enc.enc.layers)); log(f"인코더 ← {a.init}(키 {n_keys}, 동결 fp32) · 탭 {list(taps)} · world {world}")
    torch.manual_seed(a.seed)
    model = DiarBranch(1 + len(taps), 1024, a.d, a.layers, a.heads, a.ffn, a.n_spk, a.dropout, lookahead=a.lookahead).to(dev)
    if a.init_branch and not last_checkpoint(a.out_dir)[1] and not a.check_final:                 # 새로 시작할 때만(재개는 checkpoint)
        xb = torch.load(a.init_branch, map_location=dev); model.load_state_dict(xb["state"]); log(f"가지 ← {a.init_branch}(step {xb.get('step')})")
    log(f"가지 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M · {model.cfg}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.98))
    total = a.smoke or a.max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))))
    step, ck = last_checkpoint(a.out_dir) if not a.smoke else (0, None)
    if ck:
        model.load_state_dict(torch.load(os.path.join(ck, "diar.pt"), map_location=dev)["state"])
        st = torch.load(os.path.join(ck, "trainer.pt"), map_location=dev, weights_only=False); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        log(f"재개 ← {ck}(step {step})")
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index]) if ddp else model
    dl = torch.utils.data.DataLoader(Stream(sources, wts, aug, f"{a.seed}:{rank}:{step}"), batch_size=a.batch, num_workers=a.num_workers, collate_fn=collate,
                                     pin_memory=True, persistent_workers=a.num_workers > 0, prefetch_factor=4 if a.num_workers else None)

    def save(tag):
        if rank != 0: return None
        d = os.path.join(a.out_dir, tag); tmp = d + f".tmp{os.getpid()}"; os.makedirs(tmp, exist_ok=True)
        torch.save(dict(state=model.state_dict(), cfg=model.cfg, taps=list(taps), init=os.path.abspath(a.init), step=step, **({"aosc": aosc_cfg().to_dict()} if a.aosc else {}),
                        **({"shift": a.shift} if a.shift else {})),
                   os.path.join(tmp, "diar.pt"))
        if tag != "final": torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), step=step), os.path.join(tmp, "trainer.pt"))
        else: json.dump(dict(cfg=model.cfg, taps=list(taps), init=os.path.abspath(a.init), step=step, args=fingerprint()), open(os.path.join(tmp, "diar.json"), "w"), indent=1)
        if os.path.exists(d): d = f"{d}-dup{step}-{os.getpid()}"; log(f"!! {tag} 가 이미 있다 — 덮어쓰지 않고 {d} 로")
        os.replace(tmp, d); log(f"저장 → {d}"); return d

    preempt = os.path.join(a.out_dir, "PREEMPT"); t0 = time.time(); acc = {}
    net.train()
    for b in dl:
        if step >= total: break
        flag = torch.tensor([1.0 if rank == 0 and os.path.exists(preempt) else 0.0], device=dev)
        if ddp: dist.all_reduce(flag)
        if flag.item() > 0: save(f"checkpoint-{step}"); log("PREEMPT → 저장하고 끝낸다"); break
        wav, wl, K = b["wav"].to(dev, non_blocking=True), b["wav_len"].to(dev), b["K"].to(dev)
        act, mask, pad = b["act"].to(dev), b["mask"].to(dev), b["pad"].to(dev)
        with torch.no_grad(): feats = enc(wav, wl, K, taps=taps)                                      # (B, 1 + 탭, K, D) fp32
        if a.aosc:                                                                                     # E7: 표본(긴 클립)마다 스트리밍 흉내
            with torch.autocast("cuda", dtype=torch.bfloat16): loss, st = net(feats, pad, aosc=aosc_kw(b, dev, f"{a.seed}:{rank}:{step}"))
            fr = max(1.0, st["frames"]); stats = dict(sort=st["sort"] / fr, pit=st["pit"] / fr, frame_acc=st["hit"] / fr, steps=float(st["steps"]), self=st["self"] / len(K))
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16): logits = net(feats, pad)
            tg, mk = shifted(act, mask, pad); loss, srt, pit = sort_pit_loss(logits.float(), tg, mk, a.alpha)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        with torch.no_grad():
            if not a.aosc:
                m = mk.float()[..., None]; pr = (logits.float() > 0).float()
                stats = dict(sort=srt.item(), pit=pit.item(), frame_acc=float(((pr == tg).float() * m).sum() / (m.sum() * a.n_spk).clamp(min=1)))
            for k, v in dict(loss=loss.item(), gn=float(gn), **stats).items():
                acc[k] = acc.get(k, 0.0) + v
        if step % a.log_every == 0 or step == total:
            n = a.log_every if step % a.log_every == 0 else (step % a.log_every or a.log_every)
            log(f"step {step}/{total} · " + " · ".join(f"{k} {v / n:.4f}" for k, v in acc.items()) + f" · lr {sched.get_last_lr()[0]:.2e} · {(time.time() - t0) / n:.2f} s/step"); acc = {}; t0 = time.time()
        if step % a.save_every == 0 and step < total: save(f"checkpoint-{step}")
    d = save("final") if step >= total else None
    if ddp: dist.barrier(); dist.destroy_process_group()                                           # 대조(인코더 다시 읽기)는 그룹을 닫은 뒤 rank 0 만 — NCCL 대기 없이
    if d:
        reload_check(d, [b], model, enc)                                                             # 어긋나면 멈추고 DONE 을 쓰지 않는다
        open(os.path.join(a.out_dir, "DONE"), "w").write(f"step {step}\n")


if __name__ == "__main__":
    main()
