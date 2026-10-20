# V2 TAXI 하락 원인 가르기(mxc 호스트 GPU): cst_stage4_taxi.run 을 그대로 돌리되 두 가지를 더한다. 산출 폴더 말고는 쓰지 않는다.
#   (1) 슬롯별 전사·번역 토큰 수 — TAXI 채점은 A·B 언어 슬롯만 본다(C·D 로 낸 말은 삭제가 된다).
#   (2) 청크 임베딩 두 항의 크기(평균 L2): adapter(마지막 층) 대 adapter_tap(L00). --no-tap 이면 adapter_tap 마지막 Linear 를 0 으로(붙인 직후와 같은 V1 경로).
#   python taxi_v2_probe.py --checkpoint <ck> --encoder <nemotron> --sessions <taxi-natural/sessions.jsonl> --out <dir> [--no-tap] [--deltas 10] [--max-window 40]
#   → <out>/hyp·asr-d*.jsonl(TAXI 전사 — mxc 에만), <out>/probe.json(수만). 채점은 cst_stage4_taxi.py score --out <dir>.
import argparse, collections, json, os, sys
CODE = os.environ.get("CODE", "/soundai/users/tskim/VAPKT"); sys.path.insert(0, os.path.join(CODE, "experiments")); sys.path.insert(0, CODE)
import torch
import cst_stage4_taxi as T
import vapasr.data.cst_eval as E
import vapasr.hf.infer as I
from vapasr.hf.modeling_vapasr import VapAsrForStreamingASR


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--encoder", default=None); ap.add_argument("--sessions", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[10])
    ap.add_argument("--max-window", type=float, default=40.0); ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--no-tap", action="store_true"); a = ap.parse_args()
    tokens, wins, norm = collections.Counter(), collections.Counter(), collections.Counter()
    parse0, load0, embed0 = E.parse_emitted4, I.load_model, VapAsrForStreamingASR.chunk_embed

    def parse(emitted, spk_ids, trans_id, eot_id, **kw):
        src, tr, eots = parse0(emitted, spk_ids, trans_id, eot_id, **kw)
        for k in range(len(spk_ids)): tokens[f"src_{'ABCD'[k]}"] += len(src[k]); tokens[f"tr_{'ABCD'[k]}"] += len(tr[k])
        wins["windows"] += 1; wins["cd_any"] += int(any(src[k] or tr[k] for k in range(2, len(spk_ids))))
        return src, tr, eots

    def load(*x, **kw):
        model, tok = load0(*x, **kw)
        if a.no_tap:
            assert model.adapter_tap is not None, "탭 없는 모델"
            with torch.no_grad(): model.adapter_tap.net[-1].weight.zero_(); model.adapter_tap.net[-1].bias.zero_()
        return model, tok

    def embed(self, feats):
        if self.adapter_tap is not None:
            with torch.no_grad():
                m, t = self.adapter(feats[:, 0].float()).norm(dim=-1), self.adapter_tap(feats[:, 1].float()).norm(dim=-1)
                norm["n"] += m.numel(); norm["main"] += float(m.sum()); norm["tap"] += float(t.sum())
        return embed0(self, feats)

    E.parse_emitted4, I.load_model, VapAsrForStreamingASR.chunk_embed = parse, load, embed
    T.run(argparse.Namespace(checkpoint=a.checkpoint, encoder=a.encoder, sessions=a.sessions, out=a.out, deltas=a.deltas, max_window=a.max_window,
                             batch_size=a.batch_size, max_flush=None, dtype="float32", asr_only=False))
    n = max(1, norm["n"])
    json.dump(dict(checkpoint=a.checkpoint, no_tap=a.no_tap, deltas=a.deltas, max_window=a.max_window, tokens=dict(tokens), windows=dict(wins),
                   embed_norm=dict(chunks=norm["n"], main=round(norm["main"] / n, 4), tap=round(norm["tap"] / n, 4))), open(os.path.join(a.out, "probe.json"), "w"), indent=1)
    print("PROBE", json.dumps(dict(tokens=dict(tokens), windows=dict(wins), embed_norm=(round(norm["main"] / n, 4), round(norm["tap"] / n, 4)))), flush=True)


if __name__ == "__main__":
    main()
