# Stage 3 GPU 스모크: s4 체크포인트 + <EOT> 켜기(재초기화) + CstDialogueDataset 배치(패딩·packing) forward/backward. 쓰기 없음(모델 저장 안 함).
import sys, json, time, torch
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.hf.infer import load_model
from vapasr.data.cst_dialogue import CstDialogueDataset, collate_cst
from vapasr.hf.packing import pack_batch
D = "/soundai/users/tskim/VAPKT-data/data/phase2"; S4 = "/soundai/Model/VAPASR/semcommit-q06-s4-ast/final"
m, tok = load_model(S4, device="cuda"); EOT = m.config.sp_ids["<EOT>"]; SEM = m.config.sp_ids["<SEM_END>"]
W = m.get_input_embeddings().weight
lg0 = float(W[EOT].float().norm()); base_mu = W[:151705].float().mean(0)
print("before: EOT blocked", EOT in m.config.blocked_ids, "sem_registry", m.config.sem_registry, "|W[EOT]-mu|", round(float((W[EOT].float() - base_mu).norm()), 3))
m.add_semantic_tokens(tok, turn=True, reinit_turn=True); m.config.turn_weight = 2.0
print("after : EOT blocked", EOT in m.config.blocked_ids, "sem_registry", m.config.sem_registry, "sem_reserved", m.config.sem_reserved, "|W[EOT]-mu|", round(float((W[EOT].float() - base_mu).norm()), 3))
if m.encoder is not None: m.encoder.set_trainable(True)
m.train(); res = {}
for corpus in ("kec", "aihub134-2"):
    ds = CstDialogueDataset([f"{D}/{corpus}.refined.dialogues.jsonl"], tok, delays=(4, 6, 8, 10), mono_cache_dir=f"{D}/mono", decision_weight=2.0, max_items=64, seed=0)
    b = collate_cst([ds[i] for i in range(6)])
    for mode in ("padded", "packed"):
        x = pack_batch(b) if mode == "packed" else b
        keys = ("wav", "wav_len", "K", "ids", "is_audio", "chunk_of", "labels", "mask", "pos_weight", "position_ids", "labels_alt", "sem_only_mask")
        inp = {k: (x[k].cuda() if torch.is_tensor(x[k]) else x[k]) for k in keys if k in x}; inp["soft_w"] = x["soft_w_full"].cuda()
        if mode == "packed": inp.pop("mask", None)
        m.zero_grad(set_to_none=True); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16): out = m(**inp, next_weight=0.3)
        out.loss.backward(); g = sum(float(p.grad.float().norm() ** 2) for p in m.parameters() if p.grad is not None) ** 0.5
        gE = float(W.grad[EOT].float().norm()) if W.grad is not None else None; gS = float(W.grad[SEM].float().norm()) if W.grad is not None else None
        r = dict(loss=round(float(out.loss), 4), loss_text=round(out.loss_text, 4), loss_next=round(out.loss_next, 4), loss_turn=out.loss_turn, n_turn=out.n_turn,
                 n_soft=out.n_soft, n_soft_batch=b["n_soft"], n_labels=out.n_labels, grad_norm=round(g, 3), grad_EOT_row=gE, grad_SEM_row=gS, sec=round(time.time() - t0, 2), L=int(inp["ids"].shape[1]))
        res[f"{corpus}/{mode}"] = r; print(corpus, mode, json.dumps(r))
pp = [res[f"{c}/padded"]["loss"] - res[f"{c}/packed"]["loss"] for c in ("kec", "aihub134-2")]
print("padded-vs-packed loss diff", [round(x, 4) for x in pp])
