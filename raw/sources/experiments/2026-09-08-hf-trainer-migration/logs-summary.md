### parity+bench (bg-hfparity2-20260908-120059.log)
legacy 로드 191s
hf 변환 219s · thinker params 596.0M · adapter 4.20M
1) 파라미터 max|Δ| thinker 0.00e+00 adapter 0.00e+00
2) loss legacy 0.523754 hf 0.523754 | next 0.1427/0.1427 text 1.0214/1.0214 top1 0.6948/0.6948 (B=12, L=1474)
3) save_pretrained 34s → /soundai/users/tskim/VAPKT-data/ckpt/hf-parity {"model.safetensors": 2.4} GB · files ['added_tokens.json', 'config.json', 'merges.txt', 'model.safetensors', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json']
   from_pretrained 10s loss 0.523754 (Δ 0.00e+00) · tied: True
4) stream_decode 동일 True · tokens 63/63 forced 0/0 rounds 1/1 tick p50 62.8/62.3 ms
   hyp: no very natural one i wish and the glance was she cast him while not meeting his eye showed that she understood the impo
   bench[no-liger] 969 ms/step · peak 21.3 GB (B=12, L=1474)
5) liger 적용 {'rms_norm': 113, 'swiglu': 28, 'rope': 1} · loss 0.523070 (Δ 6.84e-04)
   liger stream_decode 동일 False (다른 토큰 2/63)
   bench[liger] 734 ms/step · peak 21.2 GB (B=12, L=1474)
PARITY OK
[bg:hfparity2] EXIT=0 Tue Sep  8 11:09:30 AM CST 2026

### smoke 60 step (bg-hfsmoke-20260908-120125.log)
liger: {'rms_norm': 113, 'swiglu': 28, 'rope': 1}
model ← qwen /soundai/Model/Qwen3-ASR-0.6B + adapter /soundai/Model/VAPASR/uslm/s1-adapter-distill/adapter.pt (139s) · trainable 600.2M · encoder 동결 · grad ckpt True
train librispeech-dev:1470 (drop 0, no-align 0), kspon-full:619203 (drop 729, no-align 0) | dev dev-clean:4, dev-other:4, kspon-dev:8 | world 2
rank 당 배치/epoch: librispeech-dev:183, kspon-full:38700 → steps/epoch 38883 · 유효 배치 EN 8 / KO 16
{'loss': 11.9089, 'grad_norm': 600.5091552734375, 'learning_rate': 0.0, 'loss_next': 14.6558, 'loss_text': 8.3323, 'top1_text': 0.0584, 'labels_per_step': 1820, 'epoch': 2.571818018157035e-05}
{'loss': 8.272, 'grad_norm': 224.64166259765625, 'learning_rate': 0.0004, 'loss_next': 9.1374, 'loss_text': 8.8599, 'top1_text': 0.066, 'labels_per_step': 1365, 'epoch': 0.00012859090090785175}
{'loss': 5.91, 'grad_norm': 120.42637634277344, 'learning_rate': 0.0009000000000000001, 'loss_next': 4.1624, 'loss_text': 7.6179, 'top1_text': 0.0826, 'labels_per_step': 1050, 'epoch': 0.0002571818018157035}
{'loss': 4.6544, 'grad_norm': 65.533935546875, 'learning_rate': 0.0009842915805643156, 'loss_next': 1.2992, 'loss_text': 8.4356, 'top1_text': 0.0939, 'labels_per_step': 1755, 'epoch': 0.0003857727027235553}
{'loss': 3.5976, 'grad_norm': 38.418975830078125, 'learning_rate': 0.0009221639627510075, 'loss_next': 1.1246, 'loss_text': 5.5282, 'top1_text': 0.1534, 'labels_per_step': 1222, 'epoch': 0.000514363603631407}
{'loss': 3.0442, 'grad_norm': 53.16363525390625, 'learning_rate': 0.0008187119948743449, 'loss_next': 0.4367, 'loss_text': 6.312, 'top1_text': 0.076, 'labels_per_step': 959, 'epoch': 0.0006429545045392588}
{'loss': 3.0968, 'grad_norm': 27.1640682220459, 'learning_rate': 0.0006840622763423391, 'loss_next': 0.6948, 'loss_text': 5.3024, 'top1_text': 0.1127, 'labels_per_step': 1236, 'epoch': 0.0007715454054471106}
{'eval_score': 1.0, 'eval_sec': 130.4, 'eval_dev-clean_err': 1.0, 'eval_dev-clean_tok_per_chunk': 0.0, 'eval_dev-clean_tick_p99_ms': 63.9, 'eval_dev-other_err': 1.0, 'eval_dev-other_tok_per_chunk': 0.0, 'eval_dev-other_tick_p99_ms': 59.8, 'eval_kspon-dev_err': 1.0, 'eval_kspon-dev_tok_per_chunk': 0.0, 'eval_kspon-dev_tick_p99_ms
{'loss': 2.9066, 'grad_norm': 44.84323501586914, 'learning_rate': 0.0005313952597646568, 'loss_next': 0.6891, 'loss_text': 5.1577, 'top1_text': 0.1132, 'labels_per_step': 1219, 'epoch': 0.0009001363063549623}
{'loss': 2.9038, 'grad_norm': 29.316633224487305, 'learning_rate': 0.0003756550564175727, 'loss_next': 0.5729, 'loss_text': 5.1676, 'top1_text': 0.1106, 'labels_per_step': 1203, 'epoch': 0.001028727207262814}
{'loss': 2.6563, 'grad_norm': 27.935853958129883, 'learning_rate': 0.00023208660251050156, 'loss_next': 0.5797, 'loss_text': 4.8932, 'top1_text': 0.1276, 'labels_per_step': 1373, 'epoch': 0.0011573181081706658}
{'loss': 2.6287, 'grad_norm': 11.41201400756836, 'learning_rate': 0.00011474337861210544, 'loss_next': 0.487, 'loss_text': 4.7041, 'top1_text': 0.14, 'labels_per_step': 980, 'epoch': 0.0012859090090785177}
{'loss': 2.6655, 'grad_norm': 7.7066779136657715, 'learning_rate': 3.5111757055874326e-05, 'loss_next': 0.6098, 'loss_text': 4.7946, 'top1_text': 0.147, 'labels_per_step': 1323, 'epoch': 0.0014144999099863693}
{'loss': 2.611, 'grad_norm': 18.208539962768555, 'learning_rate': 9.866357858642206e-07, 'loss_next': 0.584, 'loss_text': 4.4096, 'top1_text': 0.1594, 'labels_per_step': 1129, 'epoch': 0.0015430908108942211}
{'eval_score': 1.0, 'eval_sec': 130.0, 'eval_dev-clean_err': 1.0, 'eval_dev-clean_tok_per_chunk': 0.0, 'eval_dev-clean_tick_p99_ms': 62.6, 'eval_dev-other_err': 1.0, 'eval_dev-other_tok_per_chunk': 0.0, 'eval_dev-other_tick_p99_ms': 63.6, 'eval_kspon-dev_err': 1.0, 'eval_kspon-dev_tok_per_chunk': 0.0, 'eval_kspon-dev_tick_p99_ms
{'train_runtime': 501.4967, 'train_samples_per_second': 0.239, 'train_steps_per_second': 0.12, 'train_loss': 3.806167952219645, 'epoch': 0.0015430908108942211}
[rank0]: AttributeError: 'PrinterCallback' object has no attribute 'fired'
[rank1]: AttributeError: 'PrinterCallback' object has no attribute 'fired'
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 

### resume 60→90 (bg-hfresume-20260908-122104.log)
model ← qwen /soundai/Model/Qwen3-ASR-0.6B + adapter /soundai/Model/VAPASR/uslm/s1-adapter-distill/adapter.pt (135s) · trainable 600.2M · encoder 동결 · grad ckpt True
rank 당 배치/epoch: librispeech-dev:183, kspon-full:38700 → steps/epoch 38883 · 유효 배치 EN 8 / KO 16 · 재개 ← /soundai/users/tskim/VAPKT-data/ckpt/hf-smoke/checkpoint-60
{'loss': 2.4353, 'grad_norm': 9.302233695983887, 'learning_rate': 0.00023875071764202561, 'loss_next': 0.4317, 'loss_text': 4.5822, 'top1_text': 0.1494, 'labels_per_step': 1456, 'epoch': 0.00012859090090785175}
{'loss': 2.4474, 'grad_norm': 12.282669067382812, 'learning_rate': 0.0001605996272335291, 'loss_next': 0.5602, 'loss_text': 3.9366, 'top1_text': 0.205, 'labels_per_step': 1050, 'epoch': 0.0002571818018157035}
{'loss': 2.3347, 'grad_norm': 14.396042823791504, 'learning_rate': 9.549150281252633e-05, 'loss_next': 0.466, 'loss_text': 4.296, 'top1_text': 0.1761, 'labels_per_step': 1755, 'epoch': 0.0003857727027235553}
{'loss': 2.1543, 'grad_norm': 14.537866592407227, 'learning_rate': 4.592841308745932e-05, 'loss_next': 0.5985, 'loss_text': 3.755, 'top1_text': 0.2395, 'labels_per_step': 1222, 'epoch': 0.000514363603631407}
{'loss': 1.8783, 'grad_norm': 12.886910438537598, 'learning_rate': 1.3815039801161721e-05, 'loss_next': 0.4571, 'loss_text': 3.5379, 'top1_text': 0.2886, 'labels_per_step': 959, 'epoch': 0.0006429545045392588}
{'loss': 2.0547, 'grad_norm': 13.536943435668945, 'learning_rate': 3.854818796385495e-07, 'loss_next': 0.51, 'loss_text': 3.6041, 'top1_text': 0.2373, 'labels_per_step': 1236, 'epoch': 0.0007715454054471106}
{'eval_score': 0.9961313540260909, 'eval_sec': 122.6, 'eval_dev-clean_err': 1.0, 'eval_dev-clean_tok_per_chunk': 0.001, 'eval_dev-clean_tick_p99_ms': 62.7, 'eval_dev-other_err': 0.9965, 'eval_dev-other_tok_per_chunk': 0.002, 'eval_dev-other_tick_p99_ms': 58.6, 'eval_kspon-dev_err': 0.9919, 'eval_kspon-dev_tok_per_chunk': 0.003, 
{'train_runtime': 263.7351, 'train_samples_per_second': 0.683, 'train_steps_per_second': 0.341, 'train_loss': 0.7391496976216634, 'epoch': 0.0007715454054471106}
train 종료: step 90/90 완료 · {'train_runtime': 263.7351, 'train_samples_per_second': 0.683, 'train_steps_per_second': 0.341, 'train_loss': 0.7391496976216634, 'epoch': 0.0007715454054471106}
final → /soundai/users/tskim/VAPKT-data/ckpt/hf-smoke/final
[bg:hfresume] EXIT=0 Tue Sep  8 11:34:49 AM CST 2026
[11:45:03] 선점/종료 신호 → step 107 저장 후 종료 (재시작하면 이어서 학습)
train 종료: step 107/130 (선점/중단 → 재시작 시 이어서) · {'train_runtime': 135.9624, 'train_samples_per_second': 1.912, 'train_steps_per_second': 0.956, 'train_loss': 0.32088383113112
[bg:hfpreempt] EXIT=0 Tue Sep  8 11:46:18 AM CST 2026
