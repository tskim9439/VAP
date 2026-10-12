import time, torch
from comet import load_from_checkpoint
m = load_from_checkpoint("/soundai/users/tskim/VAPKT-data/vapkt-baselines/xcomet/models/Unbabel__XCOMET-XL/checkpoints/model.ckpt", local_files_only=True)
tk = m.encoder.tokenizer
for s in ["Das Wetter ist heute schön.", "새로운 직장이 생겼어요.", "The weather is nice today."]:
    print(tk.tokenize(s), flush=True)
ref_ko = "새로운 직장이 생겼어요."
data = [dict(src="I have a new job.", mt="새로운 직장이 생겼어요.", ref=ref_ko),
        dict(src="I have a new job.", mt="저는 새 직장이 있습니다.", ref=ref_ko),
        dict(src="I have a new job.", mt="나는 오래된 자동차를 팔았다.", ref=ref_ko),
        dict(src="I have a new job.", mt="고양이가 매트 위에 앉았다.", ref=ref_ko),
        dict(src="I have a new job.", mt="새", ref=ref_ko),
        dict(src="I have a new job.", mt="", ref=ref_ko),
        dict(src="Das Wetter ist heute schön.", mt=".", ref="The weather is beautiful today."),
        dict(src="Das Wetter ist heute schön.", mt="The", ref="The weather is beautiful today."),
        dict(src="Neueste Nachrichten anzeigen.", mt="Show latest news.", ref="Show latest news."),
        dict(src="Neueste Nachrichten anzeigen.", mt="Show the oldest news.", ref="Show latest news."),
        dict(src="Show latest news.", mt="Neueste Nachrichten anzeigen.", ref="Neueste Nachrichten anzeigen."),
        dict(src="Show latest news.", mt="Zeige die ältesten Katzen.", ref="Neueste Nachrichten anzeigen.")]
o = m.predict(data, batch_size=16, gpus=1, progress_bar=False)
for d, s in zip(data, o.scores): print(round(s, 4), "|", d["mt"], flush=True)
print("spans", o.metadata.error_spans[3] if hasattr(o, "metadata") else None)
