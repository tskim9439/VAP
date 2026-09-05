"""Stage 1 §10 — 토크나이저 일치 테스트: transformers 4.57 의 'incorrect regex pattern' 경고가 실제 토큰화를 바꾸는지.
비교 대상: (a) AutoTokenizer(local dir) 기본  (b) fix_mistral_regex=True  (c) tokenizers 라이브러리로 tokenizer.json 직접  (d) qwen_asr processor.tokenizer(학습·정렬이 쓰는 것)"""
import os, sys, json, random, warnings; warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from tokenizers import Tokenizer
Q = os.environ["MXC_QWEN_ASR_DIR"]; K = os.environ["MXC_KSPONSPEECH_DIR"]; L = os.environ["MXC_LIBRISPEECH_DIR"]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from vapasr.data.kspon import read_trn, normalize_kspon
random.seed(0)
ko = [normalize_kspon(t) for _, t in random.sample(read_trn(os.path.join(K, "dev.trn")), 150)]
en = []
for tp in sorted(os.listdir(os.path.join(L, "dev-clean")))[:5]:
    d = os.path.join(L, "dev-clean", tp)
    for ch in sorted(os.listdir(d))[:2]:
        for line in open(os.path.join(d, ch, f"{tp}-{ch}.trans.txt")): en.append(line.strip().split(" ", 1)[1].lower())
en = en[:150]; texts = en + ko
ta = AutoTokenizer.from_pretrained(Q)
try: tb = AutoTokenizer.from_pretrained(Q, fix_mistral_regex=True); tb_ok = True
except Exception as e: tb = None; tb_ok = f"{type(e).__name__}: {str(e)[:60]}"
try: tc = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B")          # 허브 원본 (로컬 dir 에는 tokenizer.json 이 없고 vocab.json+merges.txt 뿐)
except Exception as e: tc = None; print("허브 로드 실패:", type(e).__name__, str(e)[:80])
from qwen_asr import Qwen3ASRModel
import torch
qm = Qwen3ASRModel.from_pretrained(Q, dtype=torch.bfloat16, device_map="cpu", max_new_tokens=8); td = qm.processor.tokenizer
def ids(t, s): return t.encode(s, add_special_tokens=False) if not isinstance(t, Tokenizer) else t.encode(s, add_special_tokens=False).ids
res = {}
for name, t in (("fix_mistral_regex", tb), ("hub_original", tc), ("qwen_asr.processor", td)):
    if t is None: res[name] = f"로드 실패: {tb_ok}"; continue
    diff = [(s, ids(ta, s)[:12], ids(t, s)[:12]) for s in texts if ids(ta, s) != ids(t, s)]
    res[name] = dict(n=len(texts), differ=len(diff), examples=[(s[:40], a, b) for s, a, b in diff[:3]])
print(json.dumps(dict(local_default_vs=res, vocab=len(ta), sample_en=ids(ta, en[0])[:10], sample_ko=ids(ta, ko[0])[:10]), ensure_ascii=False, indent=1))
rt = sum(ta.decode(ids(ta, s)) != s for s in texts); print(f"round-trip 불일치(디코드≠원문): {rt}/{len(texts)}")
