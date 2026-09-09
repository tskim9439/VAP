"""Nemotron 인코더 캐시 검증(서버, VAPASR_STAGE_DIR 필요): 첫 로드(restore_from → 캐시 저장) vs 두 번째(캐시) 시간과 출력 동일성. 2026-09-09: 73.6 s → 5.8 s, max|diff| 1.9e-5."""
import time, os, glob, torch
os.environ.setdefault("VAPASR_STAGE_DIR", "/scratch/vapasr-stage")
from vapasr.features.online import NemotronOnline, _encoder_cache_path
d = os.environ["MXC_NEMOTRON_DIR"]; nemo = sorted(glob.glob(d + "/*.nemo"))[0]; cp = _encoder_cache_path(nemo); print("cache path:", cp, "exists:", os.path.exists(cp))
t = time.time(); a = NemotronOnline(path=d); ta = time.time() - t; print(f"1st NemotronOnline: {ta:.1f}s (cache existed before: {os.path.exists(cp)})", flush=True)
t = time.time(); b = NemotronOnline(path=d); tb = time.time() - t; print(f"2nd NemotronOnline (cache): {tb:.1f}s", flush=True)
torch.manual_seed(0); wav = torch.randn(2, 16000 * 3); wl = torch.tensor([48000, 40000]); K = torch.tensor([37, 31])
with torch.no_grad(): ya = a(wav, wl, K); yb = b(wav, wl, K)
print("output shape", tuple(ya.shape), "max|diff|", float((ya - yb).abs().max()), "ctx", a.enc.att_context_size, b.enc.att_context_size)
