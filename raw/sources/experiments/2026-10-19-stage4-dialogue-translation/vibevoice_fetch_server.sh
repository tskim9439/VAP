# VibeVoice 외부 기준선 재료를 mxc 에서 직접 받는다(2026-10-19 사용자 "이번엔 서버에서 바로 받아줘" — 이번 건만). T5 사본과 같은 고정 리비전·같은 배치.
# 모델 두 개(LFS sha256 대조), Qwen2.5-7B tokenizer 파일, 소스 zip(T5 사본 sha256), 휠 셋(PyPI sha256). 기존 파일은 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; V=/soundai/users/tskim/VAPKT-data/models/vibevoice; F="python3 raw/sources/experiments/2026-10-19-stage4-dialogue-translation/hf_fetch.py"
mkdir -p $V/pydeps $V/Qwen2.5-7B-tokenizer
$F --repo microsoft/VibeVoice-ASR-Streaming-7B --rev 60d858b518b4e19d404af3737f848fc185b30177 --out $V/VibeVoice-ASR-Streaming-7B < /dev/null
$F --repo microsoft/VibeVoice-ASR --rev d0c9efdb8d614685062c04425d91e01b6f37d944 --out $V/VibeVoice-ASR < /dev/null
Q=https://huggingface.co/Qwen/Qwen2.5-7B/resolve/d149729398750b98c0af14eb82c78cfe92750796
while read f h; do $F --url $Q/$f --sha256 $h --out $V/Qwen2.5-7B-tokenizer/$f < /dev/null; done <<'L'
tokenizer.json c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539
vocab.json ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910
merges.txt 599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3
tokenizer_config.json c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9
config.json 267ce68584c5f24c3b267d934db2de68dd21d1ca677fb78ed809eb60067f7642
generation_config.json 8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8
LICENSE 832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e
L
echo "Qwen/Qwen2.5-7B d149729398750b98c0af14eb82c78cfe92750796 (tokenizer files only)" > $V/Qwen2.5-7B-tokenizer/REVISION
$F --url https://codeload.github.com/microsoft/VibeVoice/zip/16fb2cb1217c9934a886e1948ffb06120caa2df5 --sha256 bb552c43591b3a57cf08681e77caa555092f56a961786a1b474b975a211fe8b2 --out $V/VibeVoice-src-16fb2cb.zip < /dev/null
while read u h; do $F --url $u --sha256 $h --out $V/pydeps/$(basename $u) < /dev/null; done <<'L'
https://files.pythonhosted.org/packages/2a/2e/38d9824f8c6bb048c5ba21c6d4da54c29c162a46b58b3ef907a360a76d3e/diffusers-0.35.2-py3-none-any.whl d50d5e74fdd6dcf55e5c1d304bc52cc7c2659abd1752740d736d7b54078b4db5
https://files.pythonhosted.org/packages/20/b0/36bd937216ec521246249be3bf9855081de4c5e06a0c9b4219dbeda50373/importlib_metadata-8.7.0-py3-none-any.whl e5dd1551894c77868a30651cef00984d50e1002d06942a7101d34870c5f02afd
https://files.pythonhosted.org/packages/2e/54/647ade08bf0db230bfea292f893923872fd20be6ac6f53b2b936ba839d75/zipp-3.23.0-py3-none-any.whl 071652d6115ed432f5ce1d34c336c0adfd6a884660d1e9712a256d3d3bd4b14e
L
echo "VV_FETCH_DONE $(date '+%F %T')"
