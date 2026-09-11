"""NOTSOFAR-1 서브셋 하나를 HF 에서 받는다(재시도 포함). corpus-relay.sh 3b 단계에서 호출.  usage: notsofar_download.py <repo 하위 경로> <로컬 dir>"""
import os, sys, time
from huggingface_hub import snapshot_download
sub, out = sys.argv[1], sys.argv[2]
for i in range(8):
    try:
        snapshot_download("microsoft/NOTSOFAR", repo_type="dataset", token=os.environ["HF_TOKEN"], allow_patterns=[sub + "/*", "LICENSE.txt", "README.md"], local_dir=out, max_workers=4)
        print("notsofar download ok", sub, flush=True); break
    except Exception as e:
        print("notsofar retry", i, repr(e)[:200], flush=True); time.sleep(120)
else:
    sys.exit(1)
