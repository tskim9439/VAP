# 로컬 Mac 에서:  source scripts/local-env.sh
# 사내망 TLS 검사(Kt Corporate Forward Trust CA) 때문에 python/conda/pip 이 pypi·conda-forge·github 인증서를 거부한다.
# certifi 번들 + 프록시 CA 체인을 합친 ~/.vapkt-ca-bundle.pem 을 쓰도록 환경변수를 잡고, conda env vapasr-local 을 활성화한다.
#   번들 생성(최초 1 회):  cp $(python3 -c "import certifi;print(certifi.where())") ~/.vapkt-ca-bundle.pem
#                        openssl s_client -connect pypi.org:443 -servername pypi.org -showcerts </dev/null 2>/dev/null | awk '/BEGIN CERT/,/END CERT/' | sed '1,/END CERT/d' >> ~/.vapkt-ca-bundle.pem
_b="$HOME/.vapkt-ca-bundle.pem"
if [ -f "$_b" ]; then export SSL_CERT_FILE="$_b" REQUESTS_CA_BUNDLE="$_b" CURL_CA_BUNDLE="$_b" PIP_CERT="$_b"; fi
_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export VAPASR_LOCAL_MODELS="${VAPASR_LOCAL_MODELS:-$HOME/Desktop/VAPKT-models}"      # 내려받은 체크포인트·인코더 캐시 위치
export VAPASR_ENCODER_CACHE="${VAPASR_ENCODER_CACHE:-$VAPASR_LOCAL_MODELS/nemotron-encoder.pt}"
export VAPASR_STAGE_DIR=""                                                          # 로컬은 스테이징 불필요
export PYTORCH_ENABLE_MPS_FALLBACK=1                                                # MPS 미지원 연산은 CPU 로
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate vapasr-local 2>/dev/null || echo "(conda env vapasr-local 없음)"; fi
export PYTHONPATH="$_root${PYTHONPATH:+:$PYTHONPATH}"
unset _b _root
