"""체크포인트 찾기·검사·적재 (iCloud dataless 안전).

모든 파일/폴더는 열기 전에 os.stat 의 SF_DATALESS 플래그를 본다 — dataless 파일을 읽으면 무한 대기할 수 있다.
glob/listdir 대신 정확한 파일 이름만 stat 한다 (audio-ui-spec §5.4)."""
from __future__ import annotations

import json
import os
import stat as _stat_mod

SF_DATALESS = 0x40000000
REQUIRED = ("config.json", "tokenizer.json")
CANDIDATES = (
    "~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync",
    "/Volumes/Samsung_T5/vapkt-models/semcommit-q06-s4-ast",
    "~/Desktop/VAPKT-models/semcommit-q06-s4-ast",
)


class ModelNotFound(RuntimeError):
    pass


def _dataless_msg(p: str) -> str:
    return (f"iCloud 에서 내려받지 않은 파일입니다(dataless): {p} — iCloud 가 되는 네트워크(사내망은 다운로드가 실패한다 — "
            f"핫스팟 등)에서 Finder \"지금 다운로드\" 또는 `brctl download {p}` 후 다시 실행하세요. 체크포인트는 *.nosync 폴더에 두세요")


def _stat(p: str):
    """존재하지 않으면 None, dataless 면 ModelNotFound. 경로 중간이 파일이면(NotADirectoryError) 없는 것으로,
    접근 권한이 없으면(macOS 개인정보 보호 포함) 이유를 담은 ModelNotFound — 기본 후보 목록은 다음 후보로 넘어간다."""
    try:
        st = os.stat(p)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except PermissionError:
        raise ModelNotFound(f"폴더에 접근할 수 없습니다: {p} — 시스템 설정 > 개인정보 보호 및 보안 > 파일 및 폴더(또는 전체 디스크 "
                            f"접근)에서 터미널/Python 을 허용하세요") from None
    except OSError as e:
        raise ModelNotFound(f"{p}: {e.strerror or e}") from None
    if getattr(st, "st_flags", 0) & SF_DATALESS:
        raise ModelNotFound(_dataless_msg(p))
    return st


def _ancestors_ok(d: str) -> bool:
    """d 의 상위 폴더들을 / 부터 하나씩 stat: 없으면 False, dataless 면 ModelNotFound (dataless 폴더 안의 이름 찾기는
    오프라인에서 멈출 수 있어, 각 단계는 이미 확인한 부모 안에서만 찾는다)."""
    parts = [x for x in d.split(os.sep) if x]
    cur = os.sep
    for x in parts[:-1]:
        cur = os.path.join(cur, x)
        if _stat(cur) is None:
            return False
    return True


def weight_files(d: str) -> list[str]:
    """model.safetensors 또는 샤드 index 에 적힌 파일들 (정확한 이름만 stat)."""
    single = os.path.join(d, "model.safetensors")
    if _stat(single) is not None:
        return [single]
    idx = os.path.join(d, "model.safetensors.index.json")
    if _stat(idx) is None:
        raise ModelNotFound(f"가중치 파일이 없습니다: {single}")
    with open(idx) as f:
        names = sorted(set(json.load(f)["weight_map"].values()))
    out = []
    for n in names:
        p = os.path.join(d, n)
        if _stat(p) is None:
            raise ModelNotFound(f"샤드 파일이 없습니다: {p}")
        out.append(p)
    return out


def _resolve(d: str) -> str | None:
    """<d> 또는 <d>/final 중 필수 파일이 다 있는 쪽 (없으면 None)."""
    d = os.path.abspath(os.path.expanduser(d))
    if not _ancestors_ok(d):
        return None
    st = _stat(d)
    if st is None:
        return None
    mode = getattr(st, "st_mode", None)
    if mode is not None and not _stat_mod.S_ISDIR(mode):
        raise ModelNotFound(f"폴더가 아니라 파일입니다: {d} — config.json 이 있는 폴더(또는 그 상위)를 주세요")
    for cand in (d, os.path.join(d, "final")):
        if _stat(cand) is None or _stat(os.path.join(cand, "config.json")) is None:
            continue
        for n in REQUIRED:
            if _stat(os.path.join(cand, n)) is None:
                raise ModelNotFound(f"체크포인트에 {n} 이 없습니다: {cand}"
                                    + (" — 개발 환경에서 load_tokenizer(...).save_pretrained 로 만드세요" if n == "tokenizer.json" else ""))
        weight_files(cand)
        return cand
    return None


def find_model(model_dir: str | None = None) -> str:
    """--model / $VAPKT_APP_MODEL → 기본 후보 순서. 첫 유효 디렉터리(<dir> 또는 <dir>/final)."""
    explicit = model_dir or os.environ.get("VAPKT_APP_MODEL")
    if explicit:
        r = _resolve(explicit)
        if r is None:
            raise ModelNotFound(f"모델 폴더를 찾을 수 없습니다: {explicit} (config.json 이 있는 폴더 또는 그 상위)")
        return r
    first_err = None
    for c in CANDIDATES:
        try:
            r = _resolve(c)
        except ModelNotFound as e:          # dataless 등: 다음 후보를 보고, 끝내 없으면 이 오류를 알린다
            first_err = first_err or e
            continue
        if r:
            return r
    if first_err:
        raise first_err
    raise ModelNotFound("q06-s4-ast 체크포인트를 찾지 못했습니다. --model <폴더> 또는 VAPKT_APP_MODEL 로 지정하세요. 찾아본 곳: "
                        + ", ".join(CANDIDATES))


# ── config / 토크나이저 / 가중치

S4_TEXT = dict(num_hidden_layers=28, hidden_size=1024, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
               intermediate_size=3072, rms_norm_eps=1e-6, rope_theta=1e6, vocab_size=151936, hidden_act="silu")


def load_config(d: str, strict: bool = True) -> dict:
    """config.json 을 읽고 decode-contract 체크리스트의 불변식을 검사한다."""
    from .tokens import BLOCKED_EXPECTED
    path = os.path.join(d, "config.json")
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except ValueError as e:                     # JSONDecodeError·UnicodeDecodeError: 깨진 파일
        raise ModelNotFound(f"{path}: config.json 을 읽을 수 없습니다 ({e}) — 체크포인트 폴더가 온전한지 확인하세요") from None
    missing = [k for k, ok in (("thinker", isinstance(cfg, dict) and isinstance(cfg.get("thinker"), dict)),
                               ("thinker.text_config", isinstance(cfg, dict) and isinstance((cfg.get("thinker") or {}).get("text_config"), dict)),
                               ("sp_ids", isinstance(cfg, dict) and isinstance(cfg.get("sp_ids"), dict))) if not ok]
    if missing:
        raise ModelNotFound(f"{path}: q06-s4 체크포인트의 config.json 이 아닙니다 (없는 키: {', '.join(missing)})")
    tc = cfg["thinker"]["text_config"]
    errs = []
    want = dict(runaway_cap=32, max_flush_rounds=8, encoder_right_context=3, encoder_left_context=56, ast_format="ast-v1",
                lanes=0, frame_hz=12.5, audio_pad_id=151676, adapter_d_in=1024, adapter_d_out=1024, adapter_hidden=2048)
    errs += [f"{k}={cfg.get(k)!r} (기대 {v!r})" for k, v in want.items() if cfg.get(k) != v]
    if not (cfg.get("encoder_trainable") or cfg.get("encoder_saved")):
        errs.append("encoder_trainable/encoder_saved 가 모두 거짓 — 체크포인트에 학습된 인코더가 없음")
    if sorted(cfg.get("blocked_ids", [])) != BLOCKED_EXPECTED:
        errs.append(f"blocked_ids 가 §a.2 와 다름: {cfg.get('blocked_ids')}")
    for name in ("<NEXT_AUDIO>", "<SPK_A>", "<SPK_B>", "<TRANS>"):
        if cfg["sp_ids"].get(name) in cfg.get("blocked_ids", []):
            errs.append(f"{name} 가 차단됨")
    if not tc.get("tie_word_embeddings", cfg["thinker"].get("tie_word_embeddings", True)):
        errs.append("tie_word_embeddings=False")
    rs = tc.get("rope_scaling") or {}
    if rs.get("rope_type", rs.get("type", "default")) != "default":
        errs.append(f"rope_scaling 이 default 가 아님: {rs}")
    if strict:
        errs += [f"text_config.{k}={tc.get(k)!r} (기대 {v!r})" for k, v in S4_TEXT.items() if tc.get(k) != v]
    if errs:
        raise RuntimeError("config.json 이 s4 계약과 다릅니다:\n  " + "\n  ".join(errs))
    return cfg


def load_tokenizer(d: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(d, "tokenizer.json"))


def load_weights(d: str) -> dict:
    """mx.load 로 모든 텐서(fp32, 지연 적재). 정밀도 변환은 각 모듈이 한다."""
    import mlx.core as mx
    w = {}
    for p in weight_files(d):
        w.update(mx.load(p))
    return w


def split_weights(w: dict, src: str = "") -> tuple[dict, dict, dict]:
    """(thinker 'thinker.' 제거, adapter 'adapter.' 제거, encoder 'encoder.' 제거) — 키 개수 검사 포함. src: 오류 메시지용 경로."""
    th = {k[len("thinker."):]: v for k, v in w.items() if k.startswith("thinker.")}
    ad = {k[len("adapter."):]: v for k, v in w.items() if k.startswith("adapter.")}
    en = {k[len("encoder."):]: v for k, v in w.items() if k.startswith("encoder.")}
    where = f" ({src})" if src else ""
    counts = f"키 수: 전체 {len(w)}, thinker {len(th)}, adapter {len(ad)}, encoder {len(en)}"
    if sorted(ad) != sorted(f"net.{i}.{p}" for i in (0, 1, 3) for p in ("weight", "bias")):
        raise RuntimeError(f"q06-s4 가중치가 아닙니다{where}: adapter 키가 다릅니다 {sorted(ad)[:8]} — {counts}")
    if "model.embed_tokens.weight" not in th and "lm_head.weight" not in th:
        raise RuntimeError(f"thinker 임베딩(model.embed_tokens.weight / lm_head.weight)이 없습니다{where} — {counts}")
    return th, ad, en
