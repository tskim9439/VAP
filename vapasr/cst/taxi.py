"""BAS TAXI(독일어 배차원 ↔ 영어 고객, 전화 대화) → CST 공통 스키마(vapasr/cst/schema.py).

원본 배치(CLARIN 다운로드): <root>/SESxxxx/<턴>.{wav,al,par,_annot.json} + SESxxxx.cmdi.xml
  턴 이름 = 4 자리 세션 번호 + 역할(DP 배차원·독일어 / CL 고객·영어 / PD 대화 전 / HP 끊기 전) + 2 자리 턴 번호
  .wav  16-bit PCM 8 kHz 모노(전화) — 이걸 쓴다. .al 은 같은 신호의 A-law 원본
  .par  BAS Partitur(UTF-8): ORT 단어, KAN 발음, NOI 잡음, TLN 사람 번역(“DE>EN 텍스트” 한 줄, 턴 전체)
        .par 가 없는 DP·CL 턴은 검수에서 garbage(겹침·메타 발화·빈 턴·잡음만·다른 언어) → usable=False
  녹음 서버가 버튼(DTMF)으로 턴을 나눠 겹침이 없고 원본 턴 간 시각도 없다 → timing="turn_order_only"
ORT 표기(SpeechDat): [fil] 채운 쉼, ** 알아들을 수 없음, ~ 끊긴 단어(앞·뒤), * 잘 안 들리는 단어(앞).
라이선스: “서면 허락 없이 일부라도 재배포 불가” → redistributable=False(공개본에는 ID·스크립트만)."""
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import re
import wave

from .schema import Session, Translation, Turn

ROLES = {"DP": ("German", "dispatcher"), "CL": ("English", "client")}
LANG_CODE = {"DE": "German", "EN": "English"}
TURN_RE = re.compile(r"^(\d{4})(DP|CL|PD|HP)(\d{2})$")


def parse_par(text: str) -> dict:
    """BAS Partitur 문자열 → {header, ort[(i, 단어)], noise[{span, label}], tln[(단어 범위, 방향, 텍스트)]}."""
    header, ort, noise, tln = {}, [], [], []
    for line in text.splitlines():
        m = re.match(r"^([A-Z]{3}):\s?(.*)$", line)
        if not m:
            continue
        tag, rest = m.group(1), m.group(2)
        if tag == "ORT":
            i, w = rest.split("\t", 1); ort.append((int(i), w.strip()))
        elif tag == "NOI":
            span, lab = rest.split("\t", 1); noise.append({"span": span, "label": lab.strip()})
        elif tag == "TLN":
            span, direction, txt = rest.split("\t", 2); tln.append((span, direction, txt.strip()))
        elif tag not in ("KAN",):
            header[tag] = rest.strip()
    ort.sort()
    return dict(header=header, ort=ort, noise=noise, tln=tln)


def clean_ort(words: List[str]) -> str:
    """평가용 전사: [..] 표시와 ** 를 빼고, 단어에 붙은 * · ~ 기호를 떼어 낸다(끊긴 단어 조각은 남긴다)."""
    out = []
    for w in words:
        if (w.startswith("[") and w.endswith("]")) or w == "**":
            continue
        w = w.lstrip("*").strip("~")
        if w:
            out.append(w)
    return " ".join(out)


def wav_info(path: Path) -> Tuple[int, int]:
    with wave.open(str(path)) as w:
        return w.getnframes(), w.getframerate()


def load_session(ses_dir: Path, root: Path) -> Session:
    """SESxxxx 디렉토리 하나 → Session. PD·HP 는 버리고 DP·CL 턴만 index 순으로."""
    turns: List[Turn] = []
    srs = set()
    for wav in sorted(ses_dir.glob("[0-9]*.wav")):
        m = TURN_RE.match(wav.stem)
        if not m or m.group(2) not in ROLES:
            continue
        role, idx = m.group(2), int(m.group(3))
        lang, _ = ROLES[role]
        n, sr = wav_info(wav); srs.add(sr)
        t = Turn(turn_id=wav.stem, index=idx, speaker=role, lang=lang, audio=str(wav.relative_to(root)),
                 num_samples=n, duration_s=round(n / sr, 4), usable=False)
        par = wav.with_suffix(".par")
        if par.exists():
            p = parse_par(par.read_text(encoding="utf-8"))
            words = [w for _, w in p["ort"]]
            t.words, t.transcript_raw, t.transcript, t.noise = words, " ".join(words), clean_ort(words), p["noise"]
            if p["tln"]:
                _, direction, txt = p["tln"][0]
                src, tgt = direction.split(">")
                if LANG_CODE.get(src) != lang:
                    raise ValueError(f"{wav}: TLN 방향 {direction} 이 화자 언어 {lang} 와 다름")
                t.translation = Translation(lang=LANG_CODE[tgt], text=txt, source="human")
            t.usable = bool(t.transcript) and t.translation is not None
        turns.append(t)
    turns.sort(key=lambda t: t.index)
    if len(srs) > 1:
        raise ValueError(f"{ses_dir}: 샘플레이트가 섞임 {srs}")
    return Session(session_id=f"taxi/{ses_dir.name}", corpus="TAXI", redistributable=False,
                   speakers={r: {"lang": l, "role": ro} for r, (l, ro) in ROLES.items()},
                   audio={"root": str(root), "sample_rate": srs.pop() if srs else 8000, "channel": "telephone"},
                   timing="turn_order_only", turns=turns,
                   meta={"license": "BAS TAXI — no redistribution without written permission", "version": "2.5"})


def iter_sessions(root: str) -> Iterator[Session]:
    root_p = Path(root)
    for d in sorted(p for p in root_p.glob("SES[0-9]*") if p.is_dir() and not p.name.startswith("._")):
        yield load_session(d, root_p)


def stats(sessions: List[Session]) -> Dict[str, object]:
    u = [t for s in sessions for t in s.usable_turns()]
    by = lambda role: [t for t in u if t.speaker == role]
    return dict(sessions=len(sessions), turns=sum(len(s.turns) for s in sessions), usable_turns=len(u),
                usable_DP=len(by("DP")), usable_CL=len(by("CL")),
                usable_minutes=round(sum(t.duration_s for t in u) / 60, 1),
                words=sum(len(t.transcript.split()) for t in u),
                sessions_without_usable=sum(1 for s in sessions if not s.usable_turns()),
                usable_turns_per_session_median=sorted(len(s.usable_turns()) for s in sessions)[len(sessions) // 2] if sessions else 0)
