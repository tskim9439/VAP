#!/usr/bin/env python3
"""Local listening bundle with explicit human labels; no automatic training approval."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import file_digest, digest


def export_audio(row, out):
    import numpy as np
    import soundfile as sf
    from vapasr.data.selection_audio import decode
    result = dict(key=row["key"], files=[], training_eligible=False)
    try:
        x, info = decode(row)
        if info["sha256"] != row["audio_qc"].get("audio", {}).get("sha256"):
            raise ValueError("Waveform changed since full QC")
        rel = f'audio/{row["key"]}.wav'
        sf.write(out / rel, x, 16000, subtype="FLOAT")
        result.update(files=[dict(path=rel, label="현재 QC/교사 입력 채널")],
                      audio=info, edge_200ms=dict(
            first_rms=float(np.sqrt(np.mean(x[:3200].astype(float)**2))),
            last_rms=float(np.sqrt(np.mean(x[-3200:].astype(float)**2))),
            first_zero_fraction=float(np.mean(x[:3200] == 0)),
            last_zero_fraction=float(np.mean(x[-3200:] == 0))))
        if "multichannel_without_explicit_channel_mapping" in info["review"]:
            for channel in range(1, min(info["original_channels"], 4)):
                alternate = copy.deepcopy(row)
                alternate["audio"]["path"] += f"#ch{channel}"
                try:
                    y, _ = decode(alternate)
                    rel = f'audio/{row["key"]}-ch{channel}.wav'
                    sf.write(out / rel, y, 16000, subtype="FLOAT")
                    result["files"].append(dict(path=rel, label=f"진단용 채널 {channel} — 교사 입력 아님"))
                    if channel == 1 and len(x) == len(y):
                        result["channel_0_1_max_abs_diff"] = float(np.max(np.abs(x-y)))
                        result["channel_0_1_corr"] = (float(np.corrcoef(x, y)[0, 1])
                            if np.std(x) > 0 and np.std(y) > 0 else None)
                except Exception as e:
                    result.setdefault("alternate_errors", []).append(f"ch{channel}: {e}")
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def render(rows, out, teachers, input_path):
    exports = {r["key"]: r for r in map(json.loads, (out / "exports.jsonl").open())}
    evidence = {}
    for name, path in teachers:
        fp = json.loads(path.with_suffix(".fingerprint.json").read_text())
        if fp["input_sha256"] != file_digest(input_path):
            raise ValueError("Teacher input mismatch")
        evidence[name] = {}
        for r in map(json.loads, path.open()):
            if r["fingerprint"] != digest(fp):
                raise ValueError("Teacher provenance mismatch")
            if r["key"] in evidence[name]:
                raise ValueError("Duplicate teacher result")
            evidence[name][r["key"]] = r
    esc = lambda x: html.escape(str(x))
    cards = []
    for r in rows:
        e = exports[r["key"]]
        qc = r["audio_qc"]
        files = "".join(f'<div>{esc(f["label"])}<br><audio controls preload="none" src="{esc(f["path"])}"></audio></div>' for f in e["files"])
        hyp = []
        for name, records in evidence.items():
            t = records.get(r["key"], {})
            if t.get("status") == "ok" and t["audio"]["sha256"] != e.get("audio", {}).get("sha256"):
                raise ValueError("Teacher/export waveform mismatch")
            hyp.append(f'<p><b>{esc(name)}</b>: {esc(t.get("hyp", t.get("error", "미완료")))}</p>')
        selectors = "".join(f'<label>{title} <select data-axis="{axis}"><option value="unreviewed">미검수</option><option value="pass">확인됨</option><option value="fail">문제 있음</option><option value="uncertain">판단 보류</option></select></label> ' for axis, title in [("text", "전사"), ("timing", "경계/잘림"), ("speaker", "채널/화자"), ("turn", "턴 근거")])
        details = {k: v for k, v in e.items() if k not in ("files", "key")}
        cards.append(f'''<article data-key="{r['key']}" data-source="{esc(r['source'])}">
<h2>{esc(r['source'])} / {esc(r['utt_id'])}</h2>
<p>{esc(' | '.join(r['review_cell']))} · 주석 길이 {r['duration_s']:.3f}s / 원음 {qc.get('audio', {}).get('duration_s', '오류')}s</p>
<p><b>원문:</b> {esc(r['raw_text'])}</p><p><b>현재 타깃:</b> {esc(r['text'])}</p>{files}{''.join(hyp)}
<details><summary>진단값·원본 경로</summary><pre>{esc(json.dumps(details,ensure_ascii=False,indent=2))}</pre><p>{esc(r['audio'])}</p></details>
<div>{selectors}</div><textarea placeholder="청취 메모: 누락 단어, 시작/끝 잘림, 다른 화자, 확실하지 않은 이유"></textarea></article>''')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>데이터 선별 청취 검수</title>
<style>body{max-width:1100px;margin:30px auto;padding:0 16px;font-family:system-ui;background:#f5f6f8;color:#16202b}article{background:white;padding:20px;margin:20px 0;border:1px solid #cbd5e1;border-radius:10px}audio{width:100%}textarea{display:block;width:95%;margin-top:15px;height:60px}pre{white-space:pre-wrap;word-break:break-all}header{position:sticky;top:0;background:#f5f6f8;padding:12px 0;border-bottom:1px solid #bbb}button,input,select{padding:8px}h2{font-size:17px}</style>
<h1>데이터 선별: 청취 검수</h1><p>층화 표본이며 DB 통과율 추정용이 아닙니다. 교사 합의 ≠ 정답. 경계 에너지 ≠ VAD. 독립 crop만으로 의미론적 EOT를 승인하지 마세요. 원문·학습 타깃은 바뀌지 않습니다.</p>
<header>검색 <input id="search" placeholder="DB·문장·경고 유형"><button onclick="download()">검수 JSON 내려받기</button><span id="count"></span><p>입력은 새로고침하면 사라집니다. 종료 전에 JSON을 내려받아 주세요.</p></header>
''' + "\n".join(cards) + '''<script>
const cards=[...document.querySelectorAll('article')];
document.querySelector('#search').oninput=e=>{cards.forEach(c=>c.hidden=!c.textContent.toLowerCase().includes(e.target.value.toLowerCase()));};
function download(){const rows=cards.map(c=>({key:c.dataset.key,source:c.dataset.source,labels:Object.fromEntries([...c.querySelectorAll('select')].map(s=>[s.dataset.axis,s.value])),note:c.querySelector('textarea').value}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({reviewed_at:new Date().toISOString(),training_eligible:false,rows},null,2)],{type:'application/json'}));a.download='human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
document.querySelector('#count').textContent=' '+cards.length+' 표본';
</script></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--teacher", action="append", default=[], help="NAME=JSONL")
    a = ap.parse_args()
    if not 1 <= a.workers <= 32:
        raise ValueError("workers must be 1..32")
    rows = list(map(json.loads, a.input.open()))
    if a.export:
        (a.out / "audio").mkdir(parents=True, exist_ok=False)
        with ThreadPoolExecutor(max_workers=a.workers) as pool, (a.out / "exports.jsonl").open("x") as f:
            for r in pool.map(lambda row: export_audio(row, a.out), rows):
                f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n")
    teachers = [(name, Path(path)) for name, path in (t.split("=", 1) for t in a.teacher)]
    render(rows, a.out, teachers, a.input)
    print(f"COMPLETE {len(rows)} review cards: {a.out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
