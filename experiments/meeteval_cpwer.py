#!/usr/bin/env python
"""AMI cpWER 를 meeteval 로 다시 잰다(2026-10-27, 사용자 "표준 정규화로 다시 채점하는 것도 만들어줘" — 외부 AMI 수치와 같은 도구·정규화로 맞추기).
  python experiments/meeteval_cpwer.py --pred <stage3-eval 결과 폴더> [--pred …] [--norm whisper|ours|none] [--manifest <ami-bench-v1>/manifest.json] [--json <out.json>]
cst_stage3_eval 의 predictions-rank*.jsonl 행(ref_text·hyp_text — 화자별 단어열, 전사 없는 구간은 이미 뺀 것)을 화자마다 통째로 정규화해
meeteval cpWER(meeteval.wer.api.cpwer — 화자 짝은 헝가리안, 짝 없는 가설 화자의 말은 삽입)로 매기고 (코퍼스, 공식 분할, δ) 별로 합친다(오류 합 / 참조 단어 합).
  whisper = Whisper EnglishTextNormalizer(whisper_normalizer — 채움말 hmm·mm·mhm·mmm·uh·um 제거, 줄임말·영국 철자·숫자·호칭 표준화, 문장 단위라 여러 단어 숫자도 표준 그대로),
  ours = vapasr.data.textnorm.score_en(우리 채점), none = 그대로.
창 평가면 창 하나 = 세션 하나(창마다 화자 짝), 회의 통째(--window full)면 회의 하나 = 세션 하나. 비교용으로 행에 남은 cp(그 폴더를 채점한 정규화, 단어마다)도 함께 적는다.
--refs(2026-10-27, 사용자 "앞으로 cpWER 보고할 때 다른 모델들의 수치도 함께 포함해서 비교하기 쉽도록 해줘"): 끝에 외부 AMI 기준(external_ami.json — Parakeet·NVIDIA SSA·
VibeVoice·DiCoW, 조건 함께)을 붙인다. 기본은 raw/sources/experiments/2026-10-27-std-rescore/external_ami.json, 빈 값이면 붙이지 않는다.
읽기만 한다."""
import argparse, collections, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normalizer(name: str):
    if name == "whisper":
        from whisper_normalizer.english import EnglishTextNormalizer
        return EnglishTextNormalizer()
    if name == "ours":
        from vapasr.data.textnorm import score_en
        return score_en
    assert name == "none", f"--norm whisper|ours|none: {name}"
    return lambda t: t


def score(rows, norm):
    """행들 → meeteval CPErrorRate(합친 것). 세션 = 행(창 또는 회의) × δ."""
    from meeteval.io import SegLST
    from meeteval.wer import combine_error_rates
    from meeteval.wer.api import cpwer
    R, H = [], []
    for r in rows:
        sid = f"{r['id']}|d{r['delta']}"
        for i, t in enumerate(r["ref_text"]): R.append(dict(session_id=sid, speaker=f"r{i}", words=norm(t), start_time=0.0, end_time=1.0))
        for i, t in enumerate(r["hyp_text"]): H.append(dict(session_id=sid, speaker=f"h{i}", words=norm(t), start_time=0.0, end_time=1.0))
    return combine_error_rates(*cpwer(SegLST(R), SegLST(H)).values())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", action="append", required=True); ap.add_argument("--norm", default="whisper")
    ap.add_argument("--manifest", default="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json"); ap.add_argument("--json", default=None)
    ap.add_argument("--refs", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw/sources/experiments/2026-10-27-std-rescore/external_ami.json"))
    a = ap.parse_args()
    from vapasr.data.cst_eval import pool
    man = json.load(open(a.manifest)); sub = {**{m: "dev" for m in man["dev"]}, **{m: "eval" for m in man["eval"]}}
    norm = normalizer(a.norm); out = {}
    for d in a.pred:
        rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "predictions-rank*.jsonl"))) for l in open(f, encoding="utf-8")]
        assert rows, f"원출력 없음: {d}"
        g = collections.defaultdict(list)
        for r in rows:
            if r.get("lang", "English") != "English": continue
            g[(r["corpus"], sub.get(r["id"].split(":", 1)[1].split("@")[0], "?"), r["delta"])].append(r)
        print(f"== {d} — meeteval cpWER, 정규화 {a.norm}(화자 통째) · 행의 cp = 이 폴더 채점(단어마다)", flush=True)
        for (c, s, dl), rr in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1] != "eval", kv[0][2])):
            e = score(rr, norm); p = pool(rr)
            k = f"{os.path.basename(d.rstrip('/'))}/{c}/{s}/d{dl}"
            out[k] = dict(cpwer=round(100 * e.error_rate, 2), errors=e.errors, length=e.length, ins=e.insertions, dele=e.deletions, sub=e.substitutions,
                          missed_speaker=e.missed_speaker, falarm_speaker=e.falarm_speaker, sessions=len(rr), row_cp=round(p["err_cp"], 2), row_agn=round(p["err_agn"], 2))
            v = out[k]
            print(f"  {c:11s} {s:4s} δ{dl:<2d} meeteval cpWER {v['cpwer']:6.2f}(오류 {e.errors}/{e.length} · 삽입 {e.insertions} 삭제 {e.deletions} 대치 {e.substitutions} · "
                  f"화자 놓침 {e.missed_speaker} 헛화자 {e.falarm_speaker}) · 행의 cp {v['row_cp']:6.2f} · agn {v['row_agn']:6.2f} · 세션 {len(rr)}", flush=True)
    if a.refs and os.path.exists(a.refs):
        ex = json.load(open(a.refs, encoding="utf-8"))
        print("== 외부 AMI 기준(cpWER %, IHM(-mix) / SDM — 지연·화자 분리·평가 단위·정규화가 서로 다르다, wiki output-ami-external-comparison-20261027)", flush=True)
        for r in ex["rows"]: print(f"  {r['name']:40s} {r['ihm']:6.2f} / {r['sdm']:6.2f} · {r['latency']} · 화자 분리 {r['diar']} · {r['src']}", flush=True)
    if a.json: json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
