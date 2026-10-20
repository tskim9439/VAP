# V2 첫 학습 준비 — 최대 4 화자·창 20–90 s 재추출·회의 번역 (2026-10-20)

결정: [[decision-v2-run-scope-20261020]], 계획·상태: [[output-v2-run-plan-20261020]]. 두 스트림 입력(V2-a)은 [[2026-10-20-v2a-dual-stream]].

## 스크립트
- `sim_build_multi.sh train|eval`: 3–4 인 합성 대화(`--speakers 2:0.2,3:0.4,4:0.4 --p-back 0.2`). 학습은 언어마다 1 만 세션(`sim-dialogue-v3-multi`), 평가는 held-out 재료로 300 세션(`sim-dialogue-multi-eval-v1`)이다.
- `meet_targets_watch.sh <수리 job>`: 회의 번역 수리 job 이 `all: 100.0%` 로 끝나면 `ast_build_targets --pack` 으로 묶는다.
- `v2_eval_extra.sh <run> <대상…>`: 기존 평가 큐에 더하는 셋이다. TAXI 세션 통째, 긴 창(60–90 s), 회의(≤ 4 화자, 4 슬롯 선언), 3–4 인 합성, 회의 번역. V1(2 슬롯)은 모델 슬롯 수로 잰다.
- `v2_check.sh`: 학습 전 점검. 단위 시험 → dry-run → 스모크 20 step → 슬롯·탭 확인 → 평가 경로(4 슬롯 직렬 = 배치).

## 점검 기록
- 단위 시험: 로컬 `tests/test_v2_slots.py` 12 개. mxc 개발 사본(`VAPKT-dev`)에서 94 통과(varlen 포함).
  - 2 화자 동작은 예전과 같다: 합성 2 인 세션이 HEAD 와 바이트까지 같다(5 seed). 기존 대화·평가 시험도 통과했다.
- 오라클(모델 없이 정답 시퀀스 채점, st4-m1 ck1500 tokenizer):
  - V1 2 화자 기존 창(aihub134-2·otoSpeech·kec, 창 20, δ10): slot·cp 0, swap 0
  - 회의 4 화자·20–90 s 이어 덮기(AMI·ICSI·NOTSOFAR held-out, 창 40, δ4·10): slot·cp 0, swap 0
    - 창 240 개의 화자 수: 1 명 14 · 2 명 58 · 3 명 70 · 4 명 98, 길이 중앙 47.7 s
- 회의 원문 단위(`stage4-dlg-units-meet-v1`): AMI 89,803 · ICSI 111,388 · NOTSOFAR 61,189 단위, 합계 2.04 M 단어·227,935 행. 생성 apex 93807 → 수리 93808(사용자 제출).
- 3–4 인 합성 평가 셋: 영·한 300 세션씩, 실패 0, 겹침 6 %.
