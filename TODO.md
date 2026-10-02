# TODO — CST-S2TT

> **실행용 체크리스트.** 각 항목의 상세·완료 조건은 `wiki/tasks/task-cst-*.md` 가 정본이다.
> owner 별 대시보드는 `wiki/todo.md`(생성)다. 여기서 체크한 항목은 해당 태스크 파일의 `status` 도 함께 바꾼다.
>
> 근거: `wiki/sources/source-cst-s2tt-project-spec-v0-1.md` §17
> 마지막 갱신: 2026-10-02 (목표 전환)
> 이전 체크리스트: `plans/TODO-vap-asr-20260903.md`

---

## 데이터 (Phase 0) — `task-cst-data-access`

- [ ] 즉시 접근 가능한 교차언어 대화 코퍼스 확보(CS-Dialogue·Bangor 등 공개 후보부터). 로컬 T5 에만 받는다
- [ ] TAXI / Verbmobil 접근 경로 확인
- [ ] ETRI KO↔EN 배포 가능성 확인
- [ ] 코퍼스별 라이선스·변형·재배포 조건 정리

## 데이터 처리 (Phase 0–1) — `task-cst-conversation-schema-parser`, `task-cst-timeline-generator`

- [ ] unified conversation schema 설계
- [ ] dialogue parser 구현
- [ ] turn-level audio normalization
- [ ] semantic chunk segmentation(`<SEM_END>` 라벨 재사용 검토)
- [ ] controlled timeline generator(Normal·Early Turn·Interruption·Backchannel·Barge-in, 난이도 3 단계)
- [ ] 입력 채널 재결정(mono 혼합 vs 화자별 스트림)

## 평가 (Phase 2) — `task-cst-bench-v0-metrics`

- [ ] translation metrics(COMET·BLEU·chrF, 확정 텍스트만)
- [ ] streaming latency metrics(AL·LAAL·첫 토큰·semantic commit 지연)
- [ ] direction accuracy
- [ ] false stop / false switch
- [ ] missed interruption
- [ ] stop / switch latency

## 기준선 (Phase 3) — `task-cst-strong-baselines`

- [ ] strong SimulST 모델 선정
- [ ] bidirectional 2×SimulST routing
- [ ] VAD baseline
- [ ] VAP baseline(재현 VAP 재사용)
- [ ] oracle routing baseline

## 연구 검증 — `task-cst-kill-criterion-check`

- [ ] pipeline failure case 수집
- [ ] oracle gap 측정
- [ ] joint modeling 필요성 판단 → `wiki/decisions/` 기록

## 목표 전환 전 진행분 정리

- [ ] semcommit 라벨링(job 79205)을 계속할지 결정: 원천 semantic chunk 경계·commit 라벨로 쓸 수 있다
- [ ] Stage 1 본학습 결과(q17-s1m-r0·r3)를 원천 ASR 기준선으로 기록(영어 악화 분석 포함)
