---
type: task
status: open
owner: tskim
due: 2026-09-26
priority: p0
created: 2026-09-15
updated: 2026-09-15
summary: Phase 2 Q0 데이터 준비 — 확보된 DB(71631·134-1·134-2 실외·otoSpeech·AMI·ICSI·NOTSOFAR-1)만으로 대화 복원·mono 혼합·정렬·lazy-free lane 라벨·EOT soft target·manifest·QC
sources:
  - '[[output-phase2-lane-plan]]'
  - '[[output-phase2-data-inventory]]'
  - '[[output-phase2-training-db]]'
  - '[[decision-eot-immediate-soft-label]]'
---

# Phase 2 Q0 데이터 준비 (확보된 DB 한정)

## 배경
정본 [[output-phase2-lane-plan]] §8·§13 의 Q0 단계. 사용자 지시(2026-09-15): "일단 확보된 DB 로만 Phase 2 데이터 준비를 해보자." 대상은 서버에 있고 A등급이 확인된 자료뿐이며, Switchboard·CHiME-6·CANDOR 등 조건부 자료는 이 태스크에서 제외한다. 청소년 실내 조각은 대화 단위 완전본이 없어 전사·활동 감독 전용 후보로만 둔다([[output-phase2-data-inventory]] §6).

## 대상과 산출물

| 자료 | 언어 | 대화/회의 | 시간 | 입력 형태 | 이 태스크의 산출물 |
|---|---|---:|---:|---|---|
| 71631 실내 stereo 원본 | KO | 757 | 196.6 h | 2ch wav + 라벨 JSON | 채널 분리 → 화자별 정렬 → mono mix |
| 134-1 실외 조각 | KO | 1,492 | ≤344 h | 화자별 조각 + 라벨 | 시간축 복원 → 화자별 정렬 → mono mix |
| 134-2 실외 조각(완전) | KO | 523 | 90.1 h | 위와 같음 | 위와 같음 |
| otoSpeech | EN | 420 | 104.9 h | 화자별 wav + SRT | 정렬 → mono mix, actor split |
| AMI | EN | 171 | 98.1 h | Headset-N wav + NXT | 헤드셋 채널 = 화자, 정렬 → mono mix |
| NOTSOFAR-1 | EN | 237 | 24.2 h | close_talk CT_*.wav + gt_transcription(단어 시각) | CT 채널 = 화자, 정렬 검증 → mono mix |
| ICSI | EN | 75 | 71.6 h | chan*.sph + NXT | 채널↔참가자 매핑 → 정렬 → mono mix |

공통 산출물: `conversations.jsonl`(대화 단위: 화자별 채널 경로·오프셋·게인·split·품질 플래그), `segments.jsonl`(화자별 발화 구간·단어 시각·TN 텍스트), lane 라벨(lazy-free allocator, R=6), EOT 후보 위치와 p_end, 청크 단위 직렬화 결과, QC 리포트.

## 순서
1. [ ] 기존 데이터 계층 파악(스키마·71631 복원·정렬 도구·VAD·interleave) — 재사용 지점 확정
2. [ ] 코퍼스별 파서: 대화 복원 → 화자별 채널 wav(또는 구간 목록) + 참조 전사 (`vapasr/data/dialogue.py`, 코퍼스별 loader)
3. [ ] TN(asr-tn v1.3 `target(text, lang, corpus)`) 적용, fail-fast 검사
4. [ ] 화자별 forced alignment(SLURM, 사용자 제출) → 단어 시각·segment(gap 0.25 s 병합)
5. [ ] mono 혼합·관측 mask·crop 창(20–40 s, carry 60–120 s) 생성
6. [ ] `lane_alloc.py`(never_free/lazy_free)·`eot_soft.py`(구간 끝 결과 → p_end)·직렬화 → Dataset 레코드
7. [ ] 중복·노출 감사(71631 E2 노출, 134-1 원본↔조각 join), split(actor·회의 계열·공식 split)
8. [ ] 경량 QC: 밀도 p99·강제 NEXT·삭제율·lane 부족률·EOT 후보 분포·오디오 spot-check
9. [ ] 32창 overfit 용 소형 셋 추출

## 완료 조건
- [ ] 7 개 자료 모두 `conversations.jsonl`·`segments.jsonl` 생성, 대화 수·시간이 인벤토리 실측과 일치
- [ ] 정렬 coverage(단어 시각 확보 비율) 코퍼스별 보고, 실패 대화는 플래그
- [ ] N≤6 세션에서 `never_free` ≡ `lazy_free` 테스트 통과, 16+11 fixture 통과
- [ ] EOT 후보 결과 분류 분포가 어노테이션 기반 실측(§5 정본)과 같은 자릿수
- [ ] QC 리포트와 manifest 버전 고정, 32창 overfit 셋 준비

## 진행 기록
- 2026-09-15: 생성. 기존 데이터 계층 탐색 시작.
