---
title: 결정 — semantic commit 턴 종료는 <EOT> 로 통일, Phase 1 은 <SEM_END> 만, 본 학습은 전 언어·전 데이터(대화체 포함)
summary: 2026-09-24 사용자 결정. <TURN_END> 는 Phase 2 <EOT> 와 역할이 겹쳐 폐기하고 턴 종료는 <EOT> 하나로 통일한다. Phase 1(mono) semcommit 학습은 <SEM_END> 만 가르친다. 본 학습은 두 언어 모두 시간 제한 없이 전 데이터를 쓰고 대화체를 포함한다. 골드셋은 Claude 가 만든다.
type: decision
created: 2026-09-24
related: [output-semcommit-v0.2-rack4-small, output-semcommit-gold-v1]
---

# 결정: 턴 종료 토큰 통일과 semcommit 본 학습 범위

## 맥락
v0.2 소규모 실험([[output-semcommit-v0.2-rack4-small]])은 mono 시퀀스에 `<SEM_END>`(151723)와 `<TURN_END>`(151724, 발화 끝 + 0.48 s proxy)를 함께 학습했다.
`<TURN_END>` 는 스트림 길이에 기대는 조기 발사를 보였고, 역할이 Phase 2 lane 모델의 `<EOT>`(151722, 턴 종료 soft 라벨)와 겹친다.

## 결정
- **턴 종료 토큰은 `<EOT>` 하나로 통일한다.** `<TURN_END>` 는 폐기 — 새 tokenizer 에 붙이지 않는다(v0.2 체크포인트는 읽기·평가만 호환).
- **Phase 1(mono) semcommit 은 `<SEM_END>` 만 가르친다.** 턴 종료는 Phase 2(`<EOT>`)가 맡는다. mono 에서 굳이 턴 종료를 함께 가르칠 때는 `--turn-end` 로 `<EOT>` 행을 쓴다(기본 끔).
- **본 학습은 두 언어 모두 시간 제한 없이 전 데이터를 쓰고 대화체를 포함한다.**
- **골드셋은 Claude 가 만든다**(사람 주석 대신; [[output-semcommit-gold-v1]]).

## 반영
- 코드(2026-09-24, 커밋 7635e69): `SEM_SPECIALS=[<SEM_END>]`, 학습·데이터셋·평가 기본 `turn_end=False`, 평가 턴 id 는 `config.sem_registry`(새 모델 `<EOT>`, v0.2 `<TURN_END>`, 없으면 턴 지표 없음).
  v0.2 r1 재채점 결과가 원 보고서와 전 지표 동일, SEM 만 학습 스모크 PARITY OK.
- 부수 효과: SEM 만 학습하면 “발화 끝 B/N 결정 위치가 TURN 타깃과 겹쳐 확정 억제로 학습되던” 문제(v0.2 KO 학습 1,122 스트림)가 사라진다.

## 근거
- 같은 사건(턴 종료)을 두 토큰으로 가르치면 Phase 2 로 합칠 때 충돌·중복이 생긴다. mono 스트림의 턴 종료 정답은 어차피 근사(발화 끝 + hangover)라, 실제 턴 교대 데이터가 있는 Phase 2 에서 `<EOT>` 로 배우는 편이 맞다.
- 전 데이터·대화체: 골드셋 대조에서 낭독체로만 학습한 모델이 대화체 영어(GigaSpeech)에서 거의 확정하지 못했다(r1 bias 0 확정 1 개 / 골드 192) — 대화체 데이터가 필수다.
