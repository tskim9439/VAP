---
type: task
status: open
owner: tskim
due: 2026-09-19
priority: p1
created: 2026-09-03
updated: 2026-09-03
summary: /data4 여유 575G 상황에서 체크포인트 보존·정리 규칙과 실험 폴더 규약 수립
sources:
  - [[decision-compute-environment]]
---

# 체크포인트 보존 정책

## 배경

→ [[decision-compute-environment]]

`CKPT_ROOT` (`/data4/tskim/VAPASR`) 가 있는 `/data4` 는 **97% 사용 중, 여유 575G**.
0.6B 모델 체크포인트 하나가 optimizer state 포함 시 수 GB 이므로, 규칙 없이 저장하면
Stage 1 중에 디스크가 찬다.

## 완료 조건

- [ ] 실험 폴더 규약: `CKPT_EXP_DIR/<날짜>-<stage>-<backbone>-<slug>/`
- [ ] 저장 규칙: best 1개 + last 1개만 유지, 나머지 자동 삭제 (콜백 설정)
- [ ] optimizer state 는 재개가 필요한 실험에만 저장
- [ ] 논문용 최종 모델만 `CKPT_EXPORT_DIR` 로 복사 (weights only)
- [ ] frozen encoder 특징 캐시·중간 산출물은 `/data3` 로 — `/data4` 금지
- [ ] 주 1회 `du -sh CKPT_EXP_DIR/*` 점검을 린트 루틴에 포함
- [ ] `/data4` 여유 200G 아래로 떨어지면 알림 (sync-rack4.sh status 에 경고 추가)

## 진행 기록

- 2026-09-03: 생성.
