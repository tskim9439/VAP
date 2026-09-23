## [2026-09-17] query | D1 시작 ONSET과 ASR 오류 진단

- Changed: [[output-phase2-d1-onset-asr-diagnosis]] 신규; [[output-phase2-d1-lane-eval]]에 후속 진단 링크.
- Reason: 사용자 관측인 시작 ONSET 오방출·높은 WER/CER의 보완 요청. 로컬 평가 42창과 paired 24창을 집계하고, serializer 무음 처리 및 crop 밖 재개에 따른 p_end 오라벨을 기존 함수로 재현했다. 디코더·soft EOT 입력 문맥·학습 관문·평가 지표를 분리해 실험 순서를 제안했다.
- Next: 같은 체크포인트로 임계값 승격 ablation → 라벨/시퀀스 계약 보완 → 소규모 자유실행 검증. GPU 재추론·코드 수정·정본 변경·학습·커밋·푸시는 하지 않았다.
- By: tskim
