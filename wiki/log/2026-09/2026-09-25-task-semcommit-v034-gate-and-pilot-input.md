## [2026-09-25] task | semcommit v0.3.4 관문 통과·파일럿 라벨링 입력 준비

- Changed: 코드 `vapasr/data/semcommit_llm.py`·`experiments/semcommit_gold.py`(v0.3.4, 커밋 2b273f3), `experiments/semcommit_build_speechlm_candidates.py`(구두점 태그 버그 수정),
  `experiments/semcommit_approve_speechlm_words.py`(신규), `slurm/semcommit-main-label-{apex-n2.sbatch,worker.sh}`(노드 수를 할당에 맞춤), 테스트.
  mxc: `semcommit-work/gate/speechlm-all19-g1-punctcoord-v034-20260925/`(공식 CPU 마무리·수락), `semcommit-work/pilot-v034-20260925/{candidates,approved}`, `semcommit-work/claude-code-v034-20260925/`(검증·백업·로그).
- Reason: mxc 대형 교사 관문(g1)이 영어 후보 재현율 3 경계 차로 불통과, 75451(punct_coord 후보)은 other 가지에 섞여 불통과. punct_coord 단독 후보를 별도 가지로,
  가지를 켜려면 dev A ≥ 20(--min-branch-a) — 기준을 바꾸지 않고 **accepted=true(qwen)**: 영어 A 0.956 / 0.744, 후보 재현 0.863, 한국어 A 0.922 / 0.940.
  후보 빌더의 토크나이저 경로는 구두점이 단어 안에 남아 `punct_final` 이 전혀 붙지 않았다(A 의 주 출처 소실) — 가장자리 구두점을 태그로 옮기고 단독 구두점은 이웃 단어에 합친다.
  파일럿 입력: 19 개 DB 가 모두 한국어라 AI Hub 일반 자유대화(000004)·복지 콜센터(000027) 각 2 파트 → 후보 10,836 → QC 통과 승인 9,761(짧은 9,384·긴 377, 25 shard). 스트림 끝 구두점 97.4 %.
- Next: 사용자 SLURM 제출(1 노드 × 8 GPU 파일럿) → 처리량·라벨 통계·표본 점검 → 19 개 DB 전량. 확인할 것: ASR 은빛 전사의 구두점은 사람 전사가 아니다(골드가 검증한 건 사람 전사 구두점) — 파일럿 라벨로 문장 끝 A 표본 점검.
- By: tskim
