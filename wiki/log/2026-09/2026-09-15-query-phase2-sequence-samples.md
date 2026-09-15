## [2026-09-15] query | 7 DB 실제 샘플 시퀀스 생성·검증

- Changed: [[output-phase2-sequence-samples]] 신규; `raw/sources/experiments/2026-09-15-phase2-sequence-samples/`(코퍼스별 report·window 렌더·stats); `experiments/p2_show_sequence.py`·`p2_align.py`(앞 공백 토큰화)·`vapasr/data/dialogue_corpora.py`(AMI 빈 segment·NOTSOFAR 태그·ICSI flac); [[task-phase2-data-prep]] 갱신.
- Reason: 사용자 요청 "각 DB 마다 실제 샘플로 시퀀스를 짜보고 검증까지". 6 DB 는 Qwen 정렬 실물, ICSI 는 proxy 시각으로 창 3개씩 8 검사 통과.
- Next: 71631 quarantine(digit·anon) 처리 결정, 창 격자 hop 2 s·min_text_tokens, ICSI 변환 완료 후 정렬, 전량 빌드(`slurm/p2_build_dialogues.sbatch`)·정렬 제출.
- By: tskim
