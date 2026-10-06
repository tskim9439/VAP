---
author_id: tskim
created: 2026-10-07
source_type: source-drop
---

# 평가셋 조사 원자료 (2026-10-07)

- 사용자 요청: "지금 확보한 DB 뿐만 아니라 관련 논문을 탐색해서 필요한 평가셋을 조사해."
- 조사 에이전트 4 개(스트리밍 ASR · 동시 음성번역 · 한국어 ASR/번역 · 대화·다화자)의 최종 보고를 그대로 저장했다. 각 보고의 수치·분할·라이선스는 에이전트가 원문 페이지(arXiv HTML, ACL Anthology, Hugging Face, OpenSLR, AI Hub, IWSLT 페이지)를 열어 읽은 것이고, 보고마다 출처 URL 과 불확실성 절이 있다.
- 웹 검색 한도 때문에 검색 대신 arXiv API·원문 페이지 직접 열기로 조사했다. "최근 논문" 목록은 망라적이지 않다.
- 종합: [[output-eval-set-plan-asr-simulst-20261007]]
