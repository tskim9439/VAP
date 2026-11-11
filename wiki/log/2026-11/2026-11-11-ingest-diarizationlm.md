## [2026-11-11] ingest | DiarizationLM (arXiv 2401.03506 v13)

- Changed: [[source-diarizationlm-2401-03506]] 추가, [[output-cpwer-hypotheses-20261110]] 에 링크.
- Reason: 사용자가 논문을 읽고 우리가 얻을 인사이트를 확인해 달라고 했다. 글자만 보는 LLM 으로 화자 귀속을 고치는 후처리이고, 4 인 회의에서는 5 단어 이하 구간만 고치게 해야 이득이다(AMI cpWER −0.75 pt). 우리 귀속 오류를 같은 방식으로 나누니 약 70 % 가 짧은 구간이었다.
- Next: 귀속 수단을 손볼 때 구간 길이별 고침·망침 분석과 회의 단위 짝 부트스트랩을 쓴다. 논문 방법을 그대로 붙이지는 않는다.
- By: tskim
