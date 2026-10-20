## [2026-10-21] task | V2 탭 뺀 대조 v2-p1-notap 제출(apex job 93980, 93778 뒤), 평가 큐 ck2000·final

- Changed:
  - raw [[2026-10-20-v2-multispeaker]]: "v2-p1 학습·평가" 절에 대조 run 기록
  - [[status]]
- Reason: 사용자 "현재 DB 구성으로 V1 을 다시 학습시켜볼까?" → 2 슬롯 V1 은 3–4 인 창을 자르거나 버려 DB 가 같지 않으므로, DB·창·4 슬롯·seed·step 을 v2-p1 과 같게 두고 탭만 뺐다. "93778 Job ID 완료 후 수행되도록 명령어를 수정해줘." → `--after 93778`(afterany)
- 결과:
  - 제출 스크립트 dry-run 으로 인코더 탭 −1·나머지 인자 v2-p1 과 같음·`--dependency=afterany:93778` 을 확인했다. 사용자 제출 → job 93980, `/soundai/Model/VAPASR/stage3-v2-p1-notap`
  - 평가 큐(mxc 본 checkout, `TRANS_LEAD_SPACE=1`): 기존 6 단계 × ck2000·final, V2 추가 단계 × ck2000. 체크포인트가 생기면 시작한다.
- Next: TAXI 가 33 근처면 원인은 DB·창·4 슬롯 쪽(탭 아님), 28 근처면 학습된 탭이다. 2 인 대화 이득과 다화자 결과도 탭 몫과 DB 몫으로 가른다.
- By: tskim
