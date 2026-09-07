# asr-tn-v1.0.0 audit

| partition | lang | rows | kept | quarantined | reasons | idem-viol | unk | collisions | dual num/lat/plain | Latin rows |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| train-clean-100 | En | 28539 | 28539 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| train-clean-360 | En | 104014 | 104014 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| train-other-500 | En | 148688 | 148688 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| dev-clean | En | 2703 | 2703 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| dev-other | En | 2864 | 2864 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| test-clean | En | 2620 | 2620 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| test-other | En | 2939 | 2939 | 0 | {} | 0 | 0 | 0 | -/-/- | - (-) |
| KsponSpeech_01 | Ko | 124000 | 123988 | 12 | {'charset': 12, 'digit': 12} | 0 | 0 | 859 | 17755/234/3 | 1452 (0.0117) |
| KsponSpeech_02 | Ko | 124000 | 123983 | 17 | {'charset': 17, 'digit': 17} | 0 | 0 | 829 | 16937/214/2 | 1364 (0.011) |
| KsponSpeech_03 | Ko | 124000 | 123991 | 9 | {'charset': 9, 'digit': 9} | 0 | 0 | 904 | 17876/209/2 | 1420 (0.0115) |
| KsponSpeech_04 | Ko | 124000 | 123985 | 15 | {'charset': 15, 'digit': 15} | 0 | 0 | 866 | 17584/227/1 | 1459 (0.0118) |
| KsponSpeech_05 | Ko | 124000 | 123985 | 15 | {'charset': 15, 'digit': 15} | 0 | 0 | 831 | 17464/207/4 | 1385 (0.0112) |
| dev | Ko | 2545 | 2545 | 0 | {} | 0 | 0 | 12 | 412/4/- | 26 (0.0102) |
| eval_clean | Ko | 3000 | 2999 | 1 | {'charset': 1, 'digit': 1} | 0 | 0 | 33 | 287/8/293 | 18 (0.006) |
| eval_other | Ko | 3000 | 3000 | 0 | {} | 0 | 0 | 1 | 625/30/497 | 15 (0.005) |

230 h diff: {'librispeech-100': {'segments': 28539, 'same': 28525, 'text_changed': 14}, 'kspon-100': {'segments': 62000, 'same': 61995, 'now_quarantined': 5}}
