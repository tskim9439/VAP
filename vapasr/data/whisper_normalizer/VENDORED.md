# Whisper text normalizer (vendored)

- 출처: https://github.com/openai/whisper — `whisper/normalizers/{__init__.py,basic.py,english.py,english.json}` @ commit `86098128c0b4f24f0e2aa2994de830614b474227` (2026-08-31), 저장소 `LICENSE`(MIT, Copyright (c) 2022 OpenAI) 동봉.
- 변경: 각 .py 첫 줄에 출처 주석 한 줄만 붙였다. 코드·english.json 은 원본과 같다.
- 의존성: `regex`, `more_itertools`(로컬 vapasr-local, mxc conda env vapasr 에 설치돼 있음).
- 용도: `vapasr/data/aa_wer.py` — Artificial Analysis AA-WER v2 와 비교 가능한 영어 채점(VoxPopuli-Cleaned-AA). 내부 주 지표는 여전히 `textnorm.score_en` 이다.
