# CST-Bench data format

## Manifest: common conversation schema (`manifest.jsonl`)

One JSON object per session (keys sorted). Defined in `src/cstbench/schema.py`.

| Field | Type | Meaning |
|---|---|---|
| `session_id` | str | `<corpus>/<source session id>`, e.g. `taxi/SES0037` |
| `corpus` | str | source corpus name |
| `redistributable` | bool | whether source audio/text may be redistributed |
| `speakers` | {id: {lang, role}} | e.g. `{"DP": {"lang": "German", "role": "dispatcher"}, "CL": {...}}` |
| `audio` | {root, sample_rate, channel} | turn audio paths are relative to the corpus root |
| `timing` | str | `turn_order_only` (no source timing) or `absolute` |
| `turns` | list | dialogue order, see below |
| `meta` | dict | provenance (license, corpus version) |

Turn fields:

| Field | Meaning |
|---|---|
| `turn_id`, `index` | source turn id and order |
| `speaker`, `lang` | speaker id and spoken language |
| `audio`, `num_samples`, `duration_s` | relative audio path, length at the source sample rate |
| `usable` | rated usable by the corpus validators and has a reference translation |
| `words` | source word tokens with corpus notation (e.g. SpeechDat `[fil]`, `**`, `~`, `*`) |
| `transcript_raw` | `words` joined by spaces |
| `transcript` | evaluation transcript, corpus markup removed |
| `noise` | corpus noise annotations |
| `translation` | `{lang, text, source}` reference into the other speaker's language; `source` = `human` or `silver:<method>` |
| `start_s`, `end_s` | only for `timing == absolute` |

## Session timeline (`<config>/SESxxxx/timeline.json`)

| Field | Meaning |
|---|---|
| `session_id`, `corpus`, `redistributable`, `speakers` | as in the manifest |
| `channels` | speaker order of `2ch.wav` channels |
| `sample_rate`, `config`, `seed` | render settings |
| `duration_s`, `overlap_ratio` | session length; time with both speakers talking / time with any speaker talking |
| `skipped` | source turns left out (not usable) |
| `mix`, `two_channel` | audio paths relative to the config directory |
| `turns[]` | `turn_id`, `speaker`, `lang`, `start_s`, `end_s`, `gap_before_s`, `transcript`, `translation`, `translation_lang`, `trimmed_s` (seconds of leading and trailing silence removed) |

Times are seconds from the start of `mix.wav`. Turn intervals include the 0.1 s trim margin around
speech.
