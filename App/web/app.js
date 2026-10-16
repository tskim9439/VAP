// VAPKT 실시간 전사·번역 — 브라우저 클라이언트 (빌드 단계·프레임워크·외부 요청 없음; audio-ui-spec §3, §4, §6).
// 마이크: AudioWorklet(장치 레이트, Float32 20 ms) → WS 바이너리. 파일 1×: 같은 워크릿. 빠르게/평가: 320 ms 패킷,
// 흐름 제어(보낸 초 − stats.audio_done_s ≤ 10 s, bufferedAmount ≤ 512 KB). 파일은 1×·빠르게 모두 같은 표본을 원래 레이트로 보내고
// 리샘플은 서버 soxr 한 곳에서만 한다 (WAV 는 JS 로 직접 읽어 soundfile 과 비트 동일, 그 밖의 형식은 브라우저 디코드).
const $ = (id) => document.getElementById(id);
const LANG_KO = { Korean: '한국어', English: '영어', German: '독일어' };
const REASON_KO = { pause: '멈춤', soft_max: '긴 발화', max_len: '최대 길이', kv_guard: 'KV 한도', stop: '정지', file_end: '파일 끝', aborted: '중단' };
const START_KO = { vad: '', forced: '강제 이어짐', soft: '긴 발화 이어짐', kv: 'KV 이어짐', file: '파일', none: '분할 없음' };
const VAD_KO = { silence: '조용함', speech: '말하는 중', hangover: '멈춤 대기' };
const FAST_PKT_S = 0.32;
const MAX_INFLIGHT_S = 10;
const MAX_BUFFERED = 512 * 1024;
const MIN_RATE = 8000; const MAX_RATE = 96000;            // 서버가 받는 표본율 (server.py validate_start)
const rateOk = (r) => r >= MIN_RATE && r <= MAX_RATE;
// 고급 숫자 옵션: 서버 범위와 같다 (server.py validate_start, DEFAULT_OPTIONS)
const OPT_RANGE = {
  end_pause_ms: { id: 'opt-pause', lo: 300, hi: 2000, d: 800, msg: '끝 멈춤은 300–2000 ms 입니다' },
  hard_max_s: { id: 'opt-max', lo: 10, hi: 30, d: 25, msg: '최대 길이는 10–30 s 입니다' },
};

const S = {
  ws: null, wsOpen: false, retry: 0, hello: null, state: 'connecting', stateMsg: '', loadT0: performance.now(),
  sid: null, cfg: null, run: null,          // run: {kind: 'mic'|'file1x'|'fast', stopping, startSent, sid, speed}
  cap: null,                                 // {ctx, node, sink, stream, src, limit}
  fast: null,                                // {stop, abort}
  sending: false, sentSamples: 0, sentRate: 16000,
  startWait: null, retryStart: null,
  stats: null, vad: null, events: [], rows: new Map(), liveKey: null, autoScroll: true,
  zero: { t0: 0, nonzero: false, done: true }, pendingCfg: null, pingTimer: null,
  segT: -1e9, lagTicks: 0,
  prefs: loadPrefs(),
};

// ── 환경설정 (localStorage 는 편의용; 실패해도 동작) ─────────────────────────────────────────────────────
function loadPrefs() {
  try { return JSON.parse(localStorage.getItem('vapkt-prefs') || '{}') || {}; } catch (e) { return {}; }
}
function savePrefs() {
  try { localStorage.setItem('vapkt-prefs', JSON.stringify(S.prefs)); } catch (e) { /* 무시 */ }
}

// ── 작은 도우미 ─────────────────────────────────────────────────────────────────────────────────────
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function fmtT(s) {
  if (s == null || !isFinite(s)) return '—';
  const m = Math.floor(s / 60); const r = s - 60 * m;
  return `${String(m).padStart(2, '0')}:${r.toFixed(1).padStart(4, '0')}`;
}
const num = (v, d = 0) => (v == null || !isFinite(v) ? '—' : Number(v).toFixed(d));
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
function mode() { return document.querySelector('input[name="mode"]:checked').value; }
function setMode(m) { document.querySelector(`input[name="mode"][value="${m}"]`).checked = true; }
function pairKey(src, tgt) { return `${src}>${tgt}`; }
function wsSend(obj) { if (S.ws && S.wsOpen) S.ws.send(JSON.stringify(obj)); }

// 빠르게 재생 전용 대기: 전용 워커의 타이머 (배경 탭에서도 메인 스레드 setTimeout 처럼 늦춰지지 않는다). 워커가 안 되면 setTimeout.
const P = { w: null, ok: null, n: 0, wait: new Map() };
function pacerOk() {
  if (P.ok === null) {
    try {
      P.w = new Worker('/static/pacer.js');
      P.w.onmessage = (e) => { const r = P.wait.get(e.data); if (r) { P.wait.delete(e.data); r(); } };
      P.w.onerror = () => { P.ok = false; for (const r of P.wait.values()) r(); P.wait.clear(); };
      P.ok = true;
    } catch (e) { P.ok = false; }
  }
  return P.ok;
}
function psleep(ms) {
  if (!pacerOk()) return sleep(ms);
  return new Promise((r) => { const id = ++P.n; P.wait.set(id, r); P.w.postMessage({ id, ms }); });
}

// ── 배너 ─────────────────────────────────────────────────────────────────────────────────────────────
function banner(key, msg, level = 'warn', action = null) {
  let b = document.querySelector(`.banner[data-key="${key}"]`);
  if (!b) {
    b = el('div', 'banner'); b.dataset.key = key;
    b.append(el('span', 'msg'));
    const x = el('button', 'link', '×'); x.type = 'button'; x.title = '닫기';
    x.addEventListener('click', () => b.remove());
    b.append(x);
    $('banners').append(b);
  }
  b.className = `banner ${level}`;
  b.querySelector('.msg').textContent = msg;
  const old = b.querySelector('.act'); if (old) old.remove();
  if (action) {
    const a = el('button', 'act', action.label); a.type = 'button';
    a.addEventListener('click', () => { b.remove(); action.fn(); });
    b.insertBefore(a, b.lastChild);
  }
}
function unbanner(key) { const b = document.querySelector(`.banner[data-key="${key}"]`); if (b) b.remove(); }

// ── 상태 표시 ──────────────────────────────────────────────────────────────────────────────────────────
function setState(st, msg) {
  S.state = st; S.stateMsg = msg || '';
  renderState();
  updateButtons();
}
function renderState() {
  const chip = $('state-chip');
  let st = S.state; let text;
  if (!S.wsOpen) { st = 'disconnected'; text = '연결 끊김 — 다시 연결 중…'; }
  else if (S.run) { st = 'running'; text = S.run.stopping ? '마무리 중…' : (S.run.kind === 'mic' ? '듣는 중' : '파일 처리 중'); }
  else if (st === 'loading') text = `모델 로딩 중… ${Math.round((performance.now() - S.loadT0) / 1000)} s`;
  else if (st === 'warmup') text = '준비 중(커널 컴파일)';
  else if (st === 'ready') text = '준비됨';
  else if (st === 'error') text = '모델 오류';
  else text = '연결 중…';
  chip.dataset.state = st;
  $('state-text').textContent = text;
  chip.title = S.stateMsg || '';
}
setInterval(() => { if (S.state === 'loading' || S.state === 'warmup') renderState(); }, 1000);

function updateButtons() {
  const ready = S.state === 'ready' && S.wsOpen;
  const running = !!S.run;
  const hasFile = $('file').files.length > 0;
  const mic = $('btn-mic');
  if (running && S.run.kind === 'mic') {
    mic.textContent = S.run.stopping ? '정지 중…' : '■ 정지'; mic.classList.add('on'); mic.disabled = !!S.run.stopping;
  } else {
    mic.textContent = '● 마이크 시작'; mic.classList.remove('on'); mic.disabled = !ready || running;
  }
  const play = $('btn-play'); const fast = $('btn-fast');
  play.textContent = running && S.run.kind === 'file1x' ? (S.run.stopping ? '정지 중…' : '■ 정지') : '▶ 실시간 재생';
  play.disabled = running ? !(S.run.kind === 'file1x' && !S.run.stopping) : !(ready && hasFile);
  fast.textContent = running && S.run.kind === 'fast' ? (S.run.stopping ? '정지 중…' : '■ 정지') : '⏩ 빠르게';
  fast.disabled = running ? !(S.run.kind === 'fast' && !S.run.stopping) : !(ready && hasFile);
  $('btn-abort').hidden = !running;
  $('opt-seg').disabled = running; $('opt-max').disabled = running;
  // 마이크 장치·브라우저 처리는 시작할 때만 정해진다: 실행 중에는 잠근다 (미지원 항목은 늘 잠김)
  for (const id of ['mic', 'opt-ec', 'opt-ns', 'opt-agc']) $(id).disabled = running || !!$(id).dataset.unsup;
}

// ── 연결 ─────────────────────────────────────────────────────────────────────────────────────────────
function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  S.ws = ws;
  ws.addEventListener('open', () => {
    S.wsOpen = true; S.retry = 0; unbanner('disconnected'); renderState(); updateButtons();
    clearInterval(S.pingTimer);
    S.pingTimer = setInterval(() => wsSend({ type: 'ping', t: performance.now() }), 5000);
  });
  ws.addEventListener('message', (ev) => {
    if (typeof ev.data !== 'string') return;
    let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
    try { onMsg(m); } catch (e) { console.error(e); }
  });
  ws.addEventListener('close', () => {
    const was = S.wsOpen; S.wsOpen = false; clearInterval(S.pingTimer);
    if (S.run) { teardownCapture(); endRun(); banner('session', '연결이 끊겨 진행 중이던 세션이 끝났습니다', 'error'); }
    if (was || S.retry === 0) banner('disconnected', '서버 연결이 끊겼습니다 — 다시 연결하는 중…', 'error');
    renderState(); updateButtons();
    const delay = Math.min(8000, 500 * 2 ** S.retry); S.retry += 1;
    setTimeout(connect, delay);
  });
}

// ── 서버 메시지 ─────────────────────────────────────────────────────────────────────────────────────────
function record(m) { if (S.sid && (m.sid === undefined || m.sid === S.sid || m.session_id === S.sid)) S.events.push(m); }

function onMsg(m) {
  switch (m.type) {
    case 'hello': onHello(m); break;
    case 'status':
      if (m.elapsed_s != null) S.loadT0 = performance.now() - m.elapsed_s * 1000;
      if (m.state === 'error') banner('model', `모델을 불러오지 못했습니다: ${m.message}`, 'error');
      setState(m.state === 'warmup' ? 'warmup' : m.state, m.message); break;
    case 'ready':
      S.hello && (S.hello.model = m.model);
      setState('ready', `로드 ${num(m.load_ms / 1000, 1)} s · 워밍업 ${num(m.warmup_ms / 1000, 1)} s`);
      renderModel(m.model, m.load_ms, m.warmup_ms); break;
    case 'started': onStarted(m); break;
    case 'updated': record(m); break;
    case 'segment_start': record(m); onSegStart(m); break;
    case 'delta': record(m); onDelta(m); break;
    case 'segment_end': record(m); onSegEnd(m); break;
    case 'segment_final': record(m); onSegFinal(m); break;
    case 'stats': if (S.sid && m.session_id === S.sid) { record(m); onStats(m); } break;
    case 'vad': record(m); S.vad = m; kickMeter(); break;
    case 'session_end': onSessionEnd(m); break;
    case 'warning': record(m); onWarning(m); break;
    case 'error': record(m); onError(m); break;
    case 'pong': $('rtt-info').textContent = `RTT: ${num(performance.now() - m.t, 1)} ms`; break;
    default: break;
  }
}

function onHello(h) {
  const first = !S.hello;
  S.hello = h;
  if (first || h.state === 'ready') fillControls();
  renderModel(h.model, h.load_ms, h.warmup_ms);
  setState(h.state === 'loading' && h.phase === 'warmup' ? 'warmup' : h.state, h.message);
  $('server-info').textContent = `서버: v${h.version || '?'} · proto ${h.proto} · 엔진 ${h.model && h.model.engine}`;
}

function renderModel(md, loadMs, warmMs) {
  if (!md) return;
  const parts = [md.name || 'q06-s4-ast', md.engine === 'fake' ? 'FAKE' : 'MLX'];
  if (md.dtype) parts.push(md.dtype);
  $('model-chip').textContent = parts.join(' · ');
  $('model-chip').title = `폴더: ${md.dir || '—'}\nKV ${md.kv_capacity} · 인코더 ${md.encoder_dtype || '—'} · speculative ${md.speculative}` +
    (md.precision_note ? `\n${md.precision_note}` : '') +
    (loadMs != null ? `\n로드 ${loadMs} ms · 워밍업 ${warmMs} ms` : '');
}

// ── 컨트롤 ─────────────────────────────────────────────────────────────────────────────────────────────
function clampOpt(k, v) {               // 범위 밖은 끝값으로, 빈 칸·0·숫자 아님은 기본값으로
  const r = OPT_RANGE[k]; v = +v;
  if (!isFinite(v) || v === 0) return r.d;
  v = Math.min(r.hi, Math.max(r.lo, v));
  return k === 'end_pause_ms' ? Math.round(v) : v;
}

function fillControls() {
  const h = S.hello;
  const src = $('src-lang');
  const cur = src.value || S.prefs.src || 'Korean';
  src.textContent = '';
  for (const l of h.langs) src.append(new Option(LANG_KO[l] || l, l));
  src.value = h.langs.includes(cur) ? cur : h.langs[0];
  if (S.prefs.mode) setMode(S.prefs.mode);
  const o = S.prefs.options || {};
  let fixed = false;
  for (const k of Object.keys(OPT_RANGE)) {          // 저장된 값도 범위 검사 (예전 판이 저장한 잘못된 값 정리)
    if (!(k in o)) continue;
    const v = clampOpt(k, o[k]);
    if (v !== o[k]) { o[k] = v; fixed = true; }
    $(OPT_RANGE[k].id).value = v;
  }
  if (fixed) savePrefs();
  if (o.gain) $('opt-gain').checked = o.gain === 'auto';
  for (const [id, k] of [['opt-final', 'final_redecode'], ['opt-debug', 'debug_tokens'], ['opt-ec', 'ec'], ['opt-ns', 'ns'], ['opt-agc', 'agc']]) {
    if (k in o && !$(id).dataset.unsup) $(id).checked = !!o[k];
  }
  fillTargets(); fillDelays(true); applyModeUi();
  $('debug').hidden = !$('opt-debug').checked;
}

function fillTargets() {
  const src = $('src-lang').value; const tgt = $('tgt-lang'); const cur = tgt.value || S.prefs.tgt;
  tgt.textContent = '';
  for (const [a, b] of S.hello.pairs) if (a === src) tgt.append(new Option(LANG_KO[b] || b, b));
  const opts = [...tgt.options].map((x) => x.value);
  tgt.value = opts.includes(cur) ? cur : (opts[0] || '');
}

function defaultDelay() {
  const h = S.hello; const m = mode();
  if (m === 'asr') return S.prefs.asr_delay || h.defaults.asr_delay || 6;
  const k = pairKey($('src-lang').value, $('tgt-lang').value);
  return (S.prefs.tr_delay && S.prefs.tr_delay[k]) || (h.defaults.translate_delay || {})[k] || 10;
}

function fillDelays(reset) {
  const sel = $('delay'); const keep = reset ? defaultDelay() : +sel.value;
  sel.textContent = '';
  for (const d of S.hello.delays) {
    const t = S.hello.latency_table[String(d)];
    const o = new Option(`δ${d} · 약 ${(t.mean_ms / 1000).toFixed(2)} s`, d);
    o.title = `알고리즘 지연 ${(t.lo_ms / 1000).toFixed(2)}–${(t.hi_ms / 1000).toFixed(2)} s`;
    sel.append(o);
  }
  sel.value = String(S.hello.delays.includes(keep) ? keep : defaultDelay());
  renderDelayHint();
}

function laalFor(pair, d) {
  const t = (S.hello.laal_table || {})[pair]; if (!t) return null;
  if (t[String(d)]) return t[String(d)];
  const ks = Object.keys(t).map(Number).sort((a, b) => a - b);
  const lo = ks.filter((k) => k < d).pop(); const hi = ks.find((k) => k > d);
  if (lo == null || hi == null) return null;
  const s = t[lo].s + (t[hi].s - t[lo].s) * (d - lo) / (hi - lo);
  return { s, approx: true };
}

function renderDelayHint() {
  if (!S.hello) return;
  const d = +$('delay').value; const t = S.hello.latency_table[String(d)];
  let s = `알고리즘 지연 ≈ ${(t.mean_ms / 1000).toFixed(2)} s (${(t.lo_ms / 1000).toFixed(2)}–${(t.hi_ms / 1000).toFixed(2)})`;
  if (mode() === 'translate') {
    const l = laalFor(pairKey($('src-lang').value, $('tgt-lang').value), d);
    if (l) s += ` · 번역 기준 LAAL ${l.approx ? '≈' : ''}${l.s.toFixed(2)} s (평가)`;
  } else if (d < 8) {
    s += ' · δ8–10: 영어·독일어 약간 더 정확';
  }
  $('delay-hint').textContent = s;
}

function applyModeUi() {
  const tr = mode() === 'translate';
  $('tgt-lang').disabled = !tr;
  $('panes').classList.toggle('asr-only', !tr);
  const tgt = $('tgt-lang').value;
  $('tr-head').textContent = tr ? `번역 (${LANG_KO[tgt] || tgt})` : '번역';
  $('note-koen').hidden = !(tr && $('src-lang').value === 'Korean' && tgt === 'English');
  renderDelayHint();
}

function uiCfg() {
  const m = mode();
  return { src_lang: $('src-lang').value, mode: m, tgt_lang: m === 'translate' ? $('tgt-lang').value : null, delay: +$('delay').value };
}

function onCfgChange(kind) {
  if (!S.hello) return;
  if (kind === 'src') fillTargets();
  if (kind !== 'delay') fillDelays(true);
  applyModeUi();
  const c = uiCfg();
  S.prefs.src = c.src_lang; S.prefs.mode = c.mode; if (c.tgt_lang) S.prefs.tgt = c.tgt_lang;
  if (kind === 'delay') {
    if (c.mode === 'asr') S.prefs.asr_delay = c.delay;
    else { S.prefs.tr_delay = S.prefs.tr_delay || {}; S.prefs.tr_delay[pairKey(c.src_lang, c.tgt_lang)] = c.delay; }
  }
  savePrefs();
  if (S.run && !S.run.stopping) {               // 세션 중: 다음 세그먼트부터
    const u = { type: 'update', ...c }; if (!c.tgt_lang) delete u.tgt_lang;
    wsSend(u); S.pendingCfg = c; $('pending-chip').hidden = false;
  }
}

function optsFromUi() {
  return {
    end_pause_ms: clampOpt('end_pause_ms', $('opt-pause').value), hard_max_s: clampOpt('hard_max_s', $('opt-max').value),
    gain: $('opt-gain').checked ? 'auto' : 'off',
    final_redecode: $('opt-final').checked, debug_tokens: $('opt-debug').checked, save_audio: $('opt-save').checked,
    segmentation: $('opt-seg').value,
  };
}

function onOptChange(e) {
  const o = optsFromUi();
  for (const [k, r] of Object.entries(OPT_RANGE)) {   // 범위 밖 값은 고친 값을 칸에 다시 써서 보낼 값을 보여 준다
    if (e.target.id !== r.id) continue;
    const raw = String($(r.id).value).trim();
    if (raw === '' || +raw !== o[k]) {
      $(r.id).value = o[k];
      $('opt-msg').textContent = `${r.msg} (${raw === '' ? '빈 칸' : raw} → ${o[k]})`;
    } else $('opt-msg').textContent = '';
  }
  S.prefs.options = { ...o, ec: $('opt-ec').checked, ns: $('opt-ns').checked, agc: $('opt-agc').checked };
  delete S.prefs.options.segmentation; delete S.prefs.options.save_audio;
  savePrefs();
  $('debug').hidden = !o.debug_tokens;
  if (S.run && !S.run.stopping && ['opt-pause', 'opt-gain', 'opt-final', 'opt-debug'].includes(e.target.id)) {
    wsSend({ type: 'update', options: { end_pause_ms: o.end_pause_ms, gain: o.gain, final_redecode: o.final_redecode, debug_tokens: o.debug_tokens } });
    $('pending-chip').hidden = false;
  }
}

// ── 브라우저 마이크 처리 지원 여부 (Safari 는 잡음 억제·자동 이득 제약을 모른다) ───────────────────────────────
const CONSTRAINTS = [['opt-ec', 'echoCancellation', 'lbl-ec'], ['opt-ns', 'noiseSuppression', 'lbl-ns'], ['opt-agc', 'autoGainControl', 'lbl-agc']];
let SUP = {};
function applyConstraintSupport() {
  try { SUP = (navigator.mediaDevices && navigator.mediaDevices.getSupportedConstraints) ? navigator.mediaDevices.getSupportedConstraints() : {}; }
  catch (e) { SUP = {}; }
  if (!navigator.mediaDevices) return;                // 마이크 자체를 못 쓰는 주소: 시작할 때 따로 알린다
  for (const [id, k, lbl] of CONSTRAINTS) {
    if (SUP[k]) continue;
    const box = $(id); box.checked = false; box.dataset.unsup = '1'; box.disabled = true;
    $(lbl).textContent += ' (이 브라우저 미지원)';
  }
  // 에코 제거만 되는 브라우저(Safari): 에코 제거를 켜면 Apple 음성 처리가 자동 이득·잡음 억제까지 함께 켠다
  if (SUP.echoCancellation && !SUP.noiseSuppression) $('lbl-ec').textContent = '에코 제거 (Safari 음성 처리 — 자동 이득·잡음 억제 포함)';
}
function micConstraints(dev) {
  const a = { channelCount: 1 };
  if (dev) a.deviceId = { exact: dev };
  else if (S.prefs.mic) a.deviceId = { ideal: S.prefs.mic };   // 권한 전이라 목록에 없던 저장 장치: 있으면 쓰고 없으면 기본
  for (const [id, k] of CONSTRAINTS) if (SUP[k]) a[k] = $(id).checked;
  return a;
}
const setting = (st, k) => (k in st ? st[k] : '미지원');

// ── 오디오 캡처 ─────────────────────────────────────────────────────────────────────────────────────────
async function makeContext(rate = 0) {
  let ctx = null;
  if (rate) { try { ctx = new AudioContext({ latencyHint: 'interactive', sampleRate: rate }); } catch (e) { ctx = null; } }
  if (!ctx) ctx = new AudioContext({ latencyHint: 'interactive' });
  if (ctx.sampleRate > MAX_RATE) {      // 출력 장치가 176.4/192 kHz: 서버 한도 안의 48 kHz 컨텍스트로 (마이크·파일은 브라우저가 맞춘다)
    ctx.close().catch(() => {});
    ctx = new AudioContext({ latencyHint: 'interactive', sampleRate: 48000 });
  }
  try {
    await ctx.resume();
    await ctx.audioWorklet.addModule('/static/worklet.js');
  } catch (e) { ctx.close().catch(() => {}); throw e; }
  const node = new AudioWorkletNode(ctx, 'capture', {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
    processorOptions: { packet: Math.round(ctx.sampleRate * 0.02) },
  });
  const sink = ctx.createGain(); sink.gain.value = 0;      // 0 이득 싱크: 노드를 계속 돌리되 소리는 내지 않는다
  node.connect(sink); sink.connect(ctx.destination);
  node.port.onmessage = onPacket;
  return { ctx, node, sink };
}

function onPacket(e) {
  const d = e.data;
  if (d.flushed) { if (S.cap && S.cap.onFlushed) S.cap.onFlushed(); return; }
  if (!d.pcm) return;
  meterUpdate(d.peak, d.rms);
  if (!S.zero.done && S.run && S.run.kind === 'mic') {
    if (d.peak > 0) S.zero.nonzero = true;
    if (performance.now() - S.zero.t0 > 1000) {
      S.zero.done = true;
      if (!S.zero.nonzero) banner('zero_input', '마이크 입력이 0 입니다 — 시스템 설정 > 개인정보 보호 및 보안 > 마이크에서 브라우저를 허용하세요', 'error');
    }
  }
  if (S.sending && S.wsOpen) {
    let pcm = d.pcm;
    if (S.cap && S.cap.limit != null) {           // 파일 1×: 마지막 렌더 퀀텀의 0 채움은 보내지 않는다 (파일 길이까지만)
      const left = S.cap.limit - S.sentSamples;
      if (left <= 0) return;
      if (pcm.byteLength / 4 > left) pcm = pcm.slice(0, left * 4);
    }
    S.ws.send(pcm);
    S.sentSamples += pcm.byteLength / 4;
  }
}

function teardownCapture() {
  const c = S.cap; S.cap = null; S.sending = false;
  kickMeter();
  if (!c) return;
  try { if (c.src && c.src.stop) c.src.stop(); } catch (e) { /* 이미 멈춤 */ }
  try { c.node.port.postMessage({ cmd: 'close' }); c.node.disconnect(); } catch (e) { /* 무시 */ }
  if (c.stream) c.stream.getTracks().forEach((t) => t.stop());
  c.ctx.close().catch(() => {});
}

function inputPathText(rate, ctx, track) {
  let base = rate >= 88200 ? '≈ 53–99 ms' : rate > 16000 ? '≈ 31–57 ms' : '≈ 20 ms';
  const extra = [];
  if (ctx && ctx.baseLatency) extra.push(`버퍼 ${Math.round(ctx.baseLatency * 1000)} ms`);
  const st = track && track.getSettings ? track.getSettings() : {};
  if (st.latency) extra.push(`장치 ${Math.round(st.latency * 1000)} ms`);
  return `입력 경로 ${base}${extra.length ? ' (+' + extra.join(', ') + ')' : ''}`;
}

function startWaiter() {
  return new Promise((resolve, reject) => { S.startWait = { resolve, reject }; });
}

function sendStart(source, rate, extraOpts, takeover) {
  const c = uiCfg(); const o = optsFromUi();
  const options = {
    segmentation: source === 'file' ? o.segmentation : 'vad', end_pause_ms: o.end_pause_ms, hard_max_s: o.hard_max_s, gain: o.gain,
    final_redecode: o.final_redecode, debug_tokens: o.debug_tokens, save_audio: o.save_audio, ...extraOpts,
  };
  if (options.segmentation === 'none') options.gain = 'off';
  const msg = { type: 'start', ...c, source, sample_rate: rate, format: 'f32le', takeover: !!takeover, options };
  if (!c.tgt_lang) delete msg.tgt_lang;
  if (S.run) S.run.startSent = true;
  S.sid = null;                       // 이전 세션의 늦은 stats·session_end 가 새 실행에 섞이지 않게 (started 에서 새 id)
  S.events = []; S.stats = null; S.sentSamples = 0; S.sentRate = rate; S.lagTicks = 0;
  const w = startWaiter();
  wsSend(msg);
  return w;
}

function newRun(kind, extra) {
  const run = { kind, stopping: false, startSent: false, sid: null, ...extra };
  S.run = run; updateButtons(); renderState(); kickMeter();
  return run;
}

// getUserMedia·워크릿 오류 → 한국어 안내 (stage: ctx = 오디오 컨텍스트·워크릿, gum = 장치 열기, graph = 연결)
function micErrText(e, stage) {
  const n = (e && e.name) || ''; const msg = (e && e.message) || String(e);
  if (stage === 'ctx') return `오디오 처리 모듈을 불러오지 못했습니다 — 서버 연결을 확인하고 페이지를 새로 고치세요 (${n || msg})`;
  if (n === 'NotAllowedError' || n === 'SecurityError') {
    return /system/i.test(msg)
      ? '마이크 권한이 시스템에서 막혀 있습니다 — 시스템 설정 › 개인정보 보호 및 보안 › 마이크에서 브라우저를 켠 뒤 브라우저를 다시 시작하세요'
      : '마이크 권한이 거부되었습니다 — 주소창 왼쪽 사이트 설정에서 마이크를 허용하세요 (시스템 설정 › 개인정보 보호 및 보안 › 마이크도 확인)';
  }
  if (n === 'NotFoundError') return '마이크 장치가 없습니다 — 외부 마이크를 연결하세요 (덮개를 닫은 MacBook 등)';
  if (n === 'NotReadableError' || (n === 'AbortError' && stage === 'gum')) return '마이크를 열 수 없습니다 — 다른 앱이 마이크를 쓰고 있지 않은지 확인하세요';
  if (stage === 'graph' && n === 'NotSupportedError') {
    return '이 브라우저는 장치와 다른 표본율로 마이크를 연결하지 못합니다 — Audio MIDI 설정에서 출력 장치를 96 kHz 이하(48 kHz 권장)로 바꾸세요';
  }
  return `마이크를 시작하지 못했습니다: ${n && !msg.includes(n) ? n + ': ' : ''}${msg}`;
}

async function startMic(takeover = false) {
  if (S.run) return;
  if (!navigator.mediaDevices || !window.AudioWorkletNode) {
    banner('secure', '이 주소에서는 마이크를 쓸 수 없습니다 — http://127.0.0.1 로 여세요', 'error'); return;
  }
  unbanner('busy'); unbanner('zero_input'); unbanner('mic');
  const run = newRun('mic');
  let ctx = null; let stream = null; let stage = 'ctx';
  // 시작 도중 정지·중단·다른 실행으로 바뀌면 이 실행이 만든 것만 정리하고 빠진다 (다음 실행의 S.cap 은 건드리지 않는다)
  const stale = () => S.run !== run || run.stopping;
  const dropLocal = () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (S.cap && S.cap.ctx === ctx) teardownCapture(); else if (ctx) ctx.close().catch(() => {});
  };
  try {
    const c = await makeContext();
    ctx = c.ctx;
    if (stale()) { dropLocal(); return; }
    S.cap = { ctx, node: c.node, sink: c.sink, stream: null, src: null };   // 실패해도 teardownCapture 가 컨텍스트를 닫도록 먼저 둔다
    stage = 'gum';
    const dev = $('mic').value;
    stream = await navigator.mediaDevices.getUserMedia({ audio: micConstraints(dev) });
    if (stale()) { dropLocal(); return; }
    S.cap.stream = stream;
    stage = 'graph';
    const src = ctx.createMediaStreamSource(stream);
    src.connect(c.node);
    stage = 'start';
    const track = stream.getAudioTracks()[0];
    const st = track.getSettings ? track.getSettings() : {};
    $('track-info').textContent = `트랙 설정: ${track.label || '?'} · ${st.sampleRate || ctx.sampleRate} Hz · ${st.channelCount || '?'}ch · ` +
      `EC ${setting(st, 'echoCancellation')} · NS ${setting(st, 'noiseSuppression')} · AGC ${setting(st, 'autoGainControl')}` +
      (st.latency ? ` · latency ${Math.round(st.latency * 1000)} ms` : '') + ` · 컨텍스트 ${ctx.sampleRate} Hz`;
    $('st-input').textContent = inputPathText(ctx.sampleRate, ctx, track);
    if (/bluetooth|airpods|hands-?free|핸즈프리|headset/i.test(track.label || '')) {
      banner('bt', '블루투스 헤드셋 마이크는 협대역(8–16 kHz 코덱)이라 정확도가 떨어집니다');
    }
    track.addEventListener('mute', () => { if (S.run === run) banner('mute', '마이크가 음소거되었습니다(다른 앱 점유 또는 OS 권한)'); });
    track.addEventListener('unmute', () => unbanner('mute'));
    track.addEventListener('ended', () => { if (S.run === run) { banner('mute', '마이크 장치가 끊겼습니다', 'error'); stopRun(); } });
    // 목록은 실제로 열린 장치를 보여 준다 (요청한 장치가 아니면 '기본 마이크')
    listMics(st.deviceId && (st.deviceId === dev || st.deviceId === S.prefs.mic) ? st.deviceId : '');
    S.zero = { t0: performance.now(), nonzero: false, done: false };
    S.retryStart = () => startMic(true);
    S.sending = true;
    await sendStart('mic', ctx.sampleRate, {}, takeover);
  } catch (e) {
    dropLocal();
    if (S.run !== run) return;          // 이미 끝난 실행: 배너·endRun 없이 조용히
    endRun();
    if (e && (e.server || e.cancelled)) return;          // 서버 오류는 onError 가 배너를 띄운다
    if (e && e.name === 'OverconstrainedError') {        // 고른 장치가 사라짐: 기본 마이크로 되돌린다
      $('mic').value = ''; delete S.prefs.mic; savePrefs();
      banner('mic', '선택한 마이크를 찾을 수 없어 기본 마이크로 바꿨습니다 — 다시 시작하세요', 'error');
      return;
    }
    banner('mic', micErrText(e, stage), 'error');
  }
}

// ── 파일 읽기 ──────────────────────────────────────────────────────────────────────────────────────────
// WAV(RIFF PCM/float, WAVE_FORMAT_EXTENSIBLE 포함)를 직접 읽는다: soundfile(dtype='float32') 와 비트 동일
// (u8 (x−128)/128, int16/2^15, int24/2^23, int32/2^31, float32 그대로, float64 는 float32 로 반올림). 여러 채널은 평균(워크릿과 같은 방식).
function parseWav(buf) {
  const n = buf.byteLength; if (n < 12) return null;
  const dv = new DataView(buf);
  const tag = (o) => String.fromCharCode(dv.getUint8(o), dv.getUint8(o + 1), dv.getUint8(o + 2), dv.getUint8(o + 3));
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') return null;
  let fmt = null; let data = null; let o = 12;
  while (o + 8 <= n && !(fmt && data)) {
    const id = tag(o); let sz = dv.getUint32(o + 4, true); const body = o + 8;
    if (id === 'fmt ' && sz >= 16 && body + 16 <= n) {
      let code = dv.getUint16(body, true);
      if (code === 0xfffe && sz >= 26 && body + 26 <= n) code = dv.getUint16(body + 24, true);   // 확장형: 하위 형식 GUID 앞 2 바이트
      fmt = { code, ch: dv.getUint16(body + 2, true), rate: dv.getUint32(body + 4, true), align: dv.getUint16(body + 12, true), bits: dv.getUint16(body + 14, true) };
    } else if (id === 'data') {
      if (body + sz > n) sz = n - body;                // 크기 미기록·잘린 파일: 남은 바이트만
      data = { off: body, len: sz };
    }
    o = body + sz + (sz & 1);
  }
  if (!fmt || !data) return null;
  const { code, ch, rate, bits } = fmt; const bps = bits >> 3;
  const kind = code === 1 && [8, 16, 24, 32].includes(bits) ? 'int' : code === 3 && (bits === 32 || bits === 64) ? 'float' : null;
  if (!kind || !ch || !rate || fmt.align !== ch * bps) return null;
  const rd = kind === 'float' ? (bits === 32 ? (p) => dv.getFloat32(p, true) : (p) => dv.getFloat64(p, true))
    : bits === 16 ? (p) => dv.getInt16(p, true) / 32768
      : bits === 24 ? (p) => (dv.getUint8(p) | (dv.getUint8(p + 1) << 8) | (dv.getInt8(p + 2) << 16)) / 8388608
        : bits === 32 ? (p) => dv.getInt32(p, true) / 2147483648
          : (p) => (dv.getUint8(p) - 128) / 128;
  const frame = ch * bps; const nf = Math.floor(data.len / frame);
  const pcm = new Float32Array(nf);
  let p = data.off;
  if (ch === 1) {
    for (let i = 0; i < nf; i++, p += frame) pcm[i] = rd(p);
  } else {
    const ch32 = new Float32Array(1);                   // 채널 값은 먼저 float32 (soundfile 과 같은 값) 로
    for (let i = 0; i < nf; i++, p += frame) {
      let v = 0;
      for (let c = 0; c < ch; c++) { ch32[0] = rd(p + c * bps); v += ch32[0]; }
      pcm[i] = v / ch;
    }
  }
  return { pcm, rate, ch, bits, kind };
}

function monoOf(ab) {                    // AudioBuffer → 모노 Float32 (워크릿과 같은 방식: 배정밀도 합 / 채널 수)
  const n = ab.length; const nch = ab.numberOfChannels;
  if (nch === 1) return ab.getChannelData(0);
  const chs = []; for (let c = 0; c < nch; c++) chs.push(ab.getChannelData(c));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) { let v = 0; for (let c = 0; c < nch; c++) v += chs[c][i]; out[i] = v / nch; }
  return out;
}

function deviceRate() {                  // 1× 경로가 쓰는 컨텍스트 레이트 (makeContext 와 같은 규칙)
  let r = 48000;
  try { const t = new AudioContext(); r = t.sampleRate; t.close().catch(() => {}); } catch (e) { /* 기본 48 kHz */ }
  return r > MAX_RATE ? 48000 : r;
}

// 빠르게·평가용: WAV 는 원래 레이트 그대로, 그 밖의 형식은 1× 와 같은 레이트로 브라우저 디코드 → 리샘플은 서버 soxr 한 곳
async function decodeFile(file) {
  const raw = await file.arrayBuffer();
  const w = parseWav(raw);
  if (w && rateOk(w.rate)) return { pcm: w.pcm, rate: w.rate, wav: true };
  const rate = deviceRate();
  const ab = await new OfflineAudioContext(1, 1, rate).decodeAudioData(raw);
  return { pcm: monoOf(ab), rate: ab.sampleRate, wav: false };
}

function filePathText(rate, wav) {
  if (wav) return rate === 16000 ? '입력 경로: WAV 16 kHz 직접(리샘플 없음)' : `입력 경로: WAV ${rate} Hz → 서버 soxr`;
  return `입력 경로: 브라우저 디코드 ${rate} Hz → 서버 soxr (원본과 비트 동일 아님)`;
}

function fileErrText(e) {
  const n = (e && e.name) || ''; const msg = (e && e.message) || String(e);
  if (n === 'EncodingError') return '이 형식은 브라우저가 디코드하지 못합니다 — WAV 로 바꾸거나 tools/client_file.py 를 쓰세요';
  return msg;
}

async function playFile1x(takeover = false) {
  const f = $('file').files[0]; if (!f || S.run) return;
  unbanner('busy'); unbanner('file');
  const run = newRun('file1x', { speed: 1 });
  let ctx = null;
  const stale = () => S.run !== run || run.stopping;
  const dropLocal = () => { if (S.cap && S.cap.ctx === ctx) teardownCapture(); else if (ctx) ctx.close().catch(() => {}); };
  try {
    $('file-state').textContent = '디코딩…';
    const raw = await f.arrayBuffer();
    if (stale()) return;
    const w = parseWav(raw);
    const want = w && rateOk(w.rate) ? w.rate : 0;      // WAV: 파일 레이트의 컨텍스트 → 빠르게와 같은 표본이 워크릿을 그대로 지난다
    const c = await makeContext(want);
    ctx = c.ctx;
    if (stale()) { dropLocal(); return; }
    S.cap = { ctx, node: c.node, sink: c.sink, src: null, stream: null };
    let buf; let wav = false;
    if (want && ctx.sampleRate === want) {
      if (!w.pcm.length) throw new Error('오디오 표본이 없습니다');
      buf = ctx.createBuffer(1, w.pcm.length, want); buf.copyToChannel(w.pcm, 0); wav = true;
    } else {
      buf = await ctx.decodeAudioData(raw);
      if (stale()) { dropLocal(); return; }
    }
    S.cap.limit = buf.length;
    const src = ctx.createBufferSource(); src.buffer = buf;
    src.connect(c.node);
    if ($('listen').checked) src.connect(ctx.destination);
    S.cap.src = src;
    $('st-input').textContent = `${filePathText(ctx.sampleRate, wav)} · ${inputPathText(ctx.sampleRate, ctx, null).replace('입력 경로 ', '')}`;
    S.retryStart = () => playFile1x(true);
    S.sending = true;
    const started = sendStart('file', ctx.sampleRate, { speed: 1, file_name: f.name, file_duration_s: +buf.duration.toFixed(3) }, takeover);
    src.addEventListener('ended', () => finishFile1x(run));
    src.start();
    $('file-state').textContent = `${f.name} · ${buf.duration.toFixed(1)} s 재생 중`;
    await started;
  } catch (e) {
    dropLocal();
    if (S.run !== run) return;
    endRun();
    if (e && (e.server || e.cancelled)) return;
    banner('file', `파일을 재생하지 못했습니다: ${fileErrText(e)}`, 'error');
  }
}

function finishFile1x(run) {
  const c = S.cap; if (!c || !S.run || S.run !== run || run.kind !== 'file1x') return;
  run.stopping = true; updateButtons(); renderState();
  c.onFlushed = () => { c.onFlushed = null; if (S.run === run) wsSend({ type: 'stop' }); teardownCapture(); };
  try { c.node.port.postMessage({ cmd: 'flush' }); } catch (e) { wsSend({ type: 'stop' }); teardownCapture(); }
}

async function playFileFast(takeover = false) {
  const f = $('file').files[0]; if (!f || S.run) return;
  const sel = $('fast-mode').value; const evalMode = sel === 'eval'; const speed = evalMode ? 0 : +sel;
  unbanner('busy'); unbanner('file');
  const run = newRun('fast', { speed });
  const fast = { stop: false, abort: false }; S.fast = fast;
  // 이 실행이 아직 살아 있나: 서버가 세션을 끝냈거나(가져가기·치명 오류) 다음 실행이 시작되면 보내기를 멈춘다
  const mine = () => S.run === run && S.fast === fast && !fast.abort && S.wsOpen;
  try {
    $('file-state').textContent = '디코딩…';
    const { pcm, rate, wav } = await decodeFile(f);
    if (!mine()) return;
    const dur = pcm.length / rate;
    if (!pcm.length) throw new Error('오디오 표본이 없습니다');
    if (evalMode && dur > 30) throw new Error(`평가 방식은 30 s 이하 파일만 됩니다 (${dur.toFixed(1)} s)`);
    if (evalMode && !wav) banner('file', '평가 방식: WAV 가 아니라 브라우저 디코더를 거칩니다 — 오프라인 평가와 표본이 다를 수 있습니다');
    $('st-input').textContent = filePathText(rate, wav);
    S.retryStart = () => playFileFast(true);
    const extra = { speed, file_name: f.name, file_duration_s: +dur.toFixed(3) };
    if (evalMode) Object.assign(extra, { segmentation: 'none', gain: 'off' });
    await sendStart('file', rate, extra, takeover);
    if (!mine()) return;
    const pkt = Math.round(rate * FAST_PKT_S);
    let i = 0; let t0 = performance.now(); let lastUi = 0;
    while (i < pcm.length && !fast.stop && mine()) {
      const p0 = performance.now();                       // 흐름 제어: 서버가 처리를 끝낸 오디오보다 10 s 이상 앞서지 않는다
      while (mine() && !fast.stop &&
             ((i / rate) - (S.stats ? (S.stats.audio_done_s || 0) : 0) > MAX_INFLIGHT_S || S.ws.bufferedAmount > MAX_BUFFERED)) {
        await psleep(50);
      }
      t0 += performance.now() - p0;                       // 멈춘 만큼 일정을 민다 (2×/4×)
      if (fast.stop || !mine()) break;
      if (speed > 0) {
        const wait = t0 + (i / rate) * 1000 / speed - performance.now();
        if (wait > 0) await psleep(wait);
        if (fast.stop || !mine()) break;                  // 기다리는 동안 끝났으면 한 패킷도 더 보내지 않는다
      }
      const chunk = pcm.slice(i, Math.min(pcm.length, i + pkt));
      S.ws.send(chunk.buffer); i += chunk.length; S.sentSamples = i;
      if (performance.now() - lastUi > 200) {
        lastUi = performance.now();
        $('file-state').textContent = `${f.name} · 보냄 ${(i / rate).toFixed(1)} / ${dur.toFixed(1)} s`;
      }
      if (speed === 0 && (i / pkt) % 8 === 0) await psleep(0);   // 최대 속도: 이벤트 루프 양보
    }
    if (mine()) {
      run.stopping = true; updateButtons(); renderState();
      wsSend({ type: 'stop' });
      $('file-state').textContent = `${f.name} · 보냄 ${(i / rate).toFixed(1)} s — 마무리 중`;
    }
  } catch (e) {
    if (S.run !== run) return;          // 이미 끝난 실행은 상태·배너를 건드리지 않는다
    if (e && (e.server || e.cancelled)) { endRun(); return; }
    banner('file', `파일을 처리하지 못했습니다: ${fileErrText(e)}`, 'error');
    if (run.startSent) wsSend({ type: 'abort' });
    endRun();
  }
}

function stopRun() {
  const run = S.run;
  if (!run || run.stopping) return;
  if (!run.startSent) {               // 서버에 아직 start 를 안 보냄: 알릴 것 없이 바로 끝낸다 (늦게 온 스트림은 시작 함수가 정리)
    run.stopping = true; teardownCapture(); endRun(); return;
  }
  if (run.kind === 'mic') {
    run.stopping = true; wsSend({ type: 'stop' }); teardownCapture();
  } else if (run.kind === 'file1x') {
    if (S.cap && S.cap.src) { try { S.cap.src.stop(); } catch (e) { finishFile1x(run); } } else finishFile1x(run);
  } else if (run.kind === 'fast' && S.fast) {
    S.fast.stop = true;
  }
  updateButtons(); renderState();
}

function abortRun() {
  const run = S.run;
  if (!run) return;
  if (S.fast) S.fast.abort = true;
  if (!run.startSent) { run.stopping = true; teardownCapture(); endRun(); return; }
  wsSend({ type: 'abort' });
  teardownCapture();
  run.stopping = true; updateButtons(); renderState();
  setTimeout(() => { if (S.run === run) endRun(); }, 1500);   // 서버 응답이 없을 때의 안전장치 (이 실행만)
}

function endRun() {
  if (S.fast) S.fast.abort = true;    // 빠르게 보내기 루프가 다음 세션으로 넘어가지 않게
  const w = S.startWait;
  S.run = null; S.fast = null; S.sending = false; S.startWait = null;
  $('pending-chip').hidden = true; $('note-hold').hidden = true; M.hold = false;
  updateButtons(); renderState(); kickMeter();
  if (w) w.reject({ cancelled: true });
}

async function listMics(using) {
  try {
    const ds = await navigator.mediaDevices.enumerateDevices();
    const sel = $('mic'); const cur = using !== undefined ? using : (sel.value || S.prefs.mic || '');
    sel.textContent = ''; sel.append(new Option('기본 마이크', ''));
    for (const d of ds) if (d.kind === 'audioinput' && d.deviceId) sel.append(new Option(d.label || `마이크 ${d.deviceId.slice(0, 6)}`, d.deviceId));
    sel.value = [...sel.options].some((o) => o.value === cur) ? cur : '';
  } catch (e) { /* 권한 전 */ }
}

// ── 세션 이벤트 ─────────────────────────────────────────────────────────────────────────────────────────
function onStarted(m) {
  S.sid = m.session_id; S.cfg = m.config;
  if (S.run && !S.run.sid) S.run.sid = m.session_id;
  S.events = [m];
  if ($('rows').children.length) {                // 세션 구분선 (세그먼트 번호가 세션마다 1 부터)
    const c = m.config; const o = c.options || {};
    const pair = c.mode === 'translate' ? `${LANG_KO[c.src_lang]}→${LANG_KO[c.tgt_lang]}` : `${LANG_KO[c.src_lang]} 전사`;
    $('rows').append(el('div', 'sess-div', `새 세션 ${new Date().toLocaleTimeString('ko-KR')} · ${pair} · δ${c.delay}` +
      (c.source === 'file' ? ` · 파일 ${o.file_name || ''}` : ' · 마이크')));
  }
  unbanner('session'); unbanner('model_not_ready');
  if (S.startWait) { S.startWait.resolve(m); S.startWait = null; }
  $('empty').hidden = true;
  renderState();
}

function rowKey(m) { return `${m.sid || S.sid}:${m.seg}`; }

function onSegStart(m) {
  S.segT = performance.now();
  const key = rowKey(m);
  const row = el('div', 'seg-row live');
  const meta = el('div', 'meta');
  meta.append(el('span', 'no', `#${m.seg}`), document.createTextNode(' '), el('span', 'time', `${fmtT(m.t_start)}–`), el('span', 'why', ''));
  const why = [];
  if (START_KO[m.reason]) why.push(START_KO[m.reason]);
  if (m.gain_db) why.push(`이득 ${m.gain_db > 0 ? '+' : ''}${m.gain_db} dB`);
  why.push(`δ${m.config.delay}`);
  meta.querySelector('.why').textContent = why.join(' · ');
  const src = el('div', 'cell src'); src.append(el('span', 'txt'));
  const tr = el('div', 'cell tr'); tr.append(el('span', 'txt'));
  row.append(meta, src, tr);
  $('rows').append(row);
  S.rows.set(key, { row, meta, src, tr, t_start: m.t_start, cfg: m.config, srcText: '', trText: '', final: null });
  if (S.liveKey && S.rows.get(S.liveKey)) S.rows.get(S.liveKey).row.classList.remove('live');
  S.liveKey = key;
  $('empty').hidden = true;
  if (S.pendingCfg && sameCfg(S.pendingCfg, m.config)) { S.pendingCfg = null; $('pending-chip').hidden = true; }
  else if (!S.pendingCfg) $('pending-chip').hidden = true;
  scrollMaybe();
}

function sameCfg(a, b) {
  return a.src_lang === b.src_lang && a.mode === b.mode && (a.mode !== 'translate' || a.tgt_lang === b.tgt_lang) && +a.delay === +b.delay;
}

function appendText(cell, text) {
  const sp = el('span', 'fresh', text);
  sp.addEventListener('animationend', () => {
    const p = sp.parentNode; if (!p) return;
    sp.replaceWith(document.createTextNode(sp.textContent)); p.normalize();
  }, { once: true });
  cell.querySelector('.txt').append(sp);
}

function onDelta(m) {
  const r = S.rows.get(rowKey(m));
  if (r) {
    if (m.src) { appendText(r.src, m.src); r.srcText += m.src; }
    if (m.tr) { appendText(r.tr, m.tr); r.trText += m.tr; }
    if (m.src || m.tr) scrollMaybe();
  }
  if (m.tokens && !$('debug').hidden) tokLog(m);
}

function onSegEnd(m) {
  const key = rowKey(m); const r = S.rows.get(key);
  if (!r) return;
  if (m.hidden) { r.row.remove(); S.rows.delete(key); if (S.liveKey === key) S.liveKey = null; return; }
  r.row.classList.remove('live');
  if (S.liveKey === key) S.liveKey = null;
  r.meta.querySelector('.time').textContent = `${fmtT(m.t_start)}–${fmtT(m.t_end)}`;
  const why = [REASON_KO[m.reason] || m.reason];
  if (m.finalize_ms != null) why.push(`${Math.round(m.finalize_ms)} ms`);
  if (m.gain_db) why.push(`이득 ${m.gain_db > 0 ? '+' : ''}${m.gain_db} dB`);
  why.push(`δ${m.delay}`);
  r.meta.querySelector('.why').textContent = why.join(' · ');
  r.meta.title = `길이 ${num(m.audio_s, 2)} s · 첫 글자 ${num(m.first_text_ms, 0)} ms · 마무리 ${num(m.finalize_ms, 0)} ms\n` +
    `zero tail ${num(m.zero_tail_ms, 0)} ms · 강제 라운드 ${m.forced_rounds} · flush ${m.flush_rounds} · K ${m.K ?? '—'}\n` +
    `인코더 합 ${num(m.enc_ms_sum, 0)} ms · 디코더 합 ${num(m.dec_ms_sum, 0)} ms · 토큰 원문 ${m.n_tokens.src} / 번역 ${m.n_tokens.tr}` +
    (m.n_tokens.tr_dropped ? ` (전사 모드에서 버린 번역 ${m.n_tokens.tr_dropped})` : '');
  // 서버 최종 텍스트가 delta 연결과 다르면(정상이면 같다) 최종 텍스트로 맞춘다
  if (m.src_text !== r.srcText) { r.src.querySelector('.txt').textContent = m.src_text; r.srcText = m.src_text; }
  if (m.tr_text !== r.trText) { r.tr.querySelector('.txt').textContent = m.tr_text; r.trText = m.tr_text; }
  r.end = m;
}

function onSegFinal(m) {
  const r = S.rows.get(rowKey(m)); if (!r) return;
  r.final = m;
  if (m.skipped) {
    const f = el('div', 'final', `건너뜀 (${m.skipped === 'lagging' ? '처리 밀림' : m.skipped})`); r.src.append(f); return;
  }
  r.src.append(el('div', 'final', m.src || '∅'));
  if (m.tr) r.tr.append(el('div', 'final', m.tr));
  scrollMaybe();
}

function onSessionEnd(m) {
  if (!S.sid || m.session_id !== S.sid) return;
  if (S.run && S.run.sid !== m.session_id) return;     // 다음 실행이 이미 시작됨: 그 실행은 건드리지 않는다
  S.events.push(m);
  if (S.liveKey && S.rows.get(S.liveKey)) S.rows.get(S.liveKey).row.classList.remove('live');
  S.liveKey = null;
  const s = `세션 끝 · 오디오 ${num(m.audio_s, 1)} s / ${num(m.wall_s, 1)} s · 세그먼트 ${m.segments} · RTF ${num(m.rtf, 2)}` +
    (m.aborted ? ` · 중단(버림 ${num(m.discarded_s, 2)} s)` : '');
  $('file-state').textContent = s;
  teardownCapture(); endRun();             // S.sid 는 JSON 저장을 위해 남겨 둔다
}

function onWarning(m) {
  const lvl = m.code === 'zero_input' || m.code === 'kv_guard' ? 'error' : 'warn';
  if (m.code === 'binary_outside_session') return;
  if (m.code === 'lagging') return;        // 밀림은 stats 로 매 틱 갱신되는 'lag-local' 배너 하나로 보인다 (서버 값은 10 s 마다라 낡음; 기록은 남음)
  banner(m.code, m.message, lvl);
}

function onError(m) {
  if (S.startWait && ['busy', 'bad_config', 'model_not_ready', 'internal'].includes(m.code)) {
    const w = S.startWait; S.startWait = null; teardownCapture();
    if (S.fast) S.fast.abort = true;
    endRun();
    w.reject({ server: true, code: m.code, message: m.message });
  }
  if (m.code === 'busy') {
    banner('busy', '다른 탭에서 사용 중입니다 —', 'warn', { label: '가져오기', fn: () => S.retryStart && S.retryStart() });
  } else if (m.code === 'preempted') {
    banner('busy', '다른 탭이 세션을 가져갔습니다', 'error');
    teardownCapture(); endRun();
  } else {
    banner(m.code, m.message, 'error');
    if (m.fatal && S.run) { if (S.fast) S.fast.abort = true; teardownCapture(); }
  }
}

// ── 상태 표시줄 ─────────────────────────────────────────────────────────────────────────────────────────
function setStat(id, text, cls) {
  const e = $(id); e.textContent = text; e.classList.remove('good', 'warn', 'bad'); if (cls) e.classList.add(cls);
}

function onStats(m) {
  S.stats = m;
  const realtime = !(S.run && S.run.kind === 'fast');
  if (realtime) {
    const p95 = m.proc_lag_ms.p95;
    setStat('st-lag', `처리 지연 ${num(m.proc_lag_ms.p50)}/${num(p95)} ms`, p95 == null ? null : p95 < 200 ? 'good' : p95 < 500 ? 'warn' : 'bad');
  } else {
    setStat('st-lag', `처리 속도 ${num(m.throughput_x, 1)}×`, null);
  }
  const b = m.backlog_s || 0;
  // 세그먼트 시작 때 VAD 프리롤(~0.6 s)이 한꺼번에 넘어가 한 틱 튀는 것은 밀림이 아니다:
  // 시작 직후 0.6 s 안의 틱은 세지 않고, 0.3 s 넘는 틱이 두 번 이어져야 밀림으로 본다
  const fresh = performance.now() - S.segT < 600;
  if (b <= 0.3) S.lagTicks = 0; else if (!fresh) S.lagTicks += 1;
  const lag = realtime && S.lagTicks >= 2;
  setStat('st-backlog', `밀림 ${b.toFixed(2)} s`, lag ? (b > 1 ? 'bad' : 'warn') : null);
  if (lag && b > 1 && S.run && !S.run.stopping) banner('lag-local', `처리가 실시간보다 ${b.toFixed(1)} s 늦습니다`);
  else unbanner('lag-local');
  const rtf = m.rtf_10s;
  setStat('st-rtf', `RTF ${num(rtf, 2)}`, rtf == null ? null : rtf < 0.5 ? 'good' : rtf < 0.9 ? 'warn' : 'bad');
  setStat('st-ms', `인코더 ${num(m.enc_ms.p50, 1)} ms/청크 · 디코더 ${num(m.dec_ms.p50, 1)} ms/라운드`);
  const c95 = m.chunk_ms.p95;
  $('st-chunk').textContent = `청크 p95 ${num(c95)}/320 ms`;
  const bar = $('st-budget');
  bar.style.width = `${Math.min(100, (c95 || 0) / 320 * 100)}%`;
  bar.className = c95 == null ? '' : c95 < 160 ? '' : c95 < 288 ? 'warn' : 'bad';
  setStat('st-tok', `토큰/s ${num(m.tok_per_s, 1)}`);
  setStat('st-kv', `KV ${num(m.kv_pct)} %`, m.kv_pct > 85 ? 'bad' : m.kv_pct > 60 ? 'warn' : null);
  setStat('st-mem', `메모리 ${m.mem_gb != null ? m.mem_gb.toFixed(1) + ' GB' : '—'}`);
  $('st-lag').title = `p50/p95/max ${num(m.proc_lag_ms.p50)}/${num(m.proc_lag_ms.p95)}/${num(m.proc_lag_ms.max)} ms · 이벤트 루프 p99 ${num(m.loop_lag_ms, 1)} ms`;
  $('st-backlog').title = `받음 ${num(m.audio_rx_s, 1)} s · 세그먼트 ${num(m.audio_seg_s, 1)} s · 건너뜀 ${num(m.audio_skipped_s, 1)} s · 보류 ${num(m.audio_held_s, 2)} s · 처리 ${num(m.audio_proc_s, 1)} s · 버림 ${num(m.dropped_s, 2)} s`;
}

// ── 레벨 미터 ──────────────────────────────────────────────────────────────────────────────────────────
// rAF 는 그릴 것이 있을 때만 돈다 (입력·실행 중, 막대 하강·클립 LED 가 남았을 때). 색은 캐시, DOM 은 값이 바뀔 때만 쓴다.
const M = { cur: -120, disp: -90, peak: -90, peakT: 0, clipT: -1e9, last: performance.now(), raf: 0, col: null, w: 0, h: 0, dpr: 0,
  txt: null, clip: null, hold: false };
const toDb = (x) => 20 * Math.log10(Math.max(x, 1e-9));
function meterUpdate(peak, rms) {
  const now = performance.now(); const db = toDb(rms); const pk = toDb(peak);
  M.cur = db;
  if (db > M.disp) M.disp = db;                       // 즉시 상승
  if (pk > M.peak || now - M.peakT > 1000) { M.peak = pk; M.peakT = now; }   // 1 s 피크 유지
  if (peak >= 0.999) M.clipT = now;
  kickMeter();
}
function kickMeter() {
  if (M.raf) return;
  M.last = performance.now();
  M.raf = requestAnimationFrame(drawMeter);
}
function meterColors() {
  const cs = getComputedStyle(document.documentElement); const g = (n) => cs.getPropertyValue(n).trim();
  M.col = { bg: g('--meter-bg'), silence: g('--meter-silence'), speech: g('--meter-speech'), hang: g('--meter-hang'), mark: g('--meter-mark') };
}
function meterResize() { M.w = 0; M.col = null; kickMeter(); }
function drawMeter() {
  M.raf = 0;
  const cv = $('meter'); const dpr = window.devicePixelRatio || 1;
  if (!M.w || M.dpr !== dpr) {                       // 크기는 창 크기·화면 배율이 바뀔 때만 다시 잰다
    M.w = Math.max(60, cv.clientWidth); M.h = cv.clientHeight || 18; M.dpr = dpr;
    cv.width = Math.round(M.w * dpr); cv.height = Math.round(M.h * dpr);
  }
  if (!M.col) meterColors();
  const w = M.w; const h = M.h; const C = M.col;
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const now = performance.now(); const dt = (now - M.last) / 1000; M.last = now;
  if (!S.cap) { M.cur = -120; }
  M.disp = Math.max(M.cur, M.disp - 20 * dt);          // 20 dB/s 하강
  const x = (db) => Math.max(0, Math.min(1, (db + 90) / 90)) * w;
  g.clearRect(0, 0, w, h);
  g.fillStyle = C.bg; g.fillRect(0, 0, w, h);
  const v = S.vad; const st = v && S.run ? v.state : 'silence';
  g.fillStyle = st === 'speech' ? C.speech : st === 'hangover' ? C.hang : C.silence;
  g.fillRect(0, 3, x(M.disp), h - 6);
  g.fillStyle = C.mark;
  if (S.cap && M.peak > -90) g.fillRect(x(M.peak) - 1, 1, 2, h - 2);
  if (v && S.run) {
    g.globalAlpha = 0.55;
    if (v.floor_db != null) g.fillRect(x(v.floor_db), 0, 1, h);
    if (v.on_db != null) { g.fillRect(x(v.on_db) - 0.5, 0, 1, 4); g.fillRect(x(v.on_db) - 0.5, h - 4, 1, 4); }
    g.globalAlpha = 1;
  }
  const clip = now - M.clipT < 1000;
  if (clip !== M.clip) { M.clip = clip; $('clip-led').classList.toggle('on', clip); }
  const parts = [];
  if (S.cap) parts.push(`${num(M.disp > -89 ? M.disp : null, 0)} dBFS`);
  if (v && S.run) {
    parts.push(`바닥 ${num(v.floor_db, 0)}`);
    parts.push(`이득 ${v.gain_db > 0 ? '+' : ''}${num(v.gain_db, 0)} dB`);
    parts.push(v.calibrating ? '주변 소음 측정 중…' : (VAD_KO[v.state] || v.state));
  }
  const txt = parts.length ? parts.join(' · ') : '—';
  if (txt !== M.txt) { M.txt = txt; $('meter-text').textContent = txt; }
  // 긴 발화 대기 표시: 열린 세그먼트가 hold-back 구간에 들어감
  const hold = !!(v && S.run && v.hold);
  if (hold !== M.hold) { M.hold = hold; $('note-hold').hidden = !hold; }
  if (S.cap || S.run || M.disp > -90 || clip) M.raf = requestAnimationFrame(drawMeter);
}

// ── 스크롤, 내보내기, 토큰 로그 ───────────────────────────────────────────────────────────────────────────
function scrollMaybe() {
  const w = $('panes-wrap');
  if (S.autoScroll) w.scrollTop = w.scrollHeight;
}
function onScroll() {
  const w = $('panes-wrap');
  const atBottom = w.scrollHeight - w.scrollTop - w.clientHeight < 40;
  S.autoScroll = atBottom;
  $('to-bottom').hidden = atBottom;
}

function exportText() {
  const lines = [];
  for (const r of S.rows.values()) {
    if (!r.srcText && !r.trText) continue;
    lines.push(`[${fmtT(r.t_start)}] ${r.srcText}`);
    if (r.trText) lines.push(`→ ${r.trText}`);
  }
  return lines.join('\n');
}
async function copyText() {
  const t = exportText();
  try { await navigator.clipboard.writeText(t); }
  catch (e) {
    const ta = el('textarea'); ta.value = t; document.body.append(ta); ta.select();
    try { document.execCommand('copy'); } catch (e2) { /* 무시 */ }
    ta.remove();
  }
  toolMsg(t ? `${t.split('\n').length} 줄 복사함` : '복사할 텍스트가 없습니다');
}
function saveJson() {
  const data = { exported: new Date().toISOString(), session_id: S.sid, model: S.hello && S.hello.model, events: S.events };
  const blob = new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' });
  const a = el('a'); a.href = URL.createObjectURL(blob); a.download = `vapkt-${S.sid || 'session'}.json`;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  toolMsg(`이벤트 ${S.events.length} 개 저장`);
}
function toolMsg(t) { $('tool-msg').textContent = t; setTimeout(() => { if ($('tool-msg').textContent === t) $('tool-msg').textContent = ''; }, 3000); }
function clearRows() {
  if (S.run) return;
  $('rows').textContent = ''; S.rows.clear(); S.liveKey = null; $('empty').hidden = false; $('tok-log').textContent = '';
}

function tokLog(m) {
  const log = $('tok-log');
  const line = el('div', 'tk-line');
  line.append(document.createTextNode(`#${m.seg} k=${m.k}${m.flush ? ' flush' : m.tail ? ' tail' : ''} ${num(m.proc_lag_ms)} ms `));
  if (m.forced) line.append(el('span', 'tk-forced', '[forced] '));
  for (const [piece, stream] of m.tokens) line.append(el('span', `tk-${stream}`, piece === ' ' ? '␠' : piece), document.createTextNode(' '));
  log.append(line);
  while (log.childNodes.length > 3000) log.firstChild.remove();
  log.scrollTop = log.scrollHeight;
}

// ── 단축키: Space 시작·정지, Esc 중단 ──────────────────────────────────────────────────────────────────────
function onKey(e) {
  if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey || e.isComposing) return;   // OS·브라우저 단축키는 그대로
  const t = e.target; const tag = (t && t.tagName) || '';
  const radio = tag === 'INPUT' && t.type === 'radio';   // 모드 라디오: Space 는 이미 고른 것을 다시 고를 뿐이라 단축키로 쓴다
  const field = !radio && (['INPUT', 'SELECT', 'TEXTAREA'].includes(tag) || !!(t && t.isContentEditable));
  if (e.key === 'Escape') {
    if (!S.run) return;
    e.preventDefault();
    if (field) { t.blur(); return; }                    // 입력란의 첫 Esc 는 칸에서 나가기만 (중단 아님)
    if (!e.repeat) abortRun();
    return;
  }
  if (e.key === ' ' || e.code === 'Space') {
    if (field || tag === 'BUTTON') return;              // 컨트롤의 기본 Space 동작 유지
    e.preventDefault();                                 // 누르고 있어도 화면이 스크롤되지 않게
    if (e.repeat) return;                               // 자동 반복은 무시
    if (S.run) stopRun(); else if (S.state === 'ready' && S.wsOpen) startMic(false);
  }
}

// ── 이벤트 연결 ─────────────────────────────────────────────────────────────────────────────────────────
function bind() {
  $('src-lang').addEventListener('change', () => onCfgChange('src'));
  $('tgt-lang').addEventListener('change', () => onCfgChange('tgt'));
  $('delay').addEventListener('change', () => onCfgChange('delay'));
  document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener('change', () => onCfgChange('mode')));
  for (const id of ['opt-ec', 'opt-ns', 'opt-agc', 'opt-gain', 'opt-seg', 'opt-pause', 'opt-max', 'opt-final', 'opt-debug', 'opt-save']) {
    $(id).addEventListener('change', onOptChange);
  }
  $('mic').addEventListener('change', () => { S.prefs.mic = $('mic').value; savePrefs(); });
  $('btn-mic').addEventListener('click', () => (S.run ? stopRun() : startMic(false)));
  $('btn-play').addEventListener('click', () => (S.run ? stopRun() : playFile1x(false)));
  $('btn-fast').addEventListener('click', () => (S.run ? stopRun() : playFileFast(false)));
  $('btn-abort').addEventListener('click', abortRun);
  $('file').addEventListener('change', () => { updateButtons(); $('file-state').textContent = ''; });
  $('btn-adv').addEventListener('click', () => {
    const adv = $('adv'); adv.hidden = !adv.hidden;
    $('btn-adv').textContent = adv.hidden ? '▸ 고급' : '▾ 고급'; $('btn-adv').setAttribute('aria-expanded', String(!adv.hidden));
  });
  $('panes-wrap').addEventListener('scroll', onScroll, { passive: true });
  $('to-bottom').addEventListener('click', () => { S.autoScroll = true; scrollMaybe(); $('to-bottom').hidden = true; });
  $('btn-copy').addEventListener('click', copyText);
  $('btn-json').addEventListener('click', saveJson);
  $('btn-clear').addEventListener('click', clearRows);
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) navigator.mediaDevices.addEventListener('devicechange', () => listMics());
  document.addEventListener('keydown', onKey);
  window.addEventListener('resize', meterResize);
  if (window.matchMedia) {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', meterResize);
  }
  document.addEventListener('visibilitychange', () => {   // 워커 페이서가 없을 때만: 숨은 탭에서는 빠르게 재생이 느려진다
    if (document.hidden && S.run && S.run.kind === 'fast' && P.ok === false) banner('bg', '탭을 앞에 두면 빠르게 재생이 더 빠릅니다');
    else if (!document.hidden) unbanner('bg');
  });
  $('debug').hidden = !$('opt-debug').checked;
}

applyConstraintSupport();
bind();
listMics();
connect();
kickMeter();
