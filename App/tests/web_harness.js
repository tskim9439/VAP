// web/app.js 를 node vm 에 브라우저 흉내(DOM·WebSocket·AudioContext·getUserMedia)와 함께 올려 시나리오 하나를 돌리고 JSON 한 줄을 낸다.
// test_web_client.py 가 부른다: node web_harness.js <app.js> <시나리오> [인자…]
'use strict';
const fs = require('fs'); const vm = require('vm');
const [APP, SCEN, ...ARGS] = process.argv.slice(2);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── 아주 작은 DOM 흉내: 무엇이든 받아 주는 Proxy (값은 저장한다) ──────────────────────────────────────────
function stub(name) {
  const t = function () { return stub(name + '()'); };
  return new Proxy(t, {
    get(o, k) {
      if (k in o) return o[k];
      if (k === Symbol.toPrimitive) return () => '';
      if (k === 'files') return o.files || [];
      if (k === 'children' || k === 'options' || k === 'childNodes') return [];
      if (k === 'checked' || k === 'hidden' || k === 'disabled') return o[k] || false;
      if (k === 'value') return o.value !== undefined ? o.value : '';
      if (k === 'textContent') return o.textContent || '';
      if (k === 'querySelector') return () => stub('q');
      if (k === 'querySelectorAll') return () => [];
      if (k === 'classList') { o.cls = o.cls || new Set(); const c = o.cls; return { add: (x) => c.add(x), remove: (...x) => x.forEach((y) => c.delete(y)), toggle: (x, on) => (on ? c.add(x) : c.delete(x)), contains: (x) => c.has(x) }; }
      if (k === 'dataset' || k === 'style') { o[k] = o[k] || {}; return o[k]; }
      if (k === 'getContext') return () => stub('ctx2d');
      if (k === 'clientWidth') return 300;
      if (k === 'clientHeight') return 18;
      return stub(name + '.' + String(k));
    },
    set(o, k, v) { o[k] = v; return true; },
  });
}
const els = {};
const $el = (id) => (els[id] = els[id] || stub('#' + id));

// ── WebSocket 흉내: 보낸 것을 모두 적고, 원하면 start 에 started 로 답한다 ─────────────────────────────────
const H = {
  sent: [], ws: null, autoStarted: true, nStart: 0, deviceRate: 48000, ctxs: [], streams: [], gumCalls: [], gumDelay: 0, gumErr: null,
  addModuleErr: null, addModuleDelay: 0, msErr: null, raf: 0, ctxRates: [],
  sup: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, deviceId: true },
  settings: { deviceId: 'dev-default', sampleRate: 48000, channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
};
class WS {
  constructor(url) { this.url = url; this.l = {}; this.bufferedAmount = 0; this.n = ++WS.n; H.ws = this; }
  addEventListener(t, f) { (this.l[t] = this.l[t] || []).push(f); }
  send(d) {
    H.sent.push({ ws: this.n, d });
    if (typeof d === 'string' && H.autoStarted) {
      const m = JSON.parse(d);
      if (m.type === 'start') {
        const sid = 'S' + (++H.nStart);
        setTimeout(() => this.msg({ type: 'started', session_id: sid, config: { src_lang: 'English', mode: 'asr', delay: 6, options: {} }, input_rate: m.sample_rate }), 5);
      }
    }
  }
  fire(t, ev) { (this.l[t] || []).forEach((f) => f(ev)); }
  msg(o) { this.fire('message', { data: JSON.stringify(o) }); }
}
WS.n = 0;

// ── 오디오 흉내 ─────────────────────────────────────────────────────────────────────────────────────────
class Track {
  constructor() { this.readyState = 'live'; this.label = 'Fake Mic'; }
  stop() { this.readyState = 'ended'; }
  getSettings() { return { ...H.settings }; }
  addEventListener() {}
}
class Stream {
  constructor() { this.t = [new Track()]; H.streams.push(this); }
  getTracks() { return this.t; }
  getAudioTracks() { return this.t; }
}
class Ctx {
  constructor(o) {
    this.sampleRate = (o && o.sampleRate) || H.deviceRate; this.state = 'running'; this.baseLatency = 0.01; this.destination = {};
    H.ctxs.push(this); H.ctxRates.push(this.sampleRate);
    this.audioWorklet = { addModule: async () => { if (H.addModuleDelay) await sleep(H.addModuleDelay); if (H.addModuleErr) throw H.addModuleErr; } };
  }
  async resume() {}
  close() { this.state = 'closed'; return Promise.resolve(); }
  createGain() { return { gain: { value: 1 }, connect() {} }; }
  createMediaStreamSource() { if (H.msErr) throw H.msErr; return { connect() {} }; }
}
class Node { constructor() { this.port = { postMessage() {}, onmessage: null }; } connect() {} disconnect() {} }

const banners = new Map();
const ctx = {
  console, setTimeout, clearTimeout, setInterval, clearInterval, Promise, JSON, Math, Date, Map, Set, Float32Array, Int16Array, DataView,
  ArrayBuffer, Uint8Array, Error, DOMException, String, Number, Object, isFinite,
  performance: { now: () => Number(process.hrtime.bigint() / 1000n) / 1000 },
  localStorage: { store: {}, getItem(k) { return this.store[k] ?? null; }, setItem(k, v) { this.store[k] = String(v); } },
  document: {
    getElementById: $el, hidden: false,
    querySelector: (q) => { if (q.includes('name="mode"')) { const x = stub('radio'); x.value = 'asr'; return x; } return null; },
    querySelectorAll: () => [], createElement: (t) => stub('<' + t + '>'), createTextNode: () => stub('text'), addEventListener() {}, body: stub('body'),
    documentElement: {},
  },
  getComputedStyle: () => ({ getPropertyValue: () => '#000' }),
  navigator: {
    mediaDevices: {
      getUserMedia: async (c) => { H.gumCalls.push(c); if (H.gumDelay) await sleep(H.gumDelay); if (H.gumErr) throw H.gumErr; return new Stream(); },
      enumerateDevices: async () => [], getSupportedConstraints: () => ({ ...H.sup }), addEventListener() {},
    },
  },
  location: { protocol: 'http:', host: '127.0.0.1:1' }, WebSocket: WS, AudioContext: Ctx, AudioWorkletNode: Node,
  requestAnimationFrame: () => ++H.raf, URL: { createObjectURL() { return ''; } }, Blob: function () {}, Option: function () { return stub('opt'); },
  addEventListener() {}, matchMedia: () => ({ addEventListener() {} }), devicePixelRatio: 1,
};
ctx.window = ctx;

function load(setup) {
  if (setup) setup();
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(APP, 'utf8'), ctx);
  ctx.banner = (k, m, l) => banners.set(k, `${l || 'warn'}|${m}`);
  ctx.unbanner = (k) => banners.delete(k);
}
const run = (code) => vm.runInContext(code, ctx);
const HELLO = { type: 'hello', proto: 1, state: 'ready', phase: 'ready', message: '', model: {}, langs: ['English'], pairs: [['English', 'German']],
  delays: [6, 10], defaults: { asr_delay: 6, translate_delay: {}, options: {} }, busy: false,
  latency_table: { 6: { mean_ms: 576, lo_ms: 416, hi_ms: 736 }, 10: { mean_ms: 896, lo_ms: 736, hi_ms: 1056 } }, laal_table: {} };
const FULL = { proc_lag_ms: { p50: 1, p95: 2, max: 3 }, chunk_ms: { p50: 1, p95: 2 }, enc_ms: { p50: 1, p95: 1 }, dec_ms: { p50: 1, p95: 1 },
  rtf_10s: 0.3, kv_pct: 1, mem_gb: 1, tok_per_s: 1, throughput_x: 3, loop_lag_ms: 1, audio_rx_s: 1 };
async function boot() { H.ws.fire('open', {}); H.ws.msg(HELLO); run("S.state = 'ready'"); await sleep(5); }
const texts = (arr) => arr.filter((s) => typeof s.d === 'string').map((s) => JSON.parse(s.d).type).filter((t) => t !== 'ping');
const bins = (arr) => arr.filter((s) => typeof s.d !== 'string');
const firstVals = (arr) => bins(arr).map((s) => new Float32Array(s.d)[0]);
const cnt = (b, v) => b.filter((x) => Math.abs(x - v) < 1e-6).length;
const out = (o) => { process.stdout.write(JSON.stringify({ scen: SCEN, ...o, banners: Object.fromEntries(banners) }) + '\n'); process.exit(0); };
function fakeFile(name, val, sec) { els.file = stub('#file'); els.file.files = [{ name }]; ctx.decodeFile = async () => ({ pcm: new Float32Array(Math.round(16000 * sec)).fill(val), rate: 16000, wav: true }); }
function mkEv(key, target, extra) { const e = { key, code: key === ' ' ? 'Space' : key, target, repeat: false, prevented: false, preventDefault() { this.prevented = true; }, ...extra }; return e; }

const SC = {
  // F01: 가져가기 뒤 남은 빠르게 루프가 다음 세션에 옛 파일을 섞지 않는다
  async zombie() {
    load(); await boot();
    fakeFile('old.wav', 0.1, 60); els['fast-mode'] = stub('#fast-mode'); els['fast-mode'].value = '0';
    run('playFileFast(false)'); await sleep(300);
    H.ws.msg({ type: 'stats', session_id: 'S1', audio_done_s: 1.0, backlog_s: 9, ...FULL }); await sleep(300);
    const nPre = H.sent.length;
    H.ws.msg({ type: 'error', code: 'preempted', message: 'x', fatal: true, sid: 'S1' });
    H.ws.msg({ type: 'session_end', session_id: 'S1', aborted: true });
    await sleep(800);
    const afterPreempt = H.sent.slice(nPre);
    fakeFile('new.wav', 0.2, 60);
    const n0 = H.sent.length;
    run('playFileFast(false)'); await sleep(200);
    for (let i = 1; i <= 40; i++) { H.ws.msg({ type: 'stats', session_id: 'S2', audio_done_s: i * 1.5, backlog_s: 1, ...FULL }); await sleep(25); }
    await sleep(300);
    const s2 = H.sent.slice(n0); const b = firstVals(s2);
    out({ after_preempt_frames: bins(afterPreempt).length, after_preempt_text: texts(afterPreempt), s2_old: cnt(b, 0.1), s2_new: cnt(b, 0.2), s2_text: texts(s2),
      run: run('S.run && {kind: S.run.kind, stopping: S.run.stopping}') });
  },
  // F01: 4× 도중 가져가기 → 프레임·stop·abort·오류 배너 없음
  async short4x() {
    load(); await boot();
    fakeFile('a.wav', 0.1, 6); els['fast-mode'] = stub('#fast-mode'); els['fast-mode'].value = '4';
    run('playFileFast(false)'); await sleep(500);
    H.ws.msg({ type: 'stats', session_id: 'S1', audio_done_s: 0.5, backlog_s: 1, ...FULL });
    const n1 = H.sent.length;
    H.ws.msg({ type: 'error', code: 'preempted', message: 'x', fatal: true, sid: 'S1' });
    H.ws.msg({ type: 'session_end', session_id: 'S1', aborted: true });
    await sleep(2500);
    const after = H.sent.slice(n1);
    out({ frames_after: bins(after).length, text_after: texts(after), run: run('S.run') });
  },
  // F01: 디코딩 중 연결이 끊기고 다시 이어져 새 실행 B 가 시작된 뒤 A 의 디코딩이 끝나도 A 는 start 를 보내지 않는다
  async decodeRace() {
    load(); await boot();
    els.file = stub('#file'); els.file.files = [{ name: 'a.wav' }]; els['fast-mode'] = stub('#fast-mode'); els['fast-mode'].value = '4';
    let call = 0;
    ctx.decodeFile = async () => { const me = ++call; if (me === 1) await sleep(1500); return { pcm: new Float32Array(16000 * 3).fill(me === 1 ? 0.1 : 0.2), rate: 16000, wav: true }; };
    run('playFileFast(false)'); await sleep(100);
    H.ws.fire('close', {});
    await sleep(700);                      // connect() 재시도(500 ms)
    H.ws.fire('open', {}); H.ws.msg(HELLO); run("S.state = 'ready'");
    run('playFileFast(false)');
    await sleep(2500);
    const s2 = H.sent.filter((s) => s.ws === 2); const b = firstVals(s2);
    out({ text_ws2: texts(s2), a_frames: cnt(b, 0.1), b_frames: cnt(b, 0.2) });
  },
  // F01 회귀: 보통 끝까지·사용자 정지
  async normal() {
    load(); await boot();
    fakeFile('a.wav', 0.1, 20); els['fast-mode'] = stub('#fast-mode'); els['fast-mode'].value = '0';
    run('playFileFast(false)'); await sleep(200);
    for (let i = 1; i <= 20; i++) {
      H.ws.msg({ type: 'stats', session_id: 'S1', audio_done_s: i * 1.0, backlog_s: 1, ...FULL }); await sleep(30);
      if (ARGS[0] === 'userstop' && i === 5) run('stopRun()');
    }
    await sleep(300);
    const stopping = run('S.run && S.run.stopping');
    H.ws.msg({ type: 'session_end', session_id: 'S1', aborted: false }); await sleep(50);
    out({ frames: bins(H.sent).length, samples: bins(H.sent).reduce((a, s) => a + s.d.byteLength / 4, 0), text: texts(H.sent), stopping, run: run('S.run') });
  },
  // F09: getUserMedia 대기 중 정지 → 스트림은 끝나고 UI 는 바로 준비됨, 서버에는 아무것도 안 보냄
  async stopDuringGum() {
    load(); await boot();
    H.gumDelay = 300;
    run('startMic(false)'); await sleep(50);
    const mid = run('S.run && {kind: S.run.kind, startSent: S.run.startSent}');
    run('stopRun()');
    const right = run('S.run');
    await sleep(500);
    out({ mid, run_after_stop: right, run: run('S.run'), tracks: H.streams.map((s) => s.t[0].readyState), ctxs: H.ctxs.map((c) => c.state), text: texts(H.sent), cap: run('S.cap') });
  },
  // F09: 두 번 클릭(컨텍스트 준비 중 정지) → start 를 보내지 않고 바로 준비됨
  async doubleClick() {
    load(); await boot();
    H.addModuleDelay = 100;
    run('startMic(false)'); run('stopRun()');
    await sleep(400);
    out({ run: run('S.run'), tracks: H.streams.map((s) => s.t[0].readyState), ctxs: H.ctxs.map((c) => c.state), text: texts(H.sent), gum: H.gumCalls.length });
  },
  // F09: start 를 보낸 뒤 started 전에 정지 → start, stop 순서로 가고 session_end 에 끝난다 (서버 세션이 고아가 되지 않음)
  async stopAfterStartSent() {
    load(); await boot();
    H.autoStarted = false;
    run('startMic(false)'); await sleep(50);
    const sentStart = run('S.run && S.run.startSent');
    run('stopRun()');
    const stopping = run('S.run && S.run.stopping');
    H.ws.msg({ type: 'started', session_id: 'S9', config: { src_lang: 'English', mode: 'asr', delay: 6, options: {} } }); await sleep(10);
    H.ws.msg({ type: 'session_end', session_id: 'S9', aborted: false }); await sleep(10);
    out({ sentStart, stopping, text: texts(H.sent), run: run('S.run'), tracks: H.streams.map((s) => s.t[0].readyState) });
  },
  // F09: 중단(Esc) 이 getUserMedia 대기 중에 와도 스트림이 남지 않는다
  async abortDuringGum() {
    load(); await boot();
    H.gumDelay = 300;
    run('startMic(false)'); await sleep(50); run('abortRun()');
    const right = run('S.run');
    await sleep(500);
    out({ run_after_abort: right, run: run('S.run'), tracks: H.streams.map((s) => s.t[0].readyState), text: texts(H.sent) });
  },
  // F10: getUserMedia 오류는 이름별 한국어 배너, 서버 오류는 서버 배너 하나
  async micError() {
    const kind = ARGS[0];
    load(() => {
      if (kind === 'addModule') H.addModuleErr = new DOMException('Failed to load worklet module', 'AbortError');
      else if (kind === 'graph') H.msErr = new DOMException('rate mismatch', 'NotSupportedError');
      else if (kind === 'Overconstrained') H.gumErr = Object.assign(new Error(''), { name: 'OverconstrainedError', constraint: 'deviceId' });
      else if (kind !== 'busy') H.gumErr = new DOMException(kind === 'NotAllowedSystem' ? 'Permission denied by system' : 'x', kind === 'NotAllowedSystem' ? 'NotAllowedError' : kind);
    });
    await boot();
    if (kind === 'busy') {
      H.autoStarted = false;
      run('startMic(false)'); await sleep(50);
      H.ws.msg({ type: 'error', code: 'busy', message: '다른 탭에서 사용 중' }); await sleep(20);
    } else {
      els.mic = stub('#mic'); els.mic.value = kind === 'Overconstrained' ? 'gone-device' : '';
      run('startMic(false)'); await sleep(100);
    }
    out({ code: H.gumErr && H.gumErr.code, run: run('S.run'), mic_value: els.mic ? els.mic.value : null, ctxs: H.ctxs.map((c) => c.state), text: texts(H.sent) });
  },
  // F37: 단축키
  async keys() {
    load(); await boot();
    const r = {};
    const blurred = []; const sel = { tagName: 'SELECT', blur() { blurred.push('select'); } };
    run('startMic(false)'); await sleep(50);
    let ev = mkEv('Escape', sel); ctx.__ev = ev; run('onKey(__ev)');
    r.esc_on_select = { run_alive: !!run('S.run && !S.run.stopping'), blurred: blurred.slice(), prevented: ev.prevented, text: texts(H.sent) };
    ev = mkEv(' ', { tagName: 'BODY' }, { repeat: true }); ctx.__ev = ev; run('onKey(__ev)');
    r.space_repeat = { run_alive: !!run('S.run && !S.run.stopping'), prevented: ev.prevented };
    ev = mkEv(' ', { tagName: 'BODY' }, { shiftKey: true }); ctx.__ev = ev; run('onKey(__ev)');
    r.space_shift = { run_alive: !!run('S.run && !S.run.stopping'), prevented: ev.prevented };
    ev = mkEv('Escape', { tagName: 'BUTTON' }); ctx.__ev = ev; run('onKey(__ev)');
    r.esc_on_button = { aborting: !!run('S.run && S.run.stopping'), text: texts(H.sent) };
    await sleep(1600);
    run('S.wsOpen = false');
    ev = mkEv(' ', { tagName: 'BODY' }); ctx.__ev = ev; run('onKey(__ev)');
    r.space_disconnected = { run: run('S.run'), gum: H.gumCalls.length };
    run('S.wsOpen = true');
    ev = mkEv(' ', { tagName: 'INPUT', type: 'radio' }); ctx.__ev = ev; run('onKey(__ev)');
    r.space_on_radio = { started: !!run('S.run'), prevented: ev.prevented };
    out(r);
  },
  // F38: 고급 숫자 검사
  async opts() {
    load(() => { ctx.localStorage.store['vapkt-prefs'] = JSON.stringify({ options: { end_pause_ms: 100, hard_max_s: 40 } }); });
    els['opt-pause'] = stub('#opt-pause'); els['opt-max'] = stub('#opt-max');
    await boot();                                       // hello → fillControls 가 저장값을 고친다
    const r = { loaded: { pause: els['opt-pause'].value, max: els['opt-max'].value, saved: JSON.parse(ctx.localStorage.store['vapkt-prefs']).options } };
    r.clamp = run("[clampOpt('end_pause_ms', '100'), clampOpt('end_pause_ms', ''), clampOpt('end_pause_ms', 'abc'), clampOpt('end_pause_ms', 2500), clampOpt('end_pause_ms', 812.4), clampOpt('hard_max_s', 5), clampOpt('hard_max_s', 0), clampOpt('hard_max_s', 12.5)]");
    els['opt-pause'].value = '150'; ctx.__ev = { target: { id: 'opt-pause' } }; run('onOptChange(__ev)');
    r.changed = { value: els['opt-pause'].value, msg: els['opt-msg'] ? els['opt-msg'].textContent : null, saved: JSON.parse(ctx.localStorage.store['vapkt-prefs']).options.end_pause_ms };
    els['opt-pause'].value = '900'; run('onOptChange(__ev)');
    r.valid = { value: els['opt-pause'].value, msg: els['opt-msg'].textContent };
    els['opt-max'].value = ''; H.autoStarted = false;
    run('startMic(false)'); await sleep(50);
    const st = H.sent.filter((s) => typeof s.d === 'string').map((s) => JSON.parse(s.d)).find((m) => m.type === 'start');
    r.start_opts = st && { end_pause_ms: st.options.end_pause_ms, hard_max_s: st.options.hard_max_s };
    out(r);
  },
  // F35: 서버 lagging 경고는 배너로 안 띄우고, 세그먼트 시작 직후 한 틱은 밀림으로 보지 않는다
  async lag() {
    load(); await boot();
    H.autoStarted = false;
    run('startMic(false)'); await sleep(30);
    H.ws.msg({ type: 'started', session_id: 'S1', config: { src_lang: 'English', mode: 'asr', delay: 6, options: {} } }); await sleep(5);
    const cls = () => [...(els['st-backlog'].cls || [])].join(',');
    const st = (b) => H.ws.msg({ type: 'stats', session_id: 'S1', backlog_s: b, ...FULL });
    const r = {};
    H.ws.msg({ type: 'warning', code: 'lagging', message: '처리가 실시간보다 2.1 s 늦습니다' });
    r.after_server_lagging = Object.fromEntries(banners);
    H.ws.msg({ type: 'segment_start', sid: 'S1', seg: 1, t_start: 0, reason: 'vad', config: { delay: 6 } });
    st(0.65); r.burst_tick = cls(); await sleep(260); st(0.2); r.after_burst = cls();
    await sleep(700);
    st(0.5); r.one_tick = cls(); st(0.5); r.two_ticks = cls();
    st(1.4); r.over1 = cls(); r.lag_banner = banners.has('lag-local');
    st(0.1); r.recovered = cls(); r.lag_banner_after = banners.has('lag-local');
    out(r);
  },
  // F36: 미터 rAF 는 할 일이 없으면 멈춘다
  async meter() {
    load(); await boot();
    const raf = []; ctx.requestAnimationFrame = (f) => { raf.push(f); return raf.length; };
    run('M.raf = 0; kickMeter()');
    let frames = 0;
    while (raf.length && frames < 1000) { const f = raf.shift(); frames += 1; f(); }
    const idle = frames;
    run('meterUpdate(0.5, 0.2)');                       // 입력 하나: 막대가 0 dBFS 근처에서 20 dB/s 로 내려간다
    frames = 0; const t0 = Date.now();
    while (raf.length && frames < 100000) { const f = raf.shift(); frames += 1; await sleep(16); f(); if (Date.now() - t0 > 8000) break; }
    out({ idle_frames: idle, decay_frames: frames, decay_s: (Date.now() - t0) / 1000, pending: raf.length, disp: run('M.disp') });
  },
  // F41: 잡음 억제·자동 이득 미지원 브라우저(Safari)
  async safari() {
    load(() => { H.sup = { echoCancellation: true, deviceId: true }; H.settings = { deviceId: 'dev-default', sampleRate: 48000, echoCancellation: false }; });
    els['opt-ns'] = els['opt-ns'] || stub('#opt-ns');
    await boot();
    H.autoStarted = false;
    run('startMic(false)'); await sleep(50);
    const c = H.gumCalls[0] && H.gumCalls[0].audio;
    out({ ns_disabled: els['opt-ns'].disabled, ns_unsup: els['opt-ns'].dataset.unsup, agc_unsup: els['opt-agc'].dataset.unsup,
      ec_label: els['lbl-ec'].textContent, ns_label: els['lbl-ns'].textContent, constraints: c, track_info: els['track-info'].textContent,
      mic_disabled_running: els.mic.disabled });
  },
  // F40: 출력 장치가 192 kHz → 48 kHz 컨텍스트로 시작
  async rate192() {
    load(() => { H.deviceRate = 192000; });
    await boot(); H.autoStarted = false;
    run('startMic(false)'); await sleep(50);
    const st = H.sent.filter((s) => typeof s.d === 'string').map((s) => JSON.parse(s.d)).find((m) => m.type === 'start');
    out({ ctx_rates: H.ctxRates, ctx_states: H.ctxs.map((c) => c.state), start_rate: st && st.sample_rate });
  },
  // F11: WAV 직접 해석 → float32 원시 바이트를 파일로
  async parsewav() {
    load();
    const b = fs.readFileSync(ARGS[0]); const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    ctx.__ab = ab;
    const w = run('parseWav(__ab)');
    if (!w) out({ ok: false });
    fs.writeFileSync(ARGS[1], Buffer.from(w.pcm.buffer, w.pcm.byteOffset, w.pcm.byteLength));
    out({ ok: true, rate: w.rate, ch: w.ch, bits: w.bits, kind: w.kind, n: w.pcm.length });
  },
};

if (!SC[SCEN]) { process.stderr.write(`unknown scenario ${SCEN}\n`); process.exit(2); }
SC[SCEN]().catch((e) => { process.stdout.write(JSON.stringify({ scen: SCEN, error: String(e && e.stack || e) }) + '\n'); process.exit(1); });
