// 캡처 워크릿: 입력 채널을 모노로 평균해 장치 샘플레이트 그대로 20 ms Float32 패킷으로 보낸다 (audio-ui-spec §3.3).
// 메인 스레드로 {pcm: ArrayBuffer(전송), peak, rms} 를 보내고, 메인 스레드가 ws.send 한다 (워크릿에는 WebSocket 이 없다).
// 'flush' 를 받으면 남은 부분 패킷을 보내고 {flushed: true} 로 답한다 (파일 끝).
class Capture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const p = (options && options.processorOptions) || {};
    this.n = Math.max(128, p.packet || Math.round(sampleRate * 0.02));
    this.buf = new Float32Array(this.n);
    this.i = 0;
    this.peak = 0;
    this.sq = 0;
    this.alive = true;
    this.port.onmessage = (e) => {
      const cmd = e.data && e.data.cmd;
      if (cmd === 'flush') {
        this.post(this.i);
        this.port.postMessage({ flushed: true });
      } else if (cmd === 'close') {
        this.alive = false;
      }
    };
  }

  post(len) {
    if (len <= 0) return;
    const out = len === this.n ? this.buf : this.buf.slice(0, len);
    const rms = Math.sqrt(this.sq / len);
    this.port.postMessage({ pcm: out.buffer, peak: this.peak, rms: rms }, [out.buffer]);
    this.buf = new Float32Array(this.n);
    this.i = 0;
    this.peak = 0;
    this.sq = 0;
  }

  process(inputs) {
    const inp = inputs[0];
    if (inp && inp.length > 0) {           // 입력이 연결되지 않은 렌더 퀀텀(파일 재생 전후)은 건너뛴다
      const nch = inp.length;
      const len = inp[0].length;
      for (let j = 0; j < len; j++) {
        let v = 0;
        for (let c = 0; c < nch; c++) v += inp[c][j];
        v /= nch;
        this.buf[this.i++] = v;
        const a = v < 0 ? -v : v;
        if (a > this.peak) this.peak = a;
        this.sq += v * v;
        if (this.i === this.n) this.post(this.n);
      }
    }
    return this.alive;
  }
}

registerProcessor('capture', Capture);
