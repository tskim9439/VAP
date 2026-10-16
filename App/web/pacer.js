// 빠르게 재생 페이서 (전용 워커): {id, ms} 를 받으면 ms 뒤 id 를 돌려준다.
// 메인 스레드 setTimeout 은 숨은 탭에서 1 s(5 분 뒤에는 1 분)마다로 묶이지만 전용 워커의 타이머는 그렇지 않다.
// 같은 출처의 /static/pacer.js 로 띄운다 (CSP script-src 'self' — Blob URL 워커는 막힌다).
self.onmessage = (e) => {
  const d = e.data || {};
  setTimeout(() => self.postMessage(d.id), Math.max(0, +d.ms || 0));
};
