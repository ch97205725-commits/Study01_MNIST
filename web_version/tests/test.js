// fixtures.js(PyTorch 기준값)와 JS 추론 결과를 비교한다.
// 결과는 화면과 globalThis.검증_결과에 남긴다 (자동 확인용).
(function () {
  "use strict";

  function base64_바이트(문자열) {
    const 원문 = atob(문자열);
    const 바이트 = new Uint8Array(원문.length);
    for (let i = 0; i < 원문.length; i++) 바이트[i] = 원문.charCodeAt(i);
    return 바이트;
  }

  // 28x28 이미지를 쌍선형 보간으로 키워 280x280 검은 캔버스의 (왼쪽, 위) 위치에 붙인다
  // (그림판에 크게, 한쪽으로 치우쳐 그린 상황을 흉내 낸다)
  function 캔버스에_붙이기(원본, 크기, 왼쪽, 위) {
    const 캔버스 = new Uint8Array(280 * 280);
    const 비율 = 28 / 크기;
    for (let y = 0; y < 크기; y++) {
      for (let x = 0; x < 크기; x++) {
        const sx = Math.min(27, Math.max(0, (x + 0.5) * 비율 - 0.5));
        const sy = Math.min(27, Math.max(0, (y + 0.5) * 비율 - 0.5));
        const x0 = Math.floor(sx), y0 = Math.floor(sy);
        const x1 = Math.min(27, x0 + 1), y1 = Math.min(27, y0 + 1);
        const fx = sx - x0, fy = sy - y0;
        const 값 = (1 - fy) * ((1 - fx) * 원본[y0 * 28 + x0] + fx * 원본[y0 * 28 + x1]) +
                   fy * ((1 - fx) * 원본[y1 * 28 + x0] + fx * 원본[y1 * 28 + x1]);
        캔버스[(y + 위) * 280 + (x + 왼쪽)] = Math.round(값);
      }
    }
    return 캔버스;
  }

  const 자료 = globalThis.검증_자료;
  const 층들 = 숫자인식.가중치_해독(globalThis.가중치_묶음);
  const 이미지들 = base64_바이트(자료.이미지);
  const 정답들 = base64_바이트(자료.정답);
  const 기준_로짓 = new Float32Array(base64_바이트(자료.로짓).buffer);

  const 시작_시각 = performance.now();
  let 최대_차이 = 0, 일치_수 = 0, 직접_정답 = 0, 전처리_정답 = 0;
  for (let n = 0; n < 자료.개수; n++) {
    const 원본 = 이미지들.subarray(n * 784, (n + 1) * 784);

    // 1) 28x28 원본을 그대로 넣었을 때 PyTorch 로짓과 비교
    const 입력 = Float32Array.from(원본, (값) => (값 / 255 - 숫자인식.평균) / 숫자인식.표준편차);
    const 로짓 = 숫자인식.추론(층들, 입력);
    let js_최고 = 0, 파이토치_최고 = 0;
    for (let k = 0; k < 10; k++) {
      최대_차이 = Math.max(최대_차이, Math.abs(로짓[k] - 기준_로짓[n * 10 + k]));
      if (로짓[k] > 로짓[js_최고]) js_최고 = k;
      if (기준_로짓[n * 10 + k] > 기준_로짓[n * 10 + 파이토치_최고]) 파이토치_최고 = k;
    }
    if (js_최고 === 파이토치_최고) 일치_수++;
    if (js_최고 === 정답들[n]) 직접_정답++;

    // 2) 그림판 크기로 키워 전처리까지 거친 정확도
    const 전처리_입력 = 숫자인식.전처리(캔버스에_붙이기(원본, 150, 40, 90), 280, 280);
    const 전처리_로짓 = 숫자인식.추론(층들, 전처리_입력);
    if (전처리_로짓.indexOf(Math.max(...전처리_로짓)) === 정답들[n]) 전처리_정답++;
  }
  const 걸린_시간 = performance.now() - 시작_시각;

  const 항목들 = [
    { 이름: "PyTorch 로짓과 최대 차이 < 1e-3", 값: 최대_차이.toExponential(2), 통과: 최대_차이 < 1e-3 },
    { 이름: "PyTorch와 예측 숫자 일치", 값: `${일치_수}/${자료.개수}`, 통과: 일치_수 === 자료.개수 },
    { 이름: "28x28 직접 입력 정확도 ≥ 99%", 값: `${(직접_정답 / 자료.개수 * 100).toFixed(1)}%`, 통과: 직접_정답 / 자료.개수 >= 0.99 },
    { 이름: "그림판 크기 + 전처리 정확도 ≥ 98%", 값: `${(전처리_정답 / 자료.개수 * 100).toFixed(1)}%`, 통과: 전처리_정답 / 자료.개수 >= 0.98 },
  ];

  const 모두_통과 = 항목들.every((항목) => 항목.통과);
  globalThis.검증_결과 = { 모두_통과, 항목들, 걸린_시간 };

  document.getElementById("요약").textContent =
    `${모두_통과 ? "모두 통과" : "실패 있음"} (이미지 ${자료.개수}장, ${(걸린_시간 / 1000).toFixed(1)}초)`;
  document.getElementById("요약").className = 모두_통과 ? "통과" : "실패";
  document.getElementById("결과").replaceChildren(...항목들.map((항목) => {
    const 줄 = document.createElement("li");
    줄.className = 항목.통과 ? "통과" : "실패";
    줄.textContent = `${항목.통과 ? "✔" : "✘"} ${항목.이름}: ${항목.값}`;
    return 줄;
  }));
})();
