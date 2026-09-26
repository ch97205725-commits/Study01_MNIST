// 외부 라이브러리 없이 순수 자바스크립트로 구현한 숫자인식망 추론과 전처리.
// desktop_version/model.py의 구조와 desktop_version/app.py의 전처리를 그대로 옮겼다.
// (BatchNorm은 tools/export_weights.py에서 합성곱에 미리 합쳐 두었다.)
(function () {
  "use strict";

  // MNIST 평균과 표준편차 (model.py와 같아야 한다)
  const 평균 = 0.1307;
  const 표준편차 = 0.3081;

  // weights.js의 base64 문자열을 층별 Float32Array로 풀어 놓는다
  function 가중치_해독(묶음) {
    const 문자열 = atob(묶음.데이터);
    const 바이트 = new Uint8Array(문자열.length);
    for (let i = 0; i < 문자열.length; i++) 바이트[i] = 문자열.charCodeAt(i);
    const 전체 = new Float32Array(바이트.buffer);
    const 층들 = {};
    for (const [이름, 정보] of Object.entries(묶음.목차)) {
      층들[이름] = 전체.subarray(정보.시작, 정보.시작 + 정보.길이);
    }
    return 층들;
  }

  // 3x3 합성곱(패딩 1) + ReLU. 입력은 [채널][세로][가로] 순서로 펼친 배열
  function 합성곱_렐루(입력, 입력_채널, 크기, 가중치, 편향) {
    const 출력_채널 = 편향.length;
    const 면적 = 크기 * 크기;
    const 출력 = new Float32Array(출력_채널 * 면적);
    for (let 출 = 0; 출 < 출력_채널; 출++) {
      const 출력_시작 = 출 * 면적;
      출력.fill(편향[출], 출력_시작, 출력_시작 + 면적);
      for (let 입 = 0; 입 < 입력_채널; 입++) {
        const 입력_시작 = 입 * 면적;
        const 커널_시작 = (출 * 입력_채널 + 입) * 9;
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            const 값 = 가중치[커널_시작 + ky * 3 + kx];
            // 패딩 영역을 건너뛰도록 출력 좌표 범위를 미리 좁힌다
            const y_처음 = Math.max(0, 1 - ky), y_끝 = Math.min(크기, 크기 + 1 - ky);
            const x_처음 = Math.max(0, 1 - kx), x_끝 = Math.min(크기, 크기 + 1 - kx);
            for (let y = y_처음; y < y_끝; y++) {
              const 출력_줄 = 출력_시작 + y * 크기;
              const 입력_줄 = 입력_시작 + (y + ky - 1) * 크기 + (kx - 1);
              for (let x = x_처음; x < x_끝; x++) 출력[출력_줄 + x] += 값 * 입력[입력_줄 + x];
            }
          }
        }
      }
    }
    for (let i = 0; i < 출력.length; i++) if (출력[i] < 0) 출력[i] = 0;
    return 출력;
  }

  // 2x2 최대 풀링 (홀수 크기면 PyTorch처럼 마지막 줄·칸을 버린다)
  function 최대풀링(입력, 채널, 크기) {
    const 새_크기 = Math.floor(크기 / 2);
    const 출력 = new Float32Array(채널 * 새_크기 * 새_크기);
    for (let c = 0; c < 채널; c++) {
      for (let y = 0; y < 새_크기; y++) {
        for (let x = 0; x < 새_크기; x++) {
          const 위치 = c * 크기 * 크기 + 2 * y * 크기 + 2 * x;
          출력[(c * 새_크기 + y) * 새_크기 + x] = Math.max(
            입력[위치], 입력[위치 + 1], 입력[위치 + 크기], 입력[위치 + 크기 + 1]);
        }
      }
    }
    return 출력;
  }

  // 완전연결층: 가중치 모양은 [출력][입력] (PyTorch Linear와 같음)
  function 완전연결(입력, 가중치, 편향, 렐루) {
    const 출력 = new Float32Array(편향.length);
    const 입력_수 = 입력.length;
    for (let o = 0; o < 편향.length; o++) {
      let 합 = 편향[o];
      const 시작 = o * 입력_수;
      for (let i = 0; i < 입력_수; i++) 합 += 가중치[시작 + i] * 입력[i];
      출력[o] = 렐루 && 합 < 0 ? 0 : 합;
    }
    return 출력;
  }

  // 정규화된 28x28 입력(길이 784)을 받아 숫자 0~9의 로짓을 돌려준다
  function 추론(층들, 입력) {
    let 출력 = 입력, 채널 = 1, 크기 = 28;
    for (const 번호 of [1, 2, 3]) {
      const 편향 = 층들[`합성곱${번호}.편향`];
      출력 = 합성곱_렐루(출력, 채널, 크기, 층들[`합성곱${번호}.가중치`], 편향);
      채널 = 편향.length;
      출력 = 최대풀링(출력, 채널, 크기);
      크기 = Math.floor(크기 / 2);
    }
    // 28 → 14 → 7 → 3, 펼친 순서는 PyTorch flatten과 같은 [채널][세로][가로]
    출력 = 완전연결(출력, 층들["완전연결1.가중치"], 층들["완전연결1.편향"], true);
    return 완전연결(출력, 층들["완전연결2.가중치"], 층들["완전연결2.편향"], false);
  }

  function 소프트맥스(로짓) {
    const 최대 = Math.max(...로짓);
    const 지수 = Array.from(로짓, (값) => Math.exp(값 - 최대));
    const 합 = 지수.reduce((a, b) => a + b, 0);
    return 지수.map((값) => 값 / 합);
  }

  // 면적 평균 방식으로 이미지 크기를 줄인다 (한 방향씩 두 번 적용)
  function 한_방향_축소(원본, 가로, 세로, 새_길이, 가로_방향) {
    const 원래_길이 = 가로_방향 ? 가로 : 세로;
    const 새_가로 = 가로_방향 ? 새_길이 : 가로;
    const 새_세로 = 가로_방향 ? 세로 : 새_길이;
    const 결과 = new Float32Array(새_가로 * 새_세로);
    const 비율 = 원래_길이 / 새_길이;
    for (let n = 0; n < 새_길이; n++) {
      const 시작 = n * 비율, 끝 = (n + 1) * 비율;
      // 이 출력 칸이 덮는 원본 칸들과 각 칸이 겹치는 비율(가중치)
      const 칸들 = [];
      for (let k = Math.floor(시작); k < Math.min(원래_길이, Math.ceil(끝)); k++) {
        칸들.push([k, Math.min(끝, k + 1) - Math.max(시작, k)]);
      }
      const 다른_길이 = 가로_방향 ? 세로 : 가로;
      for (let m = 0; m < 다른_길이; m++) {
        let 합 = 0;
        for (const [k, 비중] of 칸들) 합 += 비중 * (가로_방향 ? 원본[m * 가로 + k] : 원본[k * 가로 + m]);
        결과[가로_방향 ? m * 새_가로 + n : n * 새_가로 + m] = 합 / 비율;
      }
    }
    return 결과;
  }

  // 검은 배경(0)에 흰 글씨(255)인 회색조 배열을 MNIST 형식의 정규화된 입력(길이 784)으로 바꾼다.
  // desktop_version/app.py의 전처리와 같은 순서: 글씨 영역 자르기 → 긴 변 20px → 28x28 가운데 → 무게중심 이동
  function 전처리(회색조, 가로, 세로) {
    // 1) 글씨가 있는 영역(0이 아닌 픽셀) 찾기
    let 왼쪽 = 가로, 위 = 세로, 오른쪽 = -1, 아래 = -1;
    for (let y = 0; y < 세로; y++) {
      for (let x = 0; x < 가로; x++) {
        if (회색조[y * 가로 + x] > 0) {
          if (x < 왼쪽) 왼쪽 = x;
          if (x > 오른쪽) 오른쪽 = x;
          if (y < 위) 위 = y;
          if (y > 아래) 아래 = y;
        }
      }
    }
    if (오른쪽 < 0) return null; // 아무것도 그리지 않음

    // 2) 잘라낸 뒤 비율을 유지하며 긴 변을 20픽셀로 맞추기
    const 자른_가로 = 오른쪽 - 왼쪽 + 1, 자른_세로 = 아래 - 위 + 1;
    let 조각 = new Float32Array(자른_가로 * 자른_세로);
    for (let y = 0; y < 자른_세로; y++) {
      for (let x = 0; x < 자른_가로; x++) 조각[y * 자른_가로 + x] = 회색조[(y + 위) * 가로 + (x + 왼쪽)];
    }
    const 배율 = 20 / Math.max(자른_가로, 자른_세로);
    const 새_가로 = Math.max(1, Math.round(자른_가로 * 배율));
    const 새_세로 = Math.max(1, Math.round(자른_세로 * 배율));
    조각 = 한_방향_축소(조각, 자른_가로, 자른_세로, 새_가로, true);
    조각 = 한_방향_축소(조각, 새_가로, 자른_세로, 새_세로, false);

    // 3) 28x28 검은 바탕 가운데에 붙이기
    const 바탕 = new Float32Array(784);
    const 붙일_x = Math.floor((28 - 새_가로) / 2), 붙일_y = Math.floor((28 - 새_세로) / 2);
    for (let y = 0; y < 새_세로; y++) {
      for (let x = 0; x < 새_가로; x++) 바탕[(y + 붙일_y) * 28 + (x + 붙일_x)] = 조각[y * 새_가로 + x];
    }

    // 4) 무게중심을 정중앙(14, 14)으로 옮기기
    let 전체 = 0, 합_x = 0, 합_y = 0;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const 값 = 바탕[y * 28 + x];
        전체 += 값; 합_x += x * 값; 합_y += y * 값;
      }
    }
    const 이동_x = Math.round(14 - 합_x / 전체), 이동_y = Math.round(14 - 합_y / 전체);
    const 결과 = new Float32Array(784);
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const 원래_x = x - 이동_x, 원래_y = y - 이동_y;
        const 값 = 원래_x >= 0 && 원래_x < 28 && 원래_y >= 0 && 원래_y < 28 ? 바탕[원래_y * 28 + 원래_x] : 0;
        // 5) 0~1 범위로 바꾸고 MNIST 평균/표준편차로 정규화
        결과[y * 28 + x] = (값 / 255 - 평균) / 표준편차;
      }
    }
    return 결과;
  }

  globalThis.숫자인식 = { 평균, 표준편차, 가중치_해독, 추론, 소프트맥스, 전처리 };
})();
