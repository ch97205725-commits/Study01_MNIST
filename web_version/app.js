// 그림판 입력을 받아 inference.js로 숫자를 인식하고 결과를 화면에 보여 준다.
(function () {
  "use strict";

  const 붓_두께 = 18; // desktop_version/app.py와 같은 두께 (280px 기준)

  const 그림판 = document.getElementById("그림판");
  const 붓 = 그림판.getContext("2d", { willReadFrequently: true });
  const 미리보기 = document.getElementById("미리보기").getContext("2d");
  const 예측숫자 = document.getElementById("예측숫자");
  const 상태 = document.getElementById("상태");
  const 확률목록 = document.getElementById("확률목록");

  const 층들 = 숫자인식.가중치_해독(globalThis.가중치_묶음);
  let 그리는중 = false;
  let 이전_점 = null;

  function 그림판_초기화() {
    붓.fillStyle = "#000000";
    붓.fillRect(0, 0, 그림판.width, 그림판.height);
    미리보기.fillStyle = "#000000";
    미리보기.fillRect(0, 0, 28, 28);
    예측숫자.textContent = "?";
    상태.textContent = "숫자를 그려 보세요";
    확률목록.replaceChildren();
  }

  // 화면에 보이는 크기와 캔버스 내부 크기(280)가 다를 수 있으므로 좌표를 환산한다
  function 좌표(이벤트) {
    const 영역 = 그림판.getBoundingClientRect();
    return {
      x: (이벤트.clientX - 영역.left) * (그림판.width / 영역.width),
      y: (이벤트.clientY - 영역.top) * (그림판.height / 영역.height),
    };
  }

  function 선_긋기(시작, 끝) {
    붓.strokeStyle = "#ffffff";
    붓.lineWidth = 붓_두께;
    붓.lineCap = "round";
    붓.lineJoin = "round";
    붓.beginPath();
    붓.moveTo(시작.x, 시작.y);
    붓.lineTo(끝.x, 끝.y);
    붓.stroke();
  }

  function 인식하기() {
    // 캔버스의 빨강 채널을 회색조 값으로 쓴다 (흰 글씨라 세 채널 값이 같다)
    const 픽셀 = 붓.getImageData(0, 0, 그림판.width, 그림판.height).data;
    const 회색조 = new Uint8Array(그림판.width * 그림판.height);
    for (let i = 0; i < 회색조.length; i++) 회색조[i] = 픽셀[i * 4];

    const 입력 = 숫자인식.전처리(회색조, 그림판.width, 그림판.height);
    if (!입력) return;

    const 시작_시각 = performance.now();
    const 확률 = 숫자인식.소프트맥스(숫자인식.추론(층들, 입력));
    const 걸린_시간 = performance.now() - 시작_시각;

    const 순위 = 확률.map((값, 숫자) => ({ 숫자, 값 })).sort((a, b) => b.값 - a.값);
    예측숫자.textContent = String(순위[0].숫자);
    상태.textContent = `추론 시간 ${걸린_시간.toFixed(1)}ms`;
    확률목록.replaceChildren(...순위.slice(0, 3).map(확률_항목));
    미리보기_그리기(입력);
  }

  function 확률_항목({ 숫자, 값 }) {
    const 항목 = document.createElement("li");
    const 막대 = document.createElement("div");
    const 채움 = document.createElement("span");
    막대.className = "막대";
    채움.style.width = `${(값 * 100).toFixed(1)}%`;
    막대.append(채움);
    const 숫자_칸 = document.createElement("strong");
    숫자_칸.textContent = String(숫자);
    const 값_칸 = document.createElement("span");
    값_칸.textContent = `${(값 * 100).toFixed(1)}%`;
    항목.append(숫자_칸, 막대, 값_칸);
    return 항목;
  }

  // 정규화된 입력을 다시 0~255로 되돌려 28x28 미리보기에 그린다
  function 미리보기_그리기(입력) {
    const 이미지 = 미리보기.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const 값 = Math.max(0, Math.min(255, Math.round((입력[i] * 숫자인식.표준편차 + 숫자인식.평균) * 255)));
      이미지.data.set([값, 값, 값, 255], i * 4);
    }
    미리보기.putImageData(이미지, 0, 0);
  }

  // 마우스·터치·펜을 모두 포인터 이벤트로 처리한다
  그림판.addEventListener("pointerdown", (이벤트) => {
    if (이벤트.button === 2) return; // 우클릭은 지우기용
    그리는중 = true;
    그림판.setPointerCapture(이벤트.pointerId);
    이전_점 = 좌표(이벤트);
    선_긋기(이전_점, 이전_점); // 한 번 찍기만 해도 점이 남도록
  });
  그림판.addEventListener("pointermove", (이벤트) => {
    if (!그리는중) return;
    const 현재_점 = 좌표(이벤트);
    선_긋기(이전_점, 현재_점);
    이전_점 = 현재_점;
  });
  function 그리기_끝() {
    if (!그리는중) return;
    그리는중 = false;
    이전_점 = null;
    인식하기(); // 손을 뗄 때마다 결과를 갱신한다
  }
  그림판.addEventListener("pointerup", 그리기_끝);
  그림판.addEventListener("pointercancel", 그리기_끝);
  그림판.addEventListener("contextmenu", (이벤트) => {
    이벤트.preventDefault();
    그림판_초기화();
  });
  document.getElementById("지우기").addEventListener("click", 그림판_초기화);

  그림판_초기화();
})();
