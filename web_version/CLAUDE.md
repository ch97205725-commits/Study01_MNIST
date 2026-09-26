# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 개요

`desktop_version`에서 학습한 CNN을 **외부 라이브러리 없이 순수 자바스크립트로** 추론하는 정적 웹앱이다. 빌드 과정이 없고, GitHub Pages(`.github/workflows/pages.yml`)가 이 폴더를 그대로 배포한다.

## 반드시 지킬 것

- `index.html` 화면 맨 위에는 `<header id="학생정보">학번 2601990 이름 최현지</header>`가 보여야 한다(과제 제출 요건). 지우거나 숨기거나 다른 요소 아래로 옮기지 않는다.
- CDN, npm 패키지, 번들러 같은 외부 라이브러리를 쓰지 않는다.
- 코드·주석·식별자는 한글로 쓴다(파일 이름만 영어).
- `weights.js`와 `tests/fixtures.js`는 자동으로 생성되는 파일이라 직접 고치지 않는다.

## 명령

가중치를 다시 내보낸다. `desktop_version/mnist_cnn.pt`가 바뀌면 반드시 실행해야 한다.

```bash
C:\Users\hj831\.venvs\mnist\Scripts\python.exe web_version/tools/export_weights.py
```

로컬 확인은 저장소 루트의 `.claude/launch.json`(`web_version` 설정)이나 아래 명령으로 서버를 띄운 뒤 `http://localhost:8000/`을 연다. 검증 페이지는 `http://localhost:8000/tests/test.html`이다.

```bash
python -m http.server 8000 --directory web_version
```

Node.js가 설치돼 있지 않으므로 검증은 브라우저에서 `tests/test.html`을 열어 한다. 결과는 화면과 `globalThis.검증_결과`에 나온다. 통과 조건은 PyTorch 로짓과의 최대 차이 < 1e-3, 예측 1000/1000 일치, 그림판 크기로 키워 전처리까지 거친 정확도 ≥ 98%이다.

## 구조

- 읽는 순서: `weights.js` → `inference.js` → `app.js`. 모두 일반 `<script>` 태그로 불러오므로 `index.html`을 `file://`로 직접 열어도 동작한다(fetch를 쓰지 않는다).
- `tools/export_weights.py`는 BatchNorm을 앞 합성곱의 가중치·편향에 합친 뒤, 리틀 엔디언 float32를 base64로 인코딩해 `globalThis.가중치_묶음`(`목차` + `데이터`)에 담는다. 동시에 `tests/fixtures.js`(MNIST 시험 이미지 1000장과 PyTorch 로짓)도 만든다.
- `inference.js`의 `globalThis.숫자인식`은 `가중치_해독`, `추론`, `소프트맥스`, `전처리`를 제공한다. 텐서는 `[채널][세로][가로]` 순서의 Float32Array이고, 펼치는 순서는 PyTorch `flatten`과 같다.
- **모델 구조나 전처리를 바꿀 때:** `inference.js`는 `desktop_version/model.py`(층 구조), `desktop_version/app.py`의 `전처리`(영역 자르기 → 긴 변 20px → 28×28 가운데 → 무게중심 이동 → 정규화)와 짝을 이룬다. 한쪽을 고치면 다른 쪽과 `export_weights.py`도 맞춰야 한다. 축소 방법만 다르다(데스크톱은 PIL LANCZOS, 웹은 면적 평균).
- `app.js`는 280×280 캔버스에 붓 두께 18로 그린다(데스크톱과 같음). 포인터 이벤트라 터치로도 그릴 수 있다.
