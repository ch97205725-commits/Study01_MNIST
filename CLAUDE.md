# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 개요

MNIST 손글씨 숫자 인식기를 두 버전으로 제공한다. 각 폴더의 `CLAUDE.md`에 자세한 내용이 있다.

- `desktop_version/`: PyTorch 학습(`train.py`)과 tkinter 앱(`app.py`)이 있다. 학습된 가중치 `mnist_cnn.pt`의 원본이 여기 있다.
- `web_version/`: 순수 자바스크립트 추론 웹앱이다. `desktop_version/mnist_cnn.pt`를 `tools/export_weights.py`로 변환한 `weights.js`를 쓴다.
- `.github/workflows/pages.yml`: `main`에 `web_version/**` 변경이 푸시되면 이 폴더만 GitHub Pages로 배포한다.
- `CLAUDE_전역.md`: 과제로 제출하는 공통 작업 지침 문서다. 프로젝트 코드와는 관계없다.

## 두 버전 사이의 의존 관계

가중치 흐름은 `desktop_version/train.py` → `mnist_cnn.pt` → `web_version/tools/export_weights.py` → `web_version/weights.js`이다. 다시 학습했거나 모델 구조나 전처리를 바꿨다면 웹 쪽 변환·코드·검증(`web_version/tests/test.html`)까지 함께 갱신한다.

## 공통 규칙

- 코드·주석·식별자는 한글로 쓴다(파일 이름만 영어).
- Python은 `C:\Users\hj831\.venvs\mnist\Scripts\python.exe` 가상환경을 쓴다. 시스템 Python(Microsoft Store 버전)에는 경로 길이 문제로 PyTorch를 설치할 수 없다.
