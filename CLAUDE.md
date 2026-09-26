# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

PyTorch CNN으로 MNIST를 학습하고, tkinter 그림판에 마우스로 그린 숫자를 인식하는 프로젝트다.

## 작성 규칙

- **모든 코드와 주석은 한글로 작성한다.** 변수·함수·클래스 이름도 한글 식별자를 쓴다(예: `숫자인식망`, `예측하기`, `전처리`). 파일 이름과 라이브러리 API만 영어로 둔다.
- 학습된 가중치 파일 이름은 `mnist_cnn.pt`로 고정한다(`model.py`의 `가중치_파일`).

## 실행 환경과 명령

기본 Python은 Microsoft Store 버전이라서 PyTorch를 설치하면 경로가 너무 길다는 오류(`WinError 206`)로 실패한다. 그래서 짧은 경로의 가상환경 `C:\Users\hj831\.venvs\mnist`를 쓴다(torch/torchvision은 CPU 버전, pillow, numpy 포함). 시스템 `python`으로 실행하지 않는다.

```bash
C:\Users\hj831\.venvs\mnist\Scripts\python.exe train.py
```

```bash
C:\Users\hj831\.venvs\mnist\Scripts\python.exe app.py
```

- `train.py`: MNIST를 `data/`에 내려받아(처음 한 번만) 5 에포크 동안 학습한다. CPU에서 에포크당 약 75초 걸리고, 시험 정확도가 가장 좋을 때만 `mnist_cnn.pt`를 덮어쓴다. 지금까지 최고 정확도는 99.51%다.
- `app.py`: GUI 창을 연다. 창 없이 확인하려면 `app.py`에서 `모델_불러오기`와 `예측하기`를 import해 PIL 흑백 이미지(280×280, 검은 배경에 흰 글씨)를 넣어 본다.
- 테스트 스위트와 린터는 없다.
- PowerShell 5.1에서 `python -c "..."`로 여러 줄 코드를 넘기면 따옴표가 깨진다. 한글이 섞인 스크립트는 임시 `.py` 파일로 저장해서 실행하고, 출력이 깨지면 `$env:PYTHONIOENCODING="utf-8"`를 설정한다.

## 구조

- `model.py`: `숫자인식망`(합성곱+BatchNorm+최대풀링 3단, 28→14→7→3, 그다음 완전연결층 2개)을 정의한다. 학습과 추론이 함께 쓰는 상수 `가중치_파일`, `평균`, `표준편차`도 여기에 있다.
- `train.py`: 학습할 때 `RandomAffine`(회전·이동·확대) 증강을 적용해 실제 손글씨에도 잘 맞도록 한다. `state_dict`만 저장한다.
- `app.py`: 화면의 tkinter `Canvas`와 메모리 속 PIL 이미지(`self.그림`)에 같은 선을 동시에 그리고, 인식에는 PIL 이미지를 쓴다. 마우스 버튼을 뗄 때마다 인식한다.

**전처리를 바꿀 때 주의:** `app.py`의 `전처리`는 MNIST 원본의 형식을 흉내 낸다. 글씨 영역만 잘라 긴 변을 20px로 줄이고, 28×28 바탕에 붙인 뒤, 무게중심을 (14,14)로 옮기고, `평균`/`표준편차`로 정규화한다. 학습 데이터의 변환(`ToTensor` + `Normalize`)과 이 전처리가 어긋나면 GUI 인식률이 크게 떨어진다. 둘 중 하나를 고치면 다른 쪽도 맞춰야 한다. 확인할 때는 MNIST 시험 이미지를 280×280 캔버스에 크게, 한쪽으로 치우치게 붙여 `예측하기`에 넣어 본다. 이 방법으로 쟀을 때 정확도는 약 99%였다.
