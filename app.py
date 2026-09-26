# -*- coding: utf-8 -*-
"""마우스로 숫자를 그리면 학습된 CNN이 인식해 주는 프로그램."""

import tkinter as tk

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from model import 가중치_파일, 숫자인식망, 평균, 표준편차

캔버스_크기 = 280  # 그림판 한 변의 픽셀 수 (28의 10배)
붓_두께 = 18      # 그리는 선의 두께


def 모델_불러오기():
    """저장된 가중치를 읽어 평가 모드의 모델을 만든다."""
    모델 = 숫자인식망()
    모델.load_state_dict(torch.load(가중치_파일, map_location="cpu", weights_only=True))
    모델.eval()
    return 모델


def 전처리(그림):
    """검은 배경에 흰 글씨인 그림(PIL 흑백 이미지)을 MNIST 형식의 텐서로 바꾼다.

    MNIST처럼 숫자를 20x20 안에 맞추고, 무게중심이 28x28 가운데 오도록 옮긴다.
    """
    영역 = 그림.getbbox()
    if 영역 is None:
        return None  # 아무것도 그리지 않음
    숫자 = 그림.crop(영역)

    # 가로세로 비율을 유지하며 긴 변을 20픽셀로 줄인다
    가로, 세로 = 숫자.size
    배율 = 20.0 / max(가로, 세로)
    새_크기 = (max(1, round(가로 * 배율)), max(1, round(세로 * 배율)))
    숫자 = 숫자.resize(새_크기, Image.LANCZOS)

    # 28x28 검은 바탕 가운데에 붙인다
    바탕 = Image.new("L", (28, 28), 0)
    바탕.paste(숫자, ((28 - 새_크기[0]) // 2, (28 - 새_크기[1]) // 2))

    # 무게중심을 계산해 정중앙(14, 14)으로 이동시킨다
    배열 = np.asarray(바탕, dtype=np.float32)
    전체 = 배열.sum()
    세로_좌표, 가로_좌표 = np.indices(배열.shape)
    중심_x = (가로_좌표 * 배열).sum() / 전체
    중심_y = (세로_좌표 * 배열).sum() / 전체
    이동_x, 이동_y = int(round(14 - 중심_x)), int(round(14 - 중심_y))
    바탕 = 바탕.transform((28, 28), Image.AFFINE, (1, 0, -이동_x, 0, 1, -이동_y), fillcolor=0)

    # 0~1 범위로 바꾸고 MNIST 평균/표준편차로 정규화
    텐서 = torch.from_numpy(np.asarray(바탕, dtype=np.float32) / 255.0)
    텐서 = (텐서 - 평균) / 표준편차
    return 텐서.view(1, 1, 28, 28)


def 예측하기(모델, 그림):
    """그림을 인식해 (예측 숫자, 각 숫자별 확률 리스트)를 돌려준다."""
    입력 = 전처리(그림)
    if 입력 is None:
        return None, None
    with torch.no_grad():
        확률 = F.softmax(모델(입력), dim=1)[0]
    return int(확률.argmax()), 확률.tolist()


class 손글씨앱:
    """그림판과 결과 표시 영역으로 이루어진 GUI."""

    def __init__(self, 창, 모델):
        self.모델 = 모델
        self.창 = 창
        창.title("손글씨 숫자 인식기")
        창.resizable(False, False)

        # 화면에 보이는 캔버스와, 실제 인식에 쓰일 메모리 속 이미지를 함께 관리한다
        self.캔버스 = tk.Canvas(창, width=캔버스_크기, height=캔버스_크기, bg="black", cursor="cross")
        self.캔버스.grid(row=0, column=0, rowspan=3, padx=10, pady=10)
        self.그림 = Image.new("L", (캔버스_크기, 캔버스_크기), 0)
        self.붓 = ImageDraw.Draw(self.그림)
        self.이전_점 = None

        # 마우스 이벤트 연결
        self.캔버스.bind("<Button-1>", self.그리기_시작)
        self.캔버스.bind("<B1-Motion>", self.그리기)
        self.캔버스.bind("<ButtonRelease-1>", self.그리기_끝)
        self.캔버스.bind("<Button-3>", lambda 이벤트: self.지우기())

        # 결과 표시 영역
        self.결과_글자 = tk.Label(창, text="?", font=("맑은 고딕", 72, "bold"), width=3)
        self.결과_글자.grid(row=0, column=1, padx=10)
        self.확률_글자 = tk.Label(창, text="숫자를 그려 보세요", font=("맑은 고딕", 11), justify="left")
        self.확률_글자.grid(row=1, column=1, padx=10)
        tk.Button(창, text="지우기 (우클릭)", font=("맑은 고딕", 11), command=self.지우기).grid(row=2, column=1, pady=10)

    def 그리기_시작(self, 이벤트):
        self.이전_점 = (이벤트.x, 이벤트.y)
        self.점_찍기(이벤트.x, 이벤트.y)

    def 그리기(self, 이벤트):
        x, y = 이벤트.x, 이벤트.y
        if self.이전_점:
            px, py = self.이전_점
            # 화면과 메모리 이미지 양쪽에 같은 선을 그린다
            self.캔버스.create_line(px, py, x, y, fill="white", width=붓_두께, capstyle=tk.ROUND, smooth=True)
            self.붓.line([px, py, x, y], fill=255, width=붓_두께)
        self.점_찍기(x, y)
        self.이전_점 = (x, y)

    def 점_찍기(self, x, y):
        """선의 끝을 둥글게 만들기 위해 원을 찍는다."""
        반지름 = 붓_두께 / 2
        self.캔버스.create_oval(x - 반지름, y - 반지름, x + 반지름, y + 반지름, fill="white", outline="white")
        self.붓.ellipse([x - 반지름, y - 반지름, x + 반지름, y + 반지름], fill=255)

    def 그리기_끝(self, 이벤트):
        # 마우스를 뗄 때마다 인식 결과를 갱신한다
        self.이전_점 = None
        self.인식하기()

    def 인식하기(self):
        숫자, 확률 = 예측하기(self.모델, self.그림)
        if 숫자 is None:
            return
        self.결과_글자.config(text=str(숫자))
        # 확률이 높은 순서대로 상위 3개를 표시
        상위 = sorted(enumerate(확률), key=lambda 항목: 항목[1], reverse=True)[:3]
        self.확률_글자.config(text="\n".join(f"{번호}: {값 * 100:5.1f}%" for 번호, 값 in 상위))

    def 지우기(self):
        self.캔버스.delete("all")
        self.붓.rectangle([0, 0, 캔버스_크기, 캔버스_크기], fill=0)
        self.결과_글자.config(text="?")
        self.확률_글자.config(text="숫자를 그려 보세요")


if __name__ == "__main__":
    루트 = tk.Tk()
    손글씨앱(루트, 모델_불러오기())
    루트.mainloop()
