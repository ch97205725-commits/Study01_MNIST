# -*- coding: utf-8 -*-
"""MNIST 숫자 인식용 합성곱 신경망(CNN) 모델 정의."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 이 파일이 있는 폴더 (어느 위치에서 실행해도 같은 파일을 쓰도록 기준으로 삼는다)
현재_폴더 = Path(__file__).resolve().parent

# 학습된 가중치를 저장할 파일 경로
가중치_파일 = 현재_폴더 / "mnist_cnn.pt"

# MNIST 데이터를 내려받을 폴더
데이터_폴더 = 현재_폴더 / "data"

# MNIST 데이터셋의 평균과 표준편차 (정규화에 사용)
평균 = 0.1307
표준편차 = 0.3081


class 숫자인식망(nn.Module):
    """28x28 흑백 이미지를 입력받아 0~9 숫자를 분류하는 CNN."""

    def __init__(self):
        super().__init__()
        # 첫 번째 합성곱 묶음: 1채널 → 32채널
        self.합성곱1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.정규화1 = nn.BatchNorm2d(32)
        # 두 번째 합성곱 묶음: 32채널 → 64채널
        self.합성곱2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.정규화2 = nn.BatchNorm2d(64)
        # 세 번째 합성곱 묶음: 64채널 → 128채널
        self.합성곱3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.정규화3 = nn.BatchNorm2d(128)
        # 과적합 방지를 위한 드롭아웃
        self.드롭아웃 = nn.Dropout(0.3)
        # 완전연결층: 128채널 x 3 x 3 → 256 → 10
        self.완전연결1 = nn.Linear(128 * 3 * 3, 256)
        self.완전연결2 = nn.Linear(256, 10)

    def forward(self, 입력):
        # 28x28 → 14x14
        출력 = F.max_pool2d(F.relu(self.정규화1(self.합성곱1(입력))), 2)
        # 14x14 → 7x7
        출력 = F.max_pool2d(F.relu(self.정규화2(self.합성곱2(출력))), 2)
        # 7x7 → 3x3
        출력 = F.max_pool2d(F.relu(self.정규화3(self.합성곱3(출력))), 2)
        # 1차원으로 펼치기
        출력 = torch.flatten(출력, 1)
        출력 = self.드롭아웃(F.relu(self.완전연결1(출력)))
        # 각 숫자(0~9)에 대한 점수(로짓) 반환
        return self.완전연결2(출력)
