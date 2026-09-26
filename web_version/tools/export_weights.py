# -*- coding: utf-8 -*-
"""desktop_version의 mnist_cnn.pt를 웹에서 읽을 수 있는 weights.js로 변환한다.

- BatchNorm은 추론 시 고정된 선형 변환이므로 바로 앞 합성곱의 가중치·편향에 합쳐서 내보낸다.
- 모든 값은 리틀 엔디언 float32로 이어 붙인 뒤 base64 문자열로 weights.js에 넣는다.
  (script 태그로 읽으므로 index.html을 file://로 직접 열어도 동작한다.)
- 검증용으로 MNIST 시험 이미지와 PyTorch 출력값(로짓)을 tests/fixtures.js로 함께 내보낸다.

실행: C:\\Users\\hj831\\.venvs\\mnist\\Scripts\\python.exe web_version/tools/export_weights.py
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

웹_폴더 = Path(__file__).resolve().parent.parent
데스크톱_폴더 = 웹_폴더.parent / "desktop_version"
sys.path.insert(0, str(데스크톱_폴더))

from model import 가중치_파일, 데이터_폴더, 숫자인식망, 평균, 표준편차  # noqa: E402

검증_이미지_수 = 1000


def 배치정규화_합치기(합성곱, 정규화):
    """합성곱 뒤의 BatchNorm을 합성곱 가중치·편향에 합친 값을 돌려준다."""
    배율 = 정규화.weight / torch.sqrt(정규화.running_var + 정규화.eps)
    가중치 = 합성곱.weight * 배율.view(-1, 1, 1, 1)
    편향 = (합성곱.bias - 정규화.running_mean) * 배율 + 정규화.bias
    return 가중치.detach(), 편향.detach()


def 합친_모델로_추론(층들, 입력):
    """합친 가중치만으로 추론해서 원래 모델과 결과가 같은지 확인하는 용도."""
    출력 = 입력
    for 번호 in (1, 2, 3):
        출력 = F.conv2d(출력, 층들[f"합성곱{번호}.가중치"], 층들[f"합성곱{번호}.편향"], padding=1)
        출력 = F.max_pool2d(F.relu(출력), 2)
    출력 = torch.flatten(출력, 1)
    출력 = F.relu(F.linear(출력, 층들["완전연결1.가중치"], 층들["완전연결1.편향"]))
    return F.linear(출력, 층들["완전연결2.가중치"], 층들["완전연결2.편향"])


def base64_변환(배열):
    return base64.b64encode(np.ascontiguousarray(배열).tobytes()).decode("ascii")


def 메인():
    모델 = 숫자인식망()
    모델.load_state_dict(torch.load(가중치_파일, map_location="cpu", weights_only=True))
    모델.eval()

    # 1) 내보낼 층 모으기 (BatchNorm 합치기)
    층들 = {}
    for 번호 in (1, 2, 3):
        가중치, 편향 = 배치정규화_합치기(getattr(모델, f"합성곱{번호}"), getattr(모델, f"정규화{번호}"))
        층들[f"합성곱{번호}.가중치"] = 가중치
        층들[f"합성곱{번호}.편향"] = 편향
    for 번호 in (1, 2):
        완전연결 = getattr(모델, f"완전연결{번호}")
        층들[f"완전연결{번호}.가중치"] = 완전연결.weight.detach()
        층들[f"완전연결{번호}.편향"] = 완전연결.bias.detach()

    # 2) MNIST 시험 이미지로 원래 모델과 합친 모델의 출력 비교
    from torchvision import datasets
    시험_데이터 = datasets.MNIST(데이터_폴더, train=False, download=True)
    원본_이미지 = 시험_데이터.data[:검증_이미지_수].numpy().astype(np.uint8)
    정답 = 시험_데이터.targets[:검증_이미지_수].numpy().astype(np.uint8)
    입력 = (torch.from_numpy(원본_이미지).float().unsqueeze(1) / 255.0 - 평균) / 표준편차
    with torch.no_grad():
        원래_로짓 = 모델(입력)
        합친_로짓 = 합친_모델로_추론(층들, 입력)
    차이 = (원래_로짓 - 합친_로짓).abs().max().item()
    정확도 = (원래_로짓.argmax(1).numpy() == 정답).mean() * 100
    print(f"BatchNorm 합치기 전후 최대 차이: {차이:.2e}")
    print(f"PyTorch 기준 정확도 ({검증_이미지_수}장): {정확도:.2f}%")
    assert 차이 < 1e-4, "BatchNorm 합치기 결과가 원래 모델과 다릅니다"

    # 3) weights.js 쓰기
    목차, 조각들, 위치 = {}, [], 0
    for 이름, 텐서 in 층들.items():
        배열 = 텐서.numpy().astype("<f4").ravel()
        목차[이름] = {"모양": list(텐서.shape), "시작": 위치, "길이": int(배열.size)}
        조각들.append(배열)
        위치 += 배열.size
    전체 = np.concatenate(조각들)
    with open(웹_폴더 / "weights.js", "w", encoding="utf-8") as 파일:
        파일.write("// 자동 생성 파일: tools/export_weights.py가 desktop_version/mnist_cnn.pt에서 만든다. 직접 수정하지 말 것.\n")
        파일.write("// BatchNorm을 합성곱에 합친 가중치를 리틀 엔디언 float32로 이어 붙여 base64로 담았다.\n")
        파일.write(f"globalThis.가중치_묶음 = {{\n  목차: {json.dumps(목차, ensure_ascii=False)},\n")
        파일.write(f"  데이터: \"{base64_변환(전체)}\"\n}};\n")
    print(f"weights.js 저장: 값 {전체.size:,}개")

    # 4) 검증용 tests/fixtures.js 쓰기
    (웹_폴더 / "tests").mkdir(exist_ok=True)
    with open(웹_폴더 / "tests" / "fixtures.js", "w", encoding="utf-8") as 파일:
        파일.write("// 자동 생성 파일: tools/export_weights.py가 만든 검증용 데이터 (MNIST 시험 이미지와 PyTorch 출력값).\n")
        파일.write("globalThis.검증_자료 = {\n")
        파일.write(f"  개수: {검증_이미지_수},\n")
        파일.write(f"  이미지: \"{base64_변환(원본_이미지)}\",\n")
        파일.write(f"  정답: \"{base64_변환(정답)}\",\n")
        파일.write(f"  로짓: \"{base64_변환(원래_로짓.numpy().astype('<f4'))}\"\n}};\n")
    print("tests/fixtures.js 저장")


if __name__ == "__main__":
    메인()
