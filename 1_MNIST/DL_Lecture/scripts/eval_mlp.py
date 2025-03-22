import torch
import yaml
import sys
import os

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset
from DL_Lecture.models.mlp import MNIST_model


def main():
    # 평가를 위한 메시지 출력 (문장 분류 CNN이라고 되어 있으나, 실제로는 MNIST 모델 평가)
    print('CNN for sentence classification evaluation')

    # 명령행 인자를 통해 설정 파일 경로를 전달받습니다.
    # 인자가 있으면 해당 파일 경로를 사용하고, 없으면 기본 경로를 사용합니다.
    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/mnist_mlp.yaml'

    # YAML 설정 파일을 열어 파라미터들을 읽어옵니다.
    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # GPU 사용 가능 여부를 확인하고, 사용 가능하면 GPU를, 그렇지 않으면 CPU를 사용합니다.
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 결과 저장 및 체크포인트 경로를 설정하기 위해 타임스탬프를 이용한 폴더 경로를 생성합니다.
    timestamp = "1742613596"  # 하드코딩된 타임스탬프 값입니다.
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

    # 테스트 데이터셋 로드
    if params['task'] == "MNIST":
        # MNIST 테스트 데이터셋을 로드합니다 (train=False로 설정)
        mnist_test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,  # 테스트 데이터를 사용
            transform=transforms.ToTensor(),  # 이미지를 텐서로 변환
            download=True  # 데이터가 없으면 다운로드
        )
    elif params['task'] == "CIFAR10":
        # CIFAR10 작업을 위한 코드는 여기에 추가하면 됩니다.
        pass

    # 테스트 데이터셋의 총 데이터 개수를 출력합니다.
    print('The number of test data: ', len(mnist_test_dataset))

    # DataLoader를 통해 테스트 데이터를 배치 단위로 불러옵니다.
    # 테스트이므로 셔플(shuffle)은 필요하지 않습니다.
    test_loader = torch.utils.data.DataLoader(
        dataset=mnist_test_dataset,
        batch_size=params['batch_size'],
        shuffle=False
    )

    # 학습에 사용했던 MNIST 모델을 생성하고, 지정한 device(GPU/CPU)로 이동합니다.
    # 평가 시에는 dropout이 적용되지 않습니다.
    model = MNIST_model().to(device)

    # 모델을 평가 모드로 전환하여, dropout, 배치 정규화 등 평가용 설정을 활성화합니다.
    model.eval()

    # 저장된 체크포인트 파일 경로를 지정합니다. (최고 성능 모델이 저장된 파일)
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints/best.pth"))

    # 체크포인트 파일을 불러와 모델의 파라미터(state_dict)를 로드합니다.
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 맞춘 샘플 수를 누적할 변수를 초기화합니다.
    correct_cnt = 0

    # 전체 배치의 예측 결과를 저장할 변수들을 초기화합니다.
    iter = 0
    for x, y in test_loader:
        # 배치 데이터를 device(GPU 또는 CPU)로 이동합니다.
        x = x.to(device)
        y = y.to(device)

        # MNIST 이미지는 원래 [batch_size, 1, 28, 28] 형태이므로,
        # 모델에 입력하기 위해 [batch_size, 784] 형태의 1차원 벡터로 평탄화합니다.
        pred = model.forward(x.view(-1, 28 * 28))

        # 모델의 출력(pred)에서 가장 높은 값을 가진 클래스 인덱스를 선택합니다.
        _, top_pred = torch.topk(pred, k=1, dim=-1)
        top_pred = top_pred.squeeze(dim=1)  # 불필요한 차원을 제거합니다.

        # 예측 결과(top_pred)와 실제 레이블(y)이 일치하는 경우의 수를 누적합니다.
        correct_cnt += int(torch.sum(top_pred == y))

        # 첫 번째 배치일 때 초기값을 저장하고, 이후 배치의 결과와 이어 붙입니다.
        if iter == 0:
            total_pred = top_pred
            total_y = y
        else:
            total_pred = torch.cat((total_pred, top_pred))
            total_y = torch.cat((total_y, y))
        iter += 1

    # 전체 테스트 데이터에 대한 정확도를 계산합니다.
    accuracy = correct_cnt / len(mnist_test_dataset) * 100
    print("test accuracy: %.2f%%" % accuracy)


if __name__ == "__main__":
    main()
