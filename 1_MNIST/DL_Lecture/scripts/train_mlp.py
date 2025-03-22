import torch
import time
import os
import random
import sys
import yaml

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# 사용자 정의 MLP 모델: MNIST 데이터셋에 적합한 모델
from DL_Lecture.models.mlp import MNIST_model


def main():
    # 설정 파일 경로를 명령행 인자로 받을 수 있으며, 없으면 기본 파일 경로 사용
    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)  # 전달된 인자 출력
    else:
        params_filename = '../config/mnist_mlp.yaml'  # 기본 설정 파일 경로

    # YAML 파일을 열어 파라미터들을 로드 (예: 학습률, 배치 크기 등)
    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # 랜덤 시드 설정: 학습 결과의 재현성을 위해 파이썬, PyTorch, CUDA의 시드 설정
    if 'random_seed' in params:
        seed = params['random_seed']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 사용 가능한 디바이스 설정: GPU 사용 가능하면 GPU, 아니면 CPU 사용
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    torch.backends.cudnn.benchmark = True  # cuDNN 벤치마크를 활성화하여 최적의 연산 알고리즘 선택

    # 데이터 로드: MNIST 데이터셋 사용
    if params['task'] == "MNIST":
        # PyTorch 내장 MNIST 데이터셋을 로드 (train 데이터를 사용)
        mnist_train_val_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            transform=transforms.ToTensor(),  # 이미지를 텐서 형식으로 변환
            download=True  # 데이터가 없으면 다운로드
        )

        # 전체 데이터 개수 출력
        print('The number of training data : ', len(mnist_train_val_dataset))
        # 학습 데이터와 검증 데이터로 랜덤하게 분할 (50,000개와 10,000개)
        train_dataset, val_dataset = torch.utils.data.random_split(mnist_train_val_dataset, [50000, 10000])
        print('The number of training data : ', len(train_dataset))
        print('The number of validation data : ', len(val_dataset))

    elif params['task'] == "CIFAR10":
        # CIFAR10 데이터셋을 사용하려면 여기에 구현 코드를 추가하면 됩니다.
        pass

    # DataLoader 생성: 데이터를 배치 단위로 모델에 전달
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['batch_size'], shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=params['batch_size'], shuffle=False)

    # 학습할 모델 생성 및 지정한 device(GPU/CPU)로 이동
    model = MNIST_model(params['dropout_rate']).to(device)

    # 손실 함수 정의: 다중 클래스 분류 문제에 적합한 CrossEntropyLoss 사용
    criterion = nn.CrossEntropyLoss()
    # Adam 옵티마이저 설정: 학습률과 L2 정규화(weight_decay) 적용
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['l2_reg_lambda'])

    # 학습 결과와 체크포인트 저장을 위한 디렉토리 설정 (현재 시간 기준 폴더 생성)
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    summary_dir = os.path.join(out_dir, "summaries")

    # 체크포인트 디렉토리가 없으면 생성
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # TensorBoard 기록을 위한 SummaryWriter 초기화
    writer = SummaryWriter(summary_dir)

    # 학습 시작 전 초기화: 시간 기록, 최고 검증 정확도 및 전체 스텝 수 초기화
    start_time = time.time()
    highest_val_acc = 0
    global_steps = 0
    print('========================================')
    print("Start training...")

    # 에포크 반복: 전체 데이터를 여러 번 반복하여 학습
    for epoch in range(params['max_epochs']):
        train_loss = 0  # 해당 에포크 동안 누적된 손실 초기화
        train_correct_cnt = 0  # 해당 에포크 동안 맞춘 샘플 수 초기화
        train_batch_cnt = 0  # 해당 에포크 동안의 배치 수 초기화

        model.train()  # 모델을 학습 모드로 전환 (dropout, batch norm 등이 활성화)

        # 각 배치에 대해 학습 진행
        for x, y in train_loader:
            # 배치 데이터를 지정한 device로 이동 (GPU/CPU)
            x = x.to(device)
            y = y.to(device)

            # 이전 배치의 gradient를 초기화
            optimizer.zero_grad()

            # 이미지 데이터를 1차원 벡터(784차원)로 변환 후 모델에 입력하여 출력값 계산
            # 원래 x의 shape: [batch_size, 1, 28, 28] -> 변환 후: [batch_size, 784]
            outputs = model.forward(x.view(-1, 28 * 28))

            # 모델 출력과 실제 라벨(y) 간의 CrossEntropy 손실 계산
            loss = criterion(outputs, y)
            # 손실을 기준으로 역전파 수행하여 gradient 계산
            loss.backward()
            # 옵티마이저를 사용하여 모델 파라미터 업데이트
            optimizer.step()

            # 현재 배치의 손실과 배치 수 누적
            train_loss += loss
            train_batch_cnt += 1

            # 예측 결과에서 가장 높은 값을 가진 클래스 인덱스 선택
            _, top_pred = torch.topk(outputs, k=1, dim=-1)
            top_pred = top_pred.squeeze(dim=1)  # 불필요한 차원 제거

            # 현재 배치에서 맞춘 예측의 개수를 누적 (정답과 예측이 일치하는 경우)
            train_correct_cnt += int(torch.sum(top_pred == y))

            # 현재 배치의 총 샘플 수와 맞춘 샘플 수를 이용해 배치 정확도 계산
            batch_total = y.size(0)
            batch_correct = int(torch.sum(top_pred == y))
            batch_acc = batch_correct / batch_total

            # TensorBoard에 배치별 손실과 정확도 기록
            writer.add_scalar("Batch/Loss", loss.item(), global_steps)
            writer.add_scalar("Batch/Acc", batch_acc, global_steps)

            # 전체 스텝 수 증가
            global_steps += 1

            # 100 스텝마다 현재 에포크, 스텝, 손실 출력
            if (global_steps) % 100 == 0:
                print('Epoch [{}], Step [{}], Loss: {:.4f}'.format(epoch + 1, global_steps, loss.item()))

        # 에포크 종료 후, 전체 학습 데이터에 대한 평균 손실과 정확도 계산
        train_acc = train_correct_cnt / len(train_dataset) * 100  # 학습 정확도 (%)
        train_ave_loss = train_loss / train_batch_cnt  # 평균 손실
        training_time = (time.time() - start_time) / 60  # 총 학습 시간 (분 단위)

        # 에포크 단위 손실과 정확도를 TensorBoard에 기록
        writer.add_scalar("Train/Loss", train_ave_loss, epoch)
        writer.add_scalar("Train/Acc", train_acc, epoch)

        print('========================================')
        print("epoch:", epoch + 1, "/ global_steps:", global_steps)
        print("training dataset average loss: %.3f" % train_ave_loss)
        print("training_time: %.2f minutes" % training_time)

        # 검증 단계: 모델의 일반화 성능을 평가하기 위한 부분 (early stopping 기준)
        val_correct_cnt = 0  # 검증 데이터에서 맞춘 샘플 수 초기화
        val_loss = 0  # 검증 손실 누적 초기화
        val_batch_cnt = 0  # 검증 배치 수 초기화

        model.eval()  # 모델을 평가 모드로 전환 (dropout 등 비활성화)
        with torch.no_grad():  # 검증 시에는 gradient 계산 불필요
            for x, y in dev_loader:
                x = x.to(device)
                y = y.to(device)

                # 이미지를 1차원 벡터로 변환 후 모델에 입력, 출력 계산
                outputs = model.forward(x.view(-1, 28 * 28))
                # 검증 손실 계산
                loss = criterion(outputs, y)
                val_loss += loss.item()  # 손실 누적
                val_batch_cnt += 1  # 배치 수 증가

                # 예측 결과 처리: 가장 높은 값을 가진 클래스 인덱스 선택
                _, top_pred = torch.topk(outputs, k=1, dim=-1)
                top_pred = top_pred.squeeze(dim=1)
                # 맞춘 예측 개수를 누적
                val_correct_cnt += int(torch.sum(top_pred == y))

        # 전체 검증 데이터에 대한 정확도와 평균 손실 계산
        val_acc = val_correct_cnt / len(val_dataset) * 100  # 검증 정확도 (%)
        val_ave_loss = val_loss / val_batch_cnt  # 검증 평균 손실
        print("validation dataset accuracy: %.2f" % val_acc)

        # TensorBoard에 검증 손실과 정확도 기록
        writer.add_scalar("Val/Loss", val_ave_loss, epoch)
        writer.add_scalar("Val/Acc", val_acc, epoch)

        # 모델 성능이 개선되면(검증 정확도가 최고 기록을 갱신하면) 체크포인트 저장
        if val_acc > highest_val_acc:
            # 현재 에포크 모델을 저장 (에포크별 파일)
            save_path = checkpoint_dir + '/epoch_' + str(epoch + 1) + '.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict()
            }, save_path)

            # 최고 성능 모델을 best.pth로 저장 (나중에 불러오기 용이)
            save_path = checkpoint_dir + '/best.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict()
            }, save_path)
            highest_val_acc = val_acc  # 최고 검증 정확도 갱신

        # for문 내부에서 에포크 값을 증가시키는 것은 불필요합니다.
        epoch += 1


if __name__ == '__main__':
    main()
