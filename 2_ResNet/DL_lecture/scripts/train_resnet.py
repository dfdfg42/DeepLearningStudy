import torch
import time
import os
import random
import sys
import yaml

import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler

from DL_Lecture.models.resnet import ResNet32_model

def main():

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/cifar10_resnet.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # 랜덤 시드 세팅
    if 'random_seed' in params:
        seed = params['random_seed']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    torch.backends.cudnn.benchmark = True

    # 데이터 로드
    if params['task'] == "ImageNet":
        pass

    elif params['task'] == "CIFAR10":
        # 파이토치에서 제공하는 MNIST dataset
        # train data augmentation : 1) size 4만큼 패딩 후 32의 크기로 random cropping, 2) 데이터 좌우반전(2배).
        transforms_train = transforms.Compose([  # training data를 위한 transforms
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # input image 정규화 (standardization)
        ])
        transforms_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # validation data를 위한 transforms
        ])

        # CIFAR10 dataset 다운로드
        train_data = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transforms_train,
                                                  download=True)
        val_data = torchvision.datasets.CIFAR10(root='./data/', train=True, transform=transforms_val, download=True)

        # train_data, val_data의 분리
        indices = list(range(len(train_data)))
        np.random.shuffle(indices)  # training, val을 random하게 sampling하기 위해 index들을 shuffling
        train_indices, val_indices = indices[5000:], indices[:5000]  # training, val data에 대한 index들
        train_sampler = SubsetRandomSampler(
            train_indices)  # DataLoader 과정에서 training과 validation을 sampling하기 위한 Sampler
        val_sampler = SubsetRandomSampler(val_indices)

        # data 개수 확인
        print('The number of training data: ', len(train_indices))
        print('The number of validation data: ', len(val_indices))

    # 배치 단위로 네트워크에 데이터를 넘겨주는 Data loader
    train_loader = torch.utils.data.DataLoader(train_data, params['batch_size'], sampler=train_sampler)
    dev_loader = torch.utils.data.DataLoader(val_data, params['batch_size'], sampler=val_sampler)

    # 학습 모델 생성
    model = ResNet32_model().to(device)  # 모델을 지정한 device로 올려줌

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'], weight_decay=params['l2_reg_lambda'])  # model.parameters -> 가중치 w들을 의미
    decay_step = [32000, 48000]
    step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=decay_step, gamma=0.1)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    summary_dir = os.path.join(out_dir, "summaries")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    writer = SummaryWriter(summary_dir) # TensorBoard를 위한 초기화
     # training 시작
    start_time = time.time()
    highest_val_acc = 0
    global_steps = 0
    print('========================================')
    print("Start training...")
    for epoch in range(params['max_epochs']):
        train_loss = 0
        train_correct_cnt = 0
        train_batch_cnt = 0
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()# iteration 마다 gradient를 0으로 초기화
            outputs = model.forward(x) # 28 * 28 이미지를 784 features로 reshape 후 forward
            loss = criterion(outputs, y)# cross entropy loss 계산
            loss.backward()# 가중치 w에 대해 loss를 미분
            optimizer.step()# 가중치들을 업데이트
            step_lr_scheduler.step() # learning rate 업데이트

            train_loss += loss
            train_batch_cnt += 1

            _, top_pred = torch.topk(outputs, k=1, dim=-1)
            top_pred = top_pred.squeeze(dim=1)
            train_correct_cnt += int(torch.sum(top_pred == y))  # 맞춘 개수 카운트

            batch_total = y.size(0)
            batch_correct = int(torch.sum(top_pred == y))
            batch_acc = batch_correct / batch_total

            writer.add_scalar("Batch/Loss", loss.item(), global_steps)
            writer.add_scalar("Batch/Acc", batch_acc, global_steps)

            writer.add_scalar("LR/Learning_rate", step_lr_scheduler.get_last_lr()[0], global_steps)

            global_steps += 1
            if (global_steps) % 100 == 0:
                print('Epoch [{}], Step [{}], Loss: {:.4f}'.format(epoch+1, global_steps, loss.item()))

        train_acc = train_correct_cnt / len(train_indices) * 100
        train_ave_loss = train_loss / train_batch_cnt # 학습 데이터의 평균 loss
        training_time = (time.time() - start_time) / 60
        writer.add_scalar("Train/Loss", train_ave_loss, epoch)
        writer.add_scalar("Train/Acc", train_acc, epoch)
        print('========================================')
        print("epoch:", epoch + 1, "/ global_steps:", global_steps)
        print("training dataset average loss: %.3f" % train_ave_loss)
        print("training_time: %.2f minutes" % training_time)
        print("learning rate: %.6f" % step_lr_scheduler.get_last_lr()[0])

        # validation (for early stopping)
        val_correct_cnt = 0
        val_loss = 0
        val_batch_cnt = 0
        model.eval()
        for x, y in dev_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model.forward(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            val_batch_cnt += 1
            _, top_pred = torch.topk(outputs, k=1, dim=-1)
            top_pred = top_pred.squeeze(dim=1)
            val_correct_cnt += int(torch.sum(top_pred == y))# 맞춘 개수 카운트

        val_acc = val_correct_cnt / len(val_indices) * 100
        val_ave_loss = val_loss / val_batch_cnt
        print("validation dataset accuracy: %.2f" % val_acc)
        writer.add_scalar("Val/Loss", val_ave_loss, epoch)
        writer.add_scalar("Val/Acc", val_acc, epoch)

        if val_acc > highest_val_acc:# validation accuracy가 경신될 때
            save_path = checkpoint_dir + '/epoch_' + str(epoch + 1) + '.pth'
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)

            save_path = checkpoint_dir + '/best.pth'
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)  # best accuracy에 도달할 때만 모델을 저장함으로써 early stopping
            highest_val_acc = val_acc
        epoch += 1

if __name__ == '__main__':
    main()