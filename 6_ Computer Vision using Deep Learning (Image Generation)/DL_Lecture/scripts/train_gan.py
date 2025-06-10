import torch
import time
import os
import random
import sys
import yaml

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from DL_Lecture.models.gan import vanilla_D, vanilla_G, G_Loss, D_Loss
from DL_Lecture.models.cgan import conditional_G, conditional_D
from DL_Lecture.models.dcgan import dcgan_D, dcgan_G

def main():

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/fashion_mnist_gan.yaml'

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
    #전처리 설정
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
    if params['task'] == "MNIST":
        # 파이토치에서 제공하는 MNIST dataset
        # transform -> 전처리 변환 적용
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    elif params['task'] == "Fashion":
        # dataset & loader
        train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)

    # 배치 단위로 네트워크에 데이터를 넘겨주는 Data loader
    train_loader = DataLoader(dataset=train_dataset,batch_size=params['batch_size'], shuffle=True)
    # 학습 모델 생성

    if params['model'] == 'Vanilla':
        G = vanilla_G(params['z_dim'], params['img_size']).to(device)
        D = vanilla_D(params['img_size']).to(device)
    elif params['model'] == 'CGAN':
        G = conditional_G(params['z_dim'], params['img_size']).to(device)
        D = conditional_D(params['img_size']).to(device)
    elif params['model'] == 'DCGAN':
        G = dcgan_G(params['z_dim'], params['img_size']).to(device)
        D = dcgan_D(params['img_size']).to(device)

    # Loss
    G_loss = G_Loss(device)
    D_loss = D_Loss(device)

    # optimizer
    G_optim = torch.optim.Adam(G.parameters(), lr=params['lr_G'], betas=(params['beta1'], params['beta2']))
    D_optim = torch.optim.Adam(D.parameters(), lr=params['lr_D'], betas=(params['beta1'], params['beta2']))

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    images_dir = os.path.abspath(os.path.join(out_dir, "images"))
    summary_dir = os.path.join(out_dir, "summaries")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    writer = SummaryWriter(summary_dir) # TensorBoard를 위한 초기화
     # training 시작
    start_time = time.time()
    global_steps = 0
    print('========================================')
    print("Start training...")
    # 이미지 생성에 사용할 고정 random number
    if params['model'] == 'Vanilla' or params['model'] == 'DCGAN':
        eval_z = torch.randn(params['num_show_img'], params['z_dim']).to(device)
    elif params['model'] == 'CGAN':
        eval_c = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
                              dtype=torch.float32).to(device)  # 만들어야 할 condition을 지정.
        eval_z = torch.randn(eval_c.shape[0], params['z_dim']).to(device)


    def to_img(x):
        return torch.clamp((x + 1) / 2, 0, 1)  # 0~1 사이로 강제로만들기

    for epoch in range(params['max_epochs']):
        for images, labels in train_loader:
            G.train()
            D.train()
            images = images.to(device) #이미지랑 레이블 불러옴 vanilla g에선 레이블 불필요
            # 리얼 이미지로 불러옴
            labels = labels.to(device)
            ### update D ###
            z = torch.randn(params['batch_size'], params['z_dim']).to(device) #100,100차원짜리 노이즈 생성


            if params['model'] == 'Vanilla' or params['model'] == 'DCGAN':
                fake_images = G(z)       #fake 이미지 생성
                D_real = D(images)  #real이미지에대한 loss
                D_fake = D(fake_images.detach()) #fake 이미지에 대한 loss
            elif params['model'] == 'CGAN':
                fake_images = G(z, labels)
                D_real = D(images, labels)
                D_fake = D(fake_images.detach(), labels)

            d_loss, d_real_loss, d_fake_loss = D_loss(D_real, D_fake) #D_loss 까지 들어가야 로스값이 나옴 ,이전에는 확률값이였음

            D_optim.zero_grad()
            d_loss.backward()
            D_optim.step()

            ### update G ###
            z = torch.randn(params['batch_size'], params['z_dim']).to(device)
            if params['model'] == 'Vanilla' or params['model'] == 'DCGAN':
                fake_images = G(z) #fake 이미지를 만듬
                G_fake = D(fake_images) #fake 이미지를 줫을때의 확률 값 =얼마나 진짜처럼 생각하나
            elif params['model'] == 'CGAN':
                fake_images = G(z, labels)
                G_fake = D(fake_images, labels)

            g_loss = G_loss(G_fake) #그거에 대해서 loss 로 넣음

            G_optim.zero_grad()
            g_loss.backward()
            G_optim.step()

            writer.add_scalar("Batch/G_Loss", g_loss.item(), global_steps)
            writer.add_scalar("Batch/D_Loss", d_loss.item(), global_steps)

            global_steps += 1
            if (global_steps) % 300 == 0:
                print('Epoch [{}], Step [{}], G_Loss: {:.4f}, D_Loss: {:.4f}'.format(epoch+1, global_steps, g_loss.item(), d_loss.item()))
                G.eval()
                D.eval()
                if params['model'] == 'Vanilla' or params['model'] == 'DCGAN':
                    fake_images = G(eval_z) # eval을 위한 noise를 넣어서 fake 이미지 만듬
                elif params['model'] == 'CGAN':
                    fake_images = G(eval_z, eval_c)
                save_image(to_img(fake_images), images_dir+f'/gen_imgs_{global_steps}.jpg') #to _img로 픽셀값 조정

        G_save_path = checkpoint_dir + '/epoch_' + str(epoch + 1) + '_G.pth'
        D_save_path = checkpoint_dir + '/epoch_' + str(epoch + 1) + '_D.pth'
        torch.save({'epoch': epoch + 1, 'model_state_dict': G.state_dict()}, G_save_path)
        torch.save({'epoch': epoch + 1, 'model_state_dict': D.state_dict()}, D_save_path)
        #파라미터 따로 저장해줌
        training_time = (time.time() - start_time) / 60
        print('========================================')
        print("epoch:", epoch + 1, "/ global_steps:", global_steps)
        print(f'LOSS value G : {g_loss:.4f} / D(r, f) : {d_loss:.4f} ({d_real_loss:.4f} , {d_fake_loss:.4f})')
        print("training_time: %.2f minutes" % training_time)

        epoch += 1

if __name__ == '__main__':
    main()