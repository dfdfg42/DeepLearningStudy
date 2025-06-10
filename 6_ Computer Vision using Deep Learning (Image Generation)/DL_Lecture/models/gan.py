import torch
import torch.nn as nn


class vanilla_G(nn.Module):
    def __init__(self, z_dim, img_size): #노이즈가 벡터가 몇차원으로 시작하는지 입력 , 이미지사이즈 입력
        super().__init__()
        self.img_size = img_size
        self.G = nn.Sequential( # 실제 제네레이터가 연산하는 레이어를 쌓음
            nn.Linear(in_features=z_dim, out_features=256), #256차원으로 늘림
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=self.img_size * self.img_size), #이미지 사이즈로 맞춰준다.
            #이미지 사이즈인 784로 맞춰줌
            nn.Tanh() # 출력값을 -1~1 사이로 제한함
        )
    def forward(self, x): # [batch, z_dim]
        batch_size = x.shape[0]
        out = self.G(x) # out 은 784 짜리 벡터
        out = out.view(batch_size, 1, self.img_size, self.img_size)
        #view = 리사이즈 해주는 함수  채널은 1
        return out

class vanilla_D(nn.Module):
    def __init__(self, img_size): #이미지 사이즈를 받아줌
        super().__init__()
        self.img_size = img_size
        self.D = nn.Sequential(
            nn.Linear(in_features=self.img_size * self.img_size, out_features=1024),
            #이미지 사이즈로 시작
            nn.LeakyReLU(negative_slope=0.2), #LeakyReLu 사용하고 점점 사이즈를 줄여나감
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid(), #시그노이드를 씌워서 1에 가까우면 real -1 에 가까우면 fake
        )

    def forward(self, x): # [batch, 1, img_size, img_size]
        batch_size = x.shape[0] #배치 사이즈만 받아서
        out = x.view(batch_size, -1) # 리사이즈 해주는데 -1 은 3개의 숫자 계산해서 알아서 맞추라는 뜻
        #1D 이미지로 2D이미지를 평탄화
        out = self.D(out) # [batch, 1] 아웃풋 실수값 -> 확률값 Discriminator 통과 시킴
        return out

class G_Loss(nn.Module):
    def __init__(self, device):
        super(G_Loss, self).__init__()
        self.device = device
        self.criterion = nn.BCELoss() # Binary cross entropy  예측 확률과 실제 레이블 간의 차이를 측정

    def forward(self, fake):
        ones = torch.ones_like(fake).to(self.device)  # Discriminator가 생성된 이미지를 실제 이미지로 판별하도록 1로 채워진 레이블 생성
        g_loss = self.criterion(fake, ones)  # 생성된 이미지가 실제 이미지로 판별되도록 Binary Cross-Entropy Loss 계산
        return g_loss


class D_Loss(nn.Module):
    def __init__(self, device):
        super(D_Loss, self).__init__()
        self.device = device
        self.criterion = nn.BCELoss() #똑같이 BCELOSS를 사용 진짜를 진짜라고하면 1

    def forward(self, D_real, D_fake):
        ones = torch.ones_like(D_real).to(self.device)  # Discriminator가 실제 이미지를 실제 이미지로 판별하도록 1로 채워진 레이블 생성
        zeros = torch.zeros_like(D_fake).to(self.device)  # Discriminator가 생성된 이미지를 가짜 이미지로 판별하도록 0으로 채워진 레이블 생성

        d_real_loss = self.criterion(D_real, ones)  # 실제 이미지가 실제 이미지로 판별되도록 Binary Cross-Entropy Loss 계산
        d_fake_loss = self.criterion(D_fake, zeros)  # 생성된 이미지가 가짜 이미지로 판별되도록 Binary Cross-Entropy Loss 계산
        d_loss = d_real_loss + d_fake_loss  # 두 Loss를 합하여 Discriminator 전체 Loss 계산

        return d_loss, d_real_loss, d_fake_loss
