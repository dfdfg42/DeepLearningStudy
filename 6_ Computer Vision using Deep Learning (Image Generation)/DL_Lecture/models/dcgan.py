import torch.nn as nn

# Networks G & D
class dcgan_G(nn.Module):
    def __init__(self, z_dim, img_size): #노이즈의 크기 와 만들 이미지 크기
        super().__init__()
        self.img_size = img_size
        self.G = nn.Sequential( #점점
            #1x1에서 7ㅌ7 64채널로 키움
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=64, kernel_size=7,
                               stride=1, padding=0, bias=False), # [64, 7, 7]
            nn.BatchNorm2d(64), #정규화
            nn.ReLU(), #렐루 사용

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4,
                               stride=2, padding=1, bias=False), # [32, 14, 14]
            nn.BatchNorm2d(32), #정규화
            nn.ReLU(),

            #채널을 1로 흑백이미지로 최종 설정함
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4,
                               stride=2, padding=1, bias=False), # [1, 28, 28] 결과 크기
            nn.Tanh() #결과값 -1~1사이로 제한
        )

        for m in self.modules() : #가중치 초기화
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02) #평균 0 표준편차 0.02정규분포로 초기화함
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        batch_size = x.shape[0] #배치 크기 가져옴
        x = x.view(batch_size, -1, 1, 1)
        out = self.G(x)
        return out #흑백이미지 반환

class dcgan_D(nn.Module):
    def __init__(self, img_size): #이미지 사이즈 입력
        super().__init__() #점점 축소시킴
        self.img_size = img_size
        self.D = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4,
                      stride=2, padding=1, bias=False), # [32, 14, 14]
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                      stride=2, padding=1, bias=False), # [64, 7, 7]
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7,
                      stride=1, padding=0, bias=False), # [1, 1, 1]
            nn.Sigmoid(),
        )

        for m in self.modules() :
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        batch_size = x.shape[0] # [b, 1, 28, 28]
        out = self.D(x) # [batch, 1, 1, 1]
        out = out.squeeze()
        return out