import torch
import torch.nn as nn

# Networks G & D
class conditional_G(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.G = nn.Sequential(
            nn.Linear(in_features=z_dim * 2, out_features=256),  # input size is z_dim + condition size
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=self.img_size * self.img_size),
            nn.Tanh()
        )

    def forward(self, x, c):
        batch_size = x.shape[0]
        c = c.unsqueeze(1).expand(x.size())  # condition을 배치 사이즈 만큼 확장하여 x와 크기를 맞춰준다.
        #one hot 인코딩으로 만들어서 붙임
        x = torch.cat((x, c), dim=1)  # noise와 condition을 합쳐서 G의 입력으로 사용한다.
        out = self.G(x)
        out = out.view(batch_size, 1, self.img_size, self.img_size)
        return out


class conditional_D(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.D = nn.Sequential( #이미지 사이즈가 들어오면 1차원 실수값으로 만들음
            nn.Linear(in_features=self.img_size * self.img_size * 2, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x, c):
        batch_size = x.shape[0]
        out = x.view(batch_size, -1)
        c = c.unsqueeze(1).expand(out.size())
        out = torch.cat((out, c), dim=1)
        out = self.D(out)  # [batch, 1] #condition 더해서 discriminator에 전달
        return out