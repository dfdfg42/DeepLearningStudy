import torch
from torch import nn
import torch.nn.functional as F

class MNIST_model(nn.Module):
    def __init__(self, drop_prop=0.5):
        super().__init__()
        # 첫 번째 레이어: 784 → 300
        self.fc1 = nn.Linear(28 * 28, 300)
        # 두 번째 레이어: 300 → 200
        self.fc2 = nn.Linear(300, 200)
        # 세 번째 레이어: 200 → 100 (추가된 히든 레이어)
        self.fc3 = nn.Linear(200, 100)
        # 네 번째 레이어: 100 → 10 (출력 레이어)
        self.fc4 = nn.Linear(100, 10)

        # Dropout 레이어 (drop_prop=0.5 사용)
        self.dropout = nn.Dropout(drop_prop)

        # He (Kaiming) weight initialization 적용
        self._init_he()

    def _init_he(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # ReLU 활성화 함수와 함께 사용할 때 적합한 He 초기화 적용
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # 입력 x는 [batch_size, 28*28] 형태입니다.
        out = F.relu(self.fc1(x))
        out = self.dropout(out)  # 첫 번째 히든 레이어 뒤 드롭아웃 적용

        out = F.relu(self.fc2(out))
        out = self.dropout(out)  # 두 번째 히든 레이어 뒤 드롭아웃 적용

        out = F.relu(self.fc3(out))
        out = self.dropout(out)  # 세 번째 히든 레이어 뒤 드롭아웃 적용

        # 출력 레이어 (일반적으로 출력 전에 드롭아웃은 적용하지 않습니다.)
        out = self.fc4(out)
        return out
