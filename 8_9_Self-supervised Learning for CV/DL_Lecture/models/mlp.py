import torch
from torch import nn
import torch.nn.functional as F

class MNIST_model(nn.Module):
    def __init__(self, drop_prob=0):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 10)

        self.dropout = nn.Dropout(drop_prob)  # dropout
        self._init_xavier()

    def _init_xavier(self):  # xavier weight initalization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out