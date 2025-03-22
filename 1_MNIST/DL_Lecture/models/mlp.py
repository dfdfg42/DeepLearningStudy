import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

class MNIST_model(nn.Module):
    def __init__(self, drop_prop=0.5):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(drop_prop)
        self._init_he()

    def _init_he(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten input
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.dropout(out)

        return self.fc4(out)