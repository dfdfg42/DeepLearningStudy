import torch
from torch import nn
import torch.nn.functional as F

class LeNet_model(nn.Module):

    def __init__(self, drop_prob):  # drop_prob를 활용하여 dropout 추가해볼 것.
        super().__init__()

        # conv1:  5*5, input_channel: 3, output_channel(# of filters): 6
        # max pooling: size: 2, stride 2
        # conv2:  5*5, input_channel: 6, output_channel(# of filters): 16
        # fc1: (16 * 5 * 5, 120)
        # fc2: (120, 84)
        # fc3: (84, 10)

        # * hint he initialization: stddev = sqrt(2/n), filter에서 n 값은?
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # input channels, output channels, kernel size
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # kernel size, stride, padding = 0 (default)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # input features, output features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.dropout = nn.Dropout(drop_prob)  # dropout
        self._init_he()

    def _init_he(self):  # xavier weight initalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # conv1
        # relu
        # max_pooling
        # conv2
        # relu
        # max_pooling
        # reshape
        # fully connected layer1
        # relu
        # fully connected layer2
        # relu
        # fully connected layer3
        out = self.pool(F.relu(self.conv1(x))) # [batch_size,3,32,32] -> [batch_size,6,28,28] -> [batch_size,6,14,14]
        out = self.pool(F.relu(self.conv2(out))) # [batch_size,6,14,14] -> [batch_size,16,10,10] -> [batch_size,16,5,5]
        out = out.view(-1, 16 * 5 * 5)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out
