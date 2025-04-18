import numpy as np
import torch
from torch import nn


class ConvFeatures(nn.Module):
    def __init__(self, word_dimension, filter_lengths, filter_counts, dropout_rate):
        super().__init__()
        conv = [] # convolution filter 모듈을 넣어둘 list
        for size, num in zip(filter_lengths, filter_counts): # for문을 통해 filter size 별로 초기화
            conv2d = nn.Conv2d(1, num, (size, word_dimension)) # (input_channel, ouput_channel, height, width)
            nn.init.kaiming_normal_(conv2d.weight, mode='fan_out', nonlinearity='relu') # He initialization
            nn.init.zeros_(conv2d.bias) # bias는 0으로 초기화
            conv.append(nn.Sequential(conv2d, nn.ReLU(inplace=True))) # conv2d와 activation 순으로 list에 저장

        self.conv = nn.ModuleList(conv) #conv list를 ModuleList로 변환하여 인스턴스 변수로 초기화
        self.filter_sizes = filter_lengths # self.filter_sizes 인스턴스 변수를 filter_lengths로 초기화
        self.dropout = nn.Dropout(dropout_rate) #dropout_rate를 바탕으로 Dropout 모듈인 self.dropout 인스턴스 변수 초기화

    def forward(self, embedded_words):
        features = [] #
        for filter_size, conv in zip(self.filter_sizes, self.conv): #filter size 별로 convolution 수행
            # embedded_words: [batch, sentence length, embedding dimension]
            conv_output = conv(embedded_words) #[batch, sentence length - filter size + 1, 1]
            conv_output = conv_output.squeeze(-1).max(dim=-1)[0]  # max over-time pooling -> [batch, 1]
            features.append(conv_output)
            del conv_output

        features = torch.cat(features, dim=1) # 각각의 filter에서 나온 feature들을 concatenation
        dropped_features = self.dropout(features)
        return dropped_features

class SentenceCnn(nn.Module):
    def __init__(self, nb_classes, word_embedding_numpy, filter_lengths, filter_counts, dropout_rate):
        super().__init__()

        vocab_size = word_embedding_numpy.shape[0]
        word_dimension = word_embedding_numpy.shape[1]

        # Word embedding layer
        self.word_embedding = nn.Embedding(
            vocab_size,
            word_dimension,
            padding_idx=0
        )

        # word2vec 활용
        self.word_embedding.weight.detach().copy_(torch.tensor(word_embedding_numpy.astype(np.float32)))

        # Convolutional layer
        self.features = ConvFeatures(word_dimension, filter_lengths, filter_counts, dropout_rate)

        # Fully-connected layer
        nb_total_filters = sum(filter_counts)
        self.linear = nn.Linear(nb_total_filters, nb_classes)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, input_x):
        x = self.word_embedding(input_x)
        x = x.unsqueeze(1)  # 채널 1개 추가
        x = self.features(x)
        logits = self.linear(x)
        return logits