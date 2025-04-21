import torch
import numpy as np
import time
import re
import os
import random
import sys
import yaml
import smart_open
import pickle

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from DL_Lecture.models.sentence_cnn import SentenceCnn
from DL_Lecture.utils.text_prepro import load_mr_data, load_snips_data, buildVocab, text_to_indices, sequence_to_tensor


def main():

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/text_cnn_snips.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    data_params = params['data_files'][params['task']]

    # 랜덤 시드 세팅
    if 'random_seed' in params:
        seed = params['random_seed']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    nb_classes = 0
    # 데이터 로드
    if params['task'] == "MR":
        train_x_text, y = load_mr_data(data_params['pos_file'], data_params['neg_file'])
        nb_classes = max(y) + 1
        print("nb_classes: ", nb_classes)
        dev_sample_percentage = 0.1

    elif params['task'] == "SNIPS":
        label_dictionary = {}
        train_x_text, train_y, label_dictionary = load_snips_data(data_params['train_file'], label_dictionary)
        nb_classes = max(train_y) + 1
        print("nb_classes: ", nb_classes)

        dev_x_text, dev_y, label_dictionary = load_snips_data(data_params['dev_file'], label_dictionary)

    word_id_dict, _ = buildVocab(train_x_text, params['vocab_size'])  # training corpus를 토대로 단어사전 구축
    vocab_size = len(word_id_dict) + 4  # 30000 + 4
    print("vocabulary size: ", vocab_size)

    for word in word_id_dict.keys():
        word_id_dict[word] += 4  # <pad>: 0, <unk>: 1, <s>: 2 (a: 0 -> 4)
    word_id_dict['<pad>'] = 0  # zero padding을 위한 토큰
    word_id_dict['<unk>'] = 1  # OOV word를 위한 토큰
    word_id_dict['<s>'] = 2  # 문장 시작을 알리는 start 토큰
    word_id_dict['</s>'] = 3  # 문장 마침을 알리는 end 토큰

    if params['task'] == "MR":
        x_indices = text_to_indices(train_x_text, word_id_dict, params['use_unk_for_oov'])
        data = list(zip(x_indices, y))
        random.shuffle(data)
        x_indices, y = zip(*data)

        dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
        train_x, dev_x = x_indices[:dev_sample_index], x_indices[dev_sample_index:]
        train_y, dev_y = y[:dev_sample_index], y[dev_sample_index:]
    elif params['task'] == "SNIPS":
        train_x = text_to_indices(train_x_text, word_id_dict, params['use_unk_for_oov'])
        dev_x = text_to_indices(dev_x_text, word_id_dict, params['use_unk_for_oov'])

        # ───────────── embedding 전략 읽기 ─────────────
        # config 에서 rand / static / nonstatic 중 하나를 골라주세요.
    strategy = params.get('embedding_strategy', 'nonstatic').lower()

    # ───────── initW 초기화 ─────────
    if strategy == 'rand':
        # 완전 랜덤 초기화
        initW = np.random.uniform(
            -0.25, 0.25,
            (params['vocab_size'], params['embedding_dim'])
        ).astype(np.float32)

    else:
        # pretrained 불러올 두 경우(static, nonstatic) 공통: 랜덤으로 채워두고
        initW = np.random.uniform(
            -0.25, 0.25,
            (params['vocab_size'], params['embedding_dim'])
        ).astype(np.float32)

        # config 에 embedding_file 이 지정되어 있으면 실제로 pretrained 로드
        emb_file = params.get('embedding_file')
        if emb_file:
            print(f"Loading pretrained embeddings for strategy={strategy} …")
            pre_emb = KeyedVectors.load_word2vec_format(emb_file, binary=True)
            pre_emb.init_sims(replace=True)
            for w, idx in word_id_dict.items():
                key = re.sub('[^0-9a-zA-Z]+', '', w)
                vec = None
                if w in pre_emb:
                    vec = pre_emb[w]
                elif w.lower() in pre_emb:
                    vec = pre_emb[w.lower()]
                elif key in pre_emb:
                    vec = pre_emb[key]
                if vec is not None:
                    initW[idx] = vec.astype(np.float32)

    # PAD 토큰 임베딩은 항상 0으로 고정
    initW[0] = np.zeros(params['embedding_dim'], dtype=np.float32)
    # if params['embedding_file']:  # word2vec 활용 시
    #     print("Loading W2V data...")
    #     pre_emb = KeyedVectors.load_word2vec_format(params['embedding_file'], binary=True)  # pre-trained word2vec load
    #     pre_emb.init_sims(replace=True)
    #     num_keys = len(pre_emb.key_to_index)
    #     print("loaded word2vec len ", num_keys)
    #
    #     # initial matrix with random uniform, pretrained word2vec으로 vocabulary 내 단어들을 초기화하기 위핸 weight matrix 초기화
    #     initW = np.random.uniform(-0.25, 0.25, (params['vocab_size'], params['embedding_dim']))
    #     # load any vectors from the word2vec
    #     print("init initW cnn.W in FLAG")
    #     for w in word_id_dict.keys():
    #         arr = []
    #         s = re.sub('[^0-9a-zA-Z]+', '', w)
    #         if w in pre_emb:  # 직접 구축한 vocab 내 단어가 google word2vec에 존재하면
    #             arr = pre_emb[w]  # word2vec vector를 가져옴
    #         elif w.lower() in pre_emb:  # 소문자로도 확인
    #             arr = pre_emb[w.lower()]
    #         elif s in pre_emb:  # 전처리 후 확인
    #             arr = pre_emb[s]
    #         elif s.isdigit():  # 숫자이면
    #             arr = pre_emb['1']
    #         if len(arr) > 0:  # 직접 구축한 vocab 내 단어가 google word2vec에 존재하면
    #             idx = word_id_dict[w]  # 단어 index
    #             initW[idx] = np.asarray(arr).astype(np.float32)  # 적절한 index에 word2vec word 할당
    #         initW[0] = np.zeros(params['embedding_dim'])

    nb_pad = int(max(params['model_params_cnn']['filter_lengths']) / 2 + 0.5)

    # - 학습 데이터 배치 만들기
    train_x = sequence_to_tensor(train_x, nb_paddings=(nb_pad, nb_pad))
    train_y = torch.tensor(train_y)
    training_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=params['batch_size'], shuffle=True, num_workers=4)

    # - 데브 배치 만들기
    dev_x = sequence_to_tensor(dev_x, nb_paddings=(nb_pad, nb_pad))
    dev_y = torch.tensor(dev_y)
    dev_loader = DataLoader(TensorDataset(dev_x, dev_y), batch_size=params['batch_size'], shuffle=False)

    # 학습 모델 생성
    model = SentenceCnn(
        nb_classes=nb_classes,
        word_embedding_numpy=initW,
        filter_lengths=params['model_params_cnn']['filter_lengths'],
        filter_counts=params['model_params_cnn']['filter_counts'],
        dropout_rate=params['dropout_rate'],
        multi_channel=(strategy == 'multichannel')  # ★ 추가
    ).to(device)

    # ── 임베딩 학습 여부 ─────────────────────────────────────────
    if strategy in ('rand', 'nonstatic', 'multichannel'):
        model.word_embedding.weight.requires_grad = True
    else:  # static
        model.word_embedding.weight.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=params['optimizer_params'][params['optimizer']]['lr'], weight_decay=params['l2_reg_lambda']) # lamda 값과 함께 weight decay 적용
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, gamma=0.99, step_size=1000) # 1000 step 마다 lr을 99%로 감소시킴

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
        for x, y in training_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()# iteration 마다 gradient를 0으로 초기화
            outputs = model(x)
            loss = criterion(outputs, y)# cross entropy loss 계산
            loss.backward()# 가중치 w에 대해 loss를 미분
            optimizer.step()# 가중치들을 업데이트
            step_lr_scheduler.step(global_steps) # learning rate 업데이트

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

        train_acc = train_correct_cnt / len(train_y) * 100
        train_ave_loss = train_loss / train_batch_cnt # 학습 데이터의 평균 loss
        training_time = (time.time() - start_time) / 60
        writer.add_scalar("Train/Loss", train_ave_loss, epoch)
        writer.add_scalar("Train/Acc", train_acc, epoch)
        print('========================================')
        print("epoch:", epoch + 1, "/ global_steps:", global_steps)
        print("training dataset average loss: %.3f" % train_ave_loss)
        print("training_time: %.2f minutes" % training_time)
        print("learning rate: %.6f" % step_lr_scheduler.get_lr()[0])

        # validation (for early stopping)
        val_correct_cnt = 0
        val_loss = 0
        val_batch_cnt = 0
        model.eval()
        for x, y in dev_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            val_batch_cnt += 1
            _, top_pred = torch.topk(outputs, k=1, dim=-1)
            top_pred = top_pred.squeeze(dim=1)
            val_correct_cnt += int(torch.sum(top_pred == y))# 맞춘 개수 카운트

        val_acc = val_correct_cnt / len(dev_y) * 100
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

    vocab_path = os.path.abspath(os.path.join(checkpoint_dir, "vocab"))
    emb_path = os.path.abspath(os.path.join(checkpoint_dir, "emb"))
    labels_path = os.path.abspath(os.path.join(checkpoint_dir, "labels"))
    with smart_open.smart_open(vocab_path, 'wb') as f:
        pickle.dump(word_id_dict, f)
    with smart_open.smart_open(emb_path, 'wb') as f:
        pickle.dump(initW, f)
    if params['task'] == "SNIPS":
        with smart_open.smart_open(labels_path, 'wb') as f:
            pickle.dump(label_dictionary, f)

if __name__ == '__main__':
    main()