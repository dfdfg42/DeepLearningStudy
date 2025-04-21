import torch
import smart_open
import pickle
import yaml
import sys
import os
from torch.utils.data import TensorDataset, DataLoader
from DL_Lecture.models.sentence_cnn import SentenceCnn
from DL_Lecture.utils.text_prepro import load_mr_data, load_snips_data, text_to_indices, sequence_to_tensor

def main():
    print('CNN for sentence classification evaluation')

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/text_cnn_snips.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    data_params = params['data_files'][params['task']]

    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    timestamp = "1745216041"
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))
    vocab_path = os.path.abspath(os.path.join(out_dir, "checkpoints/vocab"))
    emb_path = os.path.abspath(os.path.join(out_dir, "checkpoints/emb"))

    # 데이터 로드
    if params['task'] == "MR":
        test_x_text, test_y = load_mr_data(data_params['pos_test_file'], data_params['neg_test_file'])
        nb_classes = max(test_y) + 1
        print("cfg.nb_classes: ", nb_classes)
    elif params['task'] == "SNIPS":
        label_path = os.path.abspath(os.path.join(out_dir, "checkpoints/labels"))
        with smart_open.smart_open(label_path, 'rb') as f:
            label_dictionary = pickle.load(f)
        test_x_text, test_y, label_dictionary = load_snips_data(data_params['test_file'], label_dictionary)
        nb_classes = max(test_y) + 1
        print("nb_classes: ", nb_classes)

    with smart_open.smart_open(vocab_path, 'rb') as f:
        word_id_dict = pickle.load(f)
    with smart_open.smart_open(emb_path, 'rb') as f:
        initW = pickle.load(f)

    test_x = text_to_indices(test_x_text, word_id_dict, True)

    # data 개수 확인
    print('The number of test data: ', len(test_x))

    nb_pad = int(max(params['model_params_cnn']['filter_lengths']) / 2 + 0.5)

    test_x = sequence_to_tensor(test_x, nb_paddings=(nb_pad, nb_pad))
    test_y = torch.tensor(test_y)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=params['batch_size'], shuffle=False, num_workers=4)

    # # 학습 모델 생성
    # model = SentenceCnn(nb_classes=nb_classes,
    #                     word_embedding_numpy=initW,
    #                     filter_lengths=params['model_params_cnn']['filter_lengths'],
    #                     filter_counts=params['model_params_cnn']['filter_counts'],
    #                     dropout_rate=params['dropout_rate']).to(device)

    #── 학습 모델 생성 ─────────────────────────────────────────
    model = SentenceCnn(nb_classes=nb_classes,
                        word_embedding_numpy=initW,
                        filter_lengths=params['model_params_cnn']['filter_lengths'],
                        filter_counts=params['model_params_cnn']['filter_counts'],
                        dropout_rate=params['dropout_rate'],
                        multi_channel=True  # ★ 학습 때와 동일하게
                        ).to(device)

    # test 시작
    model.eval()

    # 저장된 state 불러오기
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints/best.pth"))

    # TODO : 세팅값 마다 save_path를 바꾸어 로드
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    correct_cnt = 0

    iter = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        pred = model.forward(x)
        _, top_pred = torch.topk(pred, k=1, dim=-1)
        top_pred = top_pred.squeeze(dim=1)
        correct_cnt += int(torch.sum(top_pred == y))
        if iter == 0:
            total_pred = top_pred
            total_y = y
        total_pred = torch.cat((total_pred, top_pred))
        total_y = torch.cat((total_y, y))
        iter += 1

    accuracy = correct_cnt / len(test_x) * 100
    print("test accuracy: %.2f%%" % accuracy)

if __name__ == "__main__":
    main()