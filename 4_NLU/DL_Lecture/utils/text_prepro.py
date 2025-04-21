import re
import collections
import torch
import unicodedata
import glob
import os
import random
import json

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def clean_str(string, nmt=False):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if nmt:
        return unicodeToAscii(string.strip().lower())
    else:
        return string.strip().lower()

def load_snips_data(path, label_dictionary):
    """
    지원 형태
    1)  intent별 .txt 파일이 든 폴더 (PlayMusic.txt …)
    2)  train/valid/test 폴더 안의 {label, seq.in} 2‑파일 구조   ←★NEW
    3)  intent  \\t sentence 형식의 단일 파일
    """
    texts, labels = [], []

    # ─── case 1 : intent별 .txt 파일들 ───
    if os.path.isdir(path):
        txt_files = glob.glob(os.path.join(path, "*.txt"))
        if txt_files:                           # (1) 번 구조
            for fp in txt_files:
                intent = os.path.splitext(os.path.basename(fp))[0]
                if intent not in label_dictionary:
                    label_dictionary[intent] = len(label_dictionary)
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        texts.append(clean_str(line))
                        labels.append(label_dictionary[intent])
            return texts, labels, label_dictionary

        # (2) 번 ‑ 3‑파일 구조  -----------------
        seq_in  = os.path.join(path, "seq.in")
        label_f = os.path.join(path, "label")
        if os.path.isfile(seq_in) and os.path.isfile(label_f):
            with open(seq_in,  encoding="utf-8") as fin, \
                 open(label_f, encoding="utf-8") as flabel:
                for sent, intent in zip(fin, flabel):
                    sent   = clean_str(sent.strip())
                    intent = intent.strip()
                    if intent not in label_dictionary:
                        label_dictionary[intent] = len(label_dictionary)
                    texts.append(sent)
                    labels.append(label_dictionary[intent])
            return texts, labels, label_dictionary
        # --------------------------------------

    # ─── case 3 : 단일 TSV 파일 ───
    with open(path, encoding="utf-8") as f:
        for line in f:
            intent, sentence = line.strip().split('\t', 1)
            if intent not in label_dictionary:
                label_dictionary[intent] = len(label_dictionary)
            texts.append(clean_str(sentence))
            labels.append(label_dictionary[intent])

    return texts, labels, label_dictionary

def load_mr_data(pos_file, neg_file):
    pos_text = list(open(pos_file, "r", encoding='latin-1').readlines()) # 긍정적인 review 읽어서 list 형태로 관리
    pos_text = [clean_str(sent) for sent in pos_text] # clean_str 함수로 전처리 (소문자, 특수 기호 제거, (), 등 분리)

    neg_text = list(open(neg_file, "r", encoding='latin-1').readlines()) # 부정적인 review 읽어서 list 형태로 관리
    neg_text = [clean_str(sent) for sent in neg_text]

    positive_labels = [1 for _ in pos_text] # 긍정 review 개수만큼 ground_truth 생성
    negative_labels = [0 for _ in neg_text] # 부정 review 개수만큼 ground_truth 생성
    y = positive_labels + negative_labels

    x_final = pos_text + neg_text
    return [x_final, y]

def buildVocab(sentences, vocab_size):
    # Build vocabulary
    words = []
    for sentence in sentences:
        words.extend(sentence.split()) # i, am, a, boy, you, are, a, girl
    print("The number of words: ", len(words))
    word_counts = collections.Counter(words)
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    # vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)} # a: 0, i: 1...
    return [vocabulary, vocabulary_inv]

def text_to_indices(x_text, word_id_dict, use_unk=False):
    text_indices = []

    for text in x_text:
        words = text.split()
        ids = [2]  # <s>
        for word in words: # i, am, a, boy
            if word in word_id_dict:
                word_id = word_id_dict[word]
            else:  # oov
                if use_unk:
                    word_id = 1 # OOV (out-of-vocabulary)
                else:
                    word_id = len(word_id_dict)
                    word_id_dict[word] = word_id
            ids.append(word_id) # 5, 8, 6, 19
        ids.append(3)  # </s>
        text_indices.append(ids)
    return text_indices

def sequence_to_tensor(sequence_list, nb_paddings=(0, 0)):
    nb_front_pad, nb_back_pad = nb_paddings

    max_length = len(max(sequence_list, key=len)) + nb_front_pad + nb_back_pad
    sequence_tensor = torch.LongTensor(len(sequence_list), max_length).zero_()  # 0: <pad>
    print("\n max length: " + str(max_length))
    for i, sequence in enumerate(sequence_list):
        sequence_tensor[i, nb_front_pad:len(sequence) + nb_front_pad] = torch.tensor(sequence)
    return sequence_tensor