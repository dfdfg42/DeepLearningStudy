import re
import collections
import torch
import unicodedata
import glob
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
    string = re.sub(r"\(", r" ( ", string)
    string = re.sub(r"\)", r" ) ", string)
    string = re.sub(r"\?", r" ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if nmt:
        return unicodeToAscii(string.strip().lower())
    else:
        return string.strip().lower()

def normalizeString(s):
    """
    문자열 정규화 함수
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def load_nmt_pair_data(file_path):
    """
    NMT 페어 데이터를 로드하는 함수
    파일은 탭으로 구분된 source\ttarget 형태로 되어 있다고 가정
    """
    print(f"Reading lines from {file_path}...")

    # 파일을 읽어서 라인별로 분리
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    # 각 라인을 탭으로 분리하여 source와 target으로 나눔
    pairs = []
    for line in lines:
        if '\t' in line:  # 탭이 있는 경우에만 처리
            parts = line.split('\t')
            if len(parts) >= 2:
                source = normalizeString(parts[0])
                target = normalizeString(parts[1])
                pairs.append([source, target])

    print(f"Read {len(pairs)} sentence pairs")

    # source와 target 텍스트를 분리
    source_texts = []
    target_texts = []

    for pair in pairs:
        source_texts.append(pair[0])
        target_texts.append(pair[1])

    return source_texts, target_texts

def load_snips_data(file_path, label_dictionary):
    # Load data from files
    text = list(open(file_path+"/seq.in", "r", encoding='UTF-8').readlines())
    text = [clean_str(sent) for sent in text]
    labels_text = list(open(file_path+"/label", "r", encoding='UTF-8').readlines())
    labels_text = [label.strip() for label in labels_text]

    if len(label_dictionary) == 0:
        label_set = set(labels_text)
        for i, label in enumerate(label_set):
            label_dictionary[label] = i
    labels = [label_dictionary[label_text] for label_text in labels_text]
    return text, labels, label_dictionary

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