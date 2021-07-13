import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import collections
import pickle

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/Colab Notebooks'

# 1차 전처리가 완료된 clean_text와 sentiment 데이터를 읽어온다.
with open('data/popcorn.pkl', 'rb') as f:
    x_data, _, y_data, _ = pickle.load(f)

# 학습 데이터와 시험 데이터로 분리한다.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

# 1차 vocabulary를 생성하고, 리뷰 데이터를 인덱스로 표현한다.
vocab = collections.Counter()
for review in x_train:
    for word in review.split(' '):
        vocab[word] += 1

# 빈도가 높은 순서로 vocabulary를 생성한다. 사용 빈도가 5 이하인 단어는 제외한다.
word2idx = {}
for i, (word, count) in enumerate(vocab.most_common()):
    if count < 5:
        break
    word2idx[word] = i + 2

word2idx["<PAD>"] = 0   
word2idx["<OOV>"] = 1
idx2word = {v:k for k, v in word2idx.items()}

# review 문장을 word2idx의 인덱스로 표시한다.
train_idx = []
for review in x_train:
    tmp = []
    for word in review.split(' '):
        if word in word2idx:
            tmp.append(word2idx[word])
        # else:
        #     tmp.append(word2idx['<OOV>'])
    train_idx.append(tmp)

# [단어-label] 리스트를 만든다.
word_label = []
for review, label in zip(train_idx, y_train):
    for w in review:
        word_label.append([w, label])
        
word_label = np.array(word_label)

# 단어마다 Mutual Information (MI)을 계산한다.
#
# word_label : 단어 인덱스 - label 목록 배열
#              [[ 2, 0],
#               [85, 0], ...]
#
# mi = p(x|y=0) * p(y=0) * log(p(x|y=0) / p(x)) +
#      p(x|y=1) * p(y=1) * log(p(x|y=1) / p(x))
# -----------------------------------------------

# y = 0인 단어목록 (x[0])과 y = 1인 단어목록 (x[1])을 만든다.
x = np.array([np.where(word_label[:, 1] == i)[0] for i in [0, 1]])
py = np.array([(word_label[:, 1] == i).mean() for i in [0, 1]])
N = len(idx2word)

mi_word = []
for i in range(2, N):
    px = (word_label[:, 0] == i).mean()
    
    mi = 0
    for y in [0, 1]:
        # p(x | y)
        pxy = (word_label[x[y], 0] == i).mean()
        mi += (pxy * py[y]) * np.log(1e-8 + pxy / px)
        
    mi_word.append([mi, i])
    
    if i % 100 == 0:
        print(i, '/', N)

# mi_word 리스트를 내림차순으로 정렬한다.
mi_word.sort(reverse = True)

# MI 상위 20개 단어를 확인해 본다.
print([idx2word[y] for x, y in mi_word[:20]])

# 상위 max_vocab개의 단어로 vocabulrary를 생성한다.
max_vocab = 6000
word2idx2 = {idx2word[y]:(i+2) for i, [x, y] in enumerate(mi_word[:max_vocab])}
word2idx2['<PAD>'] = 0
word2idx2['<OOV>'] = 1
idx2word2 = {v:k for k, v in word2idx2.items()}

# MI 기반 vocabulary를 이용하여 리뷰 데이터를 다시 만든다. OOV는 제거한다.
def build_data(data):
    d_idx = []
    for sent in data:
        tmp = []
        for word in sent.split():
            if word in word2idx2:
                tmp.append(word2idx2[word])
            # else:
            #     tmp.append(word2idx2['<OOV>'])
        d_idx.append(tmp)
    return d_idx

x_train_mi = build_data(x_train)
x_test_mi = build_data(x_test)

# 학습 데이터를 저장해 둔다.
with open('data/popcorn_mi.pkl', 'wb') as f:
    pickle.dump([x_train_mi, y_train, x_test_mi, y_test, word2idx2], f, pickle.DEFAULT_PROTOCOL)

x_test_mi[0]

