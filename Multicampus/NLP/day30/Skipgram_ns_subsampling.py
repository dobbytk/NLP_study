import numpy as np
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, Dense, Dot, Activation, Flatten
from tensorflow.keras.models import Model, load_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
import pickle

# 전처리가 완료된 한글 코퍼스를 읽어온다.
# %cd '/content/drive/MyDrive/Colab Notebooks'
with open('./konovel_preprocessed.pkl', 'rb') as f:
    sent_list = pickle.load(f)

max_word = 10000
tokenizer = Tokenizer(num_words = max_word, oov_token = '<OOV>')
tokenizer.fit_on_texts(sent_list)
sent_idx = tokenizer.texts_to_sequences(sent_list)
word2idx = {k:v for (k, v) in list(tokenizer.word_index.items())[:max_word]}
idx2word = {v:k for (k, v) in word2idx.items()}

# 학습 데이터 생성
n_gram = 5      # 5-grams
target = []    # target word
context = []   # context word

# positive data
for sent in sent_idx:
    if len(sent) < n_gram:
        continue

    # 5-gram
    for w1, w2, w3, w4, w5 in nltk.ngrams(sent, n_gram):
        target.extend([w3, w3, w3, w3])   # target
        context.extend([w1, w2, w4, w5])   # context

# Subsampling of frequent words.
# [1]의 후속 논문인 [2]에 소개된 subsampling 기법을 적용한다.
def sub_sampling(x, y):
    # x, y를 합친다.
    data = np.hstack([x, y])
    # data = (x, y) 쌍을 shuffling 한다.
    np.random.shuffle(data)
    
    # data의 x 값을 기준으로 subsampling을 적용한다.
    d = np.empty(shape = (0, 2), dtype=np.int32)
    for x_set in set(data[:, 0]):
        x_tmp = data[np.where(data[:, 0] == x_set)]

        fw = 1e-8 + x_tmp.shape[0] / data.shape[0]
        pw = np.sqrt(1e-5 / fw)              # 남겨야할 비율
        cw = np.int(x_tmp.shape[0] * pw) + 1 # 남겨야할 개수 - subsampling 개수
        d = np.vstack([d, x_tmp[:cw]])

    # d[:, 1]은 0,1,2,... 순으로 되어 있어서 다시 한번 shuffle 한다.
    np.random.shuffle(d)
    return list(d[:, 0].reshape(-1)), list(d[:, 1].reshape(-1))


def train_data(t, c, voc_size):
    # subsampling
    x_target_pos, x_context_pos = sub_sampling(np.array(t).reshape(-1,1), np.array(c).reshape(-1,1))
    y_train_pos = [1] * len(x_target_pos)
    # negative sampling. random이 오래 걸려서 아래처럼 일괄처리함.
    ns_k = 2
    x_target_neg = []     # negative target word
    x_context_neg = []   # negative context word
    for k in range(ns_k):
        r = np.random.choice(range(1, voc_size), len(x_target_pos))
        x_target_neg.extend(x_target_pos.copy())
        x_context_neg.extend(list(r).copy())
    y_train_neg = [0] * len(x_target_neg)

    # positive + negative
    x_target = x_target_pos + x_target_neg
    x_context = x_context_pos + x_context_neg
    y_train = y_train_pos + y_train_neg   

    # shuffling
    x_target, x_context, y_train = shuffle(x_target, x_context, y_train)

    # list --> array 변환
    x_target = np.array(x_target).reshape(-1, 1)
    x_context = np.array(x_context).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)

    return x_target, x_context, y_train

VOC_SIZE = len(word2idx) + 1
EMB_SIZE = 4

LOAD_MODEL = False

if LOAD_MODEL:
    # 학습된 모델을 읽어온다.
    model = load_model("./skipgram_model_ns.h5")    
else:
    x_input_t = Input(batch_shape=(None, 1))
    x_input_c = Input(batch_shape=(None, 1))

    SharedEmb = Embedding(VOC_SIZE, EMB_SIZE, name='emb_vec')
    x_emb_t = SharedEmb(x_input_t)
    x_emb_c = SharedEmb(x_input_c)

    y_output = Dot(axes=(2,2))([x_emb_t, x_emb_c])
    y_output = Activation('sigmoid')(y_output)

    model = Model([x_input_t, x_input_c], y_output)
    model.compile(loss = 'binary_crossentropy', optimizer='adam')
model.summary()

for i in range(1):
    x_target, x_context, y_train = train_data(target, context, len(word2idx))
    model.fit([x_target, x_context], y_train, batch_size=10240, epochs=10)


# 학습 결과를 저장해 둔다.
model.save("./skipgram_model_ns.h5")

wv = model.get_layer('emb_vec').get_weights()[0]


# 주어진 단어의 word2vec 확인
def get_word2vec(word, wv):
    if word in word2idx:
        x = np.array(word2idx[word]).reshape(-1,1)
    else:
        x = np.array(word2idx['<OOV>']).reshape(-1,1)
    return wv[x, :][0][0]

word2vec = get_word2vec('아버지', wv)
print(np.round(word2vec, 4))