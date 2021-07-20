# Skipgram-2 : Skipgram with subsampling.
#
# Skipgram으로 한글 코퍼스를 학습하고,
# 1) 워드 벡터를 구해보고,
# 2) 단어간 의미적 유사도를 확인한다.
#
# 관련 논문 : [1] Tomas Mikolov, et. al., 2013, Efficient Estimation of Word 
#                 Representations in Vector Space
#            [2] Tomas Mikolov, et. al., 2013, distributed representations of words 
#                and phrases and their compositionality          

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Dense, Embedding
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import nltk

# 전처리가 완료된 한글 코퍼스를 읽어온다.
with open('../data/konovel_preprocessed.pickle', 'rb') as f:
    sent_list = pickle.load(f)

max_word = 10000
tokenizer = Tokenizer(num_words = max_word, oov_token = '<OOV>')
tokenizer.fit_on_texts(sent_list)
sent_idx = tokenizer.texts_to_sequences(sent_list)
word2idx = {k:v for (k, v) in list(tokenizer.word_index.items())[:max_word]}
idx2word = {v:k for (k, v) in word2idx.items()}

# 5-gram으로 학습 데이터를 생성한다.
x_data = []     # 입력 데이터
y_data = []     # 출력 데이터
for sentence in sent_idx:
    # 5-gram으로 주변 단어들을 묶는다. 가운데 단어와 다른 단어들의 쌍을 만든다.
    contexts = nltk.ngrams(sentence, 5)
    pairs = [[(c[2], c[0]), (c[2], c[1]), (c[2], c[3]), (c[2], c[4])] for c in contexts]
    for pair in pairs:
        for p in pair:
            if word2idx['<OOV>'] not in p:  # oov가 포함된 쌍은 제외한다.
                x_data.append(p[0])
                y_data.append(p[1])

x = np.array(x_data).reshape(-1, 1)
y = np.array(y_data).reshape(-1, 1)
x.shape, y.shape


# Subsampling of frequent words.
# [1]의 후속 논문인 [2]에 소개된 subsampling 기법을 적용한다.
# x : target, y : context
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
    return d[:, 0].reshape(-1, 1), d[:, 1].reshape(-1, 1)

# skipgram 모델을 생성한다.
VOCAB_SIZE = len(word2idx) + 1
EMB_SIZE = 64
LOAD_MODEL = True

if LOAD_MODEL:
    # 학습된 모델을 읽어온다.
    model = load_model("../data/skipgram_model.h5")    
else:
    x_input = Input(batch_shape = (None, 1))
    wv_layer = Embedding(VOCAB_SIZE, EMB_SIZE, name='emb_vec')(x_input)
    y_output = Dense(VOCAB_SIZE, activation='softmax')(wv_layer)
    
    model = Model(x_input, y_output)     # 학습용 모델
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.001))
model.summary()

# 학습.
for i in range(1):
    x_train, y_train = sub_sampling(x, y)
    model.fit(x_train, y_train, batch_size=1024, epochs=10)
    
# 학습 결과를 저장해 둔다.
model.save("../data/skipgram_model.h5")

# 어휘 사전인 word2idx도 저장해 둔다.
with open('../data/skipgram_word2idx.pkl', 'wb') as f:
    pickle.dump([word2idx, idx2word], f, pickle.DEFAULT_PROTOCOL)

# 주어진 단어의 주변 단어 (context) 확인
def get_contexts(word, top_n=10):
    if word in word2idx:
        x = np.array(word2idx[word]).reshape(-1,1)
    else:
        x = np.array(word2idx['<OOV>']).reshape(-1,1)

    context_prob = model.predict(x)[0][0]
    top_idx = np.argsort(context_prob)[::-1][:top_n]
    return [idx2word[i] for i in top_idx]

context = get_contexts('사랑')
print(context)

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

# 단어간 유사도 측정
doctor = get_word2vec('의사', wv)
patient = get_word2vec('환자', wv)
sea = get_word2vec('김치', wv)

print('\n의사 - 환자 :', np.round(cosine_similarity([doctor, patient])[0, 1], 4))
print('의사 - 김치 :', np.round(cosine_similarity([doctor, sea])[0, 1], 4))

father = get_word2vec('아빠', wv)
mother = get_word2vec('엄마', wv)
daughter = get_word2vec('딸', wv)

print('\n아빠 - 딸 :', np.round(cosine_similarity([father, daughter])[0, 1], 4))
print('엄마 - 딸 :', np.round(cosine_similarity([mother, daughter])[0, 1], 4))