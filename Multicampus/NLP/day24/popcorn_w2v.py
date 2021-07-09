# Word2Vec을 이용한 감성분석
import numpy as np
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from nltk.stem import LancasterStemmer
from gensim.models import word2vec
import matplotlib.pyplot as plt
import pickle

nltk.download('punkt')

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/Colab Notebooks'

# 학습, 시험 데이터를 읽어온다.
with open('popcorn.pkl', 'rb') as f:
    xd_train, yd_train, xd_test, yd_test = pickle.load(f)

# 문장의 문자열을 token으로 분리한다.
sent_tok = [nltk.word_tokenize(sent) for sent in xd_train + xd_test]
sent_tok[0]

# word2vec 모델 생성
EMB_SIZE = 32
model = word2vec.Word2Vec(sent_tok, size =EMB_SIZE, min_count=1, window=1, sg=1, negative=2)
word2idx = model.wv.vocab

print("사전 크기 =", len(word2idx))

# 학습 데이터 문자열을 문장 벡터를 생성한다.
def sentence_vectorize(data):
    vector = []
    for sent in data:
        sent_vect = np.zeros((EMB_SIZE))
        sent_tok = sent.split(' ')
        for word in sent_tok:
            # pdb.set_trace()
            sent_vect += model.wv[word]
        vector.append(sent_vect / len(sent_tok))
    return vector

xd_train_vec = sentence_vectorize(xd_train)
xd_test_vec = sentence_vectorize(xd_test)

np.array(xd_train_vec).shape

x_input = Input(batch_shape = (None, EMB_SIZE))
h_layer = Dense(128, activation='relu')(x_input)
y_output = Dense(1, activation = 'sigmoid')(h_layer)
model = Model(x_input, y_output)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
model.summary()

x_train = np.array(xd_train_vec)
x_test = np.array(xd_test_vec)
y_train = np.array(yd_train).reshape(-1,1)
y_test = np.array(yd_test).reshape(-1,1)

# 학습
hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 1024, epochs = 100)

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label = 'Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 시험 데이터로 학습 성능을 평가한다
pred = model.predict(x_test)
y_pred = np.where(pred > 0.5, 1, 0)
accuracy = (y_pred == y_test).mean()
print("\nAccuracy = %.2f %s" % (accuracy * 100, '%'))

