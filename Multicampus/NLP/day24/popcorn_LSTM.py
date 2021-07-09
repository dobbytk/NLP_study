import pandas as pd
import re
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import PorterStemmer
import pickle

nltk.download('punkt')
nltk.download('stopwords')
# train, test tsv 파일 불러오기
df_train = pd.read_csv('labeledTrainData.tsv', header=0, sep='\t', quoting=3)
df_test = pd.read_csv('testData.tsv', header=0, sep='\t', quoting=3)

print(df_train['review'][0])
print(df_train['sentiment'][0])


# Pre-processing
# PorterStemmer()
stemmer = PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')

processed_train_text = []
processed_test_text = []
all_train_tokens = []
all_test_tokens = []
# processed_train_text 
for review in df_train['review']:
    # 1. 영문자와 숫자만 사용한다. 그 이외의 문자는 공백 문자로 대체한다.
    review = review.replace('<br />', ' ')   # <br> --> space
    review = re.sub("[^a-zA-Z]", " ", review)    # 영문자만 사용

    tmp = []
    for word in nltk.word_tokenize(review):
        # 2. 길이가 2 이하인 단어와 stopword는 제거한다.
        if len(word) > 2 and word not in stopwords:
            # 3. Lemmatize
            # tmp.append(stemmer.stem(word.lower()))
            tmp.append(stemmer.stem(word.lower()))
    processed_train_text.append(tmp)

# processed_test_text
for review in df_test['review']:
  review = review.replace('<br />', ' ')
  review = re.sub("[^a-zA-Z]", " ", review)

  tmp = []
  for word in nltk.word_tokenize(review):
    if len(word) > 2 and word not in stopwords:
      tmp.append(stemmer.stem(word.lower()))
  processed_test_text.append(tmp)

# print(processed_train_text[:2])
# print(processed_test_text[:2])

# Tokenizer 객체를 만든 후 idx 벡터로 변환 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_train_text)

word2idx = tokenizer.word_index
idx2word = {v:k for k, v in word2idx.items()}

to_idx_train = tokenizer.texts_to_sequences(processed_train_text)
to_idx_test = tokenizer.texts_to_sequences(processed_test_text)

# 패딩 처리 - max_len == 174
train_inputs = pad_sequences(to_idx_train, maxlen=174, padding='post')
test_inputs = pad_sequences(to_idx_test, maxlen=174, padding='post')
train_inputs[0]
test_inputs[0]
print(train_inputs.shape)
print(test_inputs.shape)
# 학습 데이터
X_train, X_test, y_train, y_test = train_test_split(np.array(train_inputs), np.array(df_train['sentiment']), test_size = 0.3)


# # 학습 데이터를 저장해 둔다.
# with open('./popcorn_train.pkl', 'wb') as f:
#     pickle.dump([X_train, y_train, X_test, y_test], f, pickle.DEFAULT_PROTOCOL) # pickle.DEFAULT_PROTOCOL

# with open('./popcorn_test.pkl', 'wb') as f:
#   pickle.dump(processed_test_text, f, pickle.DEFAULT_PROTOCOL)

# with open('./popcorn_train.pkl', 'rb') as f:
#   X_train, y_train, X_test, y_test = pickle.load(f)

# with open('./popcorn_test.pkl', 'rb') as f:
#   processed_test_text = pickle.load(f)
X_train.shape
vocab_size = len(word2idx)+1
emb_size = 32
# 모델링

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, LSTM, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 모델링
x_input = Input(batch_shape=(None, X_train.shape[1]))
x_emb = Embedding(input_dim = vocab_size, output_dim = emb_size)(x_input)
x_emb = LSTM(64, dropout=0.3)(x_emb) 
# return_sequences = True 중간 출력을 사용하겠다는 뜻. Many_to_Many/2층 이상 구성할 때 사용. TimeDistributed와 짝꿍으로 쓴다. 
# 각 메모리 셀마다 FFN을 달아서 중간 출력값을 다 쓰겠다. return_sequences는 쓰고 TimeDistributed를 안써주면 중간 출력값을 다 모은다. 
# (None, 174, 64)가 output이 되고 얘를 -> Dense -> (None, 174, l)이 된다. 하나의 값(y-hat)이 나와야하는데 중간값을 다 모았기 때문에 행이 174개, 열이 1개인 값이 data수 만큼나온다.
x_emb = BatchNormalization()(x_emb)
y_output = Dense(1, activation='sigmoid')(x_emb)

model = Model(x_input, y_output)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

# 처음 설정
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
# mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
es = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True)
hist = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.2, callbacks=[es, mc])
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], 'r', label='val_loss')
plt.legend()
plt.show()

# 정확도 구하기
from sklearn.metrics import accuracy_score

y_train_pred = model.predict(X_train)
accuracy_score(y_train, y_train_pred.round(0))

y_pred = model.predict(X_test)
y_pred = y_pred.reshape(-1)
y_pred = y_pred.round(0)
accuracy = accuracy_score(y_pred, y_test)
print(accuracy)
result = model.predict(test_inputs)