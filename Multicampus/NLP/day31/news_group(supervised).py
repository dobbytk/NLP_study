import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# news data를 읽어와서 저장해 둔다.
newsData = fetch_20newsgroups(shuffle=True, 
                             random_state=1,
                             remove=('footers', 'quotes'))
stemmer = PorterStemmer()


# 첫 번째 news를 조회해 본다.
news = newsData['data']
topic = newsData['target']
topic_name = newsData['target_names']

n = 1
print(len(news))
print(news[n])
print('topic = ', topic[n], topic_name[topic[n]])

# 정규표현식을 사용하여 Subject: ~ 에 해당하는 문장만 추출한다.
subjects = []
for s in news:
  s = re.sub('[^a-zA-Z\n]', ' ', s)
  s = re.findall('Subject.+', s)
  for t in s:
    k = t.lower().split()
  subjects.append(k)

# 불필요한 토큰 'subject', 're'를 제거
for s in subjects:
  if 'subject' in s:
    s.remove('subject')
  if 're' in s:
    s.remove('re')

# PorterStemming & stop_word 적용
stop_words = stopwords.words('english')
processed_text = []
tmp = []
for i in range(len(subjects)):
  for j in range(len(subjects[i])):
    if len(subjects[i][j]) > 3 and subjects[i][j] not in stop_words:
      tmp.append(stemmer.stem(subjects[i][j]))
  processed_text.append(tmp)
  tmp = []

with open('./processed_text_topic_modeling.pkl', 'wb') as f:
  pickle.dump(processed_text, f, pickle.DEFAULT_PROTOCOL)

x_train, x_test, y_train, y_test = train_test_split(np.array(processed_text), topic, test_size=0.2)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
word2idx = tokenizer.word_index
idx2word = {v:k for k, v in word2idx.items()}

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=11, padding='post')
x_test = pad_sequences(x_test, maxlen=11, padding='post')

x_train.shape, x_test.shape

vocab_size = len(word2idx) + 1
EMB_SIZE = 32

# 모델링
x_input = Input(batch_shape=(None, x_train.shape[1]))
x_emb = Embedding(input_dim=vocab_size, output_dim=EMB_SIZE)(x_input)
x_emb = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True, activation='relu'), merge_mode='sum')(x_emb)
x_emb = Flatten()(x_emb)
y_output = Dense(20, activation='softmax')(x_emb)

model = Model(x_input, y_output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
model.summary()

es = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True)
hist = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_test, y_test), callbacks=[es, mc])

plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


y_pred = model.predict(x_test)
prediction = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, prediction)
print(accuracy)