import numpy as np
import pandas as pd
import sentencepiece as spm
import re
import pickle
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download('punkt')
nltk.download('stopwords')

newsData = fetch_20newsgroups(shuffle=True, random_state=1, remove=('fotters', 'quotes'))

news = newsData['data']
topic = newsData['target']
topic_name = newsData['target_names']

n = 0
print(len(news))
print(news[n])
print('topic = ', topic[n], topic_name[topic[n]])

subjects = []
for text in news:
  for sent in text.split('\n'):
    idx = sent.find('Subject:')
    if idx >= 0:
      subjects.append(sent[(idx + 9):].replace('Re: ', '').lower())
      break

subjects = [re.sub('[^a-zA-Z]', ' ', s) for s in subjects]

# Sentenpiece용 사전을 만들기 위해 corpusQA를 저장해둔다.
data_file = "./news_group20.txt"
with open(data_file, 'w', encoding='utf-8') as f:
  for sent in subjects:
    f.write(sent + '\n')

# Sentencepiece 파라미터 설정

templates= "--input={0:} \
            --pad_id=0 --pad_piece=<PAD>\
            --unk_id=1 --unk_piece=<UNK>\
            --bos_id=2 --bos_piece=<START>\
            --eos_id=3 --eos_piece=<END>\
            --model_prefix={1:} \
            --vocab_size={2:} \
            --character_coverage=0.9995 \
            --model_type=unigram"

VOCAB_SIZE = 5000
model_prefix = "./news_group20_model"
params = templates.format(data_file, model_prefix, VOCAB_SIZE)

spm.SentencePieceTrainer.Train(params)
sp = spm.SentencePieceProcessor()
sp.Load(model_prefix + '.model')

with open(model_prefix + '.vocab', encoding='utf-8') as f:
  vocab = [doc.strip().split('\t') for doc in f]


word2idx = {k:v for v, [k, _] in enumerate(vocab)}
idx2word = {v:k for k, v in word2idx.items()}
word2idx

# word index로 변환한다.
news_group20_idx = [sp.encode_as_ids(subject) for subject in subjects]

# 문장 최대 길이 확인
check_len = []
for s in news_group20_idx:
  check_len.append(len(s))
max(check_len)

# word index 값을 저장한다.
with open('./news_group20_idx.pkl', 'wb') as f:
  pickle.dump(news_group20_idx, f)

with open('./news_group20_idx.pkl', 'rb') as f:
    news_group20_idx = pickle.load(f)

n_topic = len(set(topic))
n_topic

x_data = pad_sequences(news_group20_idx, maxlen=17, padding='post', truncating='post')
y_data = topic

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

EMB_SIZE = 200
VOCAB_SIZE = len(word2idx) + 1
x_input = Input(batch_shape=(None, x_train.shape[1]))
x_embed = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_SIZE)(x_input)
x_embed = Dropout(0.5)(x_embed)
x_lstm = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True))(x_embed)
x_lstm = Bidirectional(LSTM(64))(x_lstm)
y_output = Dense(n_topic, activation='softmax')(x_lstm)

model = Model(x_input, y_output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])
model.summary()

es = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10)
hist = model.fit(x_train, y_train, epochs=100, batch_size=512, validation_data=(x_test, y_test), callbacks=[es])

plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1).reshape(-1, 1)
accuracy = (y_pred == y_test).mean()
print("\nAccurcay = %.2f %s" % (accuracy * 100, '%'))