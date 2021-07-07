# negative sampling skip_gram 모델 low-level로 구현하기

from nltk.stem import LancasterStemmer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Activation, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

nltk.download('punkt')
nltk.download('gutenberg')
text_id = nltk.corpus.gutenberg.fileids()
print(text_id)

# 영어 소설 10개 불러오기
text1 = nltk.corpus.gutenberg.raw('austen-emma.txt')
text2 = nltk.corpus.gutenberg.raw('austen-persuasion.txt')
text3 = nltk.corpus.gutenberg.raw('austen-sense.txt')
text4 = nltk.corpus.gutenberg.raw('bible-kjv.txt')
text5 = nltk.corpus.gutenberg.raw('blake-poems.txt')
text6 = nltk.corpus.gutenberg.raw('bryant-stories.txt')
text7 = nltk.corpus.gutenberg.raw('burgess-busterbrown.txt')
text8 = nltk.corpus.gutenberg.raw('carroll-alice.txt')
text9 = nltk.corpus.gutenberg.raw('chesterton-ball.txt')
text10 = nltk.corpus.gutenberg.raw('chesterton-brown.txt')

text = text1 + ' ' + text2 + ' ' + text3 + ' ' + text4 + ' ' + text5 + ' ' + text6 + ' ' + text7 + ' ' + text8 + ' ' + text9 + ' ' + text10

sentences = nltk.sent_tokenize(text)
print(len(sentences))
print(sentences[:10])

sen = []
for s in sentences:
  s = s.replace('\n', ' ').replace("'", '').replace('.', '').replace('-', '').replace('!', '').replace('?', '').replace(';', '').replace(',', '').replace('_', '').replace('(', '').replace(')', '').replace(':', '').replace('"', "").replace('[', '').replace(']', '').strip().lower()
  sen.append(s)
 
stemmer = LancasterStemmer()
word_tokens = [nltk.word_tokenize(s) for s in sen]

# stemming해 stem 리스트에 저장
stem = []
tmp = []
for s in word_tokens:
  for t in s:
    tmp.append(stemmer.stem(t))
  stem.append(tmp)
  tmp = []

# trigram을 만들 때 문장의 길이가 3 미만인 경우 에러가 발생하므로 삭제
for s in stem:
  if len(s) < 3:
    stem.remove(s)
len(stem)

# 전체 단어 사전 생성
all_tokens = []
for s in stem:
  for t in s:
    all_tokens.append(t)
len(all_tokens)

# word -> idx
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_tokens)

word2idx = tokenizer.word_index
idx2word = {v:k for k, v in word2idx.items()}

to_idx = tokenizer.texts_to_sequences(stem)

# 문장 별로 tri-gram 생성
trigram = []
tmp = []
for s in to_idx:
  # trigram을 만들 수 없는 경우 (1388, 21, None)으로 패딩 pad_right=True
  for a, b, c in nltk.ngrams(s, 3, pad_right=True): 
    tmp.append((a, b, c))
  trigram.append(tmp)
  tmp = []
len(trigram)

# k = 2라고 가정했을 때,
x_1 = [] # love, love, love, love, love, love
x_2 = [] # I, you, rand_sample, rand_sample, rand_sample, rand_sample
y = [] # 1, 1, 0, 0, 0, 0 이 들어가도록

# x_1 데이터 만들기
for i in range(len(trigram)):
  for j in range(len(trigram[i])):
    for k in range(6):
      x_1.append(trigram[i][j][1])
  
print(x_1[:200])

import random
# x_2 데이터 만들기 - Positive sampling
for i in range(len(trigram)):
  for j in range(len(trigram[i])):
    for k in range(0, 3, 2):
      x_2.append(trigram[i][j][k])
      y.append(1)
    
    # Negative sampling 4번
    for q in range(4):
      x_2.append(random.randint(1, 21948)) # max(word2idx.values()) == 21948
      y.append(0)
      
print(x_2[:200])
print(y[:200])

# 리스트를 가지고 데이터프레임 구성
df = pd.DataFrame({
    'input_1' : x_1,
    'input_2' : x_2,
    'label' : y
})

df.dropna(axis=0, inplace=True)
df = df.sample(frac=1).reset_index(drop=True)

# 데이터셋 구성
X_1_train = np.array(df['input_1']).reshape(-1, 1)
X_2_train = np.array(df['input_2']).reshape(-1, 1)
y_train = np.array(df['label']).reshape(-1, 1)
vocab_size = len(word2idx)+1 # word2idx가 0부터 시작했다면 +1을 안해줘도 됨. Embedding layer는 0부터 처리하므로 +1을 해준다.

# 학습 모델링
x_1_input = Input(batch_shape = (None, X_1_train.shape[1]))
x_2_input = Input(batch_shape = (None, X_2_train.shape[1]))

x_1_emb = Embedding(input_dim = vocab_size, output_dim = 32)(x_1_input)
x_1_emb = Flatten()(x_1_emb)

x_2_emb = Embedding(input_dim = vocab_size, output_dim = 32)(x_2_input)
x_2_emb = Flatten()(x_2_emb)

hidden = Dot(axes=1)([x_1_emb, x_2_emb])
y_output = Activation('sigmoid')(hidden)

model = Model([x_1_input, x_2_input], y_output)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

# word --> word2vec을 확인하기 위한 모델 (predict용 모델)
model_w = Model(x_2_input, x_2_emb) # Model(x_1_input, x_1_emb)을 해도 상관없다.

model.summary()

hist = model.fit([X_1_train, X_2_train], y_train, epochs=20, batch_size=8192, validation_split=0.2)

father = model_w.predict(np.array(word2idx[stemmer.stem('father')]).reshape(1, 1))
print(father)

mother = model_w.predict(np.array(word2idx[stemmer.stem('mother')]).reshape(1, 1))
print(mother)

doctor = model_w.predict(np.array(word2idx[stemmer.stem('doctor')]).reshape(1, 1))
print(doctor)

father_mother = cosine_similarity(mother, father)
print(father_mother)

mother_doctor = cosine_similarity(mother, doctor)
print(mother_doctor)

father_doctor = cosine_similarity(father, doctor)
print(father_doctor)