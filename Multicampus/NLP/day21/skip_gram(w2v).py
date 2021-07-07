# skip_gram, low-level로 구현하기
"""
skipgram 실습 과제
1. 영어소설 corpus, gutenberg corpus -> 영어소설 10개
2. working, worked => 1단어인 work로 사용
3. vocabulary 생성
4. 소설 문장을 하나씩 읽어서 tri-gram 생성
5. tri-gram으로 학습 데이터 생성 
입력       출력
love       I
love      you

6. 각 단어를 vocabulary의 index로 표현 (ex) love = 32 index로 표현. 이후 one-hot encoding
7. 네트워크 구성 - input_dim = vocab_size
word vector size 32
8. 단어의 one-hot을 입력하면 -> 은닉층 출력
9. father, mother, doctor 의 word vector 를 구하고 각 벡터끼리의 코사인 유사도 구하기
"""
from nltk.stem import LancasterStemmer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten
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
print(word_tokens[0])

# stemming을 진행해서 stem 리스트에 저장
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
print(len(stem))

# 전체 단어 사전 생성
all_tokens = []
for s in stem:
  for t in s:
    all_tokens.append(t)
print(len(all_tokens))

# word -> idx
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_tokens)

word2idx = tokenizer.word_index
idx2word = {v:k for k, v in word2idx.items()}

to_idx = tokenizer.texts_to_sequences(stem)

# 문장 별로 trigram 생성
trigram = []
tmp = []
for s in to_idx:
  # trigram을 만들 수 없는 경우 (1388, 21, None)으로 패딩 pad_right=True
  for a, b, c in nltk.ngrams(s, 3, pad_right=True): 
    tmp.append((a, b, c))
  trigram.append(tmp)
  tmp = []
len(trigram)

# input 데이터 만들기
input = []
output = []

for i in range(len(trigram)):
  for j in range(len(trigram[i])):
    for k in range(2):
      input.append(trigram[i][j][1])


print(input[:200])

# output 데이터 만들기
for i in range(len(trigram)):
  for j in range(len(trigram[i])):
    for k in range(0, 3, 2):
      output.append(trigram[i][j][k])
    
print(output[:200])

df = pd.DataFrame({
    'input' : input,
    'output' : output
})
print(df)

# 결측값 제거(Nan값 있는 행 제거)
df.dropna(axis=0, inplace=True)

# 데이터셋 구성
X_train = np.array(df['input']).reshape(-1, 1)
y_train = np.array(df['output']).reshape(-1, 1)
vocab_size = len(word2idx)+1 # word2idx가 0부터 시작했다면 +1을 안해줘도 됨. Embedding layer는 0부터 처리하므로 +1을 해준다.
X_train.shape

x_input= Input(batch_shape = (None, X_train.shape[1]))
hidden = Embedding(input_dim = vocab_size, output_dim = 32)(x_input)
hidden = Flatten()(hidden)
y_output = Dense(vocab_size, activation='softmax')(hidden)

model = Model(x_input, y_output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.01))

# word --> word2vec을 확인하기 위한 모델 (predict용 모델)
model_w = Model(x_input, hidden)

model.summary()

hist = model.fit(X_train, y_train, epochs=30, batch_size=4096, validation_split=0.2)

# father에 stem이 적용됐으므로 idx값을 찾을 때도 stem을 적용해서 찾아야함.
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

print(father + mother)
