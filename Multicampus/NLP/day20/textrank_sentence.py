import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


sentences = []

f = open('./text_rank(문장).txt')
f = f.read()
f = sent_tokenize(f)
for s in f:
  # 문장부호 없애기 + 양쪽 공백제거 + 소문자 변환
  sentences.append(s.replace('\\', '').replace('.', '').replace(',', '').replace('?', '').replace(':', '').replace('\n', '').replace("'", '').strip().lower())
# sentences = sentences[:len(sentences)]
sentences[0] = sentences[0][5:]
sentences

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(sentences)

tfidf_matrix.shape

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
np.fill_diagonal(cosine_sim, 0) # 대각을 0으로 변환

C = cosine_sim.round(3)
C

tr = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

d = 0.85
weight = 0
history = []

for i in range(80):
  ex_tr = np.array(tr)
  for j in range(len(C)):
    for k in range(len(C[j])): 
      weight += (C[j][k]/np.sum(C[k])) * tr[k]
    tr[j] = weight * d + (1-d)
    weight = 0 # weight값 초기화
  
  history.append(np.sum(abs(ex_tr - tr))) # loss값 계산

tr

hist = np.array(history).T

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
fig.set_facecolor('white')
ax = fig.add_subplot()

ax.plot(range(80), hist, marker='o', label='TR_A')

ax.legend()
plt.show()

tr

summary = np.array(tr).argsort()[::-1][:3]

for i in summary:
  print(sentences[i])