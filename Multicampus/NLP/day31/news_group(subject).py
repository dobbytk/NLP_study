import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# %cd '/content/drive/MyDrive/Colab Notebooks'

# news data를 읽어온다. subject 분석용.
newsData = fetch_20newsgroups(shuffle=True, random_state=1, remove=('footers', 'quotes'))

# 첫 번째 news를 조회해 본다.
news = newsData['data']
topic = newsData['target']
topic_name = newsData['target_names']
n=0
print(len(news))
print(news[n])
print('topic = ', topic[n], topic_name[topic[n]])

# Subject만 추출한다.
subjects = []
for text in news:
    for sent in text.split('\n'):
        idx = sent.find('Subject:')
        if idx >= 0:       # found
            subjects.append(sent[(idx + 9):].replace('Re: ', ''))
            break

# subject를 전처리한다.
stemmer = PorterStemmer()
stop_words = stopwords.words('english')

# 1. 영문자가 아닌 문자를 모두 제거한다.
subject1 = [re.sub("[^a-zA-Z]", " ", s) for s in subjects]

# 2. 불용어를 제거하고, 모든 단어를 소문자로 변환하고, 길이가 2 이하인 
# 단어를 제거한다
# 3. Porterstemmer를 적용한다.
subject2 = []
for sub in subject1:
    tmp = []
    for w in sub.split():
        w = w.lower()
        if len(w) > 2 and w not in stop_words:
            tmp.append(stemmer.stem(w))
    subject2.append(' '.join(tmp))

# news data를 다시 읽어온다. news의 body 부분 처리.
newsData = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
news = newsData['data']

# body 부분을 전처리한다.

# 1. 영문자가 아닌 문자를 모두 제거한다.
news1 = [re.sub("[^a-zA-Z]", " ", s) for s in subjects]

# 2. 불용어를 제거하고, 모든 단어를 소문자로 변환하고, 길이가 3 이하인 
# 단어를 제거한다
# 3. Porterstemmer를 적용한다.
news2 = []
for doc in news1:
    doc1 = []
    for w in doc.split():
        w = w.lower()
        if len(w) > 3 and w not in stop_words:
            doc1.append(stemmer.stem(w))
    news2.append(' '.join(doc1))

# 전처리가 완료된 데이터를 저장한다.
with open('./newsgroup20.pkl', 'wb') as f:
    pickle.dump([subject2, news2, topic], f, pickle.DEFAULT_PROTOCOL)

