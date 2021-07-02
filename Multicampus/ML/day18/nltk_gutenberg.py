import nltk
nltk.download('punkt')

nltk.download('gutenberg')
text_id = nltk.corpus.gutenberg.fileids()

text_id

# 특정 파일의 텍스트 문서를 조회한다.
text = nltk.corpus.gutenberg.raw('austen-emma.txt')
print(text[:600])
print("문자 개수 = ", len(text))

# 특정 문서를 word 단위로 읽어온다.
word = nltk.corpus.gutenberg.words('austen-emma.txt')
print(word)
print("word 개수 = ", len(word))

# 특정 문서를 문장 단위로 읽어온다.
sentence = nltk.corpus.gutenberg.sents('austen-emma.txt')
for i in range(5):
    print(sentence[i])
print("문장 개수 = ", len(sentence))

# 가장 긴 문장을 출력한다
longestLen = max(len(s) for s in sentence)
longestSent = [s for s in sentence if len(s) == longestLen]
print(longestSent)
print('단어 개수 =', len(longestSent[0]))

# 각 문장에 사용된 단어 개수의 분포를 확인해 본다.
import numpy as np
import seaborn as sns
countWord = [len(n) for n in sentence]
sns.displot(countWord, bins=50)
print("문장 당 평균 단어수 = %.2f" % np.mean(countWord))

# Gutenberg 사이트에서 직접 소설 텍스트를 읽어온다
# 제인 오스틴의 오만과 편견
from urllib import request
url = "http://www.gutenberg.org/files/1342/1342-0.txt"
response = request.urlopen(url)
text = response.read().decode('utf8')
print(text[:500])

# 문서에서 특정 단어 위치 검색
# text.find() : 앞에서부터 검색
# text.rfind() : 뒤에서부터 검색 (reverse)
idx = text.find("Author")
text[idx:(idx + 20)]

