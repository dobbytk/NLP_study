import re
from nltk.corpus import stopwords
from konlpy.tag import Okt 

sentences = []
f = open('./ko_novel', encoding='utf-8-sig')
f = f.readlines()
for s in f:
  sentences.append(s.strip())
sentences[:5]

okt = Okt()
stop_words=['은','는','이','가', '하','아','것','들','의','있','되','수','보','주','등','한','그', '나']
def preprocessing(sentences):
  sen = []
  after_pos = []
  for s in sentences: 
    s = re.sub('[^ㄱ-ㅎ가-힣]', ' ', s)
    sen.append(s)

  for s in sen:
    pos = okt.pos(s)
    after_pos.append(pos)

  return after_pos

pos = preprocessing(sentences)

adj_noun = []
tmp = []
for i in range(len(pos)):
  for j in range(len(pos[i])):
    if pos[i][j][1] == 'Adjective' or pos[i][j][1] == 'Noun':
      tmp.append(pos[i][j])
  adj_noun.append(tmp)
  tmp = []

all_tokens = []
tmp = []
for i in range(len(adj_noun)):
  for j in range(len(adj_noun[i])):
    tmp.append(adj_noun[i][j][0])
  all_tokens.append(tmp)
  tmp = []

all_tokens_removed = []
tmp = []
for i in range(len(all_tokens)):
  for j in range(len(all_tokens[i])):
    if all_tokens[i][j] not in stop_words:
      tmp.append(all_tokens[i][j])
  all_tokens_removed.append(tmp)
  tmp = []

from collections import Counter

def freq_with_word(query, all_tokens):
  a = Counter()
  for s in all_tokens:
    if query in s:
      a.update(s)
    else:
      pass
  result = a.most_common()
  result = result[1:len(result)]
  return result[:20]

mother = freq_with_word('어머니', all_tokens_removed)
father = freq_with_word('아버지', all_tokens_removed)
doctor = freq_with_word('의사', all_tokens_removed)

def to_jaccard(lst):
    result = []
    for t in lst:
        result.append(t[0])
    return result

m = to_jaccard(mother)
f = to_jaccard(father)
d = to_jaccard(doctor)

def jaccard(x, y):
    hab = set(x) | set(y)
    gyo = set(x) & set(y)
    return len(gyo) / len(hab)

jaccard(m, f) # mother와 father의 자카드 유사도
jaccard(f, d) # father와 doctor의 자카드 유사도