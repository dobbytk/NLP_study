import nltk
nltk.download('punkt')      # '/root/nltk_data/tokenizers'
nltk.download('averaged_perceptron_tagger')

text1 = nltk.corpus.gutenberg.raw('austen-emma.txt')
text2 = nltk.corpus.gutenberg.raw('austen-persuasion.txt')
text3 = nltk.corpus.gutenberg.raw('austen-sense.txt')
text4 = nltk.corpus.gutenberg.raw('bible-kjv.txt')
text5 = nltk.corpus.gutenberg.raw('blake-poems.txt')

full_text = text1 + ' ' + text2 + ' ' + text3 + ' ' + text4 + ' ' + text5

sentences = nltk.sent_tokenize(full_text)
sentences[:5]

# 개행문자 제거
sentences_removed = []
for s in sentences:
  sentences_removed.append(s.replace('\n', ' ').strip())

  # 임의의 10개 문장만 추출해서 테스트
temp = sentences_removed

pos_tag_result = []
for s in temp:
  s = nltk.word_tokenize(s)
  pos_tag_result.append(nltk.pos_tag(s))
pos_tag_result


# 1. 명사와 형용사만 추출해서 배열을 재구성
noun_adj = []
tmp = []
for i in range(len(pos_tag_result)):
  for j in range(len(pos_tag_result[i])):
    if pos_tag_result[i][j][1] == 'NN' or pos_tag_result[i][j][1] == 'JJ':
      tmp.append(pos_tag_result[i][j])
  noun_adj.append(tmp)
  tmp = []

  # 2단계 명사와 형용사만 가지고 전체 사전을 구성한다.
all_tokens = []
tmp = []
for i in range(len(noun_adj)):
  for j in range(len(noun_adj[i])):
    tmp.append(noun_adj[i][j][0])
  all_tokens.append(tmp)
  tmp = []

  # 사전 생성
word2idx = {}
n_idx = 0
for sent in all_tokens:
    for word in sent:
        if word.lower() not in word2idx:
            word2idx[word.lower()] = n_idx
            n_idx += 1
idx2word = {v:k for k, v in word2idx.items()}

word2idx


# 3. 한 문장 안에서 father와 가장 많이 사용된 단어 검색
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

mother = freq_with_word('mother', all_tokens)

father = freq_with_word('father', all_tokens)

doctor = freq_with_word('doctor', all_tokens)