import nltk
nltk.download('punkt')      # '/root/nltk_data/tokenizers'
nltk.download('stopwords')  # 불용어 목록

sentence = """
Natural language processing (NLP) is a subfield of computer science, information engineering, 
and artificial intelligence concerned with the interactions between computers and human (natural) languages, 
in particular how to program computers to process and analyze large amounts of natural language data.
"""
word_tok = nltk.word_tokenize(sentence)
print(word_tok)

text = """
Natural language processing (NLP) is a subfield of computer science, information engineering, 
and artificial intelligence concerned with the interactions between computers and human (natural) languages, 
in particular how to program computers to process and analyze large amounts of natural language data. 
Challenges in natural language processing frequently involve speech recognition, natural language understanding, 
and natural language generation.
"""

sent_tok = nltk.sent_tokenize(text)
print(sent_tok)

sent_tok[0]

# 텍스트에서 단어 분리
word_tokens = [nltk.word_tokenize(x) for x in nltk.sent_tokenize(text)]
word_tokens[0]

# stop words 제거
stopwords = nltk.corpus.stopwords.words('english')  # 등록된 stop word
stopwords

all_tokens = []
for word in word_tokens:
    all_tokens.append([w for w in word if w.lower() not in stopwords])
all_tokens

# Stemming
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

for word in ['working', 'works', 'worked']:
    print(stemmer.stem(word))

# Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemma = WordNetLemmatizer()
for word in ['working', 'works', 'worked']:
    print(lemma.lemmatize(word, 'v'))

for word in ['happier', 'happiest']:
    print(lemma.lemmatize(word, 'a'))

# 사전 (vocabulary) 생성
word2idx = {}
n_idx = 0
for sent in all_tokens:
    for word in sent:
        if word.lower() not in word2idx:
            word2idx[word.lower()] = n_idx
            n_idx += 1
idx2word = {v:k for k, v in word2idx.items()}

word2idx

# text를 사전의 index로 표현
text_idx = []
for sent in all_tokens:
    sent_idx = []
    for word in sent:
        sent_idx.append(word2idx[word.lower()])
    text_idx.append(sent_idx)
print(text_idx[0])

# text_idx를 다시 단어로 표시
text = []
for sent_idx in text_idx:
    sent = []
    for word_idx in sent_idx:
        sent.append(idx2word[word_idx])
    text.append(sent)
print(text[0])

