import numpy as np
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem import LancasterStemmer
from gensim.models import word2vec
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('gutenberg')

sent_stem = []

# 영문 소설 10개만 사용한다.
n = 10
stemmer = LancasterStemmer()
for i, text_id in enumerate(nltk.corpus.gutenberg.fileids()[:n]):
    text = nltk.corpus.gutenberg.raw(text_id)
    sentences = nltk.sent_tokenize(text)

    # 각 문장에서 명사와 형용사만 발췌한다.
    for sentence in sentences:
        word_tok = nltk.word_tokenize(sentence)
        stem = [stemmer.stem(word) for word in nltk.word_tokenize(sentence)]
        sent_stem.append(stem)
    print('{}: {} ----- processed.'.format(i+1, text_id))

print("총 문장 개수 =", len(sent_stem))
print(sent_stem[0])

model = word2vec.Word2Vec(sent_stem, size =32, window=1, sg=1, negative=1) # 단어 하나를 32차원의 벡터로 표현, sg=1(skipgram), 0(CBOW), negative=2 -> K, 0 -> softmax, hs=1

print("사전 크기 =", len(model.wv.vocab))

def stem(word):
    stem_word = stemmer.stem(word)
    if stem_word not in word2idx:
        print('{}가 없습니다.'.format(word))
        return '_'
    else:
        return stem_word

def get_word2vec(word):
    stem_word = stem(word)
    if stem_word == '_':
        return
    
    word2vec = model.wv[stem_word]
    return word2vec

def most_similar(word):
    stem_word = stem(word)
    if stem_word == '_':
        return
    
    return model.wv.most_similar(stem_word)

def similarity(w1, w2):
    stem_w1 = stem(w1)
    stem_w2 = stem(w2)
    if stem_w1 == '_' or stem_w2 == '_':
        return

    return model.wv.similarity(stem_w1, stem_w2)

get_word2vec('father')

similarity('father', 'mother')

similarity('father', 'doctor')

most_similar('father')

queen1 = get_word2vec('queen').reshape(1, -1)
queen2 = (get_word2vec('king') - get_word2vec('man') + get_word2vec('woman')).reshape(1, -1)
cosine_similarity(queen1, queen2)

similarity('dog', 'cat')

similarity('dog', 'wolf')

get_word2vec('.')

