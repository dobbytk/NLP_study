# 영문 소설 코퍼스를 통해 특정 단어의 context word를 확인한다.
import numpy as np
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('gutenberg')

sent_nj = []

# 영문 소설 5개만 사용한다.
n = 5
for i, text_id in enumerate(nltk.corpus.gutenberg.fileids()[:n]):
    text = nltk.corpus.gutenberg.raw(text_id)
    sentences = nltk.sent_tokenize(text)

    # 각 문장에서 명사와 형용사만 발췌한다.
    for sentence in sentences:
        word_tok = nltk.word_tokenize(sentence)
        njv = [word for word, pos in nltk.pos_tag(word_tok) if pos=='NN' or pos=='NNS' or pos=='JJ']
        sent_nj.append(njv)
    print('{}: {} ----- processed.'.format(i+1, text_id))

print("총 문장 개수 =", len(sent_nj))
print(sent_nj[0])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sent_nj)

# 단어사전
word2idx = tokenizer.word_index
idx2word = {v:k for k, v in word2idx.items()}

print("사전 크기 =", len(word2idx))

# 문장을 단어의 인덱스로 표현
sent_idx = tokenizer.texts_to_sequences(sent_nj)
sent_idx[0]

def get_context(x, count = True):
    idx = word2idx[x]
    word_count = {v:0 for k,v in word2idx.items()}
    for s_idx in sent_idx:
        if idx in s_idx:
            for i in s_idx:
                word_count[i] += 1

    result = sorted(word_count.items(), key=(lambda x: x[1]), reverse=True)
    result = [(idx2word[i], c) for i, c in result[:20]]

    if count:
        return result
    else:
        return [w for w, c in result]

# context = get_context('apostles', count=True)
    context = get_context('sanguine', count=True)
    context

mother = get_context('mother', count=False)
father = get_context('father', count=False)
doctor = get_context('doctor', count=False)

def jaccard(x, y):
    hab = set(x) | set(y)
    gyo = set(x) & set(y)
    return len(gyo) / len(hab)

jaccard(mother, father)

jaccard(mother, doctor)

