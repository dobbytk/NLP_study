from gensim.models.fasttext import FastText
from hangul_utils import split_syllables, join_jamos
import numpy as np
import pickle

with open('./konovel_preprocessed.pkl', 'rb') as f:
    texts = pickle.load(f)

# 자모 분해 예시
jamo = split_syllables('안녕하세요')
word = join_jamos(jamo)
print(jamo)
print(word)
result = []
tmp = []

for sentence in texts:
    for word in sentence:
        texts_jamo = split_syllables(word)
        tmp.append(texts_jamo)
    result.append(tmp)
    tmp = []
print(result[:1])
model = FastText(size=100, window=5, min_count=10, sentences=result, 
                 iter=200, bucket=2000000, min_n=3, max_n=3, sg=1, negative=2, 
                 sample=1e-5, max_vocab_size=10000)

dic = model.wv.key_to_index
dic['ㅂㅏㄷㅏ']
search = split_syllables('바다')
search
r = model.wv.most_similar(search, topn = 10)

output = [(join_jamos(w), s) for (w, s) in r]
output