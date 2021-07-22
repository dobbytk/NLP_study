from gensim.models.fasttext import FastText
import numpy as np

texts = [['human', 'interface', 'computer'],
         ['survey', 'user', 'computer', 'system', 'response', 'time'],
         ['eps', 'user', 'interface', 'system'],
         ['system', 'human', 'system', 'eps'],
         ['user', 'response', 'time'],
         ['trees'],
         ['graph', 'trees'],
         ['graph', 'minors', 'trees'],
         ['graph', 'minors', 'survey']]

model = FastText(size=4, window=3, min_count=1, sentences=texts, 
                 iter=100, bucket=10, min_n=3, max_n=3, sg=0) 
# bucket = hash k값, 최신 gensim version에서는 size -> vector_size/iter -> epochs

# 워드 벡터 확인
model.wv['computer']

# oov라도 다른 벡터를 갖는다
model.wv['comoklksjd']     # 'com' 성분이 포함돼 있다.
model.wv['omplkasjdflkd']  # 'omp' 성분이 포함돼 있다.


# 유사도 확인
model.wv.most_similar('computer', topn = 5)

# 어휘 사전 확인 - subword 사전은 아님
model.wv.vocab['eps']

# hash table (bucket) 확인. subword들의 워드 벡터가 저장된 공간.
model.wv.vectors_ngrams

"""
ex) computer
<co + omp + mpu + put + ute + ter + er> + <computer>
각각의 서브워드 벡터값을 더해서 computer에 대한 임베딩 벡터 구성
"""
