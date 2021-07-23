# Latent Semantic Analysis (LSA)
# ------------------------------
import numpy as np
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 전처리가 완료된 한글 코퍼스를 읽어온다.
with open('./newsgroup20.pkl', 'rb') as f:
    subject, text, target = pickle.load(f)

n_target = len(set(target))

# TF-IDF matrix를 생성한다.
tf_vector = TfidfVectorizer(max_features = 500)
tfidf = tf_vector.fit_transform(text)
print(tfidf.shape)

vocab = tf_vector.get_feature_names()
print(vocab[:20])

# Latent Semantic Analysis (LSA) - 행렬 분해를 통해서 문서를 주제별로 클러스터링하겠다
# ------------------------------
svd = TruncatedSVD(n_components = n_target, n_iter=1000)
svd.fit(tfidf)

U = svd.fit_transform(tfidf) / svd.singular_values_
VT = svd.components_
S = np.diag(svd.singular_values_)
U.shape, S.shape, VT.shape


# SVD decomposes the original DTM into three matrices S=U.(sigma).(V.T). 
# Here the matrix U denotes the document-topic matrix while (V) is the topic-term matrix.
# U, S, VT 행렬의 의미 --> Latent Semantic Analysis (LSA)
# U 행렬 ~ 차원 = (문서 개수 X topic 개수) : 문서당 topic 분포
# S 행렬 ~ 차원 = (topic 개수 X topic 개수)
# VT 행렬. 차원 = (topic 개수 X 단어 개수) : topic 당 단어 분포
U[0, :]   # 문서-0의 U

# 문서 별 Topic 번호를 확인한다. (문서 10개만 확인)
for i in range(10):
    print('문서-{:d} : topic = {:02d}, target = {:02d}'.format(i, np.argmax(U[i:(i+1), :][0]), target[i]))

text[0], text[7]

# VT 행렬에서 topic 별로 중요 단어를 표시한다
for i in range(len(VT)):
    idx = VT[i].argsort()[::-1][:10]
    print('토픽-{:2d} : '.format(i+1), end='')
    for n in idx:
        print('{:s} '.format(vocab[n]), end='')
    print()

