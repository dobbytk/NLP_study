# Google's pre-trained Word2Vec 사용 예시
# download : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# ---------------------------------------------------------------------------
import gensim
import numpy as np

# Load Google's pre-trained Word2Vec model.
path = 'd:/GoogleWord2Vec/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

cat = model['cat']
dog = model['dog']
human = model['human']

np.round(cat, 3)

# 유클리디언 거리 유사도
np.sqrt(np.sum((cat - dog) ** 2))
np.sqrt(np.sum((cat - human) ** 2))

# 코사인 유사도
def cosDistance(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cosDistance(cat, dog)
cosDistance(cat, human)

model.similarity('cat', 'dog')
model.similarity('cat', 'human')

model.similarity('father', 'daughter')
model.similarity('mother', 'daughter')