# Keras에서 Attention value를 계산하는 절차를 확인한다.
import numpy as np
from tensorflow.keras.layers import Dot, Activation
import tensorflow.keras.backend as K

e = np.array([[[0.3, 0.2, 0.1], [0.3, 0.6, 0.5], [0.3, 0.8, 0.2], [0.7, 0.2, 0.1]]])
d = np.array([[[0.1, 0.2, 0.3], [0.3, 0.2, 0.5], [0.1, 0.8, 0.4], [0.4, 0.2, 0.3]]])
e.shape, d.shape

te = K.constant(e)
td = K.constant(d)

dot_product = Dot(axes=(2,2))([te, td])    # (None, 4, 4)
dot_product

attn_score = Activation('softmax')(dot_product)
attn_score

attn_val = Dot(axes=(2,1))([attn_score, te])
attn_val

# attn_score 까지는 확실하므로, attn_val 부분만 수동으로 확인해 본다.
score = attn_score.numpy()
score

for n in range(4):
    e1 = e[0, 0, :]
    v1 = score[0, n, 0] * e1

    e2 = e[0, 1, :]
    v2 = score[0, n, 1] * e2

    e3 = e[0, 2, :]
    v3 = score[0, n, 2] * e3

    e4 = e[0, 3, :]
    v4 = score[0, n, 3] * e4
    
    # n번 row의 attention value를 계산한다.
    print(np.sum([v1, v2, v3, v4], axis=0))

# 수동으로 계산한 위의 결과와 Keras의 attn_val 결과가 일치하는지 육안으로 확인한다.
attn_val.numpy()

