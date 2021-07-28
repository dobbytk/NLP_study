# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import re
import pickle
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Dropout, Concatenate, Flatten, Average
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import seaborn as sns
import math

with open('./pv_dm_data.pkl', 'rb') as f:
    x_data, x_pv, x_seq, y_target, y_topic = pickle.load(f)

x_pv.shape, x_seq.shape, y_target.shape

# PV-DM 모델을 생성한다.
doc_size = len(set(x_pv.reshape(-1)))
vocab_size = 20000
doc_dim = 400
LOAD_MODEL = False

if LOAD_MODEL:
    # 학습된 모델을 읽어온다.
    model = load_model("./pv_dm.h5") 
else:
    pv_input = Input(batch_shape = (None, 1))
    sq_input = Input(batch_shape = (None, 9))

    pv_emb = Embedding(doc_size, doc_dim, name='doc2vec')(pv_input) # 내부 weight가 D행렬
    pv_emb = Flatten()(pv_emb)
    
    sq_emb = Embedding(vocab_size, doc_dim, name='word2vec')(sq_input) # 내부 weight가 C행렬
    sq_emb = LSTM(doc_dim)(sq_emb)  # context의 흐름을 분석하려면 이렇게 한다.
#   sq_emb = Flatten()(sq_emb)    # 논문 내용대로 구현하려면 위의 LSTM()대신 이렇게 한다.

    e_merge = Concatenate()([pv_emb, sq_emb])
    y_output = Dense(vocab_size, activation='softmax')(e_merge)

    model = Model([pv_input, sq_input], y_output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.001))
model.summary()

# 학습할 때 epoch가 증가할 때마다 learning rate를 decay 시킨다.
def lr_decay(epoch):
    init_lr = 0.05
    min_lr = 0.001
    drop = 0.5
    epochs_drop = 3.0
    lr = init_lr * pow(drop, math.floor((1 + epoch) / epochs_drop))
    return max([lr, min_lr])

a = []
for i in np.arange(20):
    a.append(lr_decay(i))

plt.plot(a)
print(a)

# 학습
l_rate = LearningRateScheduler(lr_decay, verbose=1)
hist = model.fit([x_pv, x_seq], y_target, batch_size=4096, epochs=50, callbacks = [l_rate])

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 학습 결과를 저장해 둔다.
model.save("./pv_dm.h5")

# 모델에서 paragraph vector (D)와 word vector (W)를 읽어온다.
D = model.get_layer('doc2vec').get_weights()[0]
W = model.get_layer('word2vec').get_weights()[0]
D.shape, W.shape

if D.shape[0] == len(y_topic):
    print('ok')

# paragraph vector (D)와 y_topic을 저장한다.
with open('./pv_dm_D.pkl', 'wb') as f:
    pickle.dump([x_data, D, np.array(y_topic).reshape(-1, 1)], f, pickle.DEFAULT_PROTOCOL)

