import json
import numpy as np
import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.layers import Input, Dense, Flatten, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pickle

# 학습이 완료된 down stream task의 weights를 읽어온다.
with open('data/weights.pickle', 'rb') as f:
    weights_1, weights_2 = pickle.load(f)
    
# 시험 데이터를 읽어온다.
with open('data/test_encoded.pickle', 'rb') as f:
    x_test_ids, x_test_msk, x_test_typ, y_test_start, y_test_end = pickle.load(f)

# 결과 확인을 위해 vocabulary를 읽어온다.
with open('data/vocabulary.pickle', 'rb') as f:
    word2idx = pickle.load(f)

MAX_LEN = 128
idx2word = {v:k for k, v in word2idx.items()}

# Bert 모델을 생성한다.
K.clear_session()
bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')

# KorQuAD 모델을 생성한다.
# BERT 입력
# ---------
x_input_ids = Input(batch_shape = (None, MAX_LEN), dtype = tf.int32)
x_input_msk = Input(batch_shape = (None, MAX_LEN), dtype = tf.int32)
x_input_typ = Input(batch_shape = (None, MAX_LEN), dtype = tf.int32)

# BERT 출력
# ---------
output_bert = bert_model([x_input_ids, x_input_msk, x_input_typ])[0]

# Downstream task : KorQuAD
# -------------------------
start_layer = Dense(1, use_bias=False, name='start_layer', weights=weights_1)(output_bert)  # (None, 128, 1)
start_layer = Flatten()(start_layer)

end_layer = Dense(1, use_bias=False, name='end_layer', weights=weights_2)(output_bert)  # (None, 128, 1)
end_layer = Flatten()(end_layer)

y_start = Activation('softmax')(start_layer)
y_end = Activation('softmax')(end_layer)

model = Model(inputs = [x_input_ids, x_input_msk, x_input_typ], outputs = [y_start, y_end])
model.summary()

# 평가
y_prob = model.predict([x_test_ids, x_test_msk, x_test_typ])

# argmax
y_pred_start = np.argmax(y_prob[0], axis=1)
y_pred_end = np.argmax(y_prob[1], axis=1)

match_cnt = 0
for i in range(y_test_start.shape[0]):
    # if y_test_start[i] == y_pred_start[i] and \
    #    y_test_end[i] == y_pred_end[i]:
    if y_test_start[i] == y_pred_start[i]:
           match_cnt += 1

acc = match_cnt / y_test_start.shape[0]
print('정확도 = {}%'.format(100* np.round(acc, 3)))

# 본문 (아래 text)에 [UNK]가 있으면 위치가 달라진다. <-- 해결은 보류함.
# 우선, [UNK]가 없는 경우만 확인해 보자.
def check(n):
    s = ' '.join([idx2word[x] for x in x_test_ids[n]])
    s = s.replace(' ##', '')
    context = s.split('[SEP] ')
    question = context[0].replace('[CLS] ', '')
    text = context[1].replace(' [SEP]', '')
    print(text, '\n')
    print(question, '\n')
    print(' actual start-end: ({}-{})'.format(y_test_start[n], y_test_end[n]))
    print('predict start-end: ({}-{})\n'.format(y_pred_start[n], y_pred_end[n]))
    print(' actual answer: {}'.format(text[y_test_start[n]:y_test_end[n]]))
    print('predict answer: {}'.format(text[y_pred_start[n]:y_pred_end[n]]))
