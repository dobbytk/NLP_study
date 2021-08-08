import numpy as np
import tensorflow as tf
from transformers import TFBertModel
from tensorflow.keras.layers import Input, Dense, Flatten, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pickle


# 학습 데이터를 읽어온다.
with open('data/train_encoded.pickle', 'rb') as f:
    x_train_ids, x_train_msk, x_train_typ, y_train_start, y_train_end = pickle.load(f)

MAX_SEQ_LEN = 128

# Bert 모델을 생성한다.
K.clear_session()
bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
bert_model.summary() # bert_model을 확인한다. trainable params = 177,854,978

# 1차 학습시에는 TFBertMainLayer를 fine-tuning하지 않는다.
bert_model.trainable = False

# KorQuAD 모델을 생성한다.
# BERT 입력
# ---------
x_input_ids = Input(batch_shape = (None, MAX_SEQ_LEN), dtype = tf.int32)
x_input_msk = Input(batch_shape = (None, MAX_SEQ_LEN), dtype = tf.int32)
x_input_typ = Input(batch_shape = (None, MAX_SEQ_LEN), dtype = tf.int32)

# BERT 출력
# ---------
output_bert = bert_model([x_input_ids, x_input_msk, x_input_typ])[0]

# Downstream task : KorQuAD
# -------------------------
start_layer = Dense(1, use_bias=False, name='start_layer')(output_bert)  # (None, 128, 1)
start_layer = Flatten()(start_layer)

end_layer = Dense(1, use_bias=False, name='end_layer')(output_bert)  # (None, 128, 1)
end_layer = Flatten()(end_layer)

y_start = Activation('softmax')(start_layer)
y_end = Activation('softmax')(end_layer)

model = Model(inputs = [x_input_ids, x_input_msk, x_input_typ], outputs = [y_start, y_end])
model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy')
model.summary()

# 학습
hist = model.fit([x_train_ids, x_train_msk, x_train_typ], 
                 [y_train_start, y_train_end], 
                 batch_size = 64, 
                 epochs=1,
                 validation_split = 0.2)

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label='Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 학습된 출력층 weights를 저장해 둔다
W1 = model.get_layer('start_layer').get_weights()
W2 = model.get_layer('end_layer').get_weights()

with open('data/weights.pickle', 'wb') as f:
    pickle.dump([W1, W2], f, pickle.DEFAULT_PROTOCOL)