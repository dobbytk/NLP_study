# A Neural Probabilistic Language Model (NPLM)
#
# NPLM 논문 : Yoshua Bengio, et. al., 2003, A Neural Probabilistic Language Model


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

data = ["The cat is walking in the bedroom",
        "A dog was running in a room",
        "The cat is running in a room",
        "A dog is walking in a bedroom",
        "The dog was walking in the room"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
word2idx = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(data)

# sequences 뒤에 <EOS>를 추가한다.
word2idx_len = len(word2idx)
word2idx['<EOS>'] = word2idx_len + 1  # end of sentence 추가
idx2word = {v: k for (k, v) in word2idx.items()}
sequences = [s + [word2idx['<EOS>']] for s in sequences]

def prepare_sentence(seq, maxlen):
    # Pads seq and slides windows
    x = []
    y = []
    for i, w in enumerate(seq[1:], 1):
        x.append(pad_sequences([seq[:i]], maxlen=maxlen - 1)[0])
        y.append(w)
    return x, y


# 학습 데이터를 생성한다.
maxlen = max([len(s) for s in sequences])
x = []
y = []
for seq in sequences:
    x_, y_ = prepare_sentence(seq, maxlen)
    x += x_
    y += y_
    
x_train = np.array(x)
y_train = np.array(y)

# NPLM 모델을 생성한다.
EMB_SIZE = 8
VOCAB_SIZE = len(word2idx) + 1
x_input = Input(batch_shape = (None, x_train.shape[1]))
x_embed = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_SIZE, name='emb')(x_input)

x_lstm = LSTM(10)(x_embed)
y_output = Dense(VOCAB_SIZE, activation = 'softmax')(x_lstm)

model = Model(x_input, y_output)     # 학습, 예측용 모델
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.01))
model.summary()

# 모델을 학습한다.
hist = model.fit(x_train, y_train, epochs=300, verbose=0)

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

C = model.get_layer('emb').get_weights()[0]

# 한 단어의 워드 벡터를 조회한다.
word = 'dog'
w_idx = word2idx[word]
wv = C[w_idx, :]  # look-up
print('\n단어 :', word)
print(np.round(wv, 3))

def get_prediction(model, sent):
    x = tokenizer.texts_to_sequences(sent)[0]
    x = pad_sequences([x], maxlen=maxlen - 1)[0]
    x = np.array(x).reshape(1, -1)
    return model.predict(x)[0]

# 주어진 문장 다음에 나올 단어를 예측한다.
x_test = ['A dog is walking in a']
p = get_prediction(model, x_test)
n = np.argmax(p)
prob = p[n]
next_word = idx2word[n]
print("\n{} --> '{}', probability = {:.2f}%".format(x_test, next_word, prob * 100))