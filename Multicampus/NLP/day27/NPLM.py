import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

nltk.download('punkt')

corpus = ['The cat is walking in the bedroom.',
          'A dog was running in a room.',
          'The cat is running in a room.',
          'A dog is walking in a bedroom.',
          'The dog was walking in the room.']

sentences = []
for c in corpus:
  sentences.append(nltk.word_tokenize(c))

# 단어 사전 구축
all_tokens = []
for i in range(len(sentences)):
  for j in range(len(sentences[i])):
    all_tokens.append(sentences[i][j])
all_tokens[:10]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word2idx = tokenizer.word_index
idx2word = {v:k for k, v in word2idx.items()}

train_to_idx = tokenizer.texts_to_sequences(sentences)

train_inputs = []
y_label = []

for i in range(len(train_to_idx)):
  for j in range(1, len(train_to_idx[i])):
    train_inputs.append(train_to_idx[i][:j])
    y_label.append(train_to_idx[i][j])

train_inputs_pad = pad_sequences(train_inputs, maxlen=7)

df_train = pd.DataFrame({
    'x_emb':train_inputs_pad,
    'label':y_label
})

train = np.array(train_inputs_pad)
label = np.array(y_label).reshape(-1, 1)
vocab_size = len(word2idx)+1

x_train, x_test, y_train, y_test = train_test_split(train, label, test_size=0.1)

x_input = Input(shape=(7, )) # batch_shape하고 shape을 쓸 때의 차이가 무엇인가?
x_emb = Embedding(input_dim=vocab_size, output_dim = 8, name='emb')(x_input)

# H-network
x_emb_H = Dense(10, use_bias=True, activation='tanh')(x_emb)

# U-network
x_emb_UH = Dense(10, use_bias=False)(x_emb_H)

x_emb_W = Dense(10, use_bias=True)(x_emb)


y_output = Add()([x_emb_W, x_emb_UH])
y_output = Flatten()(y_output)
y_output = Dense(vocab_size, activation='softmax')(y_output)

model = Model(x_input, y_output)
model_w = Model(x_input, x_emb_H)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
model.summary()

hist = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

C = model.get_layer('emb').get_weights()[0] # 0이면 가중치값, 1이면 bias값을 뽑아낸다.

# 한 단어의 워드 벡터를 조회한다.
word = 'dog'
w_idx = word2idx[word]
wv = C[w_idx, :]  # look-up
print('\n단어 :', word)
print(np.round(wv, 3))

def get_prediction(model, sent):
    x = tokenizer.texts_to_sequences(sent)[0]
    x = pad_sequences([x], maxlen= 8 - 1)[0]
    x = np.array(x).reshape(1, -1)
    return model.predict(x)[0]

# 주어진 문장 다음에 나올 단어를 예측한다.
x_test = ['A dog is walking in a']
p = get_prediction(model, x_test)
n = np.argmax(p)
prob = p[n]
next_word = idx2word[n]
print("\n{} --> '{}', probability = {:.2f}%".format(x_test, next_word, prob * 100))