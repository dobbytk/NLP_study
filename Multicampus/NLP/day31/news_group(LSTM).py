# LSTM으로 subject를 classification한다.
# ------------------------------------
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# %cd '/content/drive/MyDrive/Colab Notebooks'

# 전처리가 완료된 한글 코퍼스를 읽어온다.
with open('./newsgroup20.pkl', 'rb') as f:
    subject, text, topic = pickle.load(f)

n_topic = len(set(topic))

max_word = 5000
tokenizer = Tokenizer(num_words = max_word, oov_token = '<OOV>')
tokenizer.fit_on_texts(subject)
sent_idx = tokenizer.texts_to_sequences(subject)
word2idx = {k:v for (k, v) in list(tokenizer.word_index.items())[:max_word]}
word2idx['<PAD>'] = 0
idx2word = {v:k for (k, v) in word2idx.items()}

# subject을 길이 분포를 확인한다.
n_len = [len(sub) for sub in sent_idx]
sns.displot(n_len)
plt.show()

# 문장의 길이를 6으로 맞춘다.
x_data = pad_sequences(sent_idx, maxlen=6, padding='post', truncating='post')
y_data = topic

# 학습 데이터와 시험데이터로 분리한다.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# NPLM 모델을 생성한다.
EMB_SIZE = 32
VOCAB_SIZE = len(word2idx) + 1
x_input = Input(batch_shape = (None, x_train.shape[1]))
x_embed = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_SIZE)(x_input) # weights 옵션으로 C행렬을 넣어줄 수 있다. - 사전학습
x_embed = Dropout(0.5)(x_embed)
x_lstm = LSTM(64, dropout=0.5)(x_embed)
y_output = Dense(n_topic, activation = 'softmax')(x_lstm)

model = Model(x_input, y_output)     # 학습, 예측용 모델
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.01))
model.summary()

# 모델을 학습한다.
hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 512, epochs = 30) # 사전학습해서 Fine-tuning 

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label = 'Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 시험 데이터로 학습 성능을 평가한다
pred = model.predict(x_test)
y_pred = np.argmax(pred, axis=1).reshape(-1, 1)
accuracy = (y_pred == y_test).mean()
print("\nAccuracy = %.2f %s" % (accuracy * 100, '%'))

# 잘못 분류된 subject들을 살펴본다. 어떤 특징을 발견할 수 있을까?
mis_idx = [list(x_test[m]) for m in np.where(y_pred != y_test)[0]]

mis_sent = []
for sent in mis_idx:
    tmp = []
    for i in sent:
        if i > 0:      # pad가 아니면
            tmp.append(idx2word[i])
    mis_sent.append(tmp)
mis_sent[:10]

# 잘못 분류된 subject들을 살펴본다. 어떤 특징을 발견할 수 있을까?
good_idx = [list(x_test[m]) for m in np.where(y_pred == y_test)[0]]

good_sent = []
for sent in good_idx:
    tmp = []
    for i in sent:
        if i > 0:      # pad가 아니면
            tmp.append(idx2word[i])
    good_sent.append(tmp)
good_sent[:10]

sns.displot(y_pred)
plt.show()

