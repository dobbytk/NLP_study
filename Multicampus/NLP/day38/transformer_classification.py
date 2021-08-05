""" news_group classification using transformer """
import numpy as np
import re
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, GlobalAveragePooling1D, Add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization
from tensorflow.keras.layers import Permute, Reshape
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
import nltk
from Transformer import Encoder, PaddingMask


nltk.download('punkt')
nltk.download('stopwords')

# %cd '/content/drive/MyDrive/Colab Notebooks'

# news data를 읽어온다. subject 분석용.
newsData = fetch_20newsgroups(shuffle=True, random_state=1, remove=('footers', 'quotes'))

# 첫 번째 news를 조회해 본다.
news = newsData['data']
topic = newsData['target']
topic_name = newsData['target_names']
n=0
print(len(news))
print(news[n])
print('topic = ', topic[n], topic_name[topic[n]])

# Subject만 추출한다.
subjects = []
for text in news:
    for sent in text.split('\n'):
        idx = sent.find('Subject:')
        if idx >= 0:       # found
            subjects.append(sent[(idx + 9):].replace('Re: ', ''))
            break

# subject를 전처리한다.
stemmer = PorterStemmer()
stop_words = stopwords.words('english')

# 1. 영문자가 아닌 문자를 모두 제거한다.
subject1 = [re.sub("[^a-zA-Z]", " ", s) for s in subjects]

# 2. 불용어를 제거하고, 모든 단어를 소문자로 변환하고, 길이가 2 이하인 
# 단어를 제거한다
# 3. Porterstemmer를 적용한다.
subject2 = []
for sub in subject1:
    tmp = []
    for w in sub.split():
        w = w.lower()
        if len(w) > 2 and w not in stop_words:
            tmp.append(stemmer.stem(w))
    subject2.append(' '.join(tmp))

# news data를 다시 읽어온다. news의 body 부분 처리.
newsData = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
news = newsData['data']

# body 부분을 전처리한다.

# 1. 영문자가 아닌 문자를 모두 제거한다.
news1 = [re.sub("[^a-zA-Z]", " ", s) for s in subjects]

# 2. 불용어를 제거하고, 모든 단어를 소문자로 변환하고, 길이가 3 이하인 
# 단어를 제거한다
# 3. Porterstemmer를 적용한다.
news2 = []
for doc in news1:
    doc1 = []
    for w in doc.split():
        w = w.lower()
        if len(w) > 3 and w not in stop_words:
            doc1.append(stemmer.stem(w))
    news2.append(' '.join(doc1))

# 전처리가 완료된 데이터를 저장한다.
with open('./newsgroup20.pkl', 'wb') as f:
    pickle.dump([subject2, news2, topic], f, pickle.DEFAULT_PROTOCOL)

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

n_len = [len(sub) for sub in sent_idx]

x_data = pad_sequences(sent_idx, maxlen=15, padding='post', truncating='post')
y_data = topic

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Transformer Encoder + FFN = classifier 
# 대량의 corpus로 사전학습을 시켜놓는다면 BERT
K.clear_session()
x_input = Input(batch_shape=(None, x_train.shape[1]), dtype="int32")

padding_mask = PaddingMask()(x_input)
encoder = Encoder(num_layers=2, d_model=64, num_heads=4, d_ff=32, vocab_size=len(word2idx), dropout_rate=0.3)
enc_output, _ = encoder(x_input, padding_mask) 
enc_output = GlobalAveragePooling1D()(enc_output)

y_output = Dense(n_topic, activation='softmax')(enc_output)

model = Model(inputs=x_input, outputs=y_output)
model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy')
model.summary()

es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)
hist = model.fit(x_train, y_train, batch_size=512, epochs=100, validation_data=(x_test, y_test), shuffle=True, callbacks=[es])

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label='Val loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

pred = model.predict(x_test)
y_pred = np.argmax(pred, axis=1).reshape(-1, 1)
accuracy = (y_pred == y_test).mean()
print("\nAccuracy = %.2f %s" % (accuracy * 100, '%'))