%cd '/content/drive/MyDrive/머신러닝(멀티캠퍼스)/nsmc'
!pip install konlpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('./ratings_train.txt', sep='\t')
df_test = pd.read_csv('./ratings_test.txt', sep='\t')
df_train.head(5)
df_test.head(5)
train_length = df_train['document'].astype(str).apply(len)
train_length
plt.figure(figsize=(12, 5))
plt.hist(train_length, bins=200, alpha=0.5, color='r', label='word')
plt.yscale('log', nonposy='clip')

# 그래프 제목
plt.title('Log-Histogram of length of review')
plt.xlabel('Length of review')
plt.ylabel('Number of review')
print('리뷰 길이 최댓값: {}'.format(np.max(train_length)))
print('리뷰 길이 최솟값: {}'.format(np.max(train_length)))
print('리뷰 길이 평균값: {}'.format(np.mean(train_length)))
print('리뷰 길이 표준편차: {}'.format(np.std(train_length)))
print('리뷰 길이 중간값: {}'.format(np.median(train_length)))
print('리뷰 길이 제1사분위: {}'.format(np.percentile(train_length, 25)))
print('리뷰 길이 제3사분위: {}'.format(np.percentile(train_length, 75)))

fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6, 3)
sns.countplot(df_train['label'])
print('긍정 리뷰 개수: {}'.format(df_train['label'].value_counts()[1]))
print('부정 리뷰 개수: {}'.format(df_train['label'].value_counts()[0]))
train_word_counts = df_train['document'].astype(str).apply(lambda x:len(x.split(' ')))
print('리뷰 단어 개수 최댓값: {}'.format(np.max(train_word_counts)))
print('리뷰 단어 개수 최솟값: {}'.format(np.min(train_word_counts)))
print('리뷰 단어 개수 평균값: {:.2f}'.format(np.mean(train_word_counts)))
df_train['document'][:5]
import re

okt = Okt()
stop_words = set(['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '동', '한'])
def preprocessing(review, okt, remove_stopwords=False, stop_words =[]):
  review_text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]', '', review)
  word_review = okt.morphs(review_text, stem=True)

  if remove_stopwords:
    # 불용어 제거 (optional)
    word_review = [token for token in word_review if not token in stop_words]

  return word_review
clean_train_review = []

for review in df_train['document']:
  if type(review) == str:
    clean_train_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
    
  else:
    clean_train_review.append([]) # string이 아니면 비어있는 값 추가

clean_train_review[:4]
clean_test_review = []

for review in df_test['document']:
  if type(review) == str:
    clean_test_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
  
  else:
    clean_test_review.append([])
clean_test_review[:4]
import pickle

with open('./clean_train_review_okt.pkl', 'wb') as f:
  pickle.dump(clean_train_review, f)

with open('./clean_test_review_okt.pkl', 'wb') as f:
  pickle.dump(clean_test_review, f)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_review)
train_sequences = tokenizer.texts_to_sequences(clean_train_review)
test_sequences = tokenizer.texts_to_sequences(clean_test_review)
word_vocab = tokenizer.word_index

MAX_SEQUENCE_LENGTH = 8

train_inputs = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
train_labels = np.array(df_train['label']) # 학습 데이터의 라벨

test_inputs = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
test_labels = np.array(df_test['label'])

x_train, x_val, y_train, y_val = train_test_split(train_inputs, train_labels, test_size=0.2)
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Flatten, Concatenate
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
x_train[0]
vocab_size = len(word_vocab) + 1
EMBEDDING_DIM = 128
HIDDEN_DIM = 100
conv_feature_maps = []
x_input = Input(batch_shape=(None, x_train.shape[1]))
e_layer = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM)(x_input)
e_layer = Dropout(rate=0.5)(e_layer)
for ks in [3, 4, 5]:
  r_layer = Conv1D(filters=HIDDEN_DIM, 
                   kernel_size=ks, 
                   padding='valid', 
                   activation='relu')(e_layer)
  max_pool = GlobalMaxPooling1D()(r_layer)
  flatten = Flatten()(max_pool)
  conv_feature_maps.append(flatten)
r_layer = Concatenate()(conv_feature_maps)
r_layer = Dropout(rate=0.5)(r_layer)
y_output = Dense(250, activation='relu')(r_layer)
y_output = Dense(1, activation='sigmoid')(r_layer)

model = Model(x_input, y_output)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])
model.summary()


# 학습
es = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=2)
cp = ModelCheckpoint(filepath='./', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
hist = model.fit(x_train, y_train, validation_data = (x_val, y_val), batch_size = 512, epochs = 30, callbacks=[es, cp])


# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label = 'Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title('accuracy history')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
test_input = pad_sequences(test_inputs, maxlen=test_inputs.shape[1])
model.evaluate(test_input, test_labels)