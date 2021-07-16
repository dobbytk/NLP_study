
import pandas as pd
import numpy as np
import re
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle

# Commented out IPython magic to ensure Python compatibility.
# Quora question 데이터를 읽어온다.
# %cd '/content/drive/My Drive/Colab Notebooks'
df = pd.read_csv('./quora_question_pairs.csv')
df.head()

# 중복된 페어와 중복되지 않은 페어로 분리한다.
pos_data = df.loc[df['is_duplicate'] == 1]
neg_data = df.loc[df['is_duplicate'] == 0]

# 중복되지 않은 페어가 많으므로 둘의 비율이 비슷하도록 양을 조절한다.
# dample_frac (%) 만큼 샘플링
sample_frac = len(pos_data) / len(neg_data)
print("before : %.2f" % sample_frac)

neg_data = neg_data.sample(frac = sample_frac)
sample_frac = len(pos_data) / len(neg_data)
print("after : %.2f" % sample_frac)

# 두 데이터를 다시 합친다
df = pd.concat([neg_data, pos_data])

# 간단히 전처리를 수행한다. FILTERS에 포함된 문자 제거, 소문자로 변환.
FILTERS = "([~.,!?\"':;)(])"
change_filter = re.compile(FILTERS)

questions1 = [str(s) for s in df['question1']]
questions2 = [str(s) for s in df['question2']]

filtered_questions1 = list()
filtered_questions2 = list()

for q in questions1:
     filtered_questions1.append(re.sub(change_filter, "", q).lower())
        
for q in questions2:
     filtered_questions2.append(re.sub(change_filter, "", q).lower())

# vocabulary를 구축하고 단어들을 워드 인덱스로 변환한다.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(filtered_questions1 + filtered_questions2)
questions1_sequence = tokenizer.texts_to_sequences(filtered_questions1)
questions2_sequence = tokenizer.texts_to_sequences(filtered_questions2)
word2idx = tokenizer.word_index

# 한 문장의 길이는 31개로 제한한다.
MAX_SEQUENCE_LENGTH = 31
q1_data = pad_sequences(questions1_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
q2_data = pad_sequences(questions2_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
labels = np.array(df['is_duplicate'], dtype=int)

# 전처리 결과를 저장해 둔다.
with open("./qqp.pkl", 'wb') as f:
    pickle.dump([q1_data, q2_data, labels, word2idx], f, pickle.DEFAULT_PROTOCOL)