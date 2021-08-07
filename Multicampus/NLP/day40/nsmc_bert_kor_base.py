# BERT를 이용한 네이버 영화 감성분석
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

# huggingface.co --> (우측상단) models --> (왼쪽 메뉴) languages에서 ko --> (오른쪽) kykim/bert-kor-base 클릭
tokenizer = BertTokenizer.from_pretrained("kykim/bert-kor-base", cache_dir='kykim_bert_ckpt', do_lower_case=False)

df = pd.read_csv('data/naver_movie/ratings.txt', header=0, delimiter='\t', quoting=3)
df = df.dropna()
df.head()

# "\d+"는 숫자 1개 이상을 의미함. 모든 숫자를 공백으로 치환
df['document'] = df['document'].apply(lambda x: re.sub(r"\d+", " ", x))
df.drop('id', axis = 1, inplace = True)

document = list(df['document'])
label = list(df['label'])
x_train, x_test, y_train, y_test = train_test_split(document, label, test_size=0.2)

MAX_LEN = 60

# Bert Tokenizer
# 참조: https://huggingface.co/transformers/main_classes/tokenizer.html?highlight=encode_plus#transformers.PreTrainedTokenizer.encode_plus
def bert_tokenizer(sent):
    
    encoded_dict = tokenizer.encode_plus(
        text = sent,
        add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
        max_length = MAX_LEN,           # Pad & truncate all sentences.
        pad_to_max_length = True,
        return_attention_mask = True    # Construct attn. masks.
    )
    
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask'] # And its attention mask (simply differentiates padding from non-padding).
    token_type_id = encoded_dict['token_type_ids']  # differentiate two sentences
    
    return input_id, attention_mask, token_type_id

def build_data(doc):
    x_ids = []
    x_msk = []
    x_typ = []

    for sent in tqdm(doc):
        input_id, attention_mask, token_type_id = bert_tokenizer(sent)
        x_ids.append(input_id)
        x_msk.append(attention_mask)
        x_typ.append(token_type_id)

    x_ids = np.array(x_ids, dtype=int)
    x_msk = np.array(x_msk, dtype=int)
    x_typ = np.array(x_typ, dtype=int)

    return x_ids, x_msk, x_typ

x_train_ids, x_train_msk, x_train_typ = build_data(x_train)
x_test_ids, x_test_msk, x_test_typ = build_data(x_test)

y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

x_train_ids.shape, y_train.shape
x_test_ids.shape, y_test.shape

bert_model = TFBertModel.from_pretrained("kykim/bert-kor-base", cache_dir='kykim_bert_ckpt')
bert_model.summary() # bert_model을 확인한다. trainable params = 177,854,978

# TFBertMainLayer는 fine-tuning을 하지 않는다. (시간이 오래 걸림)
bert_model.trainable = False
bert_model.summary() # bert_model을 다시 확인한다. trainable params = 0

# BERT 입력
# ---------
x_input_ids = Input(batch_shape = (None, MAX_LEN), dtype = tf.int32)
x_input_msk = Input(batch_shape = (None, MAX_LEN), dtype = tf.int32)
x_input_typ = Input(batch_shape = (None, MAX_LEN), dtype = tf.int32)

# BERT 출력
# ---------
output_bert = bert_model([x_input_ids, x_input_msk, x_input_typ])[1]

# Downstream task : 네이버 영화 감성분석
# -------------------------------------
y_output = Dense(1, activation = 'sigmoid')(output_bert)
model = Model([x_input_ids, x_input_msk, x_input_typ], y_output)
model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.01))
model.summary()

x_train = [x_train_ids, x_train_msk, x_train_typ]
x_test = [x_test_ids, x_test_msk, x_test_typ]
hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=3, batch_size=1024)

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
y_pred = np.where(pred > 0.5, 1, 0)
accuracy = (y_pred == y_test).mean()
print("\nAccuracy = %.2f %s" % (accuracy * 100, '%'))

# bert_model.trainable = True로 바꾸고, learning-rate를 작게 적용해서
# 전체를 다시 학습한다. (미세 조정)