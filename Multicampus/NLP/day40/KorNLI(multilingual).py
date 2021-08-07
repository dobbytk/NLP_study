import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt

df = pd.read_csv('data/multinli.train.koNLI.tsv', delimiter = '\t', quoting = 3)
df = df.dropna()

# gold_label을 수치화한다. [contradiction (모순 관계) = 0, entailment (얽힘) = 1, neutral (중립) = 2]
df['label'] = LabelEncoder().fit_transform(df['gold_label'])

df = df[:10000]   # 시험용으로 조금만 사용
df.head()

df['label'] = LabelEncoder().fit_transform(df['gold_label'])

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir='bert_ckpt', do_lower_case=False)

# Bert Tokenizer
# 참조: https://huggingface.co/transformers/main_classes/tokenizer.html?highlight=encode_plus#transformers.PreTrainedTokenizer.encode_plus
def bert_tokenizer_v2(sent1, sent2):
    
    encoded_dict = tokenizer.encode_plus(
        text = sent1,
        text_pair = sent2,
        add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
        max_length = MAX_LEN,           # Pad & truncate all sentences.
        pad_to_max_length = True,
        return_attention_mask = True    # Construct attn. masks.
    )
    
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask'] # And its attention mask (simply differentiates padding from non-padding).
    token_type_id = encoded_dict['token_type_ids']  # differentiate two sentences
    
    return input_id, attention_mask, token_type_id

def build_data(sent1, sent2):
    x_ids = []
    x_msk = []
    x_typ = []

    for s1, s2 in tqdm(zip(sent1, sent2)):
        input_id, attention_mask, token_type_id = bert_tokenizer_v2(s1, s2)
        x_ids.append(input_id)
        x_msk.append(attention_mask)
        x_typ.append(token_type_id)

    x_ids = np.array(x_ids, dtype=int)
    x_msk = np.array(x_msk, dtype=int)
    x_typ = np.array(x_typ, dtype=int)

    return x_ids, x_msk, x_typ

sent1 = df['sentence1']
sent2 = df['sentence2']
y_label = df['label']
MAX_LEN = 24 * 2

x_train1, x_test1, x_train2, x_test2, y_train, y_test = train_test_split(sent1, sent2, y_label, test_size=0.1, random_state = 0)

x_train_ids, x_train_msk, x_train_typ = build_data(x_train1, x_train2)
x_test_ids, x_test_msk, x_test_typ = build_data(x_test1, x_test2)

y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

x_train_ids.shape, y_train.shape, x_test_ids.shape, y_test.shape

word2idx = tokenizer.vocab
idx2word = {v:k for k, v in word2idx.items()}
print([idx2word[x] for x in x_train_ids[1]])

bert_model = TFBertModel.from_pretrained("bert-base-multilingual-cased", cache_dir='bert_ckpt')
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

# Downstream task : 자언어 추론 (NLI)
# ----------------------------------
y_output = Dense(3, activation = 'softmax')(output_bert)
model = Model([x_input_ids, x_input_msk, x_input_typ], y_output)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(learning_rate = 0.001))
model.summary()

x_train = [x_train_ids, x_train_msk, x_train_typ]
x_test = [x_test_ids, x_test_msk, x_test_typ]
hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=1, batch_size=1024)

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