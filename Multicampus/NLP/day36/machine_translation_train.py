import pandas as pd
import numpy as np
import re
import sentencepiece as spm
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('./machine_trans.csv')

data.head()

print("전체 샘플의 개수 =", len(data))

src_data, tar_data = list(data['source']), list(data['target'])

# 인코더용 sentencepiece 단어 사전
enc_data_file = "./enc_data.txt"
with open(enc_data_file, 'w', encoding='utf-8') as f:
    for sent in src_data:
        f.write(sent + '\n')
        
# Google의 Sentencepiece를 이용해서 vocabulary를 생성한다.
# -----------------------------------------------------
templates= "--input={} \
            --pad_id=0 --pad_piece=<PAD>\
            --unk_id=1 --unk_piece=<UNK>\
            --bos_id=2 --bos_piece=<BOS>\
            --eos_id=3 --eos_piece=<EOS>\
            --model_prefix={} \
            --vocab_size={} \
            --character_coverage=1.0 \
            --model_type=unigram"

ENC_VOCAB_SIZE = 9766
enc_model_prefix = "./enc_model"
params = templates.format(enc_data_file, enc_model_prefix, ENC_VOCAB_SIZE)

spm.SentencePieceTrainer.Train(params)
enc_sp = spm.SentencePieceProcessor()
enc_sp.Load(enc_model_prefix + '.model')

with open(enc_model_prefix + '.vocab', encoding='utf-8') as f:
    enc_vocab = [doc.strip().split('\t') for doc in f]

enc_word2idx = {k:v for v, [k, _] in enumerate(enc_vocab)}
enc_idx2word = {v:k for v, [k, _] in enumerate(enc_vocab)}

# 디코더용 sentencepiece 단어 사전
dec_data_file = "./dec_data.txt"
with open(dec_data_file, 'w', encoding='utf-8') as f:
    for sent in tar_data:
        f.write(sent + '\n')
        
# Google의 Sentencepiece를 이용해서 vocabulary를 생성한다.
# -----------------------------------------------------
templates= "--input={} \
            --pad_id=0 --pad_piece=<PAD>\
            --unk_id=1 --unk_piece=<UNK>\
            --bos_id=2 --bos_piece=<BOS>\
            --eos_id=3 --eos_piece=<EOS>\
            --model_prefix={} \
            --vocab_size={} \
            --character_coverage=1.0 \
            --model_type=unigram"

DEC_VOCAB_SIZE = 4855
dec_model_prefix = "./dec_model"
params = templates.format(dec_data_file, dec_model_prefix, DEC_VOCAB_SIZE)

spm.SentencePieceTrainer.Train(params)
dec_sp = spm.SentencePieceProcessor()
dec_sp.Load(dec_model_prefix + '.model')

with open(dec_model_prefix + '.vocab', encoding='utf-8') as f:
    dec_vocab = [doc.strip().split('\t') for doc in f]

dec_word2idx = {k:v for v, [k, _] in enumerate(dec_vocab)}
dec_idx2word = {v:k for v, [k, _] in enumerate(dec_vocab)}

ENC_MAX_LEN = max([len(enc_sp.encode_as_ids(s)) for s in src_data])
DEC_MAX_LEN = max([len(dec_sp.encode_as_ids(s)) for s in tar_data])
ENC_MAX_LEN, DEC_MAX_LEN

# 데이터셋 구성하기
enc_input = []
dec_input = []
dec_output = []

for src, tar in zip(src_data, tar_data):
  # encoder 입력
  enc_i = enc_sp.encode_as_ids(src)
  enc_input.append(enc_i)

  # decoder 입력
  dec_i = [dec_sp.bos_id()] # <BOS>에서 시작함
  dec_o = []
  for d in dec_sp.encode_as_ids(tar):
    dec_i.append(d)
    dec_o.append(d)
  dec_o.append(dec_sp.eos_id()) # encoder 출력은 <EOS>로 끝남

  # dec_o는 <EOS>가 마지막에 들어있다. 
  # 나중에 pad_sequences()에서 <EOS>가 잘려 나가지 않도록 MAX_LEN 위치에 <EOS>를 넣어준다. 
  if len(dec_o) > DEC_MAX_LEN:
    dec_o[DEC_MAX_LEN] = dec_sp.eos_id()

  dec_input.append(dec_i)
  dec_output.append(dec_o)

enc_input = pad_sequences(enc_input, maxlen=ENC_MAX_LEN, value=enc_sp.pad_id(), padding='post', truncating='post')
dec_input = pad_sequences(dec_input, maxlen=DEC_MAX_LEN, value=dec_sp.pad_id(), padding='post', truncating='post')
dec_output = pad_sequences(dec_output, maxlen=DEC_MAX_LEN, value=dec_sp.pad_id(), padding='post', truncating='post')

print(enc_input[0])
print(dec_input[0])
print(dec_output[0])

# 사전과 학습 데이터를 저장한다.
with open('./enc_voc.pkl', 'wb') as f:
    pickle.dump([enc_word2idx, enc_idx2word], f, pickle.HIGHEST_PROTOCOL)

with open('./dec_voc.pkl', 'wb') as f:
    pickle.dump([dec_word2idx, dec_idx2word], f, pickle.HIGHEST_PROTOCOL)

with open('./enc_dec_data.pkl', 'wb') as f:
    pickle.dump([enc_input, dec_input, dec_output], f, pickle.HIGHEST_PROTOCOL)


# split train, test set
trainXE, testXE = train_test_split(enc_input, test_size=0.2, random_state=42)
trainXD, testXD = train_test_split(dec_input, test_size=0.2, random_state=42)
trainYD, testYD = train_test_split(dec_output, test_size=0.2, random_state=42)

# 모델 생성하기 및 학습 
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Dot, Activation, Concatenate
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

MODEL_PATH = './machine_trans.attention.h5'
EMB_SIZE = 128
LSTM_HIDDEN = 128
LOAD_MODEL = False

def Attention(x, y):
  score = Dot(axes=(2, 2))([y, x])
  dist = Activation('softmax')(score)
  attention = Dot(axes=(2, 1))([dist, x])
  
  return Concatenate()([y, attention])

K.clear_session()

# Encoder
encoderX = Input(batch_shape=(None, trainXE.shape[1]))
encEMB = Embedding(ENC_VOCAB_SIZE, EMB_SIZE)(encoderX)
encLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
encLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
ey1, eh1, ec1 = encLSTM1(encEMB)
ey2, eh2, ec2 = encLSTM2(ey1)

# Decoder
decoderX = Input(batch_shape=(None, trainXD.shape[1]))
decEMB = Embedding(DEC_VOCAB_SIZE, EMB_SIZE)(decoderX)
decLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
decLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
dy1, _, _ = decLSTM1(decEMB, initial_state = [eh1, ec1])
dy2, _, _ = decLSTM2(dy1, initial_state = [eh2, ec2])
att_dy2 = Attention(ey2, dy2)
decOutput = TimeDistributed(Dense(DEC_VOCAB_SIZE, activation='softmax'))
outputY = decOutput(att_dy2)

model = Model([encoderX, decoderX], outputY)
model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy')

if LOAD_MODEL:
  model.load_weights(MODEL_PATH)

hist = model.fit([trainXE, trainXD], trainYD, validation_data=([testXE, testXD], testYD), batch_size=512, epochs=300, shuffle=True)

model.save_weights(MODEL_PATH)

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label='Validation loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

with open('./translate_train_test.pkl', 'wb') as f:
    pickle.dump([trainXE, trainXD, trainYD, testXE, testXD, testYD], f, pickle.HIGHEST_PROTOCOL) 