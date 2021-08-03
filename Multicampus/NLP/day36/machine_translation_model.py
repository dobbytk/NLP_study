from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Concatenate, Dot, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import sentencepiece as spm
import numpy as np
import pickle
from nltk.translate.bleu_score import sentence_bleu

# 사전과 학습 데이터를 불러온다.
with open('./enc_voc.pkl', 'rb') as f:
  enc_word2idx, enc_idx2word = pickle.load(f)

with open('./dec_voc.pkl', 'rb') as f:
  dec_word2idx, dec_idx2word = pickle.load(f)

with open('./translate_train_test.pkl', 'rb') as f:
  trainXE, trainXD, trainYD, testXE, testXD, testYD = pickle.load(f)


ENC_MAX_LEN = 33
DEC_MAX_LEN = 36
ENC_VOCAB_SIZE = len(enc_idx2word)
DEC_VOCAB_SIZE = len(dec_idx2word)
EMB_SIZE = 128
LSTM_HIDDEN = 128
MODEL_PATH = './machine_trans.attention.h5'

# 데이터 전처리 과정에서 생성한 SentencePiece model을 불러온다.
SPM_ENC_MODEL = "./enc_model.model"
enc_sp = spm.SentencePieceProcessor()
enc_sp.Load(SPM_ENC_MODEL)

SPM_DEC_MODEL = "./dec_model.model"
dec_sp = spm.SentencePieceProcessor()
dec_sp.Load(SPM_DEC_MODEL)

def Attention(x, y):
  score = Dot(axes=(2, 2))([y, x])
  dist = Activation('softmax')(score)
  attention = Dot(axes=(2, 1))([dist, x])
  return Concatenate()([y, attention])

K.clear_session()
# Encoder
encoderX = Input(batch_shape=(None, ENC_MAX_LEN))
encEMB = Embedding(ENC_VOCAB_SIZE, EMB_SIZE)(encoderX)
encLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True ,return_state=True)
encLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
ey1, eh1, ec1 = encLSTM1(encEMB)
ey2, eh2, ec2 = encLSTM2(ey1)

# Decoder
decoderX = Input(batch_shape=(None, 1)) # 한 단어씩 입력으로 받는다
decEMB = Embedding(DEC_VOCAB_SIZE, EMB_SIZE)(decoderX)
decLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
decLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
dy1, _, _ = decLSTM1(decEMB, initial_state=[eh1, ec1])
dy2, _, _ = decLSTM2(dy1, initial_state=[eh2, ec2])
att_dy2 = Attention(ey2, dy2)
decOutput = TimeDistributed(Dense(DEC_VOCAB_SIZE, activation='softmax'))
outputY = decOutput(att_dy2)

model = Model([encoderX, decoderX], outputY)
model.load_weights(MODEL_PATH)

# translation용 모델
model_enc = Model(encoderX, [eh1, ec1, eh2, ec2, ey2])

ih1 = Input(batch_shape = (None, LSTM_HIDDEN))
ic1 = Input(batch_shape = (None, LSTM_HIDDEN))
ih2 = Input(batch_shape = (None, LSTM_HIDDEN))
ic2 = Input(batch_shape = (None, LSTM_HIDDEN))
ey = Input(batch_shape = (None, ENC_MAX_LEN, LSTM_HIDDEN))

dec_output1, dh1, dc1 = decLSTM1(decEMB, initial_state = [ih1, ic1])
dec_output2, dh2, dc2 = decLSTM2(dec_output1, initial_state = [ih2, ic2])
dec_attention = Attention(ey, dec_output2)
dec_output = decOutput(dec_attention)
model_dec = Model([decoderX, ih1, ic1, ih2, ic2, ey], [dec_output, dh1, dc1, dh2, dc2])

def decode_sequence(src):
  src = src[np.newaxis, :]
  init_h1, init_c1, init_h2, init_c2, enc_y = model_enc.predict(src)

  # 시작 단어는 <BOS>로 한다.
  word = np.array(dec_sp.bos_id()).reshape(1, 1)

  decoded_sentence = []
  for i in range(DEC_MAX_LEN):
    dY, next_h1, next_c1, next_h2, next_c2 = model_dec.predict([word, init_h1, init_c1, init_h2, init_c2, enc_y])

    next_word = np.argmax(dY[0, 0])
    # 예상 단어가 <EOS>이거나 <PAD>이면 더 이상 예상할 게 없다.
    if next_word == dec_sp.eos_id() or next_word == dec_sp.pad_id():
      break
      
    # 다음 예상 단어인 디코더의 출력을 decoded_sentence에 추가한다.
    decoded_sentence.append(dec_idx2word[next_word])

    # 디코더의 다음 recurrent를 위해 입력 데이터와 hidden 값을 준비한다.
    # 입력은 word이고, hidden은 h와 c이다.
    word = np.array(next_word).reshape(1, 1)

    init_h1 = next_h1
    init_c1 = next_c1
    init_h2 = next_h2
    init_c2 = next_c2

  return dec_sp.decode_pieces(decoded_sentence)


def translation(str_string):
    st_idx = []
    for x in enc_sp.encode_as_pieces(str_string):
      if x in enc_word2idx:
        st_idx.append(enc_word2idx[x])
      else:
        st_idx.append(enc_sp.unk_id())

    if len(st_idx) < ENC_MAX_LEN:
      st_idx.extend([enc_sp.pad_id()] * (ENC_MAX_LEN - len(st_idx)))
    else:
      st_idx = st_idx[0:ENC_MAX_LEN]
    return st_idx

bleu_list = []
for que_str, reference in zip(src_data[:10], tar_data[:10]):
    q_idx = translation(que_str)
    candidate = decode_sequence(np.array(q_idx))

    # BLEU를 측정한다.
    # 기계번역의 reference는 어느정도 객관성이 있지만, 일상 대화용 챗봇의 reference는 매우 주관적이기 때문에,
    # test data로 측정한 챗봇의 BLEU는 매우 낮을 수밖에 없다. 특정 업무를 위한 챗봇의 reference는 어느정도
    # 객관성이 있을 수 있다.
    #
    # 1. 짧은 문장이 많기 때문에 단어가 아닌 subword 단위로 BLEU를 측정한다.
    # 2. (Papineni et al. 2002)은 micro-average를 사용했지만, 여기서는 단순 평균인 macro-average를 사용한다.
    reference = enc_sp.encode_as_pieces(reference)
    candidate = enc_sp.encode_as_pieces(candidate)

    bleu = sentence_bleu([reference], candidate, weights=[1/2., 1/2.])
    bleu_list.append(bleu)
    print(que_str, '-->', enc_sp.decode_pieces(candidate), ':', np.round(bleu, 4))
print('Average BLEU score =', np.round(np.mean(bleu_list), 4))

