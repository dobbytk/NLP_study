from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.layers import Embedding, TimeDistributed
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sentencepiece as spm
import re
import pickle

# 데이터 파일을 읽어온다.
data_df = pd.read_csv('./ChatBotData.csv', header=0)
data_df.head()

# split train & test set 
train, test = train_test_split(data_df, test_size = 0.1)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
train.head()


question, answer = list(train['Q']), list(train['A'])

# 특수 문자를 제거한다.
FILTERS = "([~.,!?\"':;)(])"
question = [re.sub(FILTERS, "", s) for s in question]
answer = [re.sub(FILTERS, "", s) for s in answer]

question[0], answer[0]


# Sentencepice용 사전을 만들기 위해 question + answer를 저장해 둔다.
data_file = "./chatbot_data.txt"
with open(data_file, 'w', encoding='utf-8') as f:
    for sent in question + answer:
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

VOCAB_SIZE = 9000
model_prefix = "./chatbot_model"
params = templates.format(data_file, model_prefix, VOCAB_SIZE)

spm.SentencePieceTrainer.Train(params)
sp = spm.SentencePieceProcessor()
sp.Load(model_prefix + '.model')

with open(model_prefix + '.vocab', encoding='utf-8') as f:
    vocab = [doc.strip().split('\t') for doc in f]

word2idx = {k:v for v, [k, _] in enumerate(vocab)}
idx2word = {v:k for v, [k, _] in enumerate(vocab)}

# 학습 데이터를 생성한다. (인코더 입력용, 디코더 입력용, 디코더 출력용)
MAX_LEN = 15
enc_input = []
dec_input = []
dec_output = []
"""
enc_input: 안녕 오랜만이야 [12, 24]
dec_input: <BOS> 그래 오랜만이야 [2, 18, 24]
dec_output: 그래 오랜만이야 <EOS> [18, 24, 3]
"""

for Q, A in zip(question, answer):
    # Encoder 입력
    enc_i = sp.encode_as_ids(Q)
    enc_input.append(enc_i)

    # Decoder 입력, 출력
    dec_i = [sp.bos_id()]   # <BOS>에서 시작함
    dec_o = []
    for ans in sp.encode_as_ids(A):
        dec_i.append(ans)
        dec_o.append(ans)
    dec_o.append(sp.eos_id())   # Encoder 출력은 <EOS>로 끝남.        
    
    # dec_o는 <EOS>가 마지막에 들어있다. 나중에 pad_sequences()에서 <EOS>가
    # 잘려 나가지 않도록 MAX_LEN 위치에 <EOS>를 넣어준다.
    # ex) dec_output의 길이가 20인데 MAX_LEN이 15일 때, <EOS>가 짤리는 현상이 발생한다.
    # MAX_LEN 자리에 <EOS>를 넣어준다.
    if len(dec_o) > MAX_LEN:
        dec_o[MAX_LEN] = sp.eos_id() 
        
    dec_input.append(dec_i)
    dec_output.append(dec_o)

# 각 문장의 길이를 맞추고 남는 부분에 padding을 삽입한다.
enc_input = pad_sequences(enc_input, maxlen=MAX_LEN, value = sp.pad_id(), padding='post', truncating='post')
dec_input = pad_sequences(dec_input, maxlen=MAX_LEN, value = sp.pad_id(), padding='post', truncating='post')
dec_output = pad_sequences(dec_output, maxlen=MAX_LEN, value = sp.pad_id(), padding='post', truncating='post')

# 사전과 학습 데이터를 저장한다.
with open('./chatbot_voc.pkl', 'wb') as f:
    pickle.dump([word2idx, idx2word], f, pickle.HIGHEST_PROTOCOL)

with open('./chatbot_train.pkl', 'wb') as f:
    pickle.dump([enc_input, dec_input, dec_output], f, pickle.HIGHEST_PROTOCOL)

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.layers import Embedding, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pickle

# Sub-word 사전 읽어온다.
with open('./chatbot_voc.pkl', 'rb') as f:
    word2idx,  idx2word = pickle.load(f)

# 학습 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 읽어온다.
with open('./chatbot_train.pkl', 'rb') as f:
    trainXE, trainXD, trainYD = pickle.load(f)

VOCAB_SIZE = len(idx2word)
EMB_SIZE = 128
LSTM_HIDDEN = 128
MODEL_PATH = './chatbot_trained.h5'
LOAD_MODEL = False

# 워드 임베딩 레이어. Encoder와 decoder에서 공동으로 사용한다.
K.clear_session() # 가비지 컬렉터

wordEmbedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_SIZE)
# Encoder
# -------
# many-to-one으로 구성한다. 중간 출력은 필요 없고 decoder로 전달할 h와 c만
# 필요하다. h와 c를 얻기 위해 return_state = True를 설정한다.
encoderX = Input(batch_shape=(None, trainXE.shape[1]))
encEMB = wordEmbedding(encoderX)
encLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state = True)  
encLSTM2 = LSTM(LSTM_HIDDEN, return_state = True)
ey1, eh1, ec1 = encLSTM1(encEMB)    # LSTM 1층 
_, eh2, ec2 = encLSTM2(ey1)       # LSTM 2층

# Decoder
# -------
# many-to-many로 구성한다. target을 학습하기 위해서는 중간 출력이 필요하다.
# 그리고 초기 h와 c는 encoder에서 출력한 값을 사용한다 (initial_state)
# 최종 출력은 vocabulary의 인덱스인 one-hot 인코더이다.
decoderX = Input(batch_shape=(None, trainXD.shape[1]))
decEMB = wordEmbedding(decoderX)
decLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
decLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
dy1, _, _ = decLSTM1(decEMB, initial_state = [eh1, ec1]) # encoder에서 넘겨받은 eh1, ec1을 가중치 초기값으로 설정.
dy2, _, _ = decLSTM2(dy1, initial_state = [eh2, ec2]) # encoder에서 넘겨받은 eh2, ec2를 가중치 초기값으로 설정.
decOutput = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))
outputY = decOutput(dy2)

# Model
# -----
model = Model([encoderX, decoderX], outputY)
model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), 
              loss='sparse_categorical_crossentropy')

if LOAD_MODEL:
    model.load_weights(MODEL_PATH)

# 학습 (teacher forcing)
# ----------------------
hist = model.fit([trainXE, trainXD], trainYD, batch_size = 512, epochs=300, shuffle=True)

# 학습 결과를 저장한다
model.save_weights(MODEL_PATH)


# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Sub-word 사전 읽어온다.
with open('./chatbot_voc.pkl', 'rb') as f:
    word2idx,  idx2word = pickle.load(f)

VOCAB_SIZE = len(idx2word)
EMB_SIZE = 128
LSTM_HIDDEN = 128
MAX_LEN = 15            # 단어 시퀀스 길이
MODEL_PATH = './chatbot_trained.h5'

# 데이터 전처리 과정에서 생성한 SentencePiece model을 불러온다.
SPM_MODEL = "./chatbot_model.model"
sp = spm.SentencePieceProcessor()
sp.Load(SPM_MODEL)

# 워드 임베딩 레이어. Encoder와 decoder에서 공동으로 사용한다.
K.clear_session()
wordEmbedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_SIZE)

# Encoder
# -------
encoderX = Input(batch_shape=(None, MAX_LEN))
encEMB = wordEmbedding(encoderX)
encLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state = True)
encLSTM2 = LSTM(LSTM_HIDDEN, return_state = True)
ey1, eh1, ec1 = encLSTM1(encEMB)    # LSTM 1층 
_, eh2, ec2 = encLSTM2(ey1)         # LSTM 2층

# Decoder
# -------
# Decoder는 1개 단어씩을 입력으로 받는다.
decoderX = Input(batch_shape=(None, 1))
decEMB = wordEmbedding(decoderX)
decLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
decLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
dy1, _, _ = decLSTM1(decEMB, initial_state = [eh1, ec1])
dy2, _, _ = decLSTM2(dy1, initial_state = [eh2, ec2])
decOutput = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))
outputY = decOutput(dy2)

# Model
# -----
model = Model([encoderX, decoderX], outputY)
model.load_weights(MODEL_PATH) # 위 단계에서 학습한 모델의 가중치로 overwrite 시켜줌.

# Chatting용 model
model_enc = Model(encoderX, [eh1, ec1, eh2, ec2]) # [eh1, ec1, eh2, ec2] 는 decoder에 전달하기 위한 context vector

# 왜 여기서 Input으로 넣어줘야할까? eh1, ec1, eh2, ec2로 넣어줘도 되지 않을까?
# 뒤에 코드에서 ih1, ic1, ih2, ic2를 eh1, ec2, eh2, ec2로 overwrite 해주고 있음. overwrite해주기 위해 변수 선언을 먼저 해준것 같음. 
ih1 = Input(batch_shape = (None, LSTM_HIDDEN))
ic1 = Input(batch_shape = (None, LSTM_HIDDEN))
ih2 = Input(batch_shape = (None, LSTM_HIDDEN))
ic2 = Input(batch_shape = (None, LSTM_HIDDEN))

dec_output1, dh1, dc1 = decLSTM1(decEMB, initial_state = [ih1, ic1])
dec_output2, dh2, dc2 = decLSTM2(dec_output1, initial_state = [ih2, ic2])

dec_output = decOutput(dec_output2)
model_dec = Model([decoderX, ih1, ic1, ih2, ic2], [dec_output, dh1, dc1, dh2, dc2])

# Question을 입력받아 Answer를 생성한다.
def genAnswer(question):
    question = question[np.newaxis, :] # question = question.reshape(1, -1)와 동일 <- (1x15) 행렬
    init_h1, init_c1, init_h2, init_c2 = model_enc.predict(question)

    # 시작 단어는 <BOS>로 한다.
    word = np.array(sp.bos_id()).reshape(1, 1)
    

    answer = []
    for i in range(MAX_LEN): # dY - 답변할 문장의 첫 번째 subword - ex) '그래'
        dY, next_h1, next_c1, next_h2, next_c2 = model_dec.predict([word, init_h1, init_c1, init_h2, init_c2])
        
        # 디코더의 출력은 vocabulary에 대응되는 one-hot이다.
        # argmax로 해당 단어를 채택한다.
        # 무조건 argmax를 쓰는게 답인가? 융통성이 없는 챗봇이 된다. 유사한 확률을 가진 다른 답변은 못하게 된다. 주사위를 던지는 방식 사용.
        # 샘플링을 하게 되면 다른 대답을 한다. 융통성의 수준 조절 장치 추가
        # nextWord = np.random.multinomial(1, dY[0, 0])
        nextWord = np.argmax(dY[0, 0])
        # pdb.set_trace()
        # 예상 단어가 <EOS>이거나 <PAD>이면 더 이상 예상할 게 없다.
        if nextWord == sp.eos_id() or nextWord == sp.pad_id():
            break
        
        # 다음 예상 단어인 디코더의 출력을 answer에 추가한다.
        answer.append(idx2word[nextWord])
        
        # 디코더의 다음 recurrent를 위해 입력 데이터와 hidden 값을
        # 준비한다. 입력은 word이고, hidden은 h와 c이다.
        word = np.array(nextWord).reshape(1,1)
    
        init_h1 = next_h1
        init_c1 = next_c1
        init_h2 = next_h2
        init_c2 = next_c2
        
    return sp.decode_pieces(answer)

prediction = []
# Chatting
# dummy : 최초 1회는 모델을 로드하는데 약간의 시간이 걸리므로 이것을 가리기 위함.
def chatting(n=100):
    for i in range(len(test['Q'])):
        question = test['Q'][i]
        
        q_idx = []
        for x in sp.encode_as_pieces(question):
            if x in word2idx:
                q_idx.append(word2idx[x])
            else:
                q_idx.append(sp.unk_id())   # out-of-vocabulary (OOV)
        
        # <PAD>를 삽입한다.
        if len(q_idx) < MAX_LEN:
            q_idx.extend([sp.pad_id()] * (MAX_LEN - len(q_idx)))
        else:
            q_idx = q_idx[0:MAX_LEN]
        
        answer = genAnswer(np.array(q_idx))
        prediction.append(answer)

####### Chatting 시작 #######
print("\nSeq2Seq ChatBot (ver. 1.0)")
print("Chatting 모듈을 로드하고 있습니다 ...")

# 처음 1회는 시간이 걸리기 때문에 dummy question을 입력한다.
answer = genAnswer(np.zeros(MAX_LEN))
print("ChatBot이 준비 됐습니다.")

chatting(100)

test['predict'] = prediction

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

bleu_list = []


for candidate, reference in zip(test['predict'][:10], test['A'][:10]):
    # print(candidate)
    # print(reference)

    # BLEU를 측정한다.
    # 기계번역의 reference는 어느정도 객관성이 있지만, 챗봇의 reference는 매우 주관적이기 때문에,
    # test data로 측정한 챗봇의 BLEU는 매우 낮을 수밖에 없다.
    #
    # 1. 짧은 문장이 많기 때문에 단어가 아닌 subword 단위로 BLEU를 측정한다.
    # 2. (Papineni et al. 2002)은 micro-average를 사용했지만, 여기서는 단순 평균인 macro-average를 사용한다.
    # reference = sp.encode_as_pieces(reference)
    # candidate = sp.encode_as_pieces(candidate)

    bleu = sentence_bleu(candidate, [reference], weights=[1/2., 1/2.])
    # print(bleu)
    bleu_list.append(bleu)
    print(reference, '<-->', candidate, ':', np.round(bleu, 4))
print('Average BLEU score =', np.round(np.mean(bleu_list), 4))
