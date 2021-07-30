# Seq2Seq 모델를 이용한 ChatBot : 채팅 모듈
#
# 관련 논문 : Kyunghyun Cho, et. al., 2014,
#            Learning Phrase Representations using RNN Encoder–Decoder 
#            for Statistical Machine Translation
#
# 저작자: 2021.05.26, 조성현 (blog.naver.com/chunjein)
# copyright: SNS 등에 공개할 때는 출처에 저작자를 명시해 주시기 바랍니다.
# ----------------------------------------------------------------------
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.layers import Embedding, TimeDistributed
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import sentencepiece as spm
import numpy as np
import pickle

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


# Chatting
# dummy : 최초 1회는 모델을 로드하는데 약간의 시간이 걸리므로 이것을 가리기 위함.
def chatting(n=100):
    for i in range(n):
        question = input('Q : ')
        
        if  question == 'quit':
            break
        
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
        print('A :', answer)

####### Chatting 시작 #######
print("\nSeq2Seq ChatBot (ver. 1.0)")
print("Chatting 모듈을 로드하고 있습니다 ...")

# 처음 1회는 시간이 걸리기 때문에 dummy question을 입력한다.
answer = genAnswer(np.zeros(MAX_LEN))
print("ChatBot이 준비 됐습니다.")

# 채팅을 시작한다.
chatting(100)