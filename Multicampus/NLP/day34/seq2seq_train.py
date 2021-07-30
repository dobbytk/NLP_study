# Seq2Seq 모델를 이용한 ChatBot : 학습 모듈 (Teacher forcing)
#
# 관련 논문 : Kyunghyun Cho, et. al., 2014,
#             Learning Phrase Representations using RNN Encoder–Decoder 
#             for Statistical Machine Translation
#
# 저작자: 2021.05.26, 조성현 (blog.naver.com/chunjein)
# copyright: SNS 등에 공개할 때는 출처에 저작자를 명시해 주시기 바랍니다.
# ----------------------------------------------------------------------
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

