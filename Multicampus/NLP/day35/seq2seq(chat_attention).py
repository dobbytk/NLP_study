# Commented out IPython magic to ensure Python compatibility.
# 작업 디렉토리를 변경한다.
# %cd '/content/drive/My Drive/Colab Notebooks'

# Seq2Seq-Attention 모델를 이용한 ChatBot : 채팅 모듈
#
# 저작자: 2021.07.30, 조성현 (blog.naver.com/chunjein)
# copyright: SNS 등에 공개할 때는 출처에 저작자를 명시해 주시기 바랍니다.
# ----------------------------------------------------------------------
from tensorflow.keras.layers import Input, LSTM, Dense, Dot, Concatenate
from tensorflow.keras.layers import Embedding, TimeDistributed, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from nltk.translate.bleu_score import sentence_bleu
import sentencepiece as spm
import numpy as np
import pickle

# Sub-word 사전 읽어온다.
with open('data/chatbot_voc.pkl', 'rb') as f:
    word2idx,  idx2word = pickle.load(f)

# BLEU 평가를 위해 시험 데이터를 읽어온다.
with open('data/chatbot_train.pkl', 'rb') as f:
    _, _, _, que_test, ans_test = pickle.load(f)

VOCAB_SIZE = len(idx2word)
EMB_SIZE = 128
LSTM_HIDDEN = 128
MAX_LEN = 15            # 단어 시퀀스 길이
MODEL_PATH = 'data/chatbot_attention.h5'

# 데이터 전처리 과정에서 생성한 SentencePiece model을 불러온다.
SPM_MODEL = "data/chatbot_model.model"
sp = spm.SentencePieceProcessor()
sp.Load(SPM_MODEL)

# Encoder 출력과 decoder 출력으로 attention value를 생성하고,
# decoder 출력 + attention value (concatenate)를 리턴한다.
# x : encoder 출력, y : decoder 출력
# LSTM time step = 4, SMB_SIZE = 3 이라면 각 텐서의 dimension은
# 아래 주석과 같다.
def Attention(x, y):
    # step-1:
    # decoder의 매 시점마다 encoder의 전체 시점과 dot-product을 수행한다.
    score = Dot(axes=(2, 2))([y, x])                   # (1, 4, 4)
    
    # step-2:
    # dot-product 결과를 확률분포로 만든다 (softmax)
    # 이것이 attention score이다.
    dist = Activation('softmax')(score)                # (1, 4, 4)

    # step-3:
    # Attention value를 계산한다.
    attention = Dot(axes=(2, 1))([dist, x])

    # step-4:
    # decoder 출력과 attention을 concatenate 한다.
    return Concatenate()([y, attention])    # (1, 4, 6)

# 워드 임베딩 레이어. Encoder와 decoder에서 공동으로 사용한다.
K.clear_session()
wordEmbedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_SIZE)

# Encoder
# -------
encoderX = Input(batch_shape=(None, MAX_LEN))
encEMB = wordEmbedding(encoderX)
encLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state = True)
encLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state = True)
ey1, eh1, ec1 = encLSTM1(encEMB)      # LSTM 1층 
ey2, eh2, ec2 = encLSTM2(ey1)         # LSTM 2층

# Decoder
# -------
# Decoder는 1개 단어씩을 입력으로 받는다. 학습 때와 달리 문장 전체를 받아
# recurrent하는 것이 아니라, 단어 1개씩 입력 받아서 다음 예상 단어를 확인한다.
# chatting()에서 for 문으로 단어 별로 recurrent 시킨다.
# 따라서 batch_shape = (None, 1)이다. 즉, time_step = 1이다. 그래도 네트워크
# 파라메터는 동일하다.
decoderX = Input(batch_shape=(None, 1))
decEMB = wordEmbedding(decoderX)
decLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
decLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
dy1, _, _ = decLSTM1(decEMB, initial_state = [eh1, ec1])
dy2, _, _ = decLSTM2(dy1, initial_state = [eh2, ec2])
att_dy2 = Attention(ey2, dy2)
decOutput = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))
outputY = decOutput(att_dy2)

# Model
# -----
model = Model([encoderX, decoderX], outputY)
model.load_weights(MODEL_PATH)

# Chatting용 model
model_enc = Model(encoderX, [eh1, ec1, eh2, ec2, ey2])

ih1 = Input(batch_shape = (None, LSTM_HIDDEN))
ic1 = Input(batch_shape = (None, LSTM_HIDDEN))
ih2 = Input(batch_shape = (None, LSTM_HIDDEN))
ic2 = Input(batch_shape = (None, LSTM_HIDDEN))
ey = Input(batch_shape = (None, MAX_LEN, LSTM_HIDDEN))

dec_output1, dh1, dc1 = decLSTM1(decEMB, initial_state = [ih1, ic1])
dec_output2, dh2, dc2 = decLSTM2(dec_output1, initial_state = [ih2, ic2])
dec_attention = Attention(ey, dec_output2)
dec_output = decOutput(dec_attention)
model_dec = Model([decoderX, ih1, ic1, ih2, ic2, ey], 
                  [dec_output, dh1, dc1, dh2, dc2])

# Question을 입력받아 Answer를 생성한다.
def genAnswer(question):
    question = question[np.newaxis, :]
    init_h1, init_c1, init_h2, init_c2, enc_y = model_enc.predict(question)

    # 시작 단어는 <BOS>로 한다.
    word = np.array(sp.bos_id()).reshape(1, 1)

    answer = []
    for i in range(MAX_LEN):
        dY, next_h1, next_c1, next_h2, next_c2 = \
            model_dec.predict([word, init_h1, init_c1, init_h2, init_c2, enc_y])
        
        # 디코더의 출력은 vocabulary에 대응되는 one-hot이다.
        # argmax로 해당 단어를 채택한다.
        nextWord = np.argmax(dY[0, 0])

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

def make_question(que_string):
    q_idx = []
    for x in sp.encode_as_pieces(que_string):
        if x in word2idx:
            q_idx.append(word2idx[x])
        else:
            q_idx.append(sp.unk_id())   # out-of-vocabulary (OOV)
    
    # <PAD>를 삽입한다.
    if len(q_idx) < MAX_LEN:
        q_idx.extend([sp.pad_id()] * (MAX_LEN - len(q_idx)))
    else:
        q_idx = q_idx[0:MAX_LEN]
    return q_idx

# BLEU 평가
bleu_list = []
for que_str, reference in zip(que_test[:10], ans_test[:10]):
    q_idx = make_question(que_str)
    candidate = genAnswer(np.array(q_idx))

    # BLEU를 측정한다.
    # 기계번역의 reference는 어느정도 객관성이 있지만, 일상 대화용 챗봇의 reference는 매우 주관적이기 때문에,
    # test data로 측정한 챗봇의 BLEU는 매우 낮을 수밖에 없다. 특정 업무를 위한 챗봇의 reference는 어느정도
    # 객관성이 있을 수 있다.
    #
    # 1. 짧은 문장이 많기 때문에 단어가 아닌 subword 단위로 BLEU를 측정한다.
    # 2. (Papineni et al. 2002)은 micro-average를 사용했지만, 여기서는 단순 평균인 macro-average를 사용한다.
    reference = sp.encode_as_pieces(reference)
    candidate = sp.encode_as_pieces(candidate)

    bleu = sentence_bleu(reference, candidate, weights=[1/2., 1/2.])
    bleu_list.append(bleu)
    print(que_str, '-->', sp.decode_pieces(candidate), ':', np.round(bleu, 4))
print('Average BLEU score =', np.round(np.mean(bleu_list), 4))

# Chatting
# dummy : 최초 1회는 모델을 로드하는데 약간의 시간이 걸리므로 이것을 가리기 위함.
def chatting(n=100):
    for i in range(n):
        question = input('Q : ')
        
        if  question == 'quit':
            break
        
        q_idx = make_question(question)
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

# train data:
# [('이별 후 1년 그리고 선물', '이별하신게 맞나요'),
#  ('허기져', '챙겨 드세요'),
#  ('맥주 소주 어떤거 마실까', '소맥이요'),
#  ('교양 수업 재밌어', '저도 듣고 싶어요'),
#  ('권태기 이별', '극복하거나 이별하거나 둘 중 하나죠'),
#  ('읽씹은 아프네', '상대방에 대한 예의가 없네요'),
#  ('신혼여행 어디로 갈까', '못 가본 곳으로 가보세요'),
#  ('반 배정 잘 될까', '잘 되길 바랍니다'),
#  ('친구가 다 떠나서 내가 못났나 싶어', '지난 인연에 연연해하지 마세요'),
#  ('뒤돌아 보지 말고 나가야 하는데', '살짝 뒤돌아봐도 괜찮아요')]

# test data:
# [('소오름 쫙', '좋은 일이길 바랍니다'),
#  ('고백은 어떻게 하는거야', '솔직한 마음으로 다가가는 거죠'),
#  ('참 잘낫네', '진정하셔요'),
#  ('늘 빡빡하게 살기 힘드네', '여유가 생기길 바랍니다'),
#  ('집까지 데려다줬는데 호감 그냥 매너', '호감이 있을 수도 있어요 그렇지만 조금 더 상황을 지켜보세요'),
#  ('짝녀가 연락 안 되고 있는데 자나', '자고 있을지도 모르겠어요'),
#  ('마음도 춥고 날씨도 춥고', '마음 감기 조심하세요'),
#  ('죽었던 연애세포가 살아나는 것 같아', '좋은 소식이네요'),
#  ('겨울에는 온천이지', '몸은 뜨겁고 머리는 차갑게'),
#  ('소개팅 하고싶다', '친구한테 부탁해보세요')]