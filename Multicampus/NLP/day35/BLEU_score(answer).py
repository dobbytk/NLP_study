# Commented out IPython magic to ensure Python compatibility.
# 작업 디렉토리를 변경한다.
# %cd '/content/drive/My Drive/Colab Notebooks'

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
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import sentencepiece as spm
import numpy as np
import pickle

# Sub-word 사전 읽어온다.
with open('./chatbot_voc.pkl', 'rb') as f:
    word2idx,  idx2word = pickle.load(f)

# BLEU 평가를 위해 시험 데이터를 읽어온다.
with open('./chatbot_train.pkl', 'rb') as f:
    _, _, _, que_test, ans_test = pickle.load(f)

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
model.load_weights(MODEL_PATH)

# Chatting용 model
model_enc = Model(encoderX, [eh1, ec1, eh2, ec2])

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
    question = question[np.newaxis, :]
    init_h1, init_c1, init_h2, init_c2 = model_enc.predict(question)

    # 시작 단어는 <BOS>로 한다.
    word = np.array(sp.bos_id()).reshape(1, 1)

    answer = []
    for i in range(MAX_LEN):
        dY, next_h1, next_c1, next_h2, next_c2 = model_dec.predict([word, init_h1, init_c1, init_h2, init_c2])
        
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
    candidate = genAnswer(np.array(q_idx)) # 질문에 대한 답

    # BLEU를 측정한다.
    # 기계번역의 reference는 어느정도 객관성이 있지만, 챗봇의 reference는 매우 주관적이기 때문에,
    # test data로 측정한 챗봇의 BLEU는 매우 낮을 수밖에 없다.
    #
    # 1. 짧은 문장이 많기 때문에 단어가 아닌 subword 단위로 BLEU를 측정한다.
    # 2. (Papineni et al. 2002)은 micro-average를 사용했지만, 여기서는 단순 평균인 macro-average를 사용한다.
    # reference = sp.encode_as_pieces(reference) # test set의 답
    # candidate = sp.encode_as_pieces(candidate)

    bleu = sentence_bleu(reference, candidate, weights=[1/2., 1/2.])
    bleu_list.append(bleu)
    print(que_str, '-->', candidate, ':', np.round(bleu, 4))
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
# [('환승할까', '환승은 30분 안에'),
#  ('슬픔활용법', '잠깐 고독한 것도 분위기 있을지도 몰라요'),
#  ('결혼 했어', '좋겠어요'),
#  ('주체가안된다', '그럴 땐 생각을 덜어봐요'),
#  ('답정녀 퇴치법', '못 이기니 피할 수 있으면 피하세요'),
#  ('짝남한테 내일 영화보자고 말해도 될까', '안되더라도 말해보세요'),
#  ('이별통보를 받았습니다', '마음의 준비가 필요했을텐데 안타까워요'),
#  ('코딩 좀 배울까', '많이 알면 도움이 되겠죠'),
#  ('친구들한테 인기 얻으려면', '성격이 좋으면 인기가 있을 거예요'),
#  ('모쏠인 사람 만나면 답답할까', '서툴러도 괜찮아요')]

# test data:
# [('다들 어떠셨어 결국 이렇게 되는건가', '당신만 아픈 것도 당신만 겪은 것도 아니에요'),
#  ('버스 멀미나', '핸드폰 만지지 마세요'),
#  ('정말 너무 힘들어', '많이 힘들었죠'),
#  ('마음 잡고 고고씽', '다 잘 될 거예요'),
#  ('7년사귄 남자친구에게 배신당했어', '헤어지세요'),
#  ('여자친구가 베지테리언이야', '함께 건강하게 먹어봐요'),
#  ('이별하려다 붙잡고 다시 사귀는데', '잘 지내길 바랍니다'),
#  ('카페에서 같이 알바하는 사람이 좋아졌어요', '자연스럽게 말을 걸어보세요'),
#  ('뭐하냐', '일해요'),
#  ('사랑하는 사람이 행복했으면 좋겠다', '당신이 그 사람 행복하게 해주세요')]
