# Commented out IPython magic to ensure Python compatibility.
# Transformer ChatBot : 채팅 모듈
# 참고 : https://github.com/suyash/transformer
#
# 2020.06.07 : 조성현 (blog.naver.com/chunjein)
# ---------------------------------------------
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K
import sentencepiece as spm
import pickle
import numpy as np

# %cd '/content/drive/MyDrive/Colab Notebooks/삼성멀캠/자연어처리/7.챗봇_번역'
from transformer import Transformer

# Commented out IPython magic to ensure Python compatibility.
# %cd '/content/drive/My Drive/Colab Notebooks'

MAX_LEN = 15
MODEL_PATH = 'data/transformer_model.h5'
SPM_MODEL = "data/chatbot_model.model"

sp = spm.SentencePieceProcessor()
sp.Load(SPM_MODEL)

# Sub-word 사전 읽어온다.
with open('data/chatbot_voc.pkl', 'rb') as f:
    word2idx,  idx2word = pickle.load(f)

K.clear_session()
src = Input((None, ), dtype="int32", name="src")
tar = Input((None, ), dtype="int32", name="tar")

logits, enc_enc_attention, dec_dec_attention, enc_dec_attention = Transformer(
    num_layers=4,
    d_model=128,
    num_heads=8,
    d_ff=512,
    input_vocab_size=len(word2idx) + 2,
    target_vocab_size=len(word2idx) + 2,
    dropout_rate=0.1)(src, tar)

model = Model(inputs=[src, tar], outputs=logits)
model.load_weights(MODEL_PATH)
model.summary()

# Question을 입력받아 Answer를 생성한다.
def genAnswer(question):
    question = question[np.newaxis, :]
    target = np.array(sp.bos_id()).reshape(1, 1)

    answer = []
    for i in range(MAX_LEN):
        preds = model.predict_on_batch([question, target])
        
        # 디코더의 출력은 vocabulary에 대응되는 one-hot이다.
        # argmax로 해당 단어를 채택한다.
        nextWord = np.argmax(preds[:, -1:, :], axis=-1)
        
        # 예상 단어가 <EOS>이거나 <PAD>이면 더 이상 예상할 게 없다.
        if nextWord == sp.eos_id() or nextWord == sp.pad_id():
            break

        # 다음 예상 단어인 디코더의 출력을 answer에 추가한다.
        answer.append(idx2word[nextWord[0][0]])
        
        # 다음 target을 준비한다.
        target = np.concatenate([target, nextWord], axis = -1)
    
    return sp.decode_pieces(answer)

# Chatting
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
print("\nTransformer ChatBot (ver. 1.0)")
print("Chatting 모듈을 로드하고 있습니다 ...")

# 처음 1회는 시간이 걸리기 때문에 dummy question을 입력한다.
answer = genAnswer(np.zeros(MAX_LEN))
print("ChatBot이 준비 됐습니다.\n")

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