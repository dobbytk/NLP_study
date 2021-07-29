# Commented out IPython magic to ensure Python compatibility.
# 작업 디렉토리를 변경한다.
# %cd '/content/drive/My Drive/Colab Notebooks'

# Seq2Seq ChatBot : 학습 데이터 모듈
# Google의 Sentencepiece를 이용해서 학습 데이터를 생성한다.
#
# 저작자: 2021.05.26, 조성현 (blog.naver.com/chunjein)
# copyright: SNS 등에 공개할 때는 출처에 저작자를 명시해 주시기 바랍니다.
# -------------------------------------------------------------------
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import sentencepiece as spm
import re
import pickle

# 데이터 파일을 읽어온다.
data_df = pd.read_csv('./ChatBotData.csv', header=0)
question, answer = list(data_df['Q']), list(data_df['A'])

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
dec_input: <BOS> 그래 오랜만이야 [2(BOS), 18, 24]
dec_output: 그래 오랜만이야 <EOS> [18, 24, 3(EOS)]
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

