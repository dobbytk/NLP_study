
# GPT2의 TFGPT2LMHeadModel 모델을 이용한 언어 생성
# TFGPT2LMHeadModel : The GPT2 Model transformer with a language modeling head on top 
#                     (linear layer with weights tied to the input embeddings).
!pip install --upgrade mxnet>=1.6.0
!pip install gluonnlp
!pip install transformers
!pip install sentencepiece

import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer, SentencepieceDetokenizer
from transformers import TFGPT2LMHeadModel
import tensorflow as tf


MY_PATH = '/content/drive/MyDrive/머신러닝(멀티캠퍼스)'
MODEL_PATH = MY_PATH + '/gpt_ckpt'
TOKENIZER_PATH = MY_PATH + '/gpt_ckpt/gpt2_kor_tokenizer.spiece'

tokenizer = SentencepieceTokenizer(TOKENIZER_PATH, num_best=0, alpha=0)
detokenizer = SentencepieceDetokenizer(TOKENIZER_PATH)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(TOKENIZER_PATH,
                                               mask_token = None,
                                               sep_token = None,
                                               cls_token = None,
                                               unknown_token = '<unk>',
                                               padding_token = '<pad>',
                                               bos_token = '<s>',
                                               eos_token = '</s>')
# vocab --> Vocab(size=50000, unk="<unk>", reserved="['<pad>', '<s>', '</s>']")

# tokenizer 연습
# 참고 : https://nlp.gluon.ai/api/modules/data.html
toked = tokenizer('안녕 하세요')   # tokenizer가 잘못 만들어진 듯 ... 한글자씩 ?
print(toked)

toked_idx = vocab(toked)
print(toked_idx)

toked = vocab.to_tokens(toked_idx)
print(toked)

detoked = detokenizer(toked)
print(detoked)

''.join(toked).replace('▁', ' ')

model = TFGPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.summary()

# 모델의 seed 입력 문장 생성
tok = tokenizer('이때')   # tok = ['▁', '이', '때']
tok_idx = [vocab[vocab.bos_token]] + vocab[tok]     # tok_idx = [0, 47437, 47438, 47675]
input_ids = tf.convert_to_tensor(tok_idx)[None, :]  # 텐서로 변환

# 모델의 출력
output = model.generate(input_ids, max_length=50)

output

# 모델의 출력을 문자열로 변환
out_tok_idx = output.numpy().tolist()[0]   # output token 인덱스
out_tok = vocab.to_tokens(out_tok_idx)     # token 인덱스를 token 문자로 변환
out_text = detokenizer(out_tok)            # 출력 문자열로 decode
print(out_text)

# Beam search
output = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

out_tok_idx = output.numpy().tolist()[0]   # output token 인덱스
out_tok = vocab.to_tokens(out_tok_idx)     # token 인덱스를 token 문자로 변환
out_text = detokenizer(out_tok)            # 출력 문자열로 decode
print(out_text)

# 연속된 단어가 나오는 것을 방지함. no_repeat_ngram_size = 2
output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

out_tok_idx = output.numpy().tolist()[0]   # output token 인덱스
out_tok = vocab.to_tokens(out_tok_idx)     # token 인덱스를 token 문자로 변환
out_text = detokenizer(out_tok)            # 출력 문자열로 decode
print(out_text)

# top_k sampling
output = model.generate(input_ids, max_length=50, do_sample = True, top_k=100, temperature=0.8)

out_tok_idx = output.numpy().tolist()[0]   # output token 인덱스
out_tok = vocab.to_tokens(out_tok_idx)     # token 인덱스를 token 문자로 변환
out_text = detokenizer(out_tok)            # 출력 문자열로 decode
print(out_text)

# top_p sampling
output = model.generate(input_ids, max_length=50, do_sample = True, top_p=0.9, temperature=0.8)

out_tok_idx = output.numpy().tolist()[0]   # output token 인덱스
out_tok = vocab.to_tokens(out_tok_idx)     # token 인덱스를 token 문자로 변환
out_text = detokenizer(out_tok)            # 출력 문자열로 decode
print(out_text)

# top_k & top_p sampling
output = model.generate(input_ids, max_length=50, do_sample = True, top_k=100, top_p=0.9, temperature=0.8)

out_tok_idx = output.numpy().tolist()[0]   # output token 인덱스
out_tok = vocab.to_tokens(out_tok_idx)     # token 인덱스를 token 문자로 변환
out_text = detokenizer(out_tok)            # 출력 문자열로 decode
print(out_text)
