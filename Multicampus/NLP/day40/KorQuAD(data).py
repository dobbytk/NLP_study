import json
import numpy as np
from transformers import BertTokenizer
from tensorflow.keras.utils import get_file
import matplotlib.pyplot as plt
import pickle

# 학습용 데이터를 읽어온다.
train_data_url = "https://korquad.github.io/dataset/KorQuAD_v1.0_train.json"
train_path = get_file("train.json", train_data_url)

# 평가용 데이터를 읽어온다.
eval_data_url = "https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json"
eval_path = get_file("eval.json", eval_data_url)

train_data = json.load(open(train_path))
dev_data = json.load(open(eval_path))

print(train_path)
print(eval_path)

dev_data['data'][0]

MAX_SEQ_LEN = 128
MAX_TRAIN_LEN = 10000  # 시간이 오래 걸려서 데이터 개수를 제한한다.
MAX_TEST_LEN = 1000
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir='bert_ckpt')

# json parsing & tokenizing * stratify
def parsing(p_data, max_len = 10000):
    context = []
    question = []
    start_idx = []
    end_idx = []
    for item in p_data["data"]:
        for para in item["paragraphs"]:
            for qa in para["qas"]:
                i_start = qa["answers"][0]["answer_start"]
                s_answer = qa["answers"][0]["text"]
                i_end = i_start + len(s_answer)
                quest = qa["question"]
                
                if i_end < MAX_SEQ_LEN - len(quest):
                    context.append(para["context"])
                    question.append(quest)
                    start_idx.append(i_start)
                    end_idx.append(i_end)
    
    # question과 paragraph으로 BERT의 입력 데이터를 생성한다.
    qa_pairs = list(zip(question, context))
    qa_enc = tokenizer.batch_encode_plus(
                qa_pairs,
                add_special_tokens = True,
                padding = True,
                truncation = True, 
                max_length = MAX_SEQ_LEN,
                return_attention_mask = True,
                return_token_type_ids=True,
                return_tensors = 'tf')        

    x_ids = qa_enc['input_ids'].numpy()
    x_msk = qa_enc['attention_mask'].numpy()
    x_typ = qa_enc['token_type_ids'].numpy()
    
    # KorQuAD 모델의 최종 출력 target
    y_start = np.array(start_idx)
    y_end = np.array(end_idx)
        
    return x_ids, x_msk, x_typ, y_start, y_end

# 학습/시험 데이터를 생성한다. 학습 데이터는 stratify를 수행한다.
x_train_ids, x_train_msk, x_train_typ, y_train_start, y_train_end = parsing(train_data, max_len = MAX_TRAIN_LEN)
x_test_ids, x_test_msk, x_test_typ, y_test_start, y_test_end = parsing(dev_data, max_len = MAX_TEST_LEN)

# 데이터 확인용 임시 함수
def dd(n):
    context = tokenizer.decode(x_train_ids[n])
    print(context)
    print(x_train_msk[n], '\n')
    print(x_train_typ[n], '\n')
    print(y_train_start[n], y_train_end[n])

    context = tokenizer.decode(x_train_ids[n]).split('[SEP] ')
    print(context[1][y_train_start[n]:y_train_end[n]])

# vocabulary를 저장한다.
with open('data/vocabulary.pickle', 'wb') as f:
    pickle.dump(tokenizer.get_vocab(), f, pickle.DEFAULT_PROTOCOL)

# 학습 데이터를 저장한다.
with open('data/train_encoded.pickle', 'wb') as f:
    pickle.dump([x_train_ids, x_train_msk, x_train_typ, y_train_start, y_train_end], f, pickle.DEFAULT_PROTOCOL)

# 시험 데이터를 저장한다.
with open('data/test_encoded.pickle', 'wb') as f:
    pickle.dump([x_test_ids, x_test_msk, x_test_typ, y_test_start, y_test_end], f, pickle.DEFAULT_PROTOCOL)
