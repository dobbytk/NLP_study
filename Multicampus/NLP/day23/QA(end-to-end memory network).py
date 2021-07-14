# Question & Answering problem : 관련 논문의 간단 & 변형 버전.
#
# 관련 논문 : [1] Sainbayar Sukhbaatar, et. al, 2015, End-To-End Memory Networks
# 코드 출처 : Krishna Bhavsar et. al., 2017, 
#            "Natural Language Processing with Python Cookbook"
#
# 코드 수정 : blog.naver.com/chunjein, 2021.04.12
#
# get_data() 함수에서 bAbi 데이터의 story 부분을 parsing할 때 일련의 episode를
# 보존하지 못한 부분을 수정했음. ID가 1로 reset되기 전까지는 하나의 episode임.
#
# 참고 사항 : 간단 버전이므로 bAbi의 'qa1_sigle_fact'는 어느 정도 분석하지만,
#            'qa2_two_facts'는 거의 분석하지 못한다. (정확도 = 30% 정도)
#             논문 [1]의 내용을 제대로 적용하면 정확도 = 90% 이상임.
# -------------------------------------------------------------------------------
import collections
import nltk
import numpy as np
from copy import deepcopy
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM, Permute
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Add, Concatenate, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import random

nltk.download('punkt')

train_file = "qa1_single-supporting-fact_train.txt"
test_file = "qa1_single-supporting-fact_test.txt"
# train_file = "data/qa2_two-supporting-facts_train.txt"
# test_file = "data/qa2_two-supporting-facts_test.txt"


# bAbi 데이터 parsing. 데이터 구조는 https://research.fb.com/downloads/babi/ 참조
def get_data(infile, max_stories=30):
    stories = []
    questions = []
    answers = []
    story_text = []
    fin = open(infile, "r")
    old_ID = 0
    for line in fin:
        # ex) IDs: 15
        # ex) text: IDs 뒤 나머지
        IDs, text = line.split(" ", 1)
        IDs = int(IDs)
        
        if IDs < old_ID:     # ID reset
            story_text = []

        if "\t" in text:
            # question 문장에만 \t가 있음. 
            s = deepcopy(story_text)

            # story sequence가 긴 경우는 88개까지 있다. 너무 길어서
            # max_stories 개수만큼으로 제한한다.
            if len(s) <= max_stories:
                question, answer, _ = text.split("\t")
                stories.append(s)
                questions.append(question)
                answers.append(answer)
        else:
            # question이 있는 line이 아니라면 story_text 리스트에 넣어라
            story_text.append(text.strip())
        old_ID = IDs
    fin.close()
    return stories, questions, answers

# bAbi 데이터를 parsing 한다.
data_train = get_data(train_file, max_stories = 20)
data_test = get_data(test_file, max_stories = 20)

data_train[1][7] # tuple - (stories, questions, answers)

# 어휘 사전을 생성한다.
vocab = collections.Counter()
for stories, questions, answers in [data_train, data_test]:
    # story 문장에 사용된 단어의 빈도를 파악한다.
    for story in stories:
        for sent in story:
            for word in nltk.word_tokenize(sent):
                vocab[word.lower()] += 1
    
    # question 문장에 사용된 단어의 빈도를 파악한다.
    for question in questions:
        for word in nltk.word_tokenize(question):
            vocab[word.lower()] += 1
    
    # answer에 사용된 단어의 빈도를 파악한다.
    for answer in answers:
        for word in nltk.word_tokenize(answer):
            vocab[word.lower()] += 1

# 빈도가 높은 것부터 어휘 사전에 등록한다. (등록 순서가 중요한 것은 아님)
# (i+1)을 지정해준 것은 <PAD> 토큰을 index 0에 부여하기 위함.
word2idx = {w:(i+1) for i,(w,_) in enumerate(vocab.most_common())} 
word2idx["<PAD>"] = 0   
vocab_size = len(word2idx)
word2idx

# 데이터셋 구성
# story, question 문장과 answer 단어를 수치 벡터로 변환한다.
def vectorize(data):
    x_story = []
    x_query = []
    y_answer = []
    stories, questions, answers = data # unpacking - (story, question, answer)
    for story, question, answer in zip(stories, questions, answers):
        # print(story, question, answer)
        # -> ['Mary went to the bedroom.', 'John journeyed to the bathroom.'] Where is John?  bathroom
        # 각 단어 토큰을 .lower() 메서드 적용해서 idx 값으로 변환       
        xs = [word2idx[w.lower()] for s in story for w in nltk.word_tokenize(s)]
        xq = [word2idx[w.lower()] for w in nltk.word_tokenize(question)]
        
        x_story.append(xs)
        x_query.append(xq)
        y_answer.append(word2idx[answer.lower()])
    
    return x_story, x_query, y_answer

xs_train_tok, xq_train_tok, y_train_tok = vectorize(data_train)
xs_test_tok, xq_test_tok, y_test_tok = vectorize(data_test)

# padding을 위해 문장의 최대 길이를 파악한다.
story_max = 0
query_max = 0
for tokens in [xs_train_tok, xs_test_tok]:
    for tok in tokens:
        if len(tok) > story_max:
            story_max = len(tok)
            
for tokens in [xq_train_tok, xq_test_tok]:
    for tok in tokens:
        if len(tok) > query_max:
            query_max = len(tok)


# padding을 적용해서 학습 데이터를 생성한다.
xs_train = pad_sequences(xs_train_tok, maxlen=story_max, padding='post') # story_max == 64
xq_train = pad_sequences(xq_train_tok, maxlen=query_max, padding='post') # query_max == 4
xs_test = pad_sequences(xs_test_tok, maxlen=story_max, padding='post')
xq_test = pad_sequences(xq_test_tok, maxlen=query_max, padding='post')
y_train = np.array(y_train_tok)
y_test = np.array(y_test_tok)

print(xq_train[4])
print(xq_train.shape)
print(story_max)
print(query_max)

# Model을 생성한다.
EMB_SIZE = 128
LATENT_SIZE = 64
BATCH_SIZE = 64
NUM_EPOCHS = 40

# Inputs
# shape: shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors
story_input = Input(shape=(story_max,)) 
question_input = Input(shape=(query_max,))
# print(story_input.shape)
# Story encoder embedding
story_encoder = Embedding(vocab_size, EMB_SIZE)(story_input)
story_encoder = Dropout(0.2)(story_encoder)
print(story_encoder.shape)
# Question encoder embedding
question_encoder = Embedding(vocab_size, EMB_SIZE)(question_input)
question_encoder = Dropout(0.3)(question_encoder)
print(question_encoder.shape)

# Match between story and question
match = Dot(axes=[2, 2])([story_encoder, question_encoder])
print(match.shape)
# Encode story into vector space of question
story_encoder_c = Embedding(vocab_size, query_max)(story_input)
story_encoder_c = Dropout(0.3)(story_encoder_c)

# Combine match and story vectors
response = Add()([match, story_encoder_c])
response = Permute((2, 1))(response)

# Combine response and question vectors to answers space
answer = Concatenate()([response, question_encoder])
question_encoder.shape
answer = LSTM(LATENT_SIZE , dropout=0.2)(answer) # time_step = 4, LATENT_SIZE(output_dim) = 64
output = Dense(vocab_size, activation='softmax')(answer)

model = Model(inputs=[story_input, question_input], outputs=output)
model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
model.summary()

# Model Training
history = model.fit([xs_train, xq_train], [y_train],
                    batch_size = BATCH_SIZE, 
                    epochs = NUM_EPOCHS,
                    validation_data=([xs_test, xq_test], [y_test]))

# loss plot
plt.title("Episodic Memory Q & A Loss")
plt.plot(history.history["loss"], color="g", label="train")
plt.plot(history.history["val_loss"], color="r", label="validation")
plt.legend(loc="best")
plt.show()

# 정확도 평가
y_pred_p = model.predict([xs_test, xq_test])
y_pred = np.argmax(y_pred_p, axis=1)
accuracy = (y_test == y_pred).mean()

# test data와 prediction 결과를 확인한다. 샘플 10개만.
idx2word = {v:k for k, v in word2idx.items()}
for i in random.sample(range(xs_test.shape[0]), 10):
    story = data_test[0][i]
    question = data_test[1][i]
    answer = data_test[2][i]
    pred_ans = idx2word[y_pred[i]]
    
    print('\n i: {}'.format(i))
    
    for k in range(len(story)):
        print('S{}: {}'.format(k+1, story[k].strip()))
        
    print(' Q: {}'.format(question))
    print(' A: {}'.format(answer))    
    print(' P: {}, prob={:.3f}'.format(pred_ans, np.max(y_pred_p[i])))

print('정확도 = {:.4f} %'.format(100 * accuracy))

