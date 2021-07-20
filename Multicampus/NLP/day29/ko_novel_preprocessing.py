# 한글 코퍼스를 전처리하고 결과를 저장한다.
# ---------------------------------------
import re
from nltk.tokenize import sent_tokenize
from konlpy.tag import Okt
import pickle
import nltk
nltk.download('punkt')

stop_words = ['것', '그', '때', '이', '나', '수', '내', '또', '듯', '알', '두', '더', '번', '온',
              '게', '몇', '저', '채', '체', '쪽', '데', '걸', '뿐', '거' '나', '너', '좀', '리',
              '준', '만', '바', '누', '의', '네', '도', '기', '난', '넌', '늘', '건', '를', '테',
              '사', '고', '임', '진', '어', '뭐', '끼', '둥', '젠', '인', '영', '조', '겸', '로',
              '잇섯', '애가', '등']
okt = Okt()

# 전처리
def preprocessing(sentence, okt, stop_words):
    # 한글만 추출한다. 영어, 한문, 숫자를 제거한다.
    text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\s]", "", sentence)

    # text에서 어간을 추출하고, 명사, 형용사, 동사만 추출한다. 그리고 불용어를 제거한다.
    tokens = []
    for word in okt.pos(text, stem=True):
        if word[1] in ['Noun', 'Adjective', 'Verb']:
            if word[0] not in stop_words:
                tokens.append(word[0])
    return tokens

# 코퍼스를 읽어와서 전처리를 수행한다.
sentence_list = []
in_f = open("data/ko_novel.txt", "r", encoding="utf-8")
for i, line in enumerate(in_f):
    if len(line) < 20:
        continue
    
    # line은 '\n'까지 읽기 때문에 한 문단 (paragraph)이다. 여러 문장이 있을 수 있다.
    # 이를 한 문장 (sentence)으로 분리해서 문장 단위로 처리한다.
    paragraph = sent_tokenize(line)
    for p in paragraph:
        sentence = preprocessing(p, okt, stop_words)
        
        if len(sentence) > 5:
            sentence_list.append(sentence)
    
    if (i+1) % 100 == 0:
        print('Processed :', i+1)
        
in_f.close()

# 전처리가 완료된 한글 코퍼스를 저장한다.
with open('data/konovel_preprocessed.pkl', 'wb') as f:
    pickle.dump(sentence_list, f, pickle.DEFAULT_PROTOCOL)