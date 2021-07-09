import pandas as pd
import re
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import LancasterStemmer
import pickle

nltk.download('punkt')
nltk.download('stopwords')

# Commented out IPython magic to ensure Python compatibility.
# 학습 데이터를 읽어온다.
# %cd '/content/drive/My Drive/Colab Notebooks'

df = pd.read_csv('labeledTrainData.tsv', header=0, sep='\t', quoting=3)
df['review'][0]
df['sentiment'][0]

# Pre-processing
# PorterStemmer()
stemmer = LancasterStemmer()
stopwords = nltk.corpus.stopwords.words('english')

processed_text = []
for review in df['review']:
    # 1. 영문자와 숫자만 사용한다. 그 이외의 문자는 공백 문자로 대체한다.
    review = review.replace('<br />', ' ')   # <br> --> space
    review = re.sub("[^a-zA-Z]", " ", review)    # 영문자만 사용

    tmp = []
    for word in nltk.word_tokenize(review):
        # 2. 길이가 2 이하인 단어와 stopword는 제거한다.
        if len(word) > 2 and word not in stopwords:
            # 3. Lemmatize
            tmp.append(stemmer.stem(word.lower()))
    processed_text.append(' '.join(tmp))

# 학습 데이터
X_train, X_test, y_train, y_test = train_test_split(processed_text, list(df['sentiment']), test_size = 0.3)


# 학습 데이터를 저장해 둔다.
with open('./popcorn.pkl', 'wb') as f:
    pickle.dump([X_train, y_train, X_test, y_test], f, pickle.DEFAULT_PROTOCOL) # pickle.DEFAULT_PROTOCOL

