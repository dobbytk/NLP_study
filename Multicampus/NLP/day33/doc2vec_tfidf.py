import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import xgboost as xgb
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. x_news를 이용해서 tfidf 변환 후 모델 학습
with open('./pv_dm_D.pkl', 'rb') as f:
    x_news, D_array, y_data = pickle.load(f)

# 추가적으로 데이터 전처리 
x_news = [re.sub('[a-zA-Z0-9_-]+@[a-z]+.[a-z]+', '', x) for x in x_news]
x_news = [re.sub("[^a-zA-Z']", ' ', x) for x in x_news]

tfidf = TfidfVectorizer(max_features=20000)
tfidf_matrix = tfidf.fit_transform(x_news)
# print(tfidf_matrix.shape)

n_topic = len(set(y_data[:, 0]))
x_train, x_test, y_train, y_test = train_test_split(tfidf_matrix, y_data, test_size=0.2)

# SVM(linear)를 이용하여 모델 학습
model = SVC(kernel='linear', gamma=0.1, C=0.5)
model.fit(x_train, y_train)
print('정확도= ',np.round(model.score(x_test, y_test), 3))

# xgboost를 이용하여 모델 학습
trainD = xgb.DMatrix(x_train, label=y_train)
testD = xgb.DMatrix(x_test, label=y_test)

param = {
    'eta' : 0.3,
    'max_depth' : 3,
    'objective' : 'multi:softprob',
    'num_class' : 20
}

model = xgb.train(params = param, dtrain = trainD, num_boost_round = 20)

y_pred = model.predict(testD)
y_pred = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_pred, y_test)
print("정확도=",np.round(accuracy, 3))

# 2. 차원 축소하여 D행렬과 combine하여 학습

from sklearn.decomposition import TruncatedSVD

tf_vector = TfidfVectorizer(max_features=20000)
tfidf_vector = tf_vector.fit_transform(x_news)
tfidf_vector.shape

vocab = tf_vector.get_feature_names()
print(vocab[:100])

svd = TruncatedSVD(n_components = 400, n_iter = 100)
svd.fit(tfidf_vector)

U = svd.fit_transform(tfidf_vector) / svd.singular_values_
VT = svd.components_
S = np.diag(svd.singular_values_)
U.shape, S.shape, VT.shape

# 차원 축소한 행렬과 Doc 행렬을 평균
combine = (U + D_array) / 2

x_train, x_test, y_train, y_test = train_test_split(combine, y_data, test_size=0.2)

# SVM(linear) 이용해서 학습하기
model = SVC(kernel='linear', gamma=0.1, C=0.5)
model.fit(x_train, y_train)
print('정확도= ',np.round(model.score(x_test, y_test), 3))

# xgboost 이용해서 학습하기
trainD = xgb.DMatrix(x_train, label=y_train)
testD = xgb.DMatrix(x_test, label=y_test)

param = {
    'eta' : 0.3,
    'max_depth' : 3,
    'objective' : 'multi:softprob',
    'num_class' : 20
}

model = xgb.train(params = param, dtrain = trainD, num_boost_round = 20)
y_pred = model.predict(testD)
y_pred = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_pred, y_test)
print("정확도=",np.round(accuracy, 3))
