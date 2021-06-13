from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np

iris = load_iris()

trainX, testX, trainY, testY = train_test_split(iris['data'], iris['target'], test_size=0.2)

# XGBoost (classifier)로 Train 데이터를 학습한다.
# 학습데이터와 시험데이터를 xgb의 데이터 형태로 변환한다.
trainD = xgb.DMatrix(trainX, label=trainY)
testD = xgb.DMatrix(testX, label=testY)

param = {
    'eta' : 0.3,
    'max_depth' : 3,
    'objective' : 'multi:softprob',
    'num_class' : 3
}

model = xgb.train(params = param, dtrain = trainD, num_boost_round = 20)

# 테스트셋의 Feature에 대한 class를 추정하고, 정확도를 계산한다.
predY = model.predict(testD)
predY = np.argmax(predY, axis=1)
accuracy = accuracy_score(testY, predY)
print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy)

# 훈련 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다.
predY = model.predict(trainD)
predY = np.argmax(predY, axis=1)
accuracy = accuracy_score(trainY, predY)
print("* 학습용 데이터로 측정한 정확도 = %.2f" % accuracy)

"""
* 시험용 데이터로 측정한 정확도 = 1.00
* 학습용 데이터로 측정한 정확도 = 1.00
"""