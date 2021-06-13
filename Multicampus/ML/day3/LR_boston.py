import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Boston housing data set을 읽어온다.
boston = load_boston()
# MinMaxScaler 적용
scaleX = MinMaxScaler()
scaleY = MinMaxScaler()

feature_data = scaleX.fit_transform(boston['data'])
target_data = scaleY.fit_transform(boston['target'].reshape(-1, 1))

# Train 데이터 세트와 test 데이터 세트를 구성한다.
trainX, testX, trainY, testY = train_test_split(feature_data, target_data, test_size = 0.2)

model = LinearRegression()
model.fit(trainX, trainY)

price = model.predict(testX)

# 시험 데이터 전체의 오류를 R-square로 표시한다.
print('\n시험 데이터 전체 오류 (R2-score) = %.4f' % model.score(testX, testY))
print('첫 번째 testX의 추정 가격 = %.2f' % scaleY.inverse_transform(price[0].reshape(1, -1)))
print('첫 번째 testX의 실제 가격 = %.2f' % scaleY.inverse_transform(testY[0].reshape(1, -1)))