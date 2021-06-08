# MinMaxScaler을 이용한 normalization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
scaler = MinMaxScaler()
z = scaler.fit_transform(cancer['data'])

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = train_test_split(z, cancer['target'], test_size = 0.2)
# train_test_split은 랜덤하게 data를 섞은 후에 split한다. 

# DT로 Train 데이터 세트를 학습한다.
dt = DecisionTreeClassifier(criterion='gini',max_depth=15)
dt.fit(trainX, trainY)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
# accuracy = knn.score(testX, testY)와 동일함.
"""
acc = dt.score(testX, testY)
print('정확도=', np.round(acc, 4))
"""
predY = dt.predict(testX) # 테스트셋을 이용한 예측값 
accuracy = (testY == predY).mean() # 정확도
print()
print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy)

# Train 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
predY = dt.predict(trainX) # 학습한 데이터로 평가 (x), 일반적으로 사용 x
accuracy = (trainY == predY).mean()
print("* 학습용 데이터로 측정한 정확도 = %.2f" % accuracy)

import numpy as np
# feature별 중요도를 파악한다
feat_impo = dt.feature_importances_
feat_name = cancer['feature_names']

# 중요도가 높은 feature 5개를 확인한다
idx = np.argsort(feat_impo)[::-1][:5]
name = cancer['feature_names'][idx]
value = feat_impo[idx]

feat_name = list(cancer['feature_names'])

plt.figure(figsize=(12, 8))
x_idx = np.arange(len(name))
plt.barh(x_idx, value, align = 'center')
plt.yticks(x_idx, name)
plt.xlabel('feature importance')
plt.ylabel('feature')
plt.show()