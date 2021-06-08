"""
데이터 하나를 예시로 찍어보면, 데이터의 분포가 불균형하다는 것을 알 수 있다.
kNN은 거리를 계산하는 머신러닝 알고리즘이기에, 길쭉한 데이터의 분포 형태는 좋지 않다.
따라서 데이터 표준화가 필요하다.
표준화 방법에는 두 가지가 있다.
1. StandardScaler (z-score normalization)
2. MinMaxScaler
적용 순서는 다음과 같다.
	데이터 읽음 -> 표준화 -> train&test set split -> 학습 -> 평가
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()

# scaler = MinMaxScaler()
scaler = StandardScaler()
z = scaler.fit_transform(cancer['data'])

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = train_test_split(z, cancer['target'], test_size = 0.2)
# train_test_split은 랜덤하게 data를 섞은 후에 split한다. 

# KNN 으로 Train 데이터 세트를 학습한다.
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski') # n_neighbors = k의 개수, p는 minkowski의 m을 의미
knn.fit(trainX, trainY)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
# accuracy = knn.score(testX, testY)와 동일함.
predY = knn.predict(testX) # 테스트셋을 이용한 예측값 
accuracy = (testY == predY).mean() # 정확도

print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy)

# Train 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
predY = knn.predict(trainX) # 학습한 데이터로 평가 (x), 일반적으로 사용 x
accuracy = (trainY == predY).mean()
print("* 학습용 데이터로 측정한 정확도 = %.2f" % accuracy)

# k를 변화시켜가면서 정확도를 측정해 본다
testAcc = []
trainAcc = []
for k in range(1, 50):
    # KNN 으로 Train 데이터 세트를 학습한다.
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    knn.fit(trainX, trainY)
    
    # Test 세트의 Feature에 대한 정확도
    predY = knn.predict(testX)
    testAcc.append((testY == predY).sum() / len(predY))
    
    # Train 세트의 Feature에 대한 정확도
    predY = knn.predict(trainX)
    trainAcc.append((trainY == predY).sum() / len(predY))

plt.figure(figsize=(8, 5))
plt.plot(testAcc, label="Test Data")
plt.plot(trainAcc, label="Train Data")
plt.legend()
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()