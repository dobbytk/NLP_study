import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import pdb
# breast cancer 데이터를 가져온다.
cancer = load_breast_cancer()

# 표준화
feature_data = StandardScaler().fit_transform(cancer.data)

# Train 데이터 세트와 Test 데이터 세트를 구성한다
trainX, testX, trainY, testY = train_test_split(feature_data, cancer.target, test_size = 0.2)

# 학습 및 평가
"""
feature 간 scale이 다름.
분기를 나누는 DT의 경우에는 표준화가 필요없었지만, 
SVM, kNN의 경우 직선의 방정식을 찾기 위해 연산하므로 표준화 필요
"""
model = SVC(kernel='rbf', gamma=1.0, C=0.5)
model.fit(trainX, trainY)
print('정확도 =', np.round(model.score(testX, testY), 3))

# gamma와 C의 조합을 바꿔가면서 학습 데이터의 정확도가 최대인 조합을 찾는다
optAcc = -999
optG = 0
optC = 0
for gamma in np.arange(0.1, 5.0, 0.1):
    for c in np.arange(0.1, 5.0, 0.1):
        model = SVC(kernel='rbf', gamma=gamma, C=c)
        model.fit(trainX, trainY)
        acc = model.score(testX, testY)
        
        if acc > optAcc:
            optG = gamma
            optC = c
            optAcc = acc
            pdf.set_trace()

print('optimal gamma = %.2f' % optG)
print('optimal C = %.2f' % optC)
print('optimal Accuracy = %.2f' % optAcc)

# 최적 조건으로 학습한 결과를 확인한다.
model = SVC(kernel='rbf', gamma=optG, C=optC)
model.fit(trainX, trainY)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
print('\n')
print("* 학습용 데이터로 측정한 정확도 = %.2f" % model.score(trainX, trainY))
print("* 시험용 데이터로 측정한 정확도 = %.2f" % model.score(testX, testY))

"""
(1) 방법 1
acc.append(model.score)
G.append(gamma) 
C.append(c)

idx = np.argmax(acc)
optC = C[idx]
optG = G[idx]
optAcc = acc[idx]

(2) 방법 2 - 따로 append를 하지 않아도 되니깐 메모리를 아낄 수 있다.
optAcc = -999 # 충분히 작은 값을 넣어줌
optG = 0
optC = 0 

(3) 방법 3 - GridSearchCV package
"""