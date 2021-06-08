from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:5000]
y_train = y_train[:5000]
x_test = x_test[:1000]
y_test = y_test[:1000]

# 3D -> 2D로 변환, -1 데이터 개수 있는대로, 784 column 개수
x_train = x_train.reshape(-1, 784) 
x_test = x_test.reshape(-1, 784)

# 0~1 사잇값으로 표준화 한다.
x_train = x_train / 255
x_test = x_test / 255

# KNN 으로 Train 데이터 세트를 학습한다.
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski') # n_neighbors = k의 개수, p는 minkowski의 m을 의미
knn.fit(x_train, y_train)

# Test 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
# accuracy = knn.score(testX, testY)와 동일함.
predY = knn.predict(x_test) # 테스트셋을 이용한 예측값 
accuracy = (y_test == predY).mean() # 정확도
print("* 시험용 데이터로 측정한 정확도 = %.2f" % accuracy)

# Train 세트의 Feature에 대한 class를 추정하고, 정확도를 계산한다
predY = knn.predict(x_train)
accuracy = (y_train == predY).mean()
print("* 학습용 데이터로 측정한 정확도 = %.2f" % accuracy)

# 어떤 값을 제대로 예측 못했는지 프린트 해보기
n_sample = 10
miss_cls = np.where(y_test != predY)[0]
miss_sam = np.random.choice(miss_cls, n_sample)

fig, ax = plt.subplots(1, n_sample, figsize=(12, 4))
for i, miss in enumerate(miss_sam):
  x = x_test[miss] * 255
  x = x.reshape(28, 28)
  ax[i].imshow(x)
  ax[i].axis('off')
  ax[i].set_title(str(y_test[miss]) + ' / ' + str(predY[miss]))
plt.show()