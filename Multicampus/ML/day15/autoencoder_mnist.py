# Autoencoder를 이용한 차원 축소 예시
# Mnist 이미지 군집 분류
# ---------------------==------------
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# mnist 데이터를 다운로드한다
(d_train, y_train), (d_test, y_test) = mnist.load_data()
d_train.shape, y_train.shape

# input data를 생성한다
x_train = d_train.reshape(-1, 784) / 255

n_input = x_train.shape[1]
n_feat = 200               # 784개 feature를 이만큼으로 줄인다.
n_output = n_input

# FFN 모델을 생성한다
x_input = Input(batch_shape=(None, n_input))
x_encoder = Dense(256, activation='relu')(x_input)
x_encoder = Dense(n_feat, activation='relu')(x_encoder) # n_feat = 200차으로 줄인다.
y_decoder = Dense(256, activation='relu')(x_encoder)
y_decoder = Dense(n_output, activation='linear')(y_decoder)
model = Model(x_input, y_decoder)
encoder = Model(x_input, x_encoder)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# autoencoder를 학습한다 
hist = model.fit(x_train, x_train, epochs=100, batch_size=100) # 입력 데이터 = 출력 데이터, 오토인코더


# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 학습된 autoencoder를 이용하여 입력 데이터의 차원을 축소한다.
x_encoded = encoder.predict(x_train)

# K-means++ 알고리즘으로 차원이 축소된 이미지를 10 그룹으로 분류한다.
km = KMeans(n_clusters=10, init='k-means++', n_init=3, max_iter=300, tol=1e-04, random_state=0)
km = km.fit(x_encoded)
clust = km.predict(x_encoded)

print(clust)


# cluster 별로 이미지를 확인한다.
f = plt.figure(figsize=(8, 2))
for k in np.unique(clust):
    # cluster가 i인 imageX image 10개를 찾는다.
    idx = np.where(clust == k)[0][:10]
    
    f = plt.figure(figsize=(8, 2))
    for i in range(10):
        image = x_train[idx[i]].reshape(28,28)
        ax = f.add_subplot(1, 10, i + 1)
        ax.imshow(image, cmap=plt.cm.bone)
        ax.grid(False)
        ax.set_title(k)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()

