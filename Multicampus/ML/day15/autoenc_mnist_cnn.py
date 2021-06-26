# Autoencoder를 이용한 차원 축소 예시
# Mnist 이미지 군집 분류
# ---------------------==------------
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pickle

# mnist 데이터를 다운로드한다
(d_train, y_train), (d_test, y_test) = mnist.load_data()
d_train.shape, y_train.shape

# input data를 생성한다
x_train = d_train / 255
x_train = x_train.reshape(-1, 28, 28, 1)  # CNN 입력을 위해 channel 축을 추가한다.

# CNN AutoEncoder.
n_height = x_train.shape[1]
n_width = x_train.shape[2]
x_input = Input(batch_shape=(None, n_height, n_width, 1))

# encoder
# (28, 28) 이미지를 (14, 14) 이미지로 줄인다.
e_conv = Conv2D(filters=10, kernel_size=(5,5), strides=1, padding = 'same', activation='relu')(x_input)
e_pool = MaxPooling2D(pool_size=(5,5), strides=1, padding='valid')(e_conv)
e_flat = Flatten()(e_pool)
e_latent = Dense(14 * 14)(e_flat) # (28, 28) --> (14, 14)로 축소
e_latent = Reshape((14, 14, 1))(e_latent)

# decoder
# 이미지를 strides = 2 배 만큼 늘린다. 결과 = (20, 8) : 원본 이미지
d_conv = Conv2DTranspose(filters=10, kernel_size=(10, 10), strides=2, padding='same', activation='relu')(e_latent) # strides = 2 2배
y_output = Conv2D(1, kernel_size=(6, 4), strides=1, padding = 'same')(d_conv) # filter 수를 1로 지정해줌으로써 채널을 1개로 맞춰준다. 
model = Model(x_input, y_output)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.005))
encoder = Model(x_input, e_latent) # e_latent까지 중간 결과

# autoencoder를 학습한다
hist = model.fit(x_train, x_train, epochs=50, batch_size=1024) # 입력값과 출력값이 같다. 

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 학습된 autoencoder를 이용하여 입력 데이터의 차원을 축소한다.
x_encoded = encoder.predict(x_train).reshape(-1, 14 * 14)

# K-means++ 알고리즘으로 차원이 축소된 이미지를 10 그룹으로 분류한다.
km = KMeans(n_clusters=10, init='k-means++', n_init=3, max_iter=300, tol=1e-04, random_state=0)
km = km.fit(x_encoded)
clust = km.predict(x_encoded)

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

x_encoded.shape

plt.imshow(x_encoded[0].reshape(14,14))

plt.imshow(x_train[0].reshape(28,28))