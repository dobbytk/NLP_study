from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Reshape, Conv2DTranspose
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
# train set을 학습에 사용하고 test set의 노이즈를 제거
x_train = d_train / 255
x_train = x_train.reshape(-1, 28, 28, 1)  # CNN 입력을 위해 channel 축을 추가한다.

x_test = d_test / 255
x_test = x_test.reshape(-1, 28, 28, 1)

# 학습 데이터와 시험 데이터에 노이즈를 삽입한다.
xn_train = x_train + 0.3 * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
xn_test = x_test + 0.3 * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# 0 이하면 '0', 1 이상이면 '1'
xn_train = np.clip(xn_train, 0., 1.) 
xn_test = np.clip(xn_test, 0., 1.)

xn_train.shape, xn_test.shape

plt.imshow(x_test[1].reshape(28, 28))
plt.imshow(xn_test[1].reshape(28, 28))


# CNN AutoEncoder.
n_height = xn_train.shape[1]
n_width = xn_train.shape[2]
x_input = Input(batch_shape=(None, n_height, n_width, 1))

# encoder
e_conv = Conv2D(filters=10, kernel_size=(3,3), strides=1, padding = 'same', activation='relu')(x_input)
e_pool = MaxPooling2D(pool_size=(2,2), strides=1, padding='same')(e_conv)

# decoder
## 입력값도 0~1 값이고 출력값도 0~1 이기 때문에, sigmoid 함수를 활성함수로 사용.
## 출력 하나고 0~1 값이면 시그모이드(binary classification), 
## 출력이 one-hot이면 softmax ~ categorical(multi-class classification) , 
## 출력이 0 1 0 1 0 이 나오면? 각 뉴런에다가 sigmoid = binary_crossentropy (Multi-classification)
d_conv = Conv2DTranspose(filters=10, kernel_size=(3,3), strides=1, padding = 'same', activation='relu')(e_pool)
y_output = Conv2D(1, kernel_size=(3,3), strides=1, padding = 'same', activation='sigmoid')(d_conv)
"""
# 그럼 정확도를 어떻게 따지는가?
one-hot: y = (0 0 1 0) \widehat y = (0 1 0 0) => 0점
one-hot이 아니면 => 열 기준으로 비교 50점. 
"""
model = Model(x_input, y_output)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.005))
model.summary()


# autoencoder를 학습한다
# 노이즈가 들어가면 정상 데이터가 나오도록 학습 (noise_data, normal_data)
hist = model.fit(xn_train, x_train, epochs=50, batch_size=1024) 

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


# 화면에 이미지를 그린다.
def showImage(x):
    n = 0
    for k in range(2):
        plt.figure(figsize=(8, 2))
        for i in range(5):
            ax = plt.subplot(1, 5, i+1)
            plt.imshow(x[n].reshape(28, 28))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            n += 1
        plt.show()

# 노이즈가 추가된 시험 데이터 10개를 그려본다.
print("\n잡음이 삽입된 이미지 :")
showImage(xn_test)

# 노이즈가 제거된 시험 데이터 10개를 그려본다.
print("\n잡음이 제거된 이미지 :")
xn_test = xn_test.reshape(-1, 28, 28, 1)      # channel 축을 추가한다.
xd_test = model.predict(xn_test) # 노이즈 테스트를 넣응면 xd_test, denoise 데이터가 나온다. 
showImage(xd_test)