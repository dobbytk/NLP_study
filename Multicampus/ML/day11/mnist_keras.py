from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 3D -> 2D로 변환, -1 데이터 개수 있는대로, 784 column 개수
x_train = x_train.reshape(-1, 784) 
x_test = x_test.reshape(-1, 784)

# 0~1 사잇값으로 표준화 한다.
x_train = x_train / 255
x_test = x_test / 255

# 784 -> 50 차원으로 축소
pc = PCA(n_components=50)
x_train = pc.fit_transform(x_train)
x_test = pc.transform(x_test)

# class를 categorical하게 바꿔준다
y_train_ohe = to_categorical(y_train)
y_test_ohe = to_categorical(y_test)

# 모델
x_input = Input(batch_shape=(None, x_train.shape[1]))
h_layer = Dense(32, activation='relu')(x_input)
h_layer = Dropout(rate=0.5)(h_layer)
h_layer = Dense(32, activation='relu')(h_layer)
h_layer = Dropout(rate=0.5)(h_layer)
h_layer = Dense(32, activation='relu')(h_layer)
h_layer = Dropout(rate=0.5)(h_layer)
y_output = Dense(10, activation='softmax')(h_layer)

model = Model(x_input, y_output)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0005))
model.summary()

# 학습
hist = model.fit(x_train, y_train_ohe, epochs=500, batch_size=64, validation_data=(x_test, y_test_ohe))

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test) # 테스트셋을 이용한 예측값

y_pred_tmp = []
for array in y_pred:
  y_pred_tmp.append(np.argmax(array))

accuracy = accuracy_score(y_test, y_pred)
print("정확도는 %.2f" % accuracy)

# 어떤 값을 제대로 예측 못했는지 프린트 해보기
n_sample = 10
miss_cls = np.where(y_test != y_pred)[0]
miss_sam = np.random.choice(miss_cls, n_sample)
len(miss_cls)

fig, ax = plt.subplots(1, n_sample, figsize=(12, 4))
for i, miss in enumerate(miss_sam):
  x = pc.inverse_transform(x_test[miss])
  # x = x_test[miss] * 255
  x = x * 255
  x = x.reshape(28, 28)
  ax[i].imshow(x)
  ax[i].axis('off')
  ax[i].set_title(str(y_test[miss]) + ' / ' + str(y_pred[miss]))
plt.show()