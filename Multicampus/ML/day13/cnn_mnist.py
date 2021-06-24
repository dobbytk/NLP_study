import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import Dropout, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# mnist 데이터를 다운로드한다
(d_train, y_train), (d_test, y_test) = mnist.load_data()

# 손글씨 이미지를 몇개만 확인해 본다.
fig, ax = plt.subplots(1, 10, figsize=(14,4))
for i in range(10):
    ax[i].imshow(d_train[i])
    ax[i].axis('off')
    ax[i].set_title('label = ' + str(y_train[i]))
plt.show()

# train 데이터를 표준화하고, Conv2D를 사용하기 위해 shape을 조정한다.
x_train = d_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = d_test.reshape(-1, 28, 28, 1).astype('float32') / 255


# CNN 모델을 생성한다. - kernel_size: filter size, conv가 feature map이 된다. 
x_input = Input(batch_shape=(None, 28, 28, 1))
conv = Conv2D(filters=20, kernel_size=(10, 8), activation='relu')(x_input)
pool = MaxPooling2D(pool_size=(8, 6), strides=1, padding='valid')(conv) # padding='valid' 패딩 x, 'same' 원래 데이터 크기에 맞게 패딩 알아서 붙여줘~
flat = Flatten()(pool) # pooling한 결과를 Flatten에 넣으면 1열 vector로 변환 
h_layer = Dense(64, activation='relu')(flat)
h_layer = Dropout(0.5)(h_layer)
y_output = Dense(10, activation='softmax')(h_layer) # 10인 이유는 0-9까지 있기 때문에. 

model = Model(x_input, y_output)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
model.summary()

# 학습
hist = model.fit(x_train, y_train,
                 validation_data = (x_test, y_test),
                 batch_size = 512, epochs = 30)

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label = 'Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# test 데이터의 정확도를 측정한다.
pred = model.predict(x_test)
y_pred = np.argmax(pred, axis=1)
acc = (y_test == y_pred).mean()
print('정확도 =', acc)

# 잘못 분류한 이미지 몇개를 확인해 본다
n_sample = 10
miss_cls = np.where(y_test != y_pred)[0]
miss_sam = np.random.choice(miss_cls, n_sample)

fig, ax = plt.subplots(1, n_sample, figsize=(12,4))
for i, miss in enumerate(miss_sam):
    ax[i].imshow(d_test[miss])
    ax[i].axis('off')
    ax[i].set_title(str(y_test[miss]) + ' / ' + str(y_pred[miss]))
plt.show()

# 옆으로 누운 이미지를 인식할 수 있을까? 학습시키지 않았기 때문에 당연히 인식할 수 없다.
img = d_train[7].T

y = model.predict(img.reshape(1, 28, 28, 1)).argmax(axis=1)
plt.imshow(img)
plt.title('predicted =' + str(y))
plt.show()