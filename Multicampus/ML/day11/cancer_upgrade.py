import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
import numpy as np

# breast cancer 데이터를 가져온다.
data = load_breast_cancer()

feature = data['data']
target = data['target']

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2)

# Z-score normalization - Feature도 train의 평균/분산으로 test를 표준화 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 
# train의 평균과 분산으로 x_test에도 계산을 해라. x_train과 x_test가 평균과 분산이 비슷하다는 전제하에..
# 이제는 이런 방식으로 표준화 시켜야한다!

scaler.mean_, scaler.var_

x_train.shape, y_train.shape

x_input = Input(batch_shape = (None, 30))
h_layer = Dense(16, activation='relu')(x_input)
y_output = Dense(1, activation='sigmoid')(h_layer)
model = Model(x_input, y_output)
# mse는 GD의 기본 Loss함수. classification, regression 다 쓸 수 있다.
model.compile(loss='binary_crossentropy', optimizer='adam')  # activation fucn 을 sigmoid로 설정하면 loss 함수는 binary_crossentropy다

model.summary()

hist = model.fit(x_train, y_train, epochs=100, validation_data = (x_test, y_test))

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(x_test).reshape(-1)
# y_pred = np.where(y_pred > 0.5, 1, 0)

acc = (y_pred == y_test).mean()
print(acc)