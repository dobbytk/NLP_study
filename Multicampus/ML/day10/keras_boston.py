from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import *
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

boston = load_boston()

scaleX = StandardScaler()
scaleY = StandardScaler()

z_feature = scaleX.fit_transform(boston['data'])
z_target = scaleY.fit_transform(boston['target'].reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(z_feature, z_target, test_size=0.2)

x_input = Input(batch_shape = (None, x_train.shape[1]))
h_layer = Dense(256, activation='relu')(x_input)
h_layer = Dropout(rate=0.3)(h_layer)
h_layer = Dense(256, activation='relu', kernel_regularizer=L2(0.01))(h_layer)
h_layer = Dense(256)(h_layer)
h_layer = BatchNormalization()(h_layer)
h_layer = Activation('relu')(h_layer)
y_output = Dense(1, activation='linear')(h_layer) # linear for regression
model = Model(x_input, y_output)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01)) # mse for regression, optimizer => loss를 찾는 방법으로 GD + Momentum + adaptive 
model.summary()

hist = model.fit(x_train, y_train, batch_size = 50, epochs=200, validation_data=(x_test, y_test)) # 50개씩 학습해서 업데이트해라

y_pred = model.predict(x_test)

score = mean_squared_error(y_pred, y_test)
print("Mean squared Error: %.4f" % score)

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred_real = scaleY.inverse_transform(y_pred)
print(y_pred_real)

# SVR로도 테스트해보기 

from sklearn.svm import SVR

x_train, x_test, y_train, y_test = train_test_split(z_feature, z_target, test_size=0.2)
model = SVR(kernel='rbf')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred = y_pred.reshape(-1, 1)

score = mean_squared_error(y_pred, y_test)
print("mean squared error: %0.4f" % score)