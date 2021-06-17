import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np

data = load_iris()

feature = data['data']
target = to_categorical(data['target'])

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size = 0.2)

# 모델 선언하는 부분
x_input = Input(batch_shape = (None, x_train.shape[1])) # 데이터 개수는 일반적으로 안 적는다. None => 데이터가 있는대로
h_layer = Dense(5, activation='relu')(x_input)
y_output = Dense(y_train.shape[1], activation='softmax')(h_layer)
model = Model(x_input, y_output)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
"""
feature 4개 - Input layer
param bias 포함 25개 - hidden_1
param bias 포함 18개 - output layer
"""

hist = model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'])
plt.legend()
plt.show()

y_pred_cat = model.predict(x_test)
y_pred = np.argmax(y_pred_cat, axis=1)
y_test = np.argmax(y_test, axis=1)

accuracy = (y_pred == y_test).mean()
print('정확도: %.4f' % accuracy)