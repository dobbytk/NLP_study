import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import relu
import numpy as np

data = load_iris()

feature = data['data']
target = to_categorical(data['target'])

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size = 0.2)

x_input = Input(batch_shape = (None, x_train.shape[1])) # 데이터 개수는 일반적으로 안 적는다. None => 데이터가 있는대로
h_layer = Dense(5, activation='relu')(x_input) # name = 'hidden_1' 처럼 이름을 줄 수도 있다. 
y_output = Dense(y_train.shape[1], activation='softmax')(h_layer)
model = Model(x_input, y_output)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# 중간 출력을 알고 싶을 때 - 실제 feature를 latent feature로 변환 
h_model = Model(x_input, h_layer) 

h_model.predict(x_train[0].reshape(1, -1))

w1 = model.layers[1].get_weights()[0]
b1 = model.layers[1].get_weights()[1]
print(np.round(w1, 3))
print(b1)

model.predict(x_train[0].reshape(1, -1))

w2 = model.layers[2].get_weights()[0]
b2 = model.layers[2].get_weights()[1]
print(w2)
print(b2)

# 수동으로 계산해보기

h = np.dot(x_train[0].reshape(1, -1), w1) + b1
h = relu(h)
output = np.dot(h, w2) + b2
print(output)
print(np.argmax(output))
print(np.argmax(y_train[0]))