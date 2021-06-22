from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2차원 배열의 feature 데이터로 LSTM 학습 데이터를 만든다.
def build_train_data(data, t_step, n_jump = 1):
    n_data = data.shape[0]   # number of data
    n_feat = data.shape[1]   # number of features

    m = np.arange(0, n_data - t_step, n_jump)   # m =  [0, 1, 2, 3, 4, 5]
    x = [data[i:(i+t_step), :] for i in m]      # feature data
    y = [data[i, :] for i in (m + t_step)]      # target data

    # shape을 조정한다. feature = 3D, target = 2D
    x_data = np.reshape(np.array(x), (len(m), t_step, n_feat))
    y_target = np.reshape(np.array(y), (len(m), n_feat))
    
    return x_data, y_target
# 시계열 데이터 (noisy sin)
sine = np.sin(2 * np.pi * 0.03 * np.arange(1001))   # sine 곡선
# sine = np.sin(2 * np.pi * 0.03 * np.arange(1001)) + np.random.random(1001) # trend & noisy sine

# 데이터가 데이터프레임 형식으로 되어 있다고 생각하자. feature가 1개이고 target이 없는 데이터임.
# 미래의 sine 값을 target으로 만들어 주고, LSTM을 학습한다.
df = pd.DataFrame({'sine':sine})
df.head()

t_step = 20

# 학습 데이터를 생성한다.
data = np.array(df)
x_train, y_train = build_train_data(data, t_step)
x_train.shape, y_train.shape

n_input = 1
n_output = 1
n_hidden = 50

# LSTM 모델을 생성한다.
x_input = Input(batch_shape=(None, t_step, n_input))
x_lstm = LSTM(n_hidden)(x_input)
y_output = Dense(n_output)(x_lstm)

model = Model(x_input, y_output)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# 학습한다
h = model.fit(x_train, y_train, epochs=20, batch_size=100, shuffle=True)

# Loss history를 그린다
plt.figure(figsize=(8, 3))
plt.plot(h.history['loss'], color='red')
plt.title("Loss History")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

prediction = model.predict(x_train)

# 학습 시킨 결과를 100개만 그래프로 찍어보자
plt.figure(figsize=(15, 7))
plt.plot(range(len(x_train[881:])), prediction[881:])
plt.show()

# 최근 t_step 기간의 데이터로 다음 기간의 sine 값을 예측한다.
# 예측한 값을 다시 data array에 넣어서 맨 뒤에서부터 20개씩 입력값으로 사용한다.
for i in range(100):
  px = data[-t_step:].reshape(1, t_step, 1)
  y_pred = model.predict(px)
  data = np.append(data, y_pred, axis=0)

plt.figure(figsize=(20, 7))
plt.plot(range(len(x_train[881:])), prediction[881:], marker='o', label='time series')
plt.plot(range(len(x_train[881:]), len(x_train[881:])+101), data[-101:], 'g', marker='o', label='estimated')
plt.axvline(x=len(x_train[881:]), linestyle='dashed', linewidth=2)
plt.legend()
plt.show()