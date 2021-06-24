import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Dropout, MaxPooling1D, Flatten, Average
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('./samsungelec.csv', encoding='cp949')
last_price = list(df['종가'])[-1]

df.drop(['날짜', '전일비', '국가지자체', '개인누적', '기관누적', '외국인누적', '금투누적', '투신누적', '연기금누적'], axis=1, inplace=True)
df.dropna(axis=0, inplace=True)

rtn_mean = df['등락율'].mean()
rtn_std = df['등락율'].std()
df = (df - df.mean()) / df.std()

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

t_step = 20
data = np.array(df)
x_train, y_train = build_train_data(data, t_step)
x_train.shape, y_train.shape
n_feat = x_train.shape[2]
n_output = y_train.shape[1]
n_hidden = 128

# 모델 구성하기 - 차원이 같으므로 x_input을 두 모델의 입력값으로 준다. 차원이 다르면 따로 모델을 학습 
# LSTM 모델
x_input = Input(batch_shape=(None, t_step, n_feat))

x_lstm = LSTM(n_hidden, return_sequences=True)(x_input)
x_lstm = LSTM(n_hidden, dropout=0.2)(x_lstm)
y_lstm = Dense(n_output)(x_lstm)

# CNN 모델
conv = Conv1D(filters=20, kernel_size=8, activation='relu')(x_input)
pool = MaxPooling1D(pool_size=3, strides=1, padding='valid')(conv)
flat = Flatten()(pool)
h_layer = Dense(64, activation='relu')(flat)
h_layer = Dropout(0.5)(h_layer)
y_result = Dense(n_output, activation='linear')(h_layer)
avg = Average()([y_lstm, y_result])

model = Model(x_input, y_result)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
model.summary(line_length=120)

# 학습한다
h = model.fit(x_train, y_train, epochs=100, batch_size=32, shuffle=True)

# Loss history를 그린다
plt.figure(figsize=(8, 3))
plt.plot(h.history['loss'], color='red')
plt.title("Loss History")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 내일의 수익률과 주가를 예측한다.
px = np.array(df.tail(20)).reshape(1, t_step, n_feat)
y_pred = model.predict(px)[0][0]
y_rtn = y_pred * rtn_std + rtn_mean

if y_rtn > 0:
    print("내일은 {:.2f}% 상승할 것으로 예측됩니다.".format(y_rtn * 100))
else:
    print("내일은 {:.2f}% 하락할 것으로 예측됩니다.".format(y_rtn * 100))
print("예상 주가 = {:.0f}".format(last_price * (1 + y_rtn)))