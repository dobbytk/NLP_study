# 행렬 분해 : R = P * Q.T
# NaN이 포함된 R이 주어졌을 때 P, Q를 추정한다.
# by Embedding layers
# -------------------------------------------
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Dot, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# User-item matrix
N = np.NaN
R = np.array([[4, N, N, 2, N],
              [N, 5, N, 3, 1],
              [N, N, 3, 4, 4],
              [5, 2, 1, 2, N]])
R = np.array(R)

n_users = R.shape[0]
n_items = R.shape[1]
n_factors = 3

# unpivoting
user_item = pd.DataFrame(R).stack().reset_index()
user_item.columns = ['user', 'item', 'rating']

x_user = np.array(user_item['user']).reshape(-1, 1) # to_categorical을 안 쓰고 배열 그 자체를 사용. sparse_categorical 
x_item = np.array(user_item['item']).reshape(-1, 1) 

# 0~5 값을 표준화를 시켜서 0~1 값으로 만들기 -> y_pred가 음수값, 5점이 넘는 걸 방지 + activation func 사용
# regression with bounded output
y_rating = np.array(user_item['rating']).reshape(-1, 1) / 5.0 
x_user.shape, x_item.shape

x_input_user = Input(batch_shape = (None, x_user.shape[1])) # Embedding layer
x_input_item = Input(batch_shape = (None, x_item.shape[1]))

# Embedding이 하는 일 
# 1. Embedding을 사용해서 내부적으로 to_categorical로 변환한다 + Mini_batch를 이용해서 out of memory 문제를 해결한다. 
# 2. 내부적으로 행렬 곱셈을 안하고 lookup process로 처리한다. 
# 3. input 2D - output 3D (None, 1, 3)로 출력한다. ex) LSTM, CNN/ * FFN에선 Flatten or Reshape으로 2D로 변환시켜준다.
x_user_emb = Embedding(input_dim = n_users, output_dim = n_factors)(x_input_user) 
x_user_emb = Flatten()(x_user_emb)

x_item_emb = Embedding(input_dim = n_items, output_dim = n_factors)(x_input_item)
x_item_emb = Flatten()(x_item_emb)

y_output = Dot(axes=1)([x_user_emb, x_item_emb])
y_otuput = Activation('sigmoid')(y_output) # <- 0~1 값으로! 

model = Model([x_input_user, x_input_item], y_output)

# 그럼 여기서 'mse'를 써도 되나? 하는 문제가 생김.
# Gradient Descent 문제 

model.compile(loss='mse', optimizer = Adam(learning_rate=0.01)) 

model_p = Model([x_input_user, x_input_item], x_user_emb)
model_q = Model([x_input_user, x_input_item], x_item_emb)

model.summary()

# embedding_2 -> 3D
# embedding_3 -> 3D
# dot_3 -> 1은 rating 값

hist = model.fit([x_user, x_item], y_rating, epochs = 500)

y_pred = model.predict([x_user, x_item])

user_item['y_pred'] = y_pred
user_item

# user-item의 전체 조합을 생성한다
users = np.arange(n_users)
items = np.arange(n_items)

print(users)
print(items)
x_tot = np.array([(x, y) for x in users for y in items])
x_tot

x_tot_user = x_tot[:, 0].reshape(-1, 1)
x_tot_item = x_tot[:, 1].reshape(-1, 1)
print(x_tot_user)
print(x_tot_item)

# user-item의 전체 조합에 대해 expected rating을 추정한다.
y_pred = model.predict([x_tot_user, x_tot_item])

df = pd.DataFrame([x_tot_user.reshape(-1), x_tot_item.reshape(-1), y_pred.reshape(-1)]).T
df.columns = ['user', 'item', 'rating']

ER = np.array(df.pivot_table('rating', index='user', columns='item'))

ER.round(2)
R
P = model_p.predict([x_tot_user, x_tot_item])
Q = model_q.predict([x_tot_user, x_tot_item])

P.round(2)

Q.T.round(2)