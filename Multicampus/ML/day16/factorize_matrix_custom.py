import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Dot, Activation
import tensorflow.keras.backend as K # 추가적으로 import 해준다.
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

x_user = np.array(user_item['user']).reshape(-1, 1)
x_item = np.array(user_item['item']).reshape(-1, 1)
y_rating = np.array(user_item['rating']).reshape(-1, 1) / 5.0
x_user.shape, x_item.shape

def my_activation(x):
  # return K.sigmoid(x)
  # return K.hard_sigmoid(x)
  return K.clip(x, 0, 1) # 0이하는 0으로 1이상은 1로

x_input_user = Input(batch_shape = (None, x_user.shape[1]))
x_input_item = Input(batch_shape = (None, x_item.shape[1]))

x_user_emb = Embedding(input_dim = n_users, output_dim = n_factors)(x_input_user)
x_user_emb = Flatten()(x_user_emb)

x_item_emb = Embedding(input_dim = n_items, output_dim = n_factors)(x_input_item)
x_item_emb = Flatten()(x_item_emb)

y_output = Dot(axes=1)([x_user_emb, x_item_emb])
# y_output = Activation('sigmoid')(y_output)
y_output = Activation(my_activation)(y_output)

model = Model([x_input_user, x_input_item], y_output)
model.compile(loss='mse', optimizer = Adam(learning_rate=0.01))

model_p = Model([x_input_user, x_input_item], x_user_emb)
model_q = Model([x_input_user, x_input_item], x_item_emb)

model.summary()

hist = model.fit([x_user, x_item], y_rating, epochs = 500)

y_pred = model.predict([x_user, x_item])

user_item['y_pred'] = y_pred
user_item

# user-item의 전체 조합을 생성한다
users = np.arange(n_users)
items = np.arange(n_items)

x_tot = np.array([(x, y) for x in users for y in items])
x_tot_user = x_tot[:, 0].reshape(-1, 1)
x_tot_item = x_tot[:, 1].reshape(-1, 1)

# user-item의 전체 조합에 대해 expected rating을 추정한다.
y_pred = model.predict([x_tot_user, x_tot_item])

df = pd.DataFrame([x_tot_user.reshape(-1), x_tot_item.reshape(-1), y_pred.reshape(-1)]).T
df.columns = ['user', 'item', 'rating']

ER = np.array(df.pivot_table('rating', index='user', columns='item')) * 5.0

ER.round(2)

R

P = model_p.predict([x_tot_user, x_tot_item])
Q = model_q.predict([x_tot_user, x_tot_item])

P.round(2)
Q.T.round(2)
ER.reshape(20)

import seaborn as sns
sns.displot(ER.reshape(20)) # 아래로 볼록한 모양이 되면 시그모이드의 영향을 받은 것으로 사용해서는 안된다.