import pandas as pd
import numpy as np 

# 시계열 데이터 input과 target 만드는 작업
def build_train_data(data, t_step, n_jump = 1):
  x_data = []
  y_target = []
  for i in range(0, len(data), n_jump):
    a = data[i:t_step+i]
    tmp = np.concatenate(a[:t_step], axis=0)
    try:
      tmp = tmp.reshape(t_step, data.shape[1])
    except:
      continue
    x_data.append(np.array(tmp))
    y_target.append(data[t_step+i-1])
  return x_data, y_target

df = pd.DataFrame({'f1':np.arange(50), 'f2':np.arange(0.0, 5, 0.1)})

x_train, y_train = build_train_data(np.array(df), t_step=3, n_jump=1)

# 강사님 코드
def build_train_data(data, t_step, n_jump=1):
  n_data = data.shape[0]
  n_feat = data.shape[1]

  m = np.arange(0, n_data - t_step, n_jump)
  x = [data[i:(i+t_step),:] for i in m]
  y = [data[i, :] for i in (m + t_step)]

  x_data = np.reshape(np.array(x), (len(m), t_step, n_feat))
  y_target = np.reshape(np.array(y), (len(m), n_feat))

  return x_data, y_target

