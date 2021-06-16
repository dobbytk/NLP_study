# 행렬 분해를 이용한 잠재 요인 협업 필터링
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Commented out IPython magic to ensure Python compatibility.
# 학습 데이터를 읽어온다.
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

df = pd.merge(ratings, movies, on='movieId')[['userId', 'movieId', 'rating', 'title']]

# userId와 movieId가 중간에 빈 값이 많으므로 순차적인 id를 다시 부여한다.
user_enc = LabelEncoder()
item_enc = LabelEncoder()

df['userId'] = user_enc.fit_transform(df['userId'])
df['movieId'] = item_enc.fit_transform(df['movieId'])
df['rating'] /= 5.0   # 0.5 ~ 5.0 --> 0.1 ~ 1.0으로 표준화.

# movieId = 163937은 movieId = 9371로 변환되었다.
# item_enc.transform([163937])[0] = 9371
# item_enc.inverse_transform([9371])[0] = 163937
df.head()

# number of users and items
n_users = df['userId'].max() + 1
n_items = df['movieId'].max() + 1
n_users, n_items

# 학습 데이터와 시험 데이터로 분리
d_train, d_test = train_test_split(df, test_size = 0.1)

# pivotting
# d_train의 pivot table과 d_test의 pivot table의 shape이 같아야 나중에 mse를 측정할 수 있음.
# pd.pivot_table()을 사용하지 않고 수동으로 만든다. 최대 크기로 만듦.
def pivot(data, u, i):
    p = np.zeros(shape=(u, i))
    for i, row in data.iterrows():
        p[int(row['userId']), int(row['movieId'])] = row['rating']
    return p

x_train = pivot(d_train, n_users, n_items)
x_test = pivot(d_test, n_users, n_items)
x_train.shape, x_test.shape

# R에서 0이 아닌 (행번호, 열번호, 데이터) => (n_row, n_col, R_data)
R = np.array(x_train, dtype=np.float32)
print(R)
non_zeros = [(i, j, R[i,j]) for i in range(n_users) for j in range(n_items) if R[i,j] > 0.0]
non_zeros[:10]

def get_mse(R, P, Q, non_zeros):
    error = 0
    
    ER = np.dot(P, Q)  # estimated R
    
    # R에서 NaN이 아닌 행, 열 번호
    n_row = [x[0] for x in non_zeros]
    n_col = [x[1] for x in non_zeros]
    
    # R에서 NaN이 아닌 데이터와 추정-R (ER)에서 해당 위치의 데이터
    R_data = R[n_row, n_col]
    ER_data = ER[n_row, n_col]

    return mean_squared_error(R_data, ER_data)

# 타겟 유저가 보지 않은 영화들에 대해 해당 유저가 부여할 rating을 추정한다.
user_id = user_enc.transform([9])[0]         # target user = 9
top_n = 10          # 추정 평정이 높은 상위 top_n개
K = 50              # latent feature 개수
steps = 20          # SGD 학습 횟수
alpha = 0.05        # learning rate
beta = 0.001         # regularization constant
err_limit = 0.0001   # error limit for early stopping

# P, Q의 초깃값으로 정규분포를 사용한다. 초깃값에 따라 발산할 수도 있으며, error 측정치로 NaN이 나올 수도 있다.
# 경험적으로 정규분포로 랜덤값을 추출하면 안정적으로 동작한다.
P = np.random.normal(loc = 0.0, scale = 1.0 / K, size = (n_users, K)).astype(np.float32)
Q = np.random.normal(loc = 0.0, scale = 1.0 / K, size = (K, n_items)).astype(np.float32)
P.shape, Q.shape

# SGD 기법으로 P, Q 행렬을 추정한다.
old_mse = 99999999 # 학습 전 err
for step in range(steps):
    for i, j, r in non_zeros:
        # 실제 값과 추정 값으로 오류 값 계산
        eij = r - np.dot(P[i, :], Q[:, j])
        
        # update
        P[i, :] += alpha * (eij * Q[:, j] - beta * P[i, :])
        Q[:, j] += alpha * (eij * P[i, :] - beta * Q[:, j])
    
    new_mse = get_mse(R, P, Q, non_zeros)

    print('Step :', step + 1, ' mse =', np.round(new_mse, 4))
    
    # early stopping
    if np.abs(old_mse - new_mse) < err_limit:
        print('early stopped')
        break
    
    old_mse = new_mse

# 추정 평점을 계산한다.
ER = np.dot(P, Q)   # estimated R

# target user가 안 본 영화의 인덱스와 추정 rating
unseen_idx = np.where(R[user_id, :] <= 0)[0]
print(unseen_idx)
pred_R = ER[user_id, unseen_idx]

# target user에게 추천할 영화 리스트
pred_sort_idx = np.array(pred_R).argsort()[::-1][:top_n]

user = user_enc.inverse_transform([user_id])[0]
print('\n영화 추천 목록 : User = {}'.format(user))
print("--- {:s} {:s}".format('-' * 35, '-' * 15))
print("No  {:35s} {:s}".format('Title', 'Expected rating'))
print("--- {:s} {:s}".format('-' * 35, '-' * 15))
for i, p in enumerate(pred_sort_idx):
    title = df[df['movieId'] == unseen_idx[p]]['title'].values[0]

    # rating: 0.1 ~ 1.0 --> 0.5 ~ 5.0으로 복원한다.
    print("{:2d} : {:40s}{:.4f}".format(i+1, title[:39], pred_R[p] * 5.0))
print("{:s}".format('-' * 55))

# 테스트 데이터로 RMSE 평가. x_test와 ER 모두 0보다 큰 걸로 MSE 계산
s = 0
n = 0
for i in range(n_users):
    for k in range(n_items):
        if x_test[i,k] > 0 and ER[i,k] > 0:
            s += (x_test[i,k] - ER[i,k]) ** 2
            n += 1

print("MSE =", s / n)