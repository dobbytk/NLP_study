# 행렬 분해 : R = P * Q.T
# NaN이 포함된 R이 주어졌을 때 P, Q를 추정한다.
# by Stochastic Gradient Descent
# -------------------------------------------
import numpy as np
# alpha: learning rate
# beta: for regularization - 너무 과하게 주면 과소적합
# err_limit: early stopping - 이 이상 안움직이면 멈춰라
stop = []
error = []
def factorize_matrix(R, K, max_iter=5000, alpha=0.01, beta=0.01, err_limit=0.0001, verbose=False):
    n_user = R.shape[0]
    n_item = R.shape[1]
    
    P = np.random.rand(n_user, K)    # user-factor matrix
    Q = np.random.rand(K, n_item)    # factor-item matrix
    old_err = 99999999 # 학습 전 err
    
    for step in range(max_iter):
        for i in range(n_user):
            for j in range(n_item):
                if np.isnan(R[i, j]) != True:  # nan이 아니면
                    eij = R[i, j] - np.dot(P[i, :], Q[:, j]) # 두 벡터의 내적임, 행렬 연산이 아니다.
                    
                    # update P, Q - 행 단위로 update
                    P[i, :] += alpha * (eij * Q[:, j] - beta * P[i, :])
                    Q[:, j] += alpha * (eij * P[i, :] - beta * Q[:, j])

        # P,Q update 후 total err
        tot_err = 0
        for i in range(n_user):
            for j in range(n_item):
                if np.isnan(R[i, j]) != True:
                    tot_err += (R[i, j] - np.dot(P[i, :], Q[:, j])) ** 2
        
        if verbose:
            print('step : {}, error = {:.4f}'.format(step, tot_err))
            stop.append(step)
            error.append(tot_err)
        # early stopping
        if np.abs(old_err - tot_err) < err_limit:
            break
        old_err = tot_err

    if step >= max_iter - 1:
        print('max_iter={}번 동안 stop하지 못했습니다.'.format(max_iter))
        print('max_iter를 늘리거나 err_limit을 늘려야 합니다.')
    
    return P, Q

# User-item matrix
N = np.NaN
R = np.array([[4, N, N, 2, N],
              [N, 5, N, 3, 1],
              [N, N, 3, 4, 4],
              [5, 2, 1, 2, N]])

k = 3   # number of factors - 사용자가 결정
P, Q = factorize_matrix(R, k, verbose=True)
ER = np.dot(P, Q)   # estimated R

print('\nR :')
print(np.round(R, 2))
print('\nEstimated R :')
print(np.round(ER, 2))   
print('\nP :')
print(np.round(P, 2))
print('\nQ.T :')
print(np.round(Q.T, 2))

# 그래프로 표현하기
import matplotlib.pyplot as plt

plt.plot(stop, error, label='Loss')
plt.xlabel('step')
plt.ylabel('error')
plt.legend()
plt.show()