# 하드코딩으로 Textrank 먼저 구현해보기

import numpy as np

C = np.array([[0, 0.2, 0, 0.3],
              [0.2, 0, 0.4, 0],
              [0, 0.4, 0, 0],
              [0.3, 0, 0, 0]])

tr = np.array([0.25, 0.25, 0.25, 0.25])

# 값이 있는 것만 이용해서 textrank 구해보기
for i in range(2):
  tr[0] = 0.15 + 0.85 * ((C[0][1]/np.sum(C[1]) * tr[1]) + (C[0][3]/np.sum(C[3]) * tr[3])) # TR(A)
  tr[1] = 0.15 + 0.85 * ((C[1][0]/np.sum(C[0]) * tr[0]) + (C[1][2]/np.sum(C[2]) * tr[2])) # TR(B)
  tr[2] = 0.15 + 0.85 * (C[2][1]/np.sum(C[1])) * tr[1] # TR(C)
  tr[3] = 0.15 + 0.85 * (C[3][0]/np.sum(C[0])) * tr[0] # TR(D)
  
print(tr)

# 규칙을 찾기 위해 다 써보기
tr = np.array([0.25, 0.25, 0.25, 0.25])

for i in range(2):
  tr[0] = 0.15 + 0.85 * ((C[0][0]/np.sum(C[0]) * tr[0]) + (C[0][1]/np.sum(C[1]) * tr[1]) + (C[0][2]/np.sum(C[2]) * tr[2]) + (C[0][3]/np.sum(C[3]) * tr[3]))
  tr[1] = 0.15 + 0.85 * ((C[1][0]/np.sum(C[0]) * tr[0]) + (C[1][1]/np.sum(C[1]) * tr[1]) + (C[1][2]/np.sum(C[2]) * tr[2]) + (C[1][3]/np.sum(C[3]) * tr[3]))
  tr[2] = 0.15 + 0.85 * ((C[2][0]/np.sum(C[0]) * tr[0]) + (C[2][1]/np.sum(C[1]) * tr[1]) + (C[2][2]/np.sum(C[2]) * tr[2]) + (C[2][3]/np.sum(C[3]) * tr[3]))
  tr[3] = 0.15 + 0.85 * ((C[3][0]/np.sum(C[0]) * tr[0]) + (C[3][1]/np.sum(C[1]) * tr[1]) + (C[3][2]/np.sum(C[2]) * tr[2]) + (C[3][3]/np.sum(C[3]) * tr[3]))
  
print(tr)

# 위에 짠 코드를 일반화시키기
C = np.array([[0, 0.2, 0, 0.3],
              [0.2, 0, 0.4, 0],
              [0, 0.4, 0, 0],
              [0.3, 0, 0, 0]])

tr = np.array([0.25, 0.25, 0.25, 0.25])

d = 0.85
weight = 0
history = []

for i in range(80):
  ex_tr = np.array(tr)
  for j in range(len(C)): # 0
    for k in range(len(C[j])): # 0, 1, 2, 3
      weight += (C[j][k]/np.sum(C[k])) * tr[k]
    tr[j] = weight * d + (1-d)
    weight = 0 # weight값 초기화
  
  history.append(np.sum(abs(ex_tr - tr))) # loss값 계산

tr

hist = np.array(history).T

hist.shape # row = TR, column = loss

# loss값 그래프로 출력
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
fig.set_facecolor('white')
ax = fig.add_subplot()

ax.plot(range(80), hist, marker='o', label='loss')

ax.legend()
plt.show()