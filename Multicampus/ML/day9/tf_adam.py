# Tensorflow 버전 : Optimizers 기능을 사용한 예시
# - Adam optimizer를 사용하고, minimize() 함수를 이용한다.
#
# x, y 데이터 세트가 있을 때, 이차 방정식 y = w1x^2 + w2x + b를 만족하는
# parameter w1, w2, b를 추정한다.
import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

# y = 2x^2 + 3x + 5 일 때 x, y 집합을 생성한다
x = np.array(np.arange(-5, 5, 0.1))
y = 2 * x * x + 3 * x + 5

# 그래프를 생성한다.   
w1 = tf.Variable(1.0)
w2 = tf.Variable(1.0)
b = tf.Variable(1.0)

def loss():
    return tf.sqrt(tf.reduce_mean(tf.square(w1 * x * x + w2 * x + b - y)))

# Adam optimizers 기능을 사용한다.
opt = optimizers.Adam(learning_rate = 0.05)

histLoss = []
for epoch in range(300):
    opt.minimize(loss, var_list = [w1, w2, b])

    histLoss.append(loss())
    if epoch % 10 == 0:
        print("epoch = %d, loss = %.4f" % (epoch, histLoss[-1]))

print("\n추정 결과 :")
print("w1 = %.2f" % w1.numpy())
print("w2 = %.2f" % w2.numpy())
print("b = %.2f" % b.numpy())

plt.plot(histLoss, color='red', linewidth=1)
plt.title("Loss function")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()