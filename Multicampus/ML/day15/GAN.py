# Keras를 이용하여 기본-GAN 모델을 연습한다.
# 1D 정규분포에서 샘플링한 데이터를 모방하여, fake data를 생성한다. fake data는 정규분포의
# 특성을 갖는다. (KL divergence, 평균, 분산, 왜도, 첨도 등)
# Discrimi의 loss는 max[log(Dx) + log(1 - DGz)]이고, Generator의 loss는
# min[log(Dx + log(1 - DGz))]이다. Tensorflow에서는 이 loss 함수를 이용하여 직접 GAN을
# 학습할 수 있지만, Keras에서는 model.fit(), model.train_on_batch() 함수에서 target 값을
# 지정해야 하기 때문에 이 loss로 GAN을 학습할 수 없다 (Keras는 기본적으로 Supervised
# learning 목적이다). Keras에서는 supervised learning 방식으로 바꿔서 binary_crossentropy
# loss 함수를 써서 GAN을 학습하는 것이 보통이다.
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 정규분포로부터 데이터를 샘플링한다 np.random.normal(loc=0.0, scale=1.0, size=None)
realData1 = np.random.normal(1, 1.0, size=1000) 
realData2 = np.random.normal(-1.0, 0.5, size=1000)
realData = np.hstack([realData1, realData2])


np.random.shuffle(realData)
realData = realData.reshape(realData.shape[0], 1)
sns.kdeplot(realData[:, 0], color='blue', bw_method=0.3, label='Real')

realData.shape


nDInput = realData.shape[1]
nDHidden = 32
nDOutput = 1
nGInput = 16
nGHidden = 32
nGOutput = nDInput

# 두 분포 (P, Q)의 KL divergence를 계산한다.
def KL(P, Q):
    # 두 데이터의 분포를 계산한다
    histP, binsP = np.histogram(P, bins=100)
    histQ, binsQ = np.histogram(Q, bins=binsP)
    
    # 두 분포를 pdf로 만들기 위해 normalization한다.
    histP = histP / (np.sum(histP) + 1e-8)
    histQ = histQ / (np.sum(histQ) + 1e-8)

    # KL divergence를 계산한다
    kld = np.sum(histP * (np.log(histP + 1e-8) - np.log(histQ + 1e-8)))
    return kld
       
def getNoise(m, n=nGInput):
    z = np.random.uniform(-1., 1., size=[m, n])
    return z

def MyOptimizer(a = 0.001):
    return RMSprop(learning_rate = a)

# Discriminator를 생성한다
def BuildDiscriminator():
    x = Input(batch_shape = (None, nDInput))
    h = Dense(nDHidden, activation = 'relu')(x)
    Dx = Dense(nDOutput, activation = 'sigmoid')(h)
    model = Model(x, Dx)
    model.compile(loss = 'binary_crossentropy', optimizer = MyOptimizer(0.001))
    return model

# Generator를 생성한다
def BuildGenerator():
    z = Input(batch_shape = (None, nGInput))
    h = Dense(nGHidden, activation = 'relu')(z)
    Gz = Dense(nGOutput, activation='linear')(h)
    return Model(z, Gz)

# Generator --> Discriminator를 연결한 모델을 생성한다.
# 아래 네트워크로 z가 들어가면 DGz = 1이 나오도록 G를 학습한다.
# D 네트워크는 업데이트하지 않고, G 네트워크만 업데이트한다.
#
#        +---+   Gz   +---+
#  z --->| G |------->| D |---> DGz
#        +---+        +---+
#      trainable   not trainable
# ----------------------------------------------------------
def BuildGAN(D, G):
    D.trainable = False     # Discriminator는 업데이트하지 않는다.
    z = Input(batch_shape=(None, nGInput))
    Gz = G(z)
    DGz = D(Gz)
    
    model = Model(z, DGz)
    model.compile(loss = 'binary_crossentropy', optimizer = MyOptimizer(0.0005))
    return model

K.clear_session()
Discriminator = BuildDiscriminator()
Generator = BuildGenerator()
GAN = BuildGAN(Discriminator, Generator)

nBatchCnt = 3       # Mini-batch를 위해 input 데이터를 n개 블록으로 나눈다.
nBatchSize = int(realData.shape[0] / nBatchCnt)  # 블록 당 Size
for epoch in range(1000):
    # Mini-batch 방식으로 학습한다
    for n in range(nBatchCnt):
        # input 데이터를 Mini-batch 크기에 맞게 자른다
        nFrom = n * nBatchSize
        nTo = n * nBatchSize + nBatchSize
        
        # 마지막 루프이면 nTo는 input 데이터의 끝까지.
        if n == nBatchCnt - 1:
            nTo = realData.shape[0]
               
        # 학습 데이터를 준비한다
        bx = realData[nFrom : nTo]
        bz = getNoise(m=bx.shape[0], n=nGInput)
        Gz = Generator.predict(bz)

        # Discriminator를 학습한다.
        # Real data가 들어가면 Discriminator의 출력이 '1'이 나오도록 학습하고,
        # Fake data (Gz)가 들어가면 Discriminator의 출력이 '0'이 나오도록 학습한다.
        target = np.zeros(bx.shape[0] * 2)
        target[ : bx.shape[0]] = 0.9     # '1' 대신 0.9로 함
        target[bx.shape[0] : ] = 0.1     # '0' 대신 0.1로 함
        bx_Gz = np.concatenate([bx, Gz])
        Dloss = Discriminator.train_on_batch(bx_Gz, target)
        
        # Generator를 학습한다.
        # Fake data (z --> Gz --> DGz)가 들어가도 Discriminator의 출력이 '1'이
        # 나오도록 Generator를 학습한다.
        target = np.zeros(bx.shape[0])
        target[:] = 0.9
        Gloss = GAN.train_on_batch(bz, target)
        
    if epoch % 10 == 0:
        z = getNoise(m=realData.shape[0], n=nGInput)
        fakeData = Generator.predict(z)
        kd = KL(realData, fakeData)
        print("epoch = %d, D-Loss = %.3f, G-Loss = %.3f, KL divergence = %.3f" % (epoch, Dloss, Gloss, kd))
    
# real data 분포 (p)와 fake data 분포 (q)를 그려본다
z = getNoise(m=realData.shape[0], n=nGInput)
fakeData = Generator.predict(z)

plt.figure(figsize=(8, 5))
sns.set_style('whitegrid')
sns.kdeplot(realData[:, 0], color='blue', bw_method=0.3, label='Real')
sns.kdeplot(fakeData[:, 0], color='red', bw_method=0.3, label='Fake')
plt.legend()
plt.title('Distibution of real and fake data')
plt.show()