from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
(x_train, y_train), (x_test, y_test) = mnist.load_data()
imageX = x_train.reshape(-1, 784)

km = KMeans(n_clusters=10, init='k-means++', n_init=10)
km = km.fit(imageX)
y_km = km.predict(imageX)

f = plt.figure(figsize=(8, 2))
for k in np.unique(y_km):
  idx = np.where(y_km == k)[0][:10]
  f = plt.figure(figsize=(8, 2))

  for i in range(10):
    image = imageX[idx[i]].reshape(28, 28)
    ax = f.add_subplot(1, 10, i + 1)
    ax.imshow(image, cmap=plt.cm.bone)
    ax.grid(False)
    ax.set_title(k)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    plt.tight_layout()