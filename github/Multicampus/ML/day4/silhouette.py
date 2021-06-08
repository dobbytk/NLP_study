# K-Means Clustering : Silrouette 계수 확인

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

# 시험용 데이터 세트를 구성한다
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

# K-means 알고리즘으로 시험용 데이터를 3 그룹으로 분류한다 (k = 3)
km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
km = km.fit(X)
y_km = km.predict(X)

# 분류 결과를 표시한다
plt.figure(figsize=(6, 5))
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=100, c='green', marker='s', alpha=0.5, label='cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=100, c='orange', marker='o', alpha=0.5, label='cluster 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=100, c='blue', marker='v', alpha=0.5, label='cluster 3')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250, marker='+', c='red', label='centroids')
plt.legend()
plt.grid()
plt.ylabel('y')
plt.xlabel('X')
plt.show()

trc_silhouette = []
for i in range(2, 10):
    # K-means 알고리즘으로 시험용 데이터를 3 그룹으로 분류한다 (k = 3)
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    km = km.fit(X)
    y_km = km.predict(X)
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    trc_silhouette.append(np.mean(silhouette_vals))

# 실루엣 계수를 확인한다.
plt.plot(np.arange(2, 10), trc_silhouette, marker='o')
plt.show()