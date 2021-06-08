# K-Means Clustering : Elbow에 의한 최적 cluster 개수
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 시험용 데이터 세트를 구성한다
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

# 엘보우 (elbow) 방법으로 최적 cluster 개수를 찾아본다.
distortions = []
for i in range(2, 11):
    km = KMeans(n_clusters=i,
                n_init=101,
                max_iter=300,
                tol=1e-04,
                random_state=0)
    km = km.fit(X)
    
    # Cluster내의 SSE를 계산한다
    # 관성 (inertia), 혹은 왜곡 (distortion)이라고도 한다.
    distortions.append(km.inertia_)

plt.plot(range(2, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()