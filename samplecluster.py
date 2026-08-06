import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(42)

low = np.random.normal(loc=[2, 2], scale=0.4, size=(50, 2))
moderate = np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
high = np.random.normal(loc=[8, 3], scale=0.5, size=(50, 2))
critical = np.random.normal(loc=[9, 8], scale=0.4, size=(50, 2))

X = np.vstack((low, moderate, high, critical))

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

risk_names = [
    "Low Risk",
    "Moderate Risk",
    "High Risk",
    "Critical Monitoring"
]

order = np.argsort(centers[:, 0] + centers[:, 1])

mapping = {}
for i, cluster in enumerate(order):
    mapping[cluster] = risk_names[i]

plt.figure(figsize=(8,6))

for cluster in range(4):
    pts = X[labels == cluster]
    plt.scatter(
        pts[:,0],
        pts[:,1],
        s=45,
        label=mapping[cluster]
    )

plt.scatter(
    centers[:,0],
    centers[:,1],
    marker="X",
    s=220,
    c="black",
    label="Cluster Centroids"
)

plt.title("Illustrative K-Means Risk Segmentation")
plt.xlabel("Feature Dimension 1")
plt.ylabel("Feature Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()