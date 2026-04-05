import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

data = load_iris()

X = data.data

model = KMeans(n_clusters=3, random_state=42)

model.fit(X)

y_pred = model.predict(X)

print("Cluster Labels:\n", y_pred)
print("\nCentroids:\n", model.cluster_centers_)
print("\nInertia:", model.inertia_)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='X')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.show()