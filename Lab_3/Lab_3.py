import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def cmeans(data, n_clusters, m, error, maxiter):
    n_samples, n_features = data.shape

    u = np.random.rand(n_clusters, n_samples)
    u = u / np.sum(u, axis=0)

    for iteration in range(maxiter):

        um = u ** m
        cntr = np.dot(um, data) / np.sum(um, axis=1, keepdims=True)

        distances = np.zeros((n_clusters, n_samples))
        for i in range(n_clusters):
            distances[i] = np.linalg.norm(data - cntr[i], axis=1)
        distances = np.fmax(distances, np.finfo(np.float64).eps)
        u_new = 1.0 / (distances ** (2 / (m - 1)))
        u_new = u_new / np.sum(u_new, axis=0)

        jm = np.sum((u ** m) * (distances ** 2))

        if np.linalg.norm(u_new - u) < error:
            break

        u = u_new

    return cntr, u, jm

iris = load_iris()
x = iris.data[:, :2]
y = iris.target

iris_type = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}

n_clusters = 3
m = 2
error = 0.001
maxiter = 1000

cntr, distribution_matrix, jm = cmeans(x, n_clusters, m, error, maxiter)

colors = ['purple', 'green', 'blue']
plt.title('Fuzzy Clustering Results')
for class_label in range(3):
    plt.scatter(x[y == class_label, 0], x[y == class_label, 1], c=colors[class_label], s=70, label=iris_type[class_label])

plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', s=200, color='red', label='Cluster Centers')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

fuzzy_labels = np.argmax(distribution_matrix, axis=0)

plt.title('Original vs Fuzzy Cluster Assignments')
for class_label in range(3):
    plt.scatter(x[y == class_label, 0], x[y == class_label, 1], c=colors[class_label], s=70, label=f'Original {iris_type[class_label]}')

for i in range(3):
    cluster = x[fuzzy_labels == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], edgecolors='black', s=100, label=f'Fuzzy {iris_type.get(i, "Cluster")}', alpha=0.5)

plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', s=200, color='red', label='Cluster Centers')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

plt.title('Objective Function over Iterations')
plt.plot([jm], marker='o', linestyle='-', color='b')
plt.xlabel('Iterations')
plt.ylabel('Objective Function Value (Jm)')
plt.grid()
plt.show()
