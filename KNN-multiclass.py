# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances

# Create a synthetic dataset with 2 classes and 2 features
X, y = make_classification(n_samples=50, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)

# Instantiate and fit the KNN model with Euclidean distance
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X, y)

# Predict a new test point
new_point = np.array([[1.5, 1.5]])  # New point to classify
predicted_class = knn.predict(new_point)

# Calculate the Euclidean distances to each training point for the new point
distances = pairwise_distances(new_point, X, metric='euclidean')

# Visualize the dataset and the decision boundary
plt.figure(figsize=(8, 6))

# Plot the training points with two different colors for two classes
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', marker='o', s=100, label='Training points')

# Plot the new point (test point)
plt.scatter(new_point[0, 0], new_point[0, 1], c='black', marker='x', s=150, label="New Point")

# Draw lines to show nearest neighbors and distances
indices = knn.kneighbors(new_point)[1][0]
for i in indices:
    plt.plot([new_point[0, 0], X[i, 0]], [new_point[0, 1], X[i, 1]], 'k--')

# Add labels, title, and legend
plt.title(f'K-Nearest Neighbors (KNN) with Euclidean Distance\nPredicted class: {predicted_class[0]}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
