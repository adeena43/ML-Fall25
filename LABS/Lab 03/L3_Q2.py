import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score

def chi_squared_distance(x1, x2):
    return np.sum((x1 - x2) ** 2 / (x1 + x2 + 1e-10))  # small epsilon to avoid div/0

class CustomKNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for sample in X:
            # compute distances to all training samples
            distances = [chi_squared_distance(sample, x_train) for x_train in self.X_train]
            # get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]
            # majority vote
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train & predict
knn = CustomKNN(k=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
