import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load train & test data
train_data = pd.read_csv("occupancy_train.txt")
test_data = pd.read_csv("occupancy_test.txt")

# Select features (Humidity, Light, HumidityRatio) and target (Occupancy)
X_train = train_data[["Humidity", "Light", "HumidityRatio"]]
y_train = train_data["Occupancy"]

X_test = test_data[["Humidity", "Light", "HumidityRatio"]]
y_test = test_data["Occupancy"]

# Store accuracies
accuracies = []

# Running KNN for k = 1 to 10
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K = {k}, Accuracy = {acc:.4f}")

# Find best accuracy
best_acc = max(accuracies)
best_k = accuracies.index(best_acc) + 1
print("\nBest Accuracy:", best_acc)
print("Best K:", best_k)
