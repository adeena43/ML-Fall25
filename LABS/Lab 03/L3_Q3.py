import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------- Load Dataset -------------------
data = pd.read_csv("cancer patient data sets.csv")

# A) Exploratory Data Analysis
print("Initial Data Exploration:")
print(data.head())
print(data.info())
print(data.describe())

# B) Data Checks
class_distribution = data["Level"].value_counts(normalize=True)
null_values = data.isnull().sum()
duplicates = data.duplicated().sum()
categorical_features = data.select_dtypes(include=["object", "category"])

print("\nClass Distribution (Level variable):\n", class_distribution)
print("\nTotal Missing Values:", null_values.sum())
print("Duplicate Rows:", duplicates)
print("\nCategorical Columns:\n", categorical_features)

# Dataset is already clean -> no nulls, no duplicates.
# Patient ID and index are identifiers, so they can be removed.
# Target column "Level" needs encoding since it is categorical.
# Class distribution looks nearly balanced (minor difference only).
data = data.drop(['Patient Id', 'index'], axis=1)

# Encode target labels (Low, Medium, High → 0,1,2)
label_encoder = OrdinalEncoder(categories=[["Low", "Medium", "High"]])
data["Level"] = label_encoder.fit_transform(data[["Level"]])

# ------------------- Correlation -------------------
correlation_matrix = data.corr()
plt.figure(figsize=(25, 25))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
print("Correlation Table:\n", correlation_matrix)

# Histogram plots → check data spread
data.hist(figsize=(12, 12))
plt.tight_layout()
plt.show()

# Scaling is required for KNN since different attributes have varying ranges
features = data.drop("Level", axis=1)
target = data["Level"]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ------------------- Train-Test Splits -------------------
# Split1 → 80% training, 20% testing
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    features_scaled, target, test_size=0.2, random_state=0
)

# Split2 → 70% training, 30% validation
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    features_scaled, target, test_size=0.3, random_state=0
)

# Why use a validation set?
# It allows us to tune model hyperparameters without touching the final test set.

# ------------------- KNN with Multiple Metrics -------------------
distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
performance = {}

# Using k=15 as example
for dist in distance_metrics:
    model = KNeighborsClassifier(n_neighbors=15, metric=dist)
    model.fit(X_train1, y_train1)

    preds1 = model.predict(X_test1)
    preds2 = model.predict(X_test2)

    acc1 = accuracy_score(y_test1, preds1)
    acc2 = accuracy_score(y_test2, preds2)

    performance[dist] = {
        'accuracy_split1': acc1,
        'accuracy_split2': acc2
    }

# Show results
print("\nAccuracy results across different distance metrics:")
performance_df = pd.DataFrame(performance).T
print(performance_df)

# Observation:
# Based on results, Manhattan distance often gives slightly better accuracy
# compared to others, though differences depend on dataset characteristics.
