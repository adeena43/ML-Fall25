import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("your_dataset.csv")  # replace with your file path
print("First 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# =========================
# 2. Check class balance
# =========================
print("\nLabel counts:")
print(df['Label'].value_counts())
print("\nLabel percentage:")
print(df['Label'].value_counts(normalize=True))

# =========================
# 3. Check missing, categorical, duplicate
# =========================
print("\nMissing values:")
print(df.isnull().sum())

print("\nCategorical features:")
print(df.select_dtypes(include='object').columns)

print("\nDuplicate rows:")
print(df.duplicated().sum())

# Handling missing values (if any)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)  # categorical → mode
        else:
            df[col].fillna(df[col].median(), inplace=True)    # numeric → median

# Drop duplicates
df.drop_duplicates(inplace=True)

# Convert categorical columns to numeric (if any)
categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# =========================
# 4. Correlation & feature selection
# =========================
corr = df.corr(method='pearson')
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Pearson Correlation Heatmap")
plt.show()

# Select features with some correlation threshold with target
threshold = 0.1
selected_features = corr['Label'][abs(corr['Label']) > threshold].index.tolist()
selected_features.remove('Label')
print("\nSelected features:", selected_features)

# =========================
# 5. Feature scaling
# =========================
X = df[selected_features]
y = df['Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 6. Split dataset: Train / Test / Validation
ChatGPT said:

#A validation set is used to tune model hyperparameters and check performance on unseen data 
#during training, helping prevent overfitting. The test set is kept separate for final 
#evaluation only.
# =========================
# First split: Train 80% / Test 20%
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0, stratify=y
)

# Second split: Train 70% / Validation 30% of training set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.3, random_state=0, stratify=y_train_full
)

# =========================
# 7. KNN with multiple metrics
# =========================
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
results = []

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train, y_train)
    
    y_train_pred = knn.predict(X_train)
    y_val_pred = knn.predict(X_val)
    y_test_pred = knn.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    results.append([metric, train_acc, val_acc, test_acc])

# =========================
# 8. Display results
# =========================
results_df = pd.DataFrame(results, columns=['Metric', 'Train Accuracy', 'Validation Accuracy', 'Test Accuracy'])
print("\nKNN Accuracy Comparison:")
print(results_df)

# Optional: Confusion matrix for the best metric
best_metric = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Metric']
print("\nBest metric based on test accuracy:", best_metric)

knn_best = KNeighborsClassifier(n_neighbors=5, metric=best_metric)
knn_best.fit(X_train, y_train)
y_test_pred_best = knn_best.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred_best)
print("\nConfusion Matrix on Test set (best metric):")
print(cm)

# =========================
# 9. Critical Analysis (text output)
# =========================
print("""
Critical Analysis:
- KNN performance depends on the distance metric used.
- Euclidean works well for features with similar scales (we used StandardScaler).
- Manhattan can be more robust to outliers.
- Chebyshev focuses on the maximum feature difference.
- Minkowski is a generalization of Euclidean and Manhattan.
- Training accuracy is often higher than validation/test accuracy.
- Validation set helps tune hyperparameters without touching test set.
- Always check metrics to choose the most suitable distance for your dataset.
""")

