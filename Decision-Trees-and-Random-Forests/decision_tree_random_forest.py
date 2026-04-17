import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset (⚠️ change file name if needed)
df = pd.read_csv("heart.csv")

# Remove missing values (if any)
df = df.dropna()

# Target column
target_column = 'target'

# Features and label
X = df.drop(columns=[target_column])
y = df[target_column]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# Decision Tree
# =========================
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", round(dt_accuracy, 2))

# Plot Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(dt, filled=True, feature_names=X.columns)
plt.title("Decision Tree")
plt.show()

# =========================
# Random Forest
# =========================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("Random Forest Accuracy:", round(rf_accuracy, 2))

# =========================
# Feature Importance
# =========================
importances = rf.feature_importances_

plt.figure(figsize=(10, 5))
plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.show()

# =========================
# Cross Validation
# =========================
cv_scores = cross_val_score(rf, X_scaled, y, cv=5)
print("Cross Validation Score:", round(cv_scores.mean(), 2))