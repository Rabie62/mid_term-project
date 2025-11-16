
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import os

# Load data
df = pd.read_csv("data/heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tune
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)

# Save best model
os.makedirs("model", exist_ok=True)
with open("model/heart_model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)

print(f"Best model saved. Accuracy: {grid.score(X_test, y_test):.3f}")