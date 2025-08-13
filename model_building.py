# model_building.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("Dataset/diabetes.csv")

# 2. Features (X) and Target (y)
X = df.drop("Outcome", axis=1)  # Outcome = 1 (diabetic), 0 (non-diabetic)
y = df["Outcome"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model training
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 6. Predictions
y_pred = model.predict(X_test_scaled)

# 7. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 8. Save model and scaler using joblib
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and Scaler saved successfully using joblib!")
