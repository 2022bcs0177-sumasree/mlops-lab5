import os
import json
import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_wine(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

# ✅ Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save trained model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save metrics
with open("metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

print("Training complete. Accuracy:", accuracy)
