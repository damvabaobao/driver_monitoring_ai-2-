import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("driver_dataset.csv")

x = df[["EAR", "MAR", "Pitch", "Yaw"]]
y = df["Label"]

# Split

x_train, x_test, y_train, y_test = train_test_split (
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
)

print("Train:", x_train.shape)
print("Test:", x_test.shape)

# Model

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train

model.fit(x_train, y_train)

# Predict

y_pred = model.predict(x_test)

# Evaluation

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy: ", acc)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature
print("\nFeature Importancw")

for name, importance in zip(
        x.columns,
        model.feature_importances_
):
        print(f"{name}: {importance:.4f}")

# Save
joblib.dump(model, "driver_drowsiness_rf.pkl")
print("\nModel saved!!!!!!!")