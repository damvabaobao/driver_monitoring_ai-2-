import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

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

#model = RandomForestClassifier(n_estimators=100, random_state=42)
params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 8, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
}
print("Best parameters:")
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced"), params, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

# Train

#model.fit(x_train, y_train)

# Predict
cv_scores = cross_val_score(
        best_model,
        x,
        y,
        cv=5,
        scoring="accuracy"
)

print("\nCross Validation Scores:")
print(cv_scores)

print("\nMean CV Accuracy:")
print(cv_scores.mean())

y_prob = best_model.predict_proba(x_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("\nROC AUC:")
print(auc)

recall = recall_score(y_test, y_pred)
print("\nRecall:")
print(recall)

f1 = f1_score(y_test, y_pred)
print("\nF1 Score:")
print(f1)
# Evaluation

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy: ", acc)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature
print("\nFeature Importancw")

importance_df = pd.DataFrame({
        "Feature": x.columns,
        "Importance": best_model.feature_importances_
})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
best_idx = (tpr - fpr).argmax()
best_threshold = thresholds[best_idx]
print("Best Tgreshold:")
print(thresholds[best_idx])
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(
        data=importance_df,
        x="Feature",
        y="Importance")
plt.title("Feature Importance - Random Forest")
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Awake", "Drowsy"], yticklabels=["Awake", "Drowsy"])
plt.xlabel("Predicted")
plt.ylabel("Actul")
plt.title("Confusion Matrix")
plt.show()

# Save
joblib.dump(grid_search.best_estimator_, "driver_drowsiness_rf.pkl")
print("\nModel saved!!!!!!!")