import pandas as pd
import joblib

from sklearn.model_selection import (
        train_test_split,
        GridSearchCV,
        cross_val_score
)

from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
        recall_score,
        f1_score,
        roc_curve
)
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
import seaborn as sns

# LOAD

df = pd.read_csv("driver_dataset.csv")

x = df[["EAR", "MAR", "Pitch", "Yaw"]]
y = df["Label"]

# SPLIT
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", x_train.shape)
print("Test:", x_test.shape)

# GRID SEARCH

params = {"n_estimators": [100, 200, 300], "max_depth": [3, 4, 5], "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.8, 1.0]}
grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric="logloss"), params, cv=5, scoring = "f1")
grid.fit(x_train, y_train)
best_model=grid.best_estimator_
y_pred = best_model.predict(x_test)
y_prob = best_model.predict_proba(x_test)[:, 1]
cv_scores = cross_val_score(best_model, x, y, cv=5, scoring="accuracy")
print("\nCross Validation Scores:")
print(cv_scores)
print("\nMean CV Accuracy:")
print(cv_scores.mean())

# METRICS

acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nAccuracy:", acc)
print("\nRecall:", recall)
print("\nF1 Score:", f1)
print("\nROC AUC:", auc)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# FEATURE IMPORTANCE

importance_df = pd.DataFrame({
        "Feature": x.columns,
        "Importance": best_model.feature_importances_
})
importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
)
print("\nFeature Importance:")
print(importance_df)
plt.figure(figsize=(6,4))
plt.bar(
        importance_df["Feature"],
        importance_df["Importance"]
)
plt.title("Feature Importance - XGBoost")
plt.xlabel("Feature")
plt.ylabel("Importance")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Awake", "Drowsy"],
        yticklabels=["Awake", "Drowsy"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.show()
print("Best Parameters:")
print(grid.best_params_)

# ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.grid(True)
plt.show()

# SAVE MODEL

joblib.dump(best_model, "driver_drowsiness_xgb.pkl")
print("\nXGBoost modle saved!!!")