import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("driver_dataset.csv")

features = ["EAR", "MAR", "Pitch", "Yaw"]
x = df[features]
y = df["Label"]

# =========================
# 2. PCA 2D VISUALIZATION
# =========================
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

plt.figure(figsize=(8, 6))
plt.scatter(x_pca[y == 0, 0], x_pca[y == 0, 1], label="Awake", alpha=0.6)
plt.scatter(x_pca[y == 1, 0], x_pca[y == 1, 1], label="Drowsy", alpha=0.6)
plt.legend()
plt.title("PCA Visualization of Driver State")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

# =========================
# 3. HISTOGRAM + KDE
# =========================
for f in features:
    plt.figure(figsize=(7, 5))
    sns.histplot(df, x=f, hue="Label", kde=True, bins=30)
    plt.title(f"Distribution of {f} by Label")
    plt.xlabel(f)
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

# =========================
# 4. BOXPLOT
# =========================
for f in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Label", y=f, data=df)
    plt.title(f"{f} Comparison Between Awake and Drowsy")
    plt.xlabel("Label (0 = Awake, 1 = Drowsy)")
    plt.ylabel(f)
    plt.grid(True)
    plt.show()

# =========================
# 5. PCA EXPLAINED VARIANCE
# =========================
plt.figure(figsize=(6, 4))
plt.bar(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_
)
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.show()