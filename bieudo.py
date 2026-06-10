import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("driver_dataset.csv")

features = ["EAR", "MAR", "Pitch", "Yaw"]
x = df[features]
y = df["Label"]

# =========================
# 2. PCA (2D VISUALIZATION)
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
# 5. VIOLIN PLOT
# =========================
for f in features:
    plt.figure(figsize=(6, 4))
    sns.violinplot(x="Label", y=f, data=df, inner="quartile")
    plt.title(f"Violin Plot of {f}")
    plt.xlabel("Label")
    plt.ylabel(f)
    plt.show()

# =========================
# 6. SCATTER: EAR vs MAR
# =========================
plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=df,
    x="EAR",
    y="MAR",
    hue="Label",
    alpha=0.6
)
plt.title("EAR vs MAR Scatter Plot")
plt.grid(True)
plt.show()

# =========================
# 7. 3D SCATTER (EAR - MAR - PITCH)
# =========================
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    df["EAR"],
    df["MAR"],
    df["Pitch"],
    c=df["Label"],
    cmap="coolwarm",
    alpha=0.6
)

ax.set_xlabel("EAR")
ax.set_ylabel("MAR")
ax.set_zlabel("Pitch")
ax.set_title("3D Scatter: EAR - MAR - Pitch")
plt.show()

# =========================
# 8. PCA EXPLAINED VARIANCE
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

# =========================
# 9. CORRELATION HEATMAP
# =========================
plt.figure(figsize=(8, 6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation Matrix")
plt.show()