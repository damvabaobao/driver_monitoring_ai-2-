import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("driver_dataset.csv")

x = df[["EAR", "MAR", "Pitch", "Yaw"]]
y = df["Label"]

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[y==0,0],
    X_pca[y==0,1],
    label="Awake",
    alpha=0.6
)
plt.scatter(
    X_pca[y==1,0],
    X_pca[y==1,1],
    label="Drowsy",
    alpha=0.6
)

plt.legend()
plt.title("PCA Visualization")
plt.xlabel("Principal component 1")
plt.ylabel("principal Component 2")
plt.grid(True)
plt.show()

plt.show()

sns.pairplot(df, vars = ["EAR", "MAR", "Pitch", "Yaw"], hue = "Label")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()
plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Label"
)

plt.title("Class Distribution")
plt.xlabel("Label")
plt.ylabel("Number of Samples")

plt.show()