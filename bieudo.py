import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv("driver_dataset.csv")

# Tách đặc trưng (x) và nhãn trạng thái (y)
x = df[["EAR", "MAR", "Pitch", "Yaw"]]
y = df["Label"]

# 2. Giảm chiều dữ liệu bằng PCA (xuống 2 chiều để vẽ đồ thị)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)


# =====================================================================
# BIỂU ĐỒ 1: Trực quan hóa dữ liệu sau khi giảm chiều bằng PCA
# =====================================================================
plt.figure(figsize=(8, 6))

plt.scatter(x_pca[y==0, 0], x_pca[y==0, 1], label="Awake", alpha=0.5, color='teal')
plt.scatter(x_pca[y==1, 0], x_pca[y==1, 1], label="Drowsy", alpha=0.5, color='crimson')

plt.legend()
plt.title("PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()


# =====================================================================
# BIỂU ĐỒ 2: Pairplot - Biểu đồ ma trận các cặp thuộc tính
# =====================================================================
sns.pairplot(df, vars=["EAR", "MAR", "Pitch", "Yaw"], hue="Label", palette="Set1")
plt.show()


# =====================================================================
# BIỂU ĐỒ 3: Ma trận tương quan giữa các đặc trưng (Heatmap)
# =====================================================================
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()


# =====================================================================
# BIỂU ĐỒ 4: Biểu đồ hộp (Boxplot) so sánh phân phối EAR và MAR
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Vẽ Boxplot cho EAR (Tỉ lệ mở mắt)
sns.boxplot(data=df, x='Label', y='EAR', ax=axes[0], palette='Set2')
axes[0].set_title('Distribution of Eye Aspect Ratio (EAR)')
axes[0].set_xlabel('Driver State (0: Awake, 1: Drowsy)')
axes[0].set_ylabel('EAR Value')

# Vẽ Boxplot cho MAR (Tỉ lệ mở miệng)
sns.boxplot(data=df, x='Label', y='MAR', ax=axes[1], palette='Set2')
axes[1].set_title('Distribution of Mouth Aspect Ratio (MAR)')
axes[1].set_xlabel('Driver State (0: Awake, 1: Drowsy)')
axes[1].set_ylabel('MAR Value')

plt.tight_layout()
plt.show()


# =====================================================================
# BIỂU ĐỒ 5: Biểu đồ mật độ (KDE Plot) cho tư thế đầu Pitch và Yaw
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Vẽ đường cong mật độ cho Pitch (Gật đầu)
sns.kdeplot(data=df, x='Pitch', hue='Label', fill=True, ax=axes[0], palette='Dark2', common_norm=False, alpha=0.5)
axes[0].set_title('Head Pitch Density Distribution')
axes[0].set_xlabel('Pitch (Degrees)')
axes[0].set_ylabel('Density')

# Vẽ đường cong mật độ cho Yaw (Quay đầu)
sns.kdeplot(data=df, x='Yaw', hue='Label', fill=True, ax=axes[1], palette='Dark2', common_norm=False, alpha=0.5)
axes[1].set_title('Head Yaw Density Distribution')
axes[1].set_xlabel('Yaw (Degrees)')
axes[1].set_ylabel('Density')

plt.tight_layout()
plt.show()