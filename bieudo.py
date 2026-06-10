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

sns.boxplot(data=df, x='Label', y='EAR', ax=axes[0], palette='Set2')
axes[0].set_title('Distribution of Eye Aspect Ratio (EAR)')
axes[0].set_xlabel('Driver State (0: Awake, 1: Drowsy)')
axes[0].set_ylabel('EAR Value')

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

sns.kdeplot(data=df, x='Pitch', hue='Label', fill=True, ax=axes[0], palette='Dark2', common_norm=False, alpha=0.5)
axes[0].set_title('Head Pitch Density Distribution')
axes[0].set_xlabel('Pitch (Degrees)')
axes[0].set_ylabel('Density')

sns.kdeplot(data=df, x='Yaw', hue='Label', fill=True, ax=axes[1], palette='Dark2', common_norm=False, alpha=0.5)
axes[1].set_title('Head Yaw Density Distribution')
axes[1].set_xlabel('Yaw (Degrees)')
axes[1].set_ylabel('Density')

plt.tight_layout()
plt.show()


# =====================================================================
# BIỂU ĐỒ 6: Jointplot kết hợp phân phối giữa EAR và MAR
# =====================================================================
g = sns.jointplot(data=df, x="EAR", y="MAR", hue="Label", palette="plasma", alpha=0.6, height=7)
g.fig.suptitle("Joint Distribution of EAR and MAR", y=1.02)
plt.show()


# =====================================================================
# BIỂU ĐỒ 7: Lmplot xu hướng tương quan tuyến tính giữa Pitch và Yaw
# =====================================================================
sns.lmplot(data=df, x="Pitch", y="Yaw", hue="Label", palette="vlag", height=6, aspect=1.2, scatter_kws={'alpha':0.5})
plt.title("Linear Regression Trend: Pitch vs Yaw")
plt.show()


# =====================================================================
# BIỂU ĐỒ 8: Strip Plot hiển thị mật độ điểm chi tiết cho EAR
# =====================================================================
plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x="Label", y="EAR", inner=None, color="lightgray")
sns.stripplot(data=df, x="Label", y="EAR", hue="Label", palette="Set1", alpha=0.5, jitter=0.25)
plt.title("Detailed Data Points Distribution for EAR")
plt.xlabel("Driver State (0: Awake, 1: Drowsy)")
plt.ylabel("EAR Value")
plt.show()


# =====================================================================
# BIỂU ĐỒ 9 (MỚI): Biểu đồ phân phối tích lũy (ECDF Plot) cho EAR
# =====================================================================
# Thể hiện tỷ lệ phần trăm tích lũy của dữ liệu. Giúp xác định chính xác 
# giá trị EAR mà tại đó phần lớn tài xế bắt đầu rơi vào trạng thái ngủ gật.
plt.figure(figsize=(8, 5))
sns.ecdfplot(data=df, x="EAR", hue="Label", palette="bright", linewidth=2)
plt.title("Empirical Cumulative Distribution Function (ECDF) of EAR")
plt.xlabel("EAR Value")
plt.ylabel("Proportion")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# =====================================================================
# BIỂU ĐỒ 10 (MỚI): Biểu đồ phân phối mật độ hai chiều (2D KDE Plot)
# =====================================================================
# Thể hiện vùng mật độ tập trung cao nhất dưới dạng các đường đồng mức (contour lines).
# Rất tốt để chỉ ra ranh giới tách biệt không gian trạng thái giữa EAR và MAR.
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x="EAR", y="MAR", hue="Label", fill=False, thresh=0.1, levels=10, palette="viridis", alpha=0.8)
plt.title("2D Contour Density: EAR vs MAR")
plt.xlabel("EAR (Eye Aspect Ratio)")
plt.ylabel("MAR (Mouth Aspect Ratio)")
plt.show()


# =====================================================================
# BIỂU ĐỒ 11 (MỚI): Biểu đồ phân đoạn dữ liệu (Histogram theo dạng Stacked)
# =====================================================================
# Chia nhỏ các khoảng giá trị của MAR thành từng cột chồng, giúp nhìn thấy rõ 
# số lượng mẫu ngáp (MAR lớn) chiếm ưu thế như thế nào ở nhãn Drowsy (1).
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="MAR", hue="Label", multiple="stack", palette="tab10", bins=30, edgecolor="white")
plt.title("Stacked Histogram of Mouth Aspect Ratio (MAR)")
plt.xlabel("MAR Value")
plt.ylabel("Count")
plt.show()


# =====================================================================
# BIỂU ĐỒ 12 (MỚI): Swarm Plot cho góc Pitch để phát hiện biên dữ liệu
# =====================================================================
# Gần giống Strip Plot nhưng sắp xếp các điểm không bị chồng lên nhau, 
# phác họa rõ nét hình dáng mật độ phân bố góc gật đầu (Pitch) của tài xế.
plt.figure(figsize=(8, 6))
sns.swarmplot(data=df, x="Label", y="Pitch", hue="Label", palette="cubehelix", size=4, alpha=0.7)
plt.title("Swarm Plot Distribution for Head Pitch")
plt.xlabel("Driver State (0: Awake, 1: Drowsy)")
plt.ylabel("Pitch (Degrees)")
plt.show()