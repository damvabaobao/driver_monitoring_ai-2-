import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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
# BIỂU ĐỒ 9: Biểu đồ phân phối tích lũy (ECDF Plot) cho EAR
# =====================================================================
plt.figure(figsize=(8, 5))
sns.ecdfplot(data=df, x="EAR", hue="Label", palette="bright", linewidth=2)
plt.title("Empirical Cumulative Distribution Function (ECDF) of EAR")
plt.xlabel("EAR Value")
plt.ylabel("Proportion")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


# =====================================================================
# BIỂU ĐỒ 10: Biểu đồ phân phối mật độ hai chiều (2D KDE Plot)
# =====================================================================
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df, x="EAR", y="MAR", hue="Label", fill=False, thresh=0.1, levels=10, palette="viridis", alpha=0.8)
plt.title("2D Contour Density: EAR vs MAR")
plt.xlabel("EAR (Eye Aspect Ratio)")
plt.ylabel("MAR (Mouth Aspect Ratio)")
plt.show()


# =====================================================================
# BIỂU ĐỒ 11: Biểu đồ phân đoạn dữ liệu (Histogram theo dạng Stacked)
# =====================================================================
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="MAR", hue="Label", multiple="stack", palette="tab10", bins=30, edgecolor="white")
plt.title("Stacked Histogram of Mouth Aspect Ratio (MAR)")
plt.xlabel("MAR Value")
plt.ylabel("Count")
plt.show()


# =====================================================================
# BIỂU ĐỒ 12: Swarm Plot cho góc Pitch để phát hiện biên dữ liệu
# =====================================================================
plt.figure(figsize=(8, 6))
sns.swarmplot(data=df, x="Label", y="Pitch", hue="Label", palette="cubehelix", size=4, alpha=0.7)
plt.title("Swarm Plot Distribution for Head Pitch")
plt.xlabel("Driver State (0: Awake, 1: Drowsy)")
plt.ylabel("Pitch (Degrees)")
plt.show()


# =====================================================================
# BIỂU ĐỒ 13 (MỚI): Split Violin Plot - Đối xứng hóa EAR và MAR
# =====================================================================
# Gom chung 2 trạng thái vào cùng một thân Violin (nửa trái là Awake, nửa phải là Drowsy)
# Cách biểu diễn này cực kỳ tiết kiệm không gian và tối ưu hóa việc so sánh trực quan.
plt.figure(figsize=(8, 6))
df_melted = pd.melt(df, id_vars=['Label'], value_vars=['EAR', 'MAR'], var_name='Feature', value_name='Value')
sns.violinplot(data=df_melted, x='Feature', y='Value', hue='Label', split=True, palette='muted', inner="quart")
plt.title("Split Violin Plot: EAR vs MAR Comparison")
plt.xlabel("Features")
plt.ylabel("Values")
plt.show()


# =====================================================================
# BIỂU ĐỒ 14 (MỚI): Boxenplot (Letter-value Plot) cho đặc trưng Yaw
# =====================================================================
# Là dạng nâng cao của Boxplot, vẽ thêm nhiều phân vị sâu hơn. 
# Cực kỳ tốt đối với dữ liệu lớn nhằm triệt tiêu khuyết điểm che giấu đuôi dữ liệu của Boxplot gốc.
plt.figure(figsize=(8, 6))
sns.boxenplot(data=df, x="Label", y="Yaw", hue="Label", palette="coolwarm")
plt.title("Boxenplot (Letter-Value) for Head Yaw")
plt.xlabel("Driver State")
plt.ylabel("Yaw (Degrees)")
plt.show()


# =====================================================================
# BIỂU ĐỒ 15 (MỚI): Biểu đồ mật độ quét thảm (KDE Plot + Rug Plot) cho MAR
# =====================================================================
# Rug plot sẽ thêm các vạch kim nhỏ ở đáy đồ thị, đại diện cho từng điểm dữ liệu đơn lẻ.
# Giúp bạn biết chính xác phân bố trơn (KDE) được dựng từ mật độ điểm dày hay mỏng ở thực tế.
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x="MAR", hue="Label", palette="crest", fill=True, common_norm=False, alpha=0.4)
sns.rugplot(data=df, x="MAR", hue="Label", palette="crest", height=0.08)
plt.title("KDE Density with Rug Plot for MAR")
plt.xlabel("MAR Value")
plt.ylabel("Density")
plt.show()


# =====================================================================
# BIỂU ĐỒ 16 (MỚI): Tách lưới phân phối (FacetGrid) cho Pitch và Yaw
# =====================================================================
# Thay vì vẽ chung một ô, FacetGrid tự động tách thành các ô biểu đồ độc lập 
# dựa theo điều kiện Nhãn (0 hoặc 1), giúp dễ dàng so sánh phân bố góc nhìn độc lập.
g = sns.FacetGrid(df, col="Label", hue="Label", palette="magma", height=5)
g.map(sns.scatterplot, "Pitch", "Yaw", alpha=0.6)
g.add_legend()
g.fig.suptitle("FacetGrid Scatter: Pitch vs Yaw separated by Label", y=1.05)
plt.show()


# =====================================================================
# BIỂU ĐỒ 17 (MỚI): Biplot ước lệ cho không gian thành phần chính PCA
# =====================================================================
# Vẽ các vectơ hướng đại diện cho các thuộc tính gốc (EAR, MAR, Pitch, Yaw) 
# chiếu lên không gian 2D PCA. Giúp giải thích thuộc tính nào đóng góp nhiều nhất vào PC1 và PC2.
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[y==0, 0], x_pca[y==0, 1], label="Awake", alpha=0.4, color='silver')
plt.scatter(x_pca[y==1, 0], x_pca[y==1, 1], label="Drowsy", alpha=0.4, color='peachpuff')

# Tính toán trọng số của các đặc trưng gốc (Loadings)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
features = ["EAR", "MAR", "Pitch", "Yaw"]

for i, feature in enumerate(features):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='red', alpha=0.8, head_width=0.05, linewidth=1.5)
    plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature, color='darkred', ha='center', va='center', weight='bold')

plt.title("PCA Biplot (Feature Contributions)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.legend()
plt.show()