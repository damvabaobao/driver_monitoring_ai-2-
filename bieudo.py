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
# BIỂU ĐỒ 13: Split Violin Plot - Đối xứng hóa EAR và MAR
# =====================================================================
plt.figure(figsize=(8, 6))
df_melted = pd.melt(df, id_vars=['Label'], value_vars=['EAR', 'MAR'], var_name='Feature', value_name='Value')
sns.violinplot(data=df_melted, x='Feature', y='Value', hue='Label', split=True, palette='muted', inner="quart")
plt.title("Split Violin Plot: EAR vs MAR Comparison")
plt.xlabel("Features")
plt.ylabel("Values")
plt.show()


# =====================================================================
# BIỂU ĐỒ 14: Boxenplot (Letter-value Plot) cho đặc trưng Yaw
# =====================================================================
plt.figure(figsize=(8, 6))
sns.boxenplot(data=df, x="Label", y="Yaw", hue="Label", palette="coolwarm")
plt.title("Boxenplot (Letter-Value) for Head Yaw")
plt.xlabel("Driver State")
plt.ylabel("Yaw (Degrees)")
plt.show()


# =====================================================================
# BIỂU ĐỒ 15: Biểu đồ mật độ quét thảm (KDE Plot + Rug Plot) cho MAR
# =====================================================================
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x="MAR", hue="Label", palette="crest", fill=True, common_norm=False, alpha=0.4)
sns.rugplot(data=df, x="MAR", hue="Label", palette="crest", height=0.08)
plt.title("KDE Density with Rug Plot for MAR")
plt.xlabel("MAR Value")
plt.ylabel("Density")
plt.show()


# =====================================================================
# BIỂU ĐỒ 16: Tách lưới phân phối (FacetGrid) cho Pitch và Yaw
# =====================================================================
g = sns.FacetGrid(df, col="Label", hue="Label", palette="magma", height=5)
g.map(sns.scatterplot, "Pitch", "Yaw", alpha=0.6)
g.add_legend()
g.fig.suptitle("FacetGrid Scatter: Pitch vs Yaw separated by Label", y=1.05)
plt.show()


# =====================================================================
# BIỂU ĐỒ 17: Biplot ước lệ cho không gian thành phần chính PCA
# =====================================================================
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[y==0, 0], x_pca[y==0, 1], label="Awake", alpha=0.4, color='silver')
plt.scatter(x_pca[y==1, 0], x_pca[y==1, 1], label="Drowsy", alpha=0.4, color='peachpuff')
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


# =====================================================================
# BIỂU ĐỒ 18 (MỚI): Lưới PairGrid tùy biến (Khác biệt hoàn toàn so với Pairplot)
# =====================================================================
# Cho phép tùy biến đồ thị ở đường chéo (KDE), nửa trên (Scatter), và nửa dưới (KDE 2D).
# Nó cung cấp cái nhìn 3 chiều khác nhau trên cùng một ma trận đặc trưng dữ liệu.
g = sns.PairGrid(df, vars=["EAR", "MAR", "Pitch", "Yaw"], hue="Label", palette="Set1")
g.map_diag(sns.kdeplot, fill=True)
g.map_upper(sns.scatterplot, alpha=0.4, size=1)
g.map_lower(sns.kdeplot, levels=4, thresh=0.2, alpha=0.7)
g.add_legend()
g.fig.suptitle("Custom PairGrid: Scatter, KDE, and Contour Combo", y=1.02)
plt.show()


# =====================================================================
# BIỂU ĐỒ 19 (MỚI): Biểu đồ phân tán có kích thước động (Bubble Chart)
# =====================================================================
# Trực quan hóa 3 đặc trưng cùng lúc: Trục X là EAR, Trục Y là MAR, và kích thước
# kích cỡ của chấm tròn (size) tỷ lệ thuận với trị tuyệt đối của góc Pitch.
plt.figure(figsize=(9, 7))
sns.scatterplot(data=df, x="EAR", y="MAR", hue="Label", size=df["Pitch"].abs(), 
                sizes=(20, 200), palette="Accent", alpha=0.6)
plt.title("Bubble Chart: EAR vs MAR (Size represents |Pitch|)")
plt.xlabel("EAR")
plt.ylabel("MAR")
plt.show()


# =====================================================================
# BIỂU ĐỒ 20 (MỚI): Biểu đồ hộp kết hợp phân vị cắt lớp (Quantile Boxplot)
# =====================================================================
# Sử dụng stripplot đặt chồng lên boxplot nhưng định dạng điểm dạng "line" (marker="|")
# Giúp theo dõi chính xác mật độ phân bố của các phân vị (Quantiles) cụ thể cho chỉ số MAR.
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Label", y="MAR", color="whitesmoke", whiskers=np.inf)
sns.stripplot(data=df, x="Label", y="MAR", hue="Label", palette="dark", alpha=0.3, marker="|", size=15)
plt.title("Quantile Layered Stripplot over Boxplot (MAR)")
plt.show()


# =====================================================================
# BIỂU ĐỒ 21 (MỚI): Biểu đồ mật độ lục giác (Hexbin Plot) cho PC1 và PC2
# =====================================================================
# Thay vì hiển thị các chấm tròn dễ bị đè lên nhau (Overplotting), Hexbin gom dữ liệu 
# vào các ô lưới lục giác. Ô nào màu càng đậm chứng tỏ mật độ điểm ở không gian PCA càng cao.
fig, ax = plt.subplots(figsize=(8, 6))
hb = ax.hexbin(x_pca[:, 0], x_pca[:, 1], gridsize=25, cmap='Blues', mincnt=1)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Sample Count per Hexagon')
ax.set_title("Hexagonal Binning of PCA Space")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
plt.show()


# =====================================================================
# BIỂU ĐỒ 22 (MỚI): Biểu đồ phân phối tần suất tương đối (Histplot dạng Poly)
# =====================================================================
# Thay vì các cột vuông, đồ thị nối các đỉnh phân phối bằng đường đa giác (element="poly").
# Giúp trực quan hóa cực tốt tần suất số lượng mẫu dữ liệu của góc Yaw lặp lại.
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Yaw", hue="Label", element="poly", fill=True, palette="autumn", alpha=0.4)
plt.title("Polygon Frequency Distribution of Head Yaw")
plt.xlabel("Yaw (Degrees)")
plt.ylabel("Count")
plt.show()


# =====================================================================
# BIỂU ĐỒ 23 (MỚI): Đồ thị phân tán phân vùng mật độ biên (Marginal Rug Scatter)
# =====================================================================
# Kết hợp biểu đồ scatter gốc giữa EAR và Pitch, nhưng ép thêm thảm mật độ (Rug)
# ở cả 2 biên cạnh trục hoành lẫn trục tung để tránh việc dữ liệu bị ảo ảnh khoảng trống.
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="EAR", y="Pitch", hue="Label", palette="cool", alpha=0.5)
sns.rugplot(data=df, x="EAR", y="Pitch", hue="Label", palette="cool", alpha=0.3)
plt.title("Scatter Plot with Two-Dimensional Marginal Rugs: EAR vs Pitch")
plt.xlabel("EAR Value")
plt.ylabel("Pitch (Degrees)")
plt.show()