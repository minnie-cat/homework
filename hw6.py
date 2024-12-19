import numpy as np
import matplotlib.pyplot as plt

# 生成資料集
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2], [0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2], [0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# 計算每一類的均值
m1 = np.mean(X1, axis=0, keepdims=True)
m2 = np.mean(X2, axis=0, keepdims=True)

# 計算類內散佈矩陣 (Sw)
S1 = (X1 - m1).T @ (X1 - m1)
S2 = (X2 - m2).T @ (X2 - m2)
Sw = S1 + S2

# 計算類間散佈矩陣 (Sb)
Sb = (m1 - m2).T @ (m1 - m2)

# 解最大化 J(w) 的問題，求解 Sw^-1 * Sb 的特徵值與特徵向量
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Sw) @ Sb)

# 選擇對應最大特徵值的特徵向量作為投影向量 w
w = eigvecs[:, np.argmax(eigvals)]

# 將資料投影到 w 上
X1_proj = X1 @ w
X2_proj = X2 @ w

# 繪圖：原始資料點與類別
plt.figure(figsize=(8, 6), dpi=120)
plt.scatter(X1[:, 0], X1[:, 1], color='red', alpha=0.6, label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='green', alpha=0.6, label='Class 2')

# 繪製投影向量
origin = np.mean(np.vstack([X1, X2]), axis=0)  # 投影向量的起點
plt.quiver(*origin, *w, color='blue', scale=3, width=0.005, label='Projection Vector (w)')

plt.title("LDA: Original Data with Projection Vector")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc='upper left')
plt.axis('equal')
plt.grid(True)
plt.show()

# 繪圖：投影後的資料分布
plt.figure(figsize=(8, 6), dpi=120)
plt.scatter(X1_proj, np.zeros_like(X1_proj), color='red', alpha=0.6, label='Class 1 Projection')
plt.scatter(X2_proj, np.zeros_like(X2_proj), color='green', alpha=0.6, label='Class 2 Projection')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

plt.title("LDA: Data Projection onto w")
plt.xlabel("Projection onto w")
plt.yticks([])  # 移除 y 軸標籤
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
