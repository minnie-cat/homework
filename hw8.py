# 匯入必要模組
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC  # 支持向量機 (SVM)

# 載入資料
file_path = 'hw8.csv'
hw8_dataset = pd.read_csv(file_path).to_numpy(dtype=np.float64)

X = hw8_dataset[:, 0:2]  # 特徵
y = hw8_dataset[:, 2]    # 標籤

# 訓練 SVM 分類器
svm_model = SVC(kernel='linear')  # 使用線性核
svm_model.fit(X, y)

# 建立網格以繪製分類邊界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# 預測網格上的點
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 圖 1：資料點分布
plt.figure(figsize=(8, 6), dpi=120)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='$\omega_1$', edgecolor='k')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', label='$\omega_2$', edgecolor='k')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Data Distribution')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()

# 圖 2：資料點與分類邊界
plt.figure(figsize=(8, 6), dpi=120)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)  # 繪製分類邊界
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='$\omega_1$', edgecolor='k')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', label='$\omega_2$', edgecolor='k')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('SVM Classification and Decision Boundary')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()
