# 匯入模組
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 載入資料
file_path = 'hw7.csv'
hw7_data = pd.read_csv(file_path).to_numpy(dtype=np.float64)
x = hw7_data[:, 0]
y = hw7_data[:, 1]

# 初始化參數
w = np.array([-0.1607108, 2.0808538, 0.3277537, -1.5511576])
alpha = 0.05
max_iters = 500

# 定義成本函數
def cost_function(w, x, y):
    return np.sum((y - (w[0] + w[1] * np.sin(w[2] * x + w[3])))**2)

# 定義梯度計算（解析法）
def gradient_analytic(w, x, y):
    e = y - (w[0] + w[1] * np.sin(w[2] * x + w[3]))
    grad = np.zeros_like(w)
    grad[0] = -2 * np.sum(e)
    grad[1] = -2 * np.sum(e * np.sin(w[2] * x + w[3]))
    grad[2] = -2 * np.sum(e * w[1] * x * np.cos(w[2] * x + w[3]))
    grad[3] = -2 * np.sum(e * w[1] * np.cos(w[2] * x + w[3]))
    return grad

# 解析法梯度下降
w_analytic = w.copy()
for _ in range(max_iters):
    grad = gradient_analytic(w_analytic, x, y)
    w_analytic -= alpha * grad

# 定義梯度計算（數值法）
def gradient_numeric(w, x, y, epsilon=1e-5):
    grad = np.zeros_like(w)
    for i in range(len(w)):
        w1 = w.copy()
        w2 = w.copy()
        w1[i] += epsilon
        w2[i] -= epsilon
        grad[i] = (cost_function(w1, x, y) - cost_function(w2, x, y)) / (2 * epsilon)
    return grad

# 數值法梯度下降
w_numeric = w.copy()
for _ in range(max_iters):
    grad = gradient_numeric(w_numeric, x, y)
    w_numeric -= alpha * grad

# 繪製結果
xmin, xmax = np.min(x) - 1, np.max(x) + 1
ymin, ymax = np.min(y) - 1, np.max(y) + 1
xt = np.linspace(xmin, xmax, 100)
yt_analytic = w_analytic[0] + w_analytic[1] * np.sin(w_analytic[2] * xt + w_analytic[3])
yt_numeric = w_numeric[0] + w_numeric[1] * np.sin(w_numeric[2] * xt + w_numeric[3])

plt.figure(dpi=120)
plt.scatter(x, y, color='black', edgecolor='white', linewidth=0.5, s=50, label='Data')
plt.plot(xt, yt_analytic, color='blue', linewidth=2, label='Analytic Method')
plt.plot(xt, yt_numeric, color='red', linestyle='--', linewidth=2, label='Numeric Method')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True)
plt.title('Gradient Descent: Analytic vs Numeric')
plt.show()
