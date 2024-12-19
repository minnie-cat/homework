import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 載入數據
file_path = 'hw5.csv'
hw5_csv_data = pd.read_csv(file_path)
hours = hw5_csv_data['Hours'].values
sulfate = hw5_csv_data['Sulfate'].values

# 1. 繪製「硫酸鹽濃度 vs 時間」圖
plt.figure(figsize=(10, 6))
plt.scatter(hours, sulfate, color='blue', label='Data')
plt.title('Sulfate Concentration vs Time')
plt.xlabel('Time (hours)')
plt.ylabel('Sulfate Concentration (x $10^{-4}$)')
plt.legend()
plt.grid(True)
plt.show()

# 2. 多項式回歸擬合
degree = 3  # 使用三次多項式回歸
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(hours.reshape(-1, 1))
model = LinearRegression()
model.fit(X_poly, sulfate)
y_pred = model.predict(X_poly)

# 繪製回歸曲線
plt.figure(figsize=(10, 6))
plt.scatter(hours, sulfate, color='blue', label='Data')
plt.plot(hours, y_pred, color='red', label=f'{degree}-Degree Polynomial Regression')
plt.title('Sulfate Concentration vs Time (Regression)')
plt.xlabel('Time (hours)')
plt.ylabel('Sulfate Concentration (x $10^{-4}$)')
plt.legend()
plt.grid(True)
plt.show()

# 3. 繪製「硫酸鹽濃度對數 vs 時間對數」圖
log_hours = np.log(hours)
log_sulfate = np.log(sulfate)

plt.figure(figsize=(10, 6))
plt.scatter(log_hours, log_sulfate, color='blue', label='Log-Log Data')
plt.title('Log(Sulfate Concentration) vs Log(Time)')
plt.xlabel('Log(Time)')
plt.ylabel('Log(Sulfate Concentration)')
plt.legend()
plt.grid(True)
plt.show()

# 4. 對數數據的線性回歸
log_model = LinearRegression()
log_model.fit(log_hours.reshape(-1, 1), log_sulfate)
log_y_pred = log_model.predict(log_hours.reshape(-1, 1))

# 繪製對數迴歸曲線
plt.figure(figsize=(10, 6))
plt.scatter(log_hours, log_sulfate, color='blue', label='Log-Log Data')
plt.plot(log_hours, log_y_pred, color='red', label='Log-Log Regression Curve')
plt.title('Log(Sulfate Concentration) vs Log(Time) (Regression)')
plt.xlabel('Log(Time)')
plt.ylabel('Log(Sulfate Concentration)')
plt.legend()
plt.grid(True)
plt.show()
