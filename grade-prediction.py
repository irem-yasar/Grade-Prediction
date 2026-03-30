import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Veri (daha gerçekçi)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([35, 45, 55, 65, 70, 78, 85, 90])

# Model
model = LinearRegression()
model.fit(X, y)

# Tahminler
predictions = model.predict(X)

# Yeni tahmin (9 saat çalışan biri)
new_prediction = model.predict([[9]])

# Hata hesaplama
mae = mean_absolute_error(y, predictions)

print("Predicted score for 9 hours:", new_prediction[0])
print("Mean Absolute Error:", mae)

# Grafik
plt.scatter(X, y, color='blue', label='Real Data')
plt.plot(X, predictions, color='red', label='Model Prediction')
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Student Performance Prediction")
plt.legend()
plt.show()