import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = load_diabetes()

X = data.data[:, np.newaxis, 2]
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficient (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("BMI Feature")
plt.ylabel("Disease Progression")
plt.title("Linear Regression on Diabetes Dataset")
plt.show()