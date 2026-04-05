import matplotlib
matplotlib.use("TkAgg")

from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

data = load_diabetes()

X = data.data
y = data.target

model = SVR(kernel='linear')

y_pred = cross_val_predict(model, X, y, cv=5)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

plt.scatter(y, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("SVM Regression with Cross-Validation")
plt.show()