import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("student_data.csv")

X = data[['StudyHours']]
y = data['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Coefficient (m):", model.coef_[0][0])
print("Intercept (c):", model.intercept_[0])

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

X_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_values = model.predict_proba(X_values)[:, 1]

plt.figure()
plt.scatter(X, y)
plt.plot(X_values, y_values)
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression: Pass/Fail Prediction")
plt.show()