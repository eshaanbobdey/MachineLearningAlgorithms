import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = load_breast_cancer()

X = data.data
y = data.target

model = SVC(kernel='linear')

y_pred = cross_val_predict(model, X, y, cv=5)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

plt.bar(["Accuracy","Precision","Recall","F1"], [accuracy, precision, recall, f1])
plt.title("SVM Cross-Validation Performance")
plt.ylabel("Score")
plt.show()