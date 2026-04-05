import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1', 'v2']]
data.columns = ['label', 'text']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['text']
y = data['label']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.bar(["Accuracy","Precision","Recall","F1"], [accuracy, precision, recall, f1])
plt.title("SVM Spam Detection")
plt.ylabel("Score")
plt.show()