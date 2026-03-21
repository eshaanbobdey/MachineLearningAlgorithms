import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

accuracy_results = {}
precision_results = {}
recall_results = {}
f1_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy_results[name] = accuracy_score(y_test, pred)
    precision_results[name] = precision_score(y_test, pred)
    recall_results[name] = recall_score(y_test, pred)
    f1_results[name] = f1_score(y_test, pred)

print("Accuracy")
for k, v in accuracy_results.items():
    print(k, ":", v)

print("\nPrecision")
for k, v in precision_results.items():
    print(k, ":", v)

print("\nRecall")
for k, v in recall_results.items():
    print(k, ":", v)

print("\nF1 Score")
for k, v in f1_results.items():
    print(k, ":", v)

labels = list(models.keys())
x = range(len(labels))

plt.figure(figsize=(10,6))
plt.plot(x, accuracy_results.values(), marker='o', label="Accuracy")
plt.plot(x, precision_results.values(), marker='o', label="Precision")
plt.plot(x, recall_results.values(), marker='o', label="Recall")
plt.plot(x, f1_results.values(), marker='o', label="F1 Score")

plt.xticks(x, labels, rotation=30)
plt.ylabel("Score")
plt.title("Model Comparison")
plt.legend()
plt.show()