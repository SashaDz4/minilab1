# Description: KNN algorithm for iris dataset
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"Predicted labels: {y_pred}")
    print(f"Accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
