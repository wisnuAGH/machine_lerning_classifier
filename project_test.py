from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from algorithms.project_perceptron import Perceptron
from algorithms.project_logistic_regression import LogisticRegressionModel
from algorithms.project_knn import KNNClassifier
from algorithms.project_svm import SVMClassifier
from utils.plot_2d import plot_decision_boundaries


# Przygotowanie danych (X i y to dane, które wcześniej zostały wygenerowane np. za pomocą make_blobs)
X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=1.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Perceptron
perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
perceptron.fit(X_train, y_train)

# 2. Logistic Regression
logistic_model = LogisticRegressionModel(learning_rate=0.01, n_iters=1000)
logistic_model.fit(X_train, y_train)

# 3. k-NN
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)

# 4. SVM
svm = SVMClassifier(learning_rate=0.001, n_iters=1000)
svm.fit(X_train, y_train)

# Modele do porównania
models = [perceptron, logistic_model, knn, svm]
model_names = ["Perceptron", "Regresja Logistyczna", "k-NN", "SVM"]

# Rysowanie granic decyzyjnych
plot_decision_boundaries(X_test, y_test, models, model_names, "Granice decyzyjne różnych algorytmów")
