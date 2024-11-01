import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from project_perceptron import Perceptron
from project_logistic_regression import LogisticRegressionModel
from project_knn import KNNClassifier
from project_svm import SVMClassifier


def plot_decision_boundaries(X, y, models, model_names, title):
    """Function provide plotting decision boundaries"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    fig, ax = plt.subplots(figsize=(8, 6))

    cmap_points = ListedColormap(['#000000', '#FFFFFF'])

    colors = ['r', 'g', 'b', 'y']  # Add more colors if needed

    for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contour(xx, yy, Z, colors=color, levels=[-1, 0, 1], alpha=1)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolor='k', marker='o')

    ax.set_title(title)
    ax.legend(handles=[plt.Line2D([], [], color=color, label=name) for color, name in zip(colors, model_names)])
    plt.show()


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
