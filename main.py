from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Import project models
from algorithms.project_perceptron import Perceptron
from algorithms.project_logistic_regression import LogisticRegressionModel
from algorithms.project_knn import KNNClassifier
from algorithms.project_svm import SVMClassifier

from utils.plot_2d import plot_decision_boundaries


def run():
    # Generate and split data
    X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=1.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
    logistic_model = LogisticRegressionModel(learning_rate=0.01, n_iters=1000)
    knn = KNNClassifier(k=5)
    svm = SVMClassifier(learning_rate=0.001, n_iters=1000)

    # Train models
    models = [perceptron, logistic_model, knn, svm]
    for model in models:
        model.fit(X_train, y_train)

    model_names = ["Perceptron", "Logistic Regression", "k-NN", "SVM"]
    plot_decision_boundaries(X_test, y_test, models, model_names, "Decision Boundaries of Different Algorithms")


if __name__ == "__main__":
    run()
