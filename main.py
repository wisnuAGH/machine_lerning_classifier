from sklearn.datasets import make_blobs  # add nist input data
from sklearn.model_selection import train_test_split

# Import project models
from algorithms.project_perceptron import Perceptron
from algorithms.project_logistic_regression import LogisticRegressionModel
from algorithms.project_knn import KNNClassifier
from algorithms.project_svm import SVMClassifier

from utils.plot_2d import plotDecisionBoundaries
from utils.plot_3d import plotDecisionBoundaries3d

FEATURES_NUMBER = 3


def run():
    # Generate and split data
    x, y = make_blobs(n_samples=300, centers=2, n_features=FEATURES_NUMBER, cluster_std=1.5, random_state=42)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize models
    perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
    logisticModel = LogisticRegressionModel(learning_rate=0.01, n_iters=1000)
    knn = KNNClassifier(k=10)
    svm = SVMClassifier(learning_rate=0.01, n_iters=1000)

    # Train models
    models = [perceptron, logisticModel, knn, svm]
    for model in models:
        model.fit(xTrain, yTrain)

    modelNames = ["Perceptron", "Logistic Regression", "k-NN", "SVM"]
    # plotDecisionBoundaries(xTest, yTest, models, modelNames, "Decision Boundaries of Different Algorithms")
    plotDecisionBoundaries3d(xTest, yTest, models, modelNames, "Decision Boundaries of Different Algorithms")


if __name__ == "__main__":
    run()
