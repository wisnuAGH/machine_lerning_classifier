from sklearn.datasets import make_blobs  # add nist input data
from sklearn.model_selection import train_test_split

# Import project models
from algorithms.project_perceptron import Perceptron
from algorithms.project_logistic_regression import LogisticRegressionModel
from algorithms.project_knn import KNNClassifier
from algorithms.project_svm import SVMClassifier

from utils.plot_2d import plotDecisionBoundaries
from utils.plot_3d import plotDecisionBoundaries3d

LEARNING_RATE = 0.01
ITERATIONS = 20000


def selectPlotType():
    while True:
        plotType = input("Choose plot type: \n Press [2] for 2D plotting or [3] for 3D.\n")
        if plotType == '2':
            return 2
        elif plotType == '3':
            return 3


def performAnalysis(plotType):
    # Generate and split data
    x, y = make_blobs(n_samples=300, centers=2, n_features=plotType, cluster_std=1.5, random_state=42)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize models
    perceptron = Perceptron(learning_rate=LEARNING_RATE, n_iters=ITERATIONS)
    logisticModel = LogisticRegressionModel(learning_rate=LEARNING_RATE, n_iters=ITERATIONS)
    knn = KNNClassifier(k=3)
    svm = SVMClassifier(learning_rate=LEARNING_RATE, n_iters=ITERATIONS)

    # Train models
    models = [perceptron, logisticModel, knn, svm]
    for model in models:
        model.fit(xTrain, yTrain)

    modelNames = ["Perceptron", "Logistic Regression", "k-NN", "SVM"]

    if plotType == 2:
        plotDecisionBoundaries(xTest, yTest, models, modelNames, "Decision Boundaries of Different Algorithms")
    elif plotType == 3:
        plotDecisionBoundaries3d(xTest, yTest, models, modelNames, "Decision Boundaries of Different Algorithms")


def run():
    plot_type = selectPlotType()
    if plot_type is not None:
        performAnalysis(plot_type)


if __name__ == "__main__":
    run()
