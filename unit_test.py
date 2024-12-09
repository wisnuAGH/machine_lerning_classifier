import unittest
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Import reference models
from sklearn.linear_model import Perceptron as PerceptronReference
from sklearn.linear_model import LogisticRegression as LogisticRegressionReference
from sklearn.neighbors import KNeighborsClassifier as KNNReference
from sklearn.svm import SVC as SVMReference

# Import project models
from algorithms.project_perceptron import Perceptron
from algorithms.project_logistic_regression import LogisticRegressionModel
from algorithms.project_knn import KNNClassifier
from algorithms.project_svm import SVMClassifier


class TestModels(unittest.TestCase):
    def setUp(self):
        # Generate a simple dataset for classification
        self.X, self.y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=1.5, random_state=42)
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
                                                                            self.X,
                                                                            self.y,
                                                                            test_size=0.2,
                                                                            random_state=42)

    def testPerceptron(self):
        perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
        perceptron.fit(self.xTrain, self.yTrain)
        predictions = perceptron.predict(self.xTest)

        # scikit-learn Perceptron
        skPerceptron = PerceptronReference(max_iter=1000, eta0=0.01, tol=1e-3)
        skPerceptron.fit(self.xTrain, self.yTrain)
        skPredictions = skPerceptron.predict(self.xTest)

        self.assertEqual(predictions.tolist(), skPredictions.tolist(),
                         "Perceptron predictions differ from scikit-learn")

    def testLogisticRegression(self):
        logistic = LogisticRegressionModel(learning_rate=0.01, n_iters=1000)
        logistic.fit(self.xTrain, self.yTrain)
        predictions = logistic.predict(self.xTest)

        # scikit-learn Logistic Regression
        skLogistic = LogisticRegressionReference(max_iter=1000)
        skLogistic.fit(self.xTrain, self.yTrain)
        skPredictions = skLogistic.predict(self.xTest)

        self.assertEqual(predictions.tolist(), skPredictions.tolist(),
                         "Logistic Regression predictions differ from scikit-learn")

    def testKnn(self):
        knn = KNNClassifier(k=3)
        knn.fit(self.xTrain, self.yTrain)
        predictions = knn.predict(self.xTest)

        # scikit-learn k-NN
        skKnn = KNNReference(n_neighbors=3)
        skKnn.fit(self.xTrain, self.yTrain)
        skPredictions = skKnn.predict(self.xTest)

        self.assertEqual(predictions.tolist(), skPredictions.tolist(), "k-NN predictions differ from scikit-learn")

    def testSvm(self):
        svm = SVMClassifier(learning_rate=0.001, n_iters=1000)
        svm.fit(self.xTrain, self.yTrain)
        predictions = svm.predict(self.xTest)

        # scikit-learn SVM
        skSvm = SVMReference(kernel='linear', C=1.0)
        skSvm.fit(self.xTrain, self.yTrain)
        skPredictions = skSvm.predict(self.xTest)

        self.assertEqual(predictions.tolist(), skPredictions.tolist(), "SVM predictions differ from scikit-learn")


if __name__ == "__main__":
    unittest.main()
