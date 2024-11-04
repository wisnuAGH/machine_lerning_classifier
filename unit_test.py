import unittest
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                                            self.X,
                                                                            self.y,
                                                                            test_size=0.2,
                                                                            random_state=42)

    def testPerceptron(self):
        perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
        perceptron.fit(self.X_train, self.y_train)
        predictions = perceptron.predict(self.X_test)

        # scikit-learn Perceptron
        sk_perceptron = PerceptronReference(max_iter=1000, eta0=0.01, tol=1e-3)
        sk_perceptron.fit(self.X_train, self.y_train)
        sk_predictions = sk_perceptron.predict(self.X_test)

        self.assertEqual(predictions.tolist(), sk_predictions.tolist(),
                         "Perceptron predictions differ from scikit-learn")

    def testLogisticRegression(self):
        logistic = LogisticRegressionModel(learning_rate=0.01, n_iters=1000)
        logistic.fit(self.X_train, self.y_train)
        predictions = logistic.predict(self.X_test)

        # scikit-learn Logistic Regression
        sk_logistic = LogisticRegressionReference(max_iter=1000)
        sk_logistic.fit(self.X_train, self.y_train)
        sk_predictions = sk_logistic.predict(self.X_test)

        self.assertEqual(predictions.tolist(), sk_predictions.tolist(),
                         "Logistic Regression predictions differ from scikit-learn")

    def testKnn(self):
        knn = KNNClassifier(k=3)
        knn.fit(self.X_train, self.y_train)
        predictions = knn.predict(self.X_test)

        # scikit-learn k-NN
        skKnn = KNNReference(n_neighbors=3)
        skKnn.fit(self.X_train, self.y_train)
        skPredictions = skKnn.predict(self.X_test)

        self.assertEqual(predictions.tolist(), skPredictions.tolist(), "k-NN predictions differ from scikit-learn")

    def testSvm(self):
        svm = SVMClassifier(learning_rate=0.001, n_iters=1000)
        svm.fit(self.X_train, self.y_train)
        predictions = svm.predict(self.X_test)

        # scikit-learn SVM
        skSvm = SVMReference(kernel='linear', C=1.0)
        skSvm.fit(self.X_train, self.y_train)
        skPredictions = skSvm.predict(self.X_test)

        self.assertEqual(predictions.tolist(), skPredictions.tolist(), "SVM predictions differ from scikit-learn")


if __name__ == "__main__":
    unittest.main()
