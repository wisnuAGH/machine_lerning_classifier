import numpy as np
from tqdm import tqdm
import sys

# Redirect standard output to the console
sys.stdout = sys.__stdout__


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Trenuj perceptron
        for _ in tqdm(range(self.n_iters), desc="Training Perceptron"):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predict = self.activation_func(self, linear_output)

                # Aktualizuj wagi
                update = self.lr * (y[idx] - y_predict)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predict = self.activation_func(self, linear_output)
        return y_predict

    @staticmethod
    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)
