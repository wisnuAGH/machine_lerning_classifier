import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_predict = [self._predict(x) for x in X]
        return np.array(y_predict)

    def _predict(self, x):
        # Oblicz odległość euklidesową między x a wszystkimi danymi treningowymi
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        # Pobierz k najbliższych sąsiadów
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Głosowanie: wybierz klasę, która występuje najczęściej
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
