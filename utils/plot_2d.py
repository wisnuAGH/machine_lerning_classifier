import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_boundaries(X, y, models, model_names, title):
    """
    Function provide plotting decision boundaries
    Parameters:
    - X (np.array): Feature matrix.
    - y (np.array): Target vector.
    - models (list): List of trained model instances with a `predict` method.
    - model_names (list): Names of the models, in the same order as `models`.
    - title (str): Title of the plot.
    """

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
