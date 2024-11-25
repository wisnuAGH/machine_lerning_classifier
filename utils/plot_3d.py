from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt  # try cborn dictionary
import numpy as np
from matplotlib.colors import ListedColormap


def plotDecisionBoundaries3d(X, y, models, model_names, title):
    """
    Plot decision boundaries for 3D data.

    Parameters:
    - X (np.array): Feature matrix (3D data).
    - y (np.array): Target vector.
    - models (list): List of trained model instances with a `predict` method.
    - model_names (list): Names of the models, in the same order as `models`.
    - title (str): Title of the plot.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the data points
    cmap_points = ListedColormap(['#FF0000', '#0000FF'])
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cmap_points, edgecolor='k')

    # Define grid for decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.5),
        np.arange(y_min, y_max, 0.5)
    )

    colors = ['r', 'g', 'b', 'y']  # Colors for models

    for model, name, color in zip(models, model_names, colors):
        # Predict values for the grid
        zz = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = model.predict(np.array([[xx[i, j], yy[i, j], (z_min + z_max) / 2]]))[0]

        # Plot decision boundary surface
        ax.plot_surface(xx, yy, zz, color=color, alpha=0.3)

    ax.set_title(title)
    ax.set_xlabel("OX axis")
    ax.set_ylabel("OY axis")
    ax.set_zlabel("OZ axis")
    plt.show()
