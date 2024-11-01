# Simple Machine Learning Classifier

This project includes the implementation of basic classification algorithms in Python: Perceptron, Logistic Regression, k-NN, and SVM. The goal is to understand these algorithms' mechanics and compare their performance with the `scikit-learn` library.

## Contents
- **Project Description**: Overview of implemented algorithms and their function.
- **Algorithms**:
  - **Perceptron**: A simple binary linear classifier adjusting weights iteratively based on error.
  - **Logistic Regression**: A linear classifier using the sigmoid function, optimized via gradient descent.
  - **k-NN (k-Nearest Neighbors)**: A non-linear classifier labeling based on neighbors' votes.
  - **SVM (Support Vector Machine)**: A linear classifier maximizing the margin between classes.
- **Requirements**:
  - Python 3.7+
  - Libraries: `numpy`, `matplotlib`, `scikit-learn`, `tqdm`

## Running the Project
1. Ensure dependencies are installed.
2. Run the main script to train algorithms and view comparative decision boundary visualizations.

## Comparison with `scikit-learn`
After training custom models, compare their performance with `scikit-learn`'s built-in implementations to validate accuracy and training time.

## Unit Testing
Unit tests verify the custom models:
1. **fit method**: Checks for proper weight initialization and updates.
2. **predict method**: Ensures predictions align with test data.
3. **Accuracy vs. `scikit-learn`**: Tests if model accuracy is comparable to `scikit-learn` versions.

## Visualization
A decision boundary plot for each model shows class regions, helping to visually compare each algorithm's classification behavior.

--- 

### Summary
This project provides a foundational understanding of classification algorithms and demonstrates model-building in Python, complemented by comparisons with `scikit-learn`.
