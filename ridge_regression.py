import numpy as np
import matplotlib.pyplot as plt

from utils import load_dataset, problem


@problem.tag("hw1-A")
def train(x: np.ndarray, y: np.ndarray, _lambda: float) -> np.ndarray:
    """Train function for the Ridge Regression problem.
    Should use observations (`x`), targets (`y`) and regularization parameter (`_lambda`)
    to train a weight matrix $$\\hat{W}$$.


    Args:
        x (np.ndarray): observations represented as `(n, d)` matrix.
            n is number of observations, d is number of features.
        y (np.ndarray): targets represented as `(n, k)` matrix.
            n is number of observations, k is number of classes.
        _lambda (float): parameter for ridge regularization.

    Raises:
        NotImplementedError: When problem is not attempted.

    Returns:
        np.ndarray: weight matrix of shape `(d, k)`
            which minimizes Regularized Squared Error on `x` and `y` with hyperparameter `_lambda`.
    """
    reg_matrix = _lambda * np.eye(x.shape[1])
    weight = np.linalg.solve(x.T @ x + reg_matrix, x.T @ y)
    return weight


    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw1-A")
def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Train function for the Ridge Regression problem.
    Should use observations (`x`), and weight matrix (`w`) to generate predicated class for each observation in x.

    Args:
        x (np.ndarray): observations represented as `(n, d)` matrix.
            n is number of observations, d is number of features.
        w (np.ndarray): weights represented as `(d, k)` matrix.
            d is number of features, k is number of classes.

    Raises:
        NotImplementedError: When problem is not attempted.

    Returns:
        np.ndarray: predictions matrix of shape `(n,)` or `(n, 1)`.
    """
    ypred = np.argmax((w.T).dot(x.T), axis =0)
    return ypred
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw1-A")
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """One hot encode a vector `y`.
    One hot encoding takes an array of integers and coverts them into binary format.
    Each number i is converted into a vector of zeros (of size num_classes), with exception of i^th element which is 1.

    Args:
        y (np.ndarray): An array of integers [0, num_classes), of shape (n,)
        num_classes (int): Number of classes in y.

    Returns:
        np.ndarray: Array of shape (n, num_classes).
        One-hot representation of y (see below for example).

    Example:
        ```python
        > one_hot([2, 3, 1, 0], 4)
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        ```
    """
    
    y_hotencoded = np.zeros((y.size,num_classes))
    y_hotencoded[np.arange(y.size), y] = 1

    return y_hotencoded
    

    raise NotImplementedError("Your Code Goes Here")


def main():

    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    # Convert to one-hot
    y_train_one_hot = one_hot(y_train.reshape(-1), 10)

    _lambda = 1e-4

    w_hat = train(x_train, y_train_one_hot, _lambda)

    y_train_pred = predict(x_train, w_hat)
    y_test_pred = predict(x_test, w_hat)

    print("Ridge Regression Problem")
    print(
        f"\tTrain Error: {np.average(1 - np.equal(y_train_pred, y_train)) * 100:.6g}%"
    )
    print(f"\tTest Error:  {np.average(1 - np.equal(y_test_pred, y_test)) * 100:.6g}%")
    
    
    #printing first 15 incorrect predictions and showing the images
    incorrect_labels = y_test_pred[np.where(y_test_pred!=y_test)][0:15]
    correct_labels = y_test[np.where(y_test_pred!=y_test)][0:15]
    incorrect_xtest = x_test[np.where(y_test_pred!=y_test)][0:15]
    reshape_Xtest = np.reshape(incorrect_xtest,(15,28,28))
    rows = 3
    columns = 5
    fig = plt.figure()

    for i in range(0,15):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(reshape_Xtest[i])
        plt.axis('off')
        plt.title(f"ylabel = {correct_labels[i]},\n ypred = {incorrect_labels[i]}")
    plt.show()

    


if __name__ == "__main__":
    main()
