import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    # Get MSE value on the validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    print("MSE:", np.mean((y_eval - y_pred)**2))

    # Plot validation predictions on top of training set
    plt.figure()
    plt.plot(x_train, y_train, 'go')
    plt.plot(x_eval, y_pred, 'rx')
    plt.show()

    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x : np.ndarray):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        y_pred = np.zeros(m)

        for i in range(m):
            W = np.diag(np.exp(-np.sum((self.x - x[i]) ** 2, axis=1) / (2 * self.tau * self.tau)))
            theta = la.inv(self.x.T @ W @ self.x) @ self.x.T @ W @ self.y
            y_pred[i] = x[i] @ theta
    
        return y_pred
        # *** END CODE HERE ***


if __name__ == '__main__':
    main(0.5, "../data/ds5_train.csv", "../data/ds5_valid.csv")
