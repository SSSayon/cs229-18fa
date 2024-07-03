import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path=None):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = clf.predict(x_eval)
    
    plt.figure()
    plt.plot(y_eval, 'go', label='label')
    plt.plot(y_pred, 'rx', label='prediction')
    plt.legend()
    plt.show()
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x : np.ndarray, y : np.ndarray):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape

        self.theta = np.zeros(n)
        delta = np.zeros(n)

        while True:

            delta = self.step_size * x.T.dot(y - np.exp(x.dot(self.theta))) / m

            self.theta += delta

            if (la.norm(delta, ord=1) < self.eps):
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***


if __name__ == "__main__":
    train_path = "../data/ds4_train.csv"
    eval_path  = "../data/ds4_valid.csv"
    main(lr=2e-7, train_path=train_path, eval_path=eval_path)
