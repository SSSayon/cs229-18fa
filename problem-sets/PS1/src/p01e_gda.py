import numpy as np
import numpy.linalg as la
import util

from linear_model import LinearModel
from p01b_logreg import LogisticRegression


def main(train_path, eval_path, pred_path=None):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)

    x2_train, y2_train = util.load_dataset(train_path, add_intercept=True)
    clf2 = LogisticRegression()
    clf2.fit(x2_train, y2_train)

    util.plot(x_train, y_train, clf.theta, clf2.theta)

    # x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    # util.plot(x_eval, y_eval, clf.theta)

    # y_pred = clf.predict(x_eval)
    # print(y_pred[:3])
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x : np.ndarray, y : np.ndarray):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n + 1)

        phi   = 1/m * np.sum(y == 1) 
        mu_0  = (np.sum(x[y == 0], axis=0)) / (np.sum(y == 0)) 
        mu_1  = (np.sum(x[y == 1], axis=0)) / (np.sum(y == 1))
        sigma = 1/m * ((x[y == 0] - mu_0).T.dot(x[y == 0] - mu_0) 
                     + (x[y == 1] - mu_1).T.dot(x[y == 1] - mu_1))
        sigma_inv = la.inv(sigma)

        self.theta[0]  = 0.5 * (mu_0.T.dot(sigma_inv).dot(mu_0) 
                              - mu_1.T.dot(sigma_inv).dot(mu_1)) - np.log((1 - phi) / phi)
        self.theta[1:] = sigma_inv.dot(mu_1 - mu_0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE


if __name__ == "__main__":
    train_path = "../data/ds1_train.csv"
    eval_path  = "../data/ds1_valid.csv"
    main(train_path, eval_path)
