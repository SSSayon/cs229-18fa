import numpy as np
import numpy.linalg as la
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path = None):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    util.plot(x_eval, y_eval, clf.theta)

    y_pred = clf.predict(x_eval)
    print(y_pred[:3])
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x : np.ndarray, y : np.ndarray):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape                  # m: #data
                                        # n: #params
        self.theta = np.zeros(n)
        delta = np.zeros(n)

        # h(x) = h_theta(x) = sigmoid(theta dot x)
        # Grad              = 1/m * x.T dot (h(x) - y) [ = 1/m * x.T * (h(x) - y) dot ones() ]
        # Hessian           = 1/m * x.T * h(x) * (1 - h(x)) dot x
        while True:   

            h       = 1 / (1 + np.exp(-x.dot(self.theta)))
            grad    = 1/m * x.T.dot(h - y)
            hessian = 1/m * (x.T * h * (1 - h)).dot(x)   
            delta   = la.inv(hessian).dot(grad)

            self.theta -= self.step_size * delta

            if (la.norm(delta, ord=1) < self.eps):
                break

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
        # *** END CODE HERE ***


if __name__ == "__main__":
    train_path = "../data/ds2_train.csv"
    eval_path  = "../data/ds2_valid.csv"
    main(train_path, eval_path)
