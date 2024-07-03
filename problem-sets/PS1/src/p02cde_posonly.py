import numpy as np
import util

from p01b_logreg import LogisticRegression

def main(train_path, valid_path, test_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
    """

    # *** START CODE HERE ***
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    _,       y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)

    x_valid, t_valid = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    _,       y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)

    x_test,  t_test  = util.load_dataset(test_path,  label_col='t', add_intercept=True)
    _,       y_test  = util.load_dataset(test_path,  label_col='y', add_intercept=True)

    # Part (c): Train and test on true labels
    clf_true = LogisticRegression()
    clf_true.fit(x_train, t_train)
    # util.plot(x_test, t_test, clf_true.theta)

    # Part (d): Train on y-labels and test on true labels
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    # util.plot(x_test, t_test, clf.theta)

    # Part (e): Apply correction factor using validation set and test on true labels
    alpha = np.average(clf.predict(x_valid)[y_valid == 1])

    # h_modified = h / alpha, where h = 1/(1 + exp(theta dot x))
    # letting h_modified = 1/2 to get the classifying line
    # ==> (theta_0 + log(2 / alpha - 1)) + theta_1 x_1 + theta_2 x_2 = 0
    util.plot(x_test, t_test, clf.theta, clf_true.theta, correction=1+np.log(2/alpha-1)/clf.theta[0])

    # *** END CODER HERE ***

if __name__ == '__main__':
    main(train_path='../data/ds3_train.csv', valid_path='../data/ds3_valid.csv', test_path='../data/ds3_test.csv')
