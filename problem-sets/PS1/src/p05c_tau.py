import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path):
    """Problem 5(c): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    # Search tau_values for the best tau (lowest MSE on the validation set)
    MSE = np.zeros_like(tau_values)
    i = 0
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_eval)

        mse = np.mean((y_eval - y_pred)**2)
        print(f"tau: {tau}, MSE: {mse}")
        MSE[i] = mse
        i += 1

    # Fit a LWR model with the best tau value
    tau = tau_values[np.argmin(MSE)]
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)

    # Run on the test set to get the MSsE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_eval_pred = model.predict(x_eval)
    y_test_pred = model.predict(x_test)

    mse = np.mean((y_test - y_test_pred)**2)
    print(f"tau: {tau}, MSE on valid data: {np.min(MSE)}, MSE on test data: {mse}")

    # Plot data
    plt.figure()
    plt.plot(x_train, y_train, 'go')
    # plt.plot(x_eval, y_eval_pred, 'rx')
    plt.plot(x_test, y_test_pred, 'rx')
    plt.show()

    # *** END CODE HERE ***


if __name__ == '__main__':
    tau_values = [5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0]
    main(tau_values, "../data/ds5_train.csv", "../data/ds5_valid.csv", "../data/ds5_test.csv")
