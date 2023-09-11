import numpy as np
import matplotlib.pyplot as plt


def plot_svm_decision_boundary(X, Y, w):
    plt.scatter(np.array(X[:, 0]), np.array(X[:, 1]), c=Y)

    # Create the hyperplane
    a = -w[0] / w[1]
    xx = np.linspace(0, 20)
    yy = a * xx

    # Plot the hyperplane
    plt.plot(xx, yy, color='red')

    # Plot support vectors
    plt.plot(xx, (yy + 1), linestyle='--', color='blue')
    plt.plot(xx, (yy - 1), linestyle='--', color='blue')

    plt.show()


def linear_binary_class(n, low_D, high_D, m, q):
    X = np.zeros((n, 2))
    Y = np.zeros(n)
    for i in range(2):
        X[:, i] = np.random.uniform(low_D, high_D, size=n)

    Y[X[:, 1] - (X[:, 0] * m + q) > 0] = 1
    Y[X[:, 1] - (X[:, 0] * m + q) < 0] = -1

    return X, Y


def svm_predict(x_train, y_train, x_test, epochs=100000, reg_param=0.01):
    n_train, d_train = np.shape(x_train)

    if any(np.abs(y_train) != 1):
        print("The values of Ytrain should be +1 or -1.")
        return -1

    w = np.zeros(d_train)
    for epoch in range(1, epochs):
        learning_rate = 1 / epoch
        for i, x in enumerate(x_train):
            if (y_train[i] * np.dot(x_train[i], w)) < 1:
                w = (1 - learning_rate) * w + learning_rate * reg_param * y_train[i] * x_train[i]
            else:
                w = (1 - learning_rate) * w
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} | w: {w}")

    Ypred = np.sign(x_test @ w)
    return Ypred, w


if __name__ == '__main__':
    n = 150
    x, y = linear_binary_class(int(n * 1.5), 0, 20, 0.8, 0)
    x_train_data, x_test_data = x[:n], x[n:]
    y_train_data, y_test_data = y[:n], y[n:]
    y_pred, w = svm_predict(x_train_data, y_train_data, x_test_data, epochs=10000, reg_param=25)
    print(f"\nPredicted labels: {y_pred}")
    print(f"Weight vector: {w}")
    plot_svm_decision_boundary(x_train_data, y_train_data, w)



