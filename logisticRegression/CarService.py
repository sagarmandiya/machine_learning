import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def loadData(name):
    data = pd.read_csv(name)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
    return X_train, X_test, y_train, y_test


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros([dim, 1])
    b = 0

    assert (w.shape == (dim, 1))

    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -np.sum((Y * np.log(A) + ((1 - Y) * (np.log(1 - A))))) / m  # compute cost using cross-entropy

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (np.dot(X, (A - Y).T)) / m
    db = (np.sum(A - Y)) / m

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i%100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
        pass

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


name = 'serviceTestData.csv'
X_train, X_test, y_train, y_test = loadData(name)

# normalize dataset
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

m_train = X_train.shape[0]
m_test = X_test.shape[0]
num_px = m_test + m_train

dict = model(X_train.T, y_train, X_test.T, y_test, num_iterations=1000, learning_rate=0.5, print_cost=False)
cost = dict['costs']
epochs = np.arange(dict['num_iterations'] / 100)

plt.plot(epochs, cost)
plt.xlabel('EPOCHS')
plt.ylabel('COST')
plt.show()
