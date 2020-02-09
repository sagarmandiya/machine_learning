import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class MyLinearRegression:
    def __init__(self, weight=np.zeros(4), bias=0, learning_rate=0.001, iterations=1000):
        self.weight = weight
        self.bias = bias
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.cost_trend = []
        self.cost = 0

    def predict(self, X):
        X = pd.DataFrame(X)
        x1, x2, x3, x4 = X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], X.iloc[:, 3]
        predicted_set = []
        for i in range(len(x1)):
            predicted_value = (self.weight[0] * x1[i] + self.weight[1] * x2[i] +
                               self.weight[2] * x3[i] + self.weight[3] * x4[i] + self.bias)
            predicted_set.append(predicted_value)
        return predicted_set

    def cost_function(self, x1, x2, x3, x4, y):
        m = len(y)
        total_error = 0.0
        for i in range(m):
            total_error += (y[i] - (self.weight[0] * x1[i] + self.weight[1] * x2[i] +
                            self.weight[2] * x3[i] + self.weight[3] * x4[i] + self.bias)) ** 2
        return float(total_error) / (2 * m)

    def update_weights(self, x1, x2, x3, x4, y):
        D_theta = [0.0] * 4
        D_bias = 0.0
        count = len(y)
        for i in range(count):
            ypred = (self.weight[0] * x1[i] + self.weight[1] * x2[i] +
                     self.weight[2] * x3[i] + self.weight[3] * x4[i] + self.bias)
            D_theta[0] += -2 * x1[i] * (y[i] - ypred)
            D_theta[1] += -2 * x2[i] * (y[i] - ypred)
            D_theta[2] += -2 * x3[i] * (y[i] - ypred)
            D_theta[3] += -2 * x4[i] * (y[i] - ypred)
            D_bias += -2 * (y[i] - ypred)
        for i in range(4):
            self.weight[i] -= (D_theta[i] / count) * self.learning_rate
        self.bias -= (D_bias / count) * self.learning_rate

    def train(self, X, y):
        X = pd.DataFrame(X)
        x1, x2, x3, x4 = X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], X.iloc[:, 3]
        for i in range(self.iterations):
            self.update_weights(x1, x2, x3, x4, y)
            # Calculating cost
            self.cost = self.cost_function(x1, x2, x3, x4, y)
            self.cost_trend.append(self.cost)
            # if i % 10000 == 0:
            print("Iteration: {}\t Weight: {}\t Bias: {}\t Cost: {}".format(i, self.weight, self.bias, self.cost))


data = pd.read_excel('basketball.xlsx')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Fitting multivariate Linear Regression to the Training set
regressor = MyLinearRegression()
regressor.train(X_train, y_train)
print('Weight: ' + str(regressor.weight) + ' Bias: ' + str(regressor.bias))
epochs = np.arange(regressor.iterations)

plt.plot(epochs, regressor.cost_trend)
plt.xlabel('EPOCHS')
plt.ylabel('COST')
plt.show()

# Predicting the Test set results
y_pred = regressor.predict(X_test)
