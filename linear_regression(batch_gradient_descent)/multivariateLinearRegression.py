import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def cost(X, y, thetas):
    m = len(y)
    J = (np.sum(y - X.dot(thetas) ** 2)) / (2 * m)
    return J


def batchGR(X, y, thetas, lr, epochs):
    costHistory = [0] * epochs
    m = len(y)

    for i in range(epochs):
        h = X.dot(thetas)
        loss = h - y
        gradient = X.T.dot(loss) / m
        thetas = thetas - (lr * gradient)
        newcost = cost(X, y, thetas)
        costHistory[i] = newcost
        # print('Thetas: ', thetas, 'Cost: ', newcost)
    return thetas, costHistory


data = pd.read_excel('basketball.xlsx')
X = data.iloc[:, :4]
y = data.iloc[:, -1]

sc = StandardScaler()
X = sc.fit_transform(X)

thetas = np.zeros(X.shape[1])
epochs = 10000
lr = 0.003

newThetas, costHistory = batchGR(X, y, thetas, lr, epochs)
print('Thetas: ', newThetas, ' Cost: ', costHistory[-1])

itr = np.arange(epochs)
plt.plot(itr, costHistory)
plt.xlabel('Epochs ')
plt.ylabel('Cost ')
plt.show()
