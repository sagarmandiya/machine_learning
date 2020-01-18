import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams['figure.figsize'] = (12.0, 7.2)

# Preprocessing input data
data = pd.read_csv('ex1data1.txt')
x = data.iloc[:, 0]
y = data.iloc[:, 1]

# Building the model
theta0 = 0
theta1 = 0
alpha = 0.003
epoche = 10000
n = len(x)

for i in range(epoche):
    ypred = theta0 + theta1 * x
    D_theta0 = (-2 * sum(y - ypred)) / n
    D_theta1 = (-2 * sum(x * (y - ypred))) / n
    theta0 = theta0 - D_theta0 * alpha
    theta1 = theta1 - D_theta1 * alpha

print(theta0, theta1)

# Making Predictions

ypred = theta0 + x * theta1
plt.scatter(x, y)
plt.plot([min(x), max(x)], [min(ypred), max(ypred)], color='red')
plt.show()
