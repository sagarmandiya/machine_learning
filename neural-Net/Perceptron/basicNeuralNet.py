import numpy as np
import matplotlib.pyplot as plt


# Class definition
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)  # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = self.sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    # Activation function
    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def sigmoid_derivative(self, p):
        return p * (1 - p)

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * self.sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * self.sigmoid_derivative(self.output),
                                                      self.weights2.T) * self.sigmoid_derivative(self.layer1))
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()


# Each row is a training example, each column is a feature  [X1, X2, X3]
X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
y = np.array(([0], [1], [1], [0]), dtype=float)

cost = []
itr = 1500
epochs = np.arange(itr)
NN = NeuralNetwork(X, y)
for i in range(itr):  # trains the NN 1,500 times
    print("for iteration # " + str(i) + "\n")
    print("Input : \n" + str(X))
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n" + str(NN.feedforward()))
    loss = np.mean(np.square(y - NN.feedforward()))
    cost.append(loss)
    print("Loss: \n" + str(loss))  # mean sum squared loss
    print("\n")
    NN.train(X, y)

plt.plot(epochs, cost)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()
