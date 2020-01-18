def gradientDescent(x, y, n):
    theta0, theta1 = 0, 0
    alpha = 0.001
    itr = 10
    dtheta0 = 0
    dtheta1 = 0
    print("theta0\ttheta1\tcost")
    for i in range(itr):
        error = 0
        for j in range(n):
            error += ((y[j] - (theta0 + theta1 * x[j])) ** 2)
        cost = error / (2 * n)
        for j in range(n):
            dtheta1 += -2 * x[j] * (y[j] - (theta0 + (theta1 * x[j])))
            dtheta0 += -2 * (y[j] - (theta0 + (theta1 * x[j])))
        theta0 -= alpha * (dtheta0 / n)
        theta1 -= alpha * (dtheta1 / n)
        print("{:.3f}".format(theta0), "\t", "{:.3f}".format(theta1), "\t","{:.3f}".format(cost))


x = [1, 2, 3, 4, 5]
y = [6, 7, 8, 9, 10]
gradientDescent(x, y, len(x))
