#Current Goals: figure out how to strcuture data array, implement generation of that structure using a sigmoid function or other random spread method, 
# and reimplement sigmoid(), cost(), deltax0(), and deltak() with new structure in mind. 

import matplotlib.pyplot as plt
import numpy as np
import sys


datasize, k, x0, data, xs= 0, 0, 0, [], []
xmax, xmin = 0, 0
learnrates = open("./LearnRates.txt", 'w+')

def generate_data(datasize, truek, truex0, xminInput, xmaxInput):
    global data, xs
    if (datasize is None):
        datasize=int(input("Enter the size of the dataset: "))
    if(truek is None):
        truek=float(input("Enter k value (small values = smoother transition): "))
    if(truex0 is None):
        truex0=float(input("Enter x0 value (x0 = the 'midpoint' of the data): "))
    if(xminInput is None):
        xmin = int(input("Enter the lower bound of the data: "))
    else: xmin = xminInput
    if(xmaxInput is None):
        xmax = int(input("Enter the upper bound of the data: "))
    else: xmax = xmaxInput
    xs = np.linspace(xmin, xmax, datasize)
    data = np.zeros_like(xs)
    probabilities = np.reciprocal(1+np.exp(-truek*(xs-truex0)))
    for i in range(0, datasize):
        # transition probability based on sigmoid, but weighted by previous state
        p = probabilities[i] * 0.7 + data[i-1] * 0.3
        data[i] = np.random.binomial(1, p)
        print(f"[{data[i]}, {p}]")
    return data


generate_data(50, .5, 5, -5, 15)
print(f"Data: {data}")

#set up graph and plot data
plt.scatter(xs, data, label='Random Data Points', color='blue')
plt.ion()
plt.title('Logistic fit to random data')  
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)


#calculates prediction value of a logistic best fit line at i index of data and k/x0 values
def sigmoid(i, k, x0):
    prediction = np.reciprocal(1 + np.exp(-k * (xs[i]-x0)))
    return np.clip(prediction, 1e-12, 1-1e-12)


#calculates mean squared error for a prediction and target array
def cost(predictions, targets):
    return np.mean((predictions - targets)**2)


initialCost = cost(np.array([sigmoid(i, k, x0) for i in range(len(data))]), data)
print("Initial Cost: " + str(initialCost))


#Partial derivative of cost with respect to k
def deltak():
    # return 0
    dk = 0
    for j in range(len(data)):
        y_hat = sigmoid(j, k, x0)
        y = data[j]

        dk += (y_hat - y) * y_hat * (1 - y_hat) * (xs[j] - x0)
    return 2*dk/len(data)

#Partial derivative of cost with respect to x0
def deltax0():
    # return 0
    dx0 = 0
    for j in range(len(data)):
        y_hat = sigmoid(j, k, x0)
        y = data[j]

        dx0 += (y_hat - y) * y_hat * (1 - y_hat) * k
    return -2*dx0/len(data)

#perform gradient descent with 
def grad_descent(): 
    global k, x0
    learnRateK = 0.03
    learnRateX0 = 0.5
    dcost = 1
    iters = 0 
    #Creating best-fit line
    y_fit = np.array([sigmoid(i, k, x0) for i in range(len(data))])

    line, = plt.plot(xs, y_fit, label = "Best fit line", color = "orange", linestyle ="--")
    while np.abs(dcost) > 1e-6 and iters < 1000:
        newK = k - learnRateK * deltak()
        newX0 = x0 - learnRateX0 * deltax0()
        dcost = cost(np.array([sigmoid(i, k, x0) for i in range(len(data))]), data)-cost(np.array([sigmoid(i, newK, newX0) for i in range(len(data))]), data)
        k, x0 = newK, newX0
        
        y_fit = np.array([sigmoid(i, k, x0) for i in range(len(data))])
        line.set_ydata(y_fit)
        plt.pause(0.01)  # This updates the plot

        print(f"x0: {x0}")
        print(f"k: {k}")
        print(f"cost: {str(cost(np.array([sigmoid(i, k, x0) for i in range(len(data))]), data))}")
        print(f"dCost: {dcost}")
        print(f"iteration: {iters}")
        sys.stdout.flush()
        if(abs(dcost) > 1e-6):
            print("\033[{}A".format(5), end='')  # Move cursor up 5 lines

        learnRateK *= 0.999
        learnRateX0 *= 0.999
        iters += 1


    return

plt.show(block=False)
grad_descent()
plt.show(block=True)