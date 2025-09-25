#Current Goals: figure out how to strcuture data array, implement generation of that structure using a sigmoid function or other random spread method, 
# and reimplement sigmoid(), cost(), deltax0(), and deltak() with new structure in mind. 

import matplotlib.pyplot as plt
import numpy as np
import sys


datasize, k, x0, data, xs= 0, 0, 0, [], []
xmax, xmin = 0, 0
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


generate_data(40, 0.4, 7, -10, 10)
print(f"Data: {data}")

plt.scatter(xs, data, label='Random Data Points', color='blue')
plt.ion()
plt.title('Logistic fit to random data')  
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)


def sigmoid(i, k, x0):
    prediction = np.reciprocal(1 + np.exp(-k * (xs[i]-x0)))
    return prediction


def cost(predictions, targets):
    mean_squared_error = 0
    for p, t in zip(predictions, targets):
        mean_squared_error += (p - t) ** 2
    return mean_squared_error/len(data)


initialCost = cost(np.array([sigmoid(i, k, x0) for i in range(len(data))]), data)
print("Initial Cost: " + str(initialCost))


def deltak():
    # return 0
    dk = 0
    for j in range(len(data)):
        
        dk += (data[j]-sigmoid(j, k, x0))*np.reciprocal(sigmoid(j, k, x0)**2)*(xs[j]-x0)*np.exp(-k*(xs[j]-x0)) #check math
    return -2*dk/len(data)

def deltax0():
    # return 0
    dx0 = 0
    for j in range(len(data)):
        dx0 += (data[j]-sigmoid(j, k, x0))*np.reciprocal(sigmoid(j, k, x0)**2)*k*np.exp(-k*(xs[j]-x0)) #check math
    return 2*dx0/len(data)

def grad_descent():
    global k, x0
    learnRateK = 0.00001
    learnRateX0 = 0.1
    dcost = 1

    #Creating best-fit line
    y_fit = np.array([sigmoid(i, k, x0) for i in range(len(data))])

    line, = plt.plot(xs, y_fit, label = "Best fit line", color = "orange", linestyle ="--")
    while np.abs(dcost) > 1e-6:
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
        sys.stdout.flush()
        if(abs(dcost) > 1e-6):
            print("\033[{}A".format(4), end='')  # Move cursor up 4 lines


    return


plt.show(block=False)
grad_descent()
plt.show(block=True)