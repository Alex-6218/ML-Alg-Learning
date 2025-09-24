import matplotlib.pyplot as plt
import numpy as np
import sys


k, x0, datasize, data= 1, 0.5, 0, []

def generate_data(datasize=int(input("Enter the size of the dataset: ")), 
                  truek=float(input("Enter k value (small values = smoother transition): ")), 
                  truex0=float(input("Enter x0 value (x0 = the 'midpoint' of the data): ")), 
                  xmin = float(input("Enter the lower bound of the data: ")), 
                  xmax = float(input("Enter the upper bound of the data: "))):
    global data
    x = np.linspace(xmin, xmax, datasize)
    data = np.linspace(datasize)
    probabilities = np.reciprocal(1+np.exp(-truek*(x-truex0)))
    for i in range(1, datasize):
        # transition probability based on sigmoid, but weighted by previous state
        p = probabilities[i] * 0.7 + data[i-1] * 0.3
        data[i] = np.random.binomial(1, p)
    return data


generate_data()
print(f"Data: {data}")

plt.scatter(np.array([i/(len(data)-1) for i in range(len(data))]), data, label='Random Data Points', color='blue')
plt.ion()
plt.title('Logistic fit to random data')  
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)


def sigmoid(i, k, x0):
    xterm = i/(len(data)-1)-x0
    prediction = np.reciprocal(1 + np.exp(-k * xterm))
    return prediction


def cost(predictions, targets):
    mean_squared_error = 0
    for p, t in zip(predictions, targets):
        mean_squared_error += (p - t) ** 2
    return mean_squared_error/len(data)


initialCost = cost(np.array([sigmoid(i, k, x0) for i in range(len(data))]), data)
print("Initial Cost: " + str(initialCost))

def deltak():
    dk = 0
    for j in range(len(data)):
        xterm = j/(len(data))-1-x0
        dk += (data[j]-sigmoid(j, k, x0))*np.reciprocal(sigmoid(j, k, x0)**2)*xterm*np.exp(-k*xterm)
    return 2*dk/len(data)

def deltax0():
    dx0 = 0
    for j in range(len(data)):
        xterm = j/(len(data)-1)-x0
        dx0 += (data[j]-sigmoid(j, k, x0))*np.reciprocal(sigmoid(j, k, x0)**2)*k*np.exp(-k*xterm)
    return 2*dx0/len(data)

def grad_descent():
    global k, x0
    learn_rate = 0.0001
    dcost = 1

    #Creating best-fit line
    y_fit = np.zeros_like(data)
    for i in range(len(data)):
        y_fit[i] += sigmoid(i, k, x0)
    # Sort samplex and y_fit for plotting
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_y_fit = y_fit[sorted_indices]
    line, = plt.plot(sorted_data, sorted_y_fit, label='Best Fit Line', color='orange', linestyle='--')
    plt.show(block = False)

    while np.abs(dcost) > 1e-6:
        newk= k+learn_rate*deltak()
        newx0 = x0+learn_rate*deltax0()
        dcost = cost(np.array([sigmoid(i, k, x0) for i in range(len(data))]), data)-cost(np.array([sigmoid(i, newk, newx0) for i in range(len(data))]), data)
        k, x0 = newk, newx0
        
        #update y values for best fit line
        y_fit = np.zeros_like(data)
        for i in range(len(data)):
            y_fit[i] += sigmoid(i, k, x0)
        # Sort for plotting
        sorted_y_fit = y_fit[sorted_indices]
        line.set_ydata(sorted_y_fit)
        plt.pause(0.03)  # This updates the plot

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
input("Finished! Press Enter to end program.")
plt.show(block=True)