import matplotlib.pyplot as plt
import numpy as np
import sys

samplex = np.array([])
sampley = np.array([])


correlation = p = 0.05  # 0 no correlation, 1 perfect correlation


real_slope = np.random.rand()*2 - 1
real_intercept = np.random.rand()
for i in range(100):
    samplex = np.append(samplex, np.random.rand())
    sampley = np.append(sampley, real_slope*samplex[i]+real_intercept+2*np.random.rand()*p-p)

dataset = np.column_stack((samplex, sampley))


coeff0 = 0
coeff1 = 0


d_error = 2


def mse_cost(m,b):
    cost  = 0
    for i in range(len(dataset)):
        x = dataset[i][0]
        y = dataset[i][1]
        prediction = m*x+b
        error = prediction - y
        squared_error = error ** 2
        cost += squared_error
    return cost / len(dataset) 

def gradient_intercept():
    global coeff0, coeff1
    d_coeff0 = 0
    for i in range(len(dataset)):
        x = dataset[i][0]
        y = dataset[i][1]
        d_coeff0 += y-(coeff0+coeff1*x)
    return -2 * d_coeff0 / len(dataset)

def gradient_slope():
    global coeff0, coeff1
    d_coeff1 = 0
    for i in range(len(dataset)):
        x = dataset[i][0]
        y = dataset[i][1]
        d_coeff1 += (y-(coeff0+coeff1*x)) * x
    return -2 * d_coeff1 / len(dataset)


plt.scatter(samplex, sampley, label='Random Data Points', color='blue')
plt.ion()
plt.title('Linear Fit to Random Data')  
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)

def grad_descent():
    global coeff0, coeff1, d_error
    learn_rate = 0.03
    line, = plt.plot(samplex, coeff1*samplex+coeff0, label='Best Fit Line', color='orange', linestyle='--')
    plt.show(block=False)
    while d_error > 0.000001 or d_error < -0.000001:
        new_coeff0 = coeff0 - learn_rate * gradient_intercept()
        new_coeff1 = coeff1 - learn_rate * gradient_slope()
        d_error = mse_cost(coeff1, coeff0) - mse_cost(new_coeff1, new_coeff0)
        coeff0 = new_coeff0
        coeff1 = new_coeff1
        line.set_ydata(coeff1 * samplex + coeff0)
        plt.pause(0.01)  # This updates the plot

        print(f"Cost: {mse_cost(coeff1, coeff0)}")
        print(f"Coeff0: {coeff0}")
        print(f"Coeff1: {coeff1}")
        print(f"d_Cost: {d_error}")
        sys.stdout.flush()
        print("\033[4A]", end='')  # Move cursor up 4 lines


print("\033c", end="")
plt.show(block=False)
grad_descent()
plt.show(block=True)
