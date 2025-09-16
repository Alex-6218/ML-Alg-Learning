import matplotlib.pyplot as plt
import numpy as np
import sys

samplex = np.array([])
sampley = np.array([])

degree = int(input("Degree:"))
real_coeffs = [np.random.rand()*2 - 1 for _ in range(degree+1)]
correlation = p = 0.05  # 0 no correlation, 1 perfect correlation

samplex = np.linspace(0, 1, 100)
for i in range(100):
    sampley = np.append(sampley, sum([real_coeffs[j]*(samplex[i]**j) for j in range(len(real_coeffs))]) + 2*np.random.rand()*p - p)

dataset = np.column_stack((samplex, sampley))
2

plt.scatter(samplex, sampley, label='Random Data Points', color='blue')
plt.ion()
plt.title('Polynomial Fit to Random Data')  
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
# plt.xlim(0, 1)
# plt.ylim(0, 1)



coeffs = np.zeros(degree+1)
d_error = 2

def poly_cost(prediction):
    cost = 0
    for i in range(len(dataset)):
        y = dataset[i][1]
        error = prediction[i] - y
        squared_error = error ** 2
        cost += squared_error
    return cost / len(dataset)

def roc_coeffn(n):
    global coeffs
    d_coeffn = 0
    for i in range(len(dataset)):
        x = dataset[i][0]
        y = dataset[i][1]
        d_coeffn += (y - sum([coeffs[j] * (x ** j) for j in range(len(coeffs))])) * (x ** n)
    return -2 * d_coeffn / len(dataset)

def poly_descent():
    global coeffs, d_error
    learn_rate = 0.1
    # Compute initial y values for the fit line
    y_fit = np.zeros_like(samplex)
    for i in range(len(coeffs)):
        y_fit += coeffs[i] * (samplex ** i)
    # Sort samplex and y_fit for plotting
    sorted_indices = np.argsort(samplex)
    sorted_samplex = samplex[sorted_indices]
    sorted_y_fit = y_fit[sorted_indices]
    line, = plt.plot(sorted_samplex, sorted_y_fit, label='Best Fit Line', color='orange', linestyle='--')
    plt.show(block=False)
    while abs(d_error) > 0.000001:
        new_coeffs = coeffs - learn_rate * np.array([roc_coeffn(i) for i in range(len(coeffs))])
        y_pred_old = [sum([coeffs[i]*(dataset[x][0]**i) for i in range(len(coeffs))]) for x in range(len(dataset))]
        y_pred_new = [sum([new_coeffs[i]*(dataset[x][0]**i) for i in range(len(new_coeffs))]) for x in range(len(dataset))]
        d_error = poly_cost(y_pred_old) - poly_cost(y_pred_new)
        coeffs = new_coeffs
        # Update y_fit for the new coefficients
        y_fit = np.zeros_like(samplex)
        for i in range(len(coeffs)):
            y_fit += coeffs[i] * (samplex ** i)
        # Sort for plotting
        sorted_y_fit = y_fit[sorted_indices]
        line.set_ydata(sorted_y_fit)
        plt.pause(0.03)  # This updates the plot

        for i in range(len(coeffs)):
            print(f"Coeff{i}: {coeffs[i]}")
        print(f"Cost: {poly_cost([sum([coeffs[i]*(dataset[x][0]**i) for i in range(len(coeffs))]) for x in range(len(dataset))])}")
        print(f"d_Cost: {d_error}")
        sys.stdout.flush()
        print("\033[{}A".format(len(coeffs)+2), end='')  # Move cursor up len(coeffs)+2 lines

print("\033c", end="")
plt.show(block=False)
poly_descent()
plt.show(block=True)