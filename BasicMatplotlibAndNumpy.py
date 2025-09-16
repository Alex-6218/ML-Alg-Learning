import matplotlib.pyplot as plt
import numpy as np

datasize = int(input("Enter the size of the dataset: "))
degree = int(input("Enter the degree of the polynomial fit: "))
samplex = np.array([])
for i in range(datasize):
    samplex = np.append(samplex, np.random.rand())

sampley = np.array([])
for i in range(datasize):
    sampley = np.append(sampley, np.random.rand())

bestfit = np.polyfit(samplex, sampley, degree)

x = np.linspace(0, 1, 100)
y = np.polyval(bestfit, x) 

plt.plot(x, y, label='Best Fit Polynomial', color='orange', linestyle='--')
plt.scatter(samplex, sampley, label='Random Data Points', color='blue')
plt.title('Polynomial Fit to Random Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()