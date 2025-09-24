import numpy as np
import matplotlib.pyplot as plt



def parametric_function(t):
    x = np.cos(t)
    y = np.sin(1.6*t)
    return x, y

domain = np.linspace(0, 2*np.pi, 100)


def plotline():
    # collect all x and y values and plot them as a continuous line
    xs, ys = [], []
    for t in domain:
        x, y = parametric_function(t)
        xs.append(x)
        ys.append(y)
        plt.plot(xs, ys, 'r-', linewidth=4)
        plt.pause(0.01)


plt.title('Parametric Line Plot')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plotline()
plt.show()