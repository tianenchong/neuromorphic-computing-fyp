import numpy as np
import copy 
from math import exp
from matplotlib import pyplot as plt

x = [i for i in range(1,201)]
yupper = [1/i for i in x]
ylower = [1/(3*i) for i in x]

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(x, yupper, label='Upper bound')  # Plot some data on the axes.
ax.plot(x, ylower, label='Lower bound')  # Plot some data on the axes.
ax.set_xlabel('R Scale Factor')  # Add an x-label to the axes.
ax.set_ylabel('Compressed Weight')  # Add a y-label to the axes.
ax.legend()  # Add a legend.

plt.show()
	