import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

# Data for plotting
t = [(i-100)/10.0 for i in range(200)]
#s = [i if i > 0 else 0 for i in t]
s = [1/(1+math.exp(-i)) for i in t]

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='Input', ylabel='Output')
ax.grid()

plt.show()