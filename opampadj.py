import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

# Data for plotting
t = [i for i in range(100000)]
#s = [i if i > 0 else 0 for i in t]
s = [(1000*i)/(1000+i) for i in t]

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='Resistance Added in Parallel', ylabel='Effective Resistance')
ax.grid()

plt.show()