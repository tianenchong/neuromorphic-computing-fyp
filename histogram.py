import numpy as np
import random
from matplotlib import pyplot as plt

def explode(string): 
    li = list(string.split(",")) 
    return li 

f= open("10 dense.txt","r")
lines = f.readlines()
count = 1
# Strips the newline character 
weights = []
for line in lines: 
    wl = explode(line.strip())
    for w in wl:
        weights.append(float(w))
    if count == 784:
        break;
    count = count + 1

bin = 0.0001

# fixed bin size
bins = np.arange(0, 1, bin) # fixed bin size

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # set up the 2nd axis



plt.xlim([0, 0.5])

ax1.hist(weights, bins=bins, label='Frequency')
ax2.hist(weights, bins=bins, cumulative=True,histtype='step', label='Cumulative')
plt.title('Trained Weights Distribution (fixed bin size)')
ax1.set_xlabel('Weights (bin size = 0.001)')
ax1.set_ylabel('count')

plt.show()