import numpy as np
import random
from matplotlib import pyplot as plt

def explode(string): 
    li = list(string.split(",")) 
    return li 

f= open("alldenseoriginal.txt","r")
lines = f.readlines()
count = 1
# Strips the newline character 
weights = []
for line in lines: 
    wl = explode(line.strip())
    for w in wl:
        weights.append(float(w))
    if count == 15880:
        break;
    count = count + 1
	
f= open("alldensemodified.txt","r")
lines = f.readlines()
count = 1
# Strips the newline character 
mweights = []
for line in lines: 
    wl = explode(line.strip())
    for w in wl:
        mweights.append(float(w))
    if count == 15880:
        break;
    count = count + 1

bin = 0.01

# fixed bin size
bins = np.arange(-0.1, 1, bin) # fixed bin size

fig, ax = plt.subplots()




plt.xlim([-0.1, 1])

ax.hist(weights, bins=bins, label='Original Weight Frequency',alpha=0.5)
ax.hist(mweights, bins=bins, label='Modified Weight Frequency',alpha=0.5)
ax.set_xlabel('Weights (bin size = 0.001)',fontsize=30)
ax.set_ylabel('count',fontsize=30)
ax.tick_params(axis='both', labelsize=30)
ax.legend(fontsize=30)  # Add a legend.


plt.show()