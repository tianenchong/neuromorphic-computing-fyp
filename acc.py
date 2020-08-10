import numpy as np
import random
import math
from matplotlib import pyplot as plt


def explode(string): 
    li = list(string.split('\t')) 
    return li 

f= open("accdata.txt","r")
lines = f.readlines()
count = 1
# Strips the newline character 
measured = []
ideal = []
percents = []
avgpercent = 0
n = 20
for line in lines: 
	wl = explode(line.strip())
	measured.append(float(wl[0]))
	ideal.append(float(wl[1]))
	percent = measured[count-1]/ideal[count-1]*100
	percents.append(percent)
	avgpercent = avgpercent + percent
	if count == n:
		break;
	count = count + 1

avgpercent = avgpercent/n
sd = 0

intermediatesd = 0
for i in range(n):
	intermediatesd = intermediatesd + (percents[i] - avgpercent)**2

sd = math.sqrt(intermediatesd/(n-1))

print("{:.2f}".format(sd))

x = range(1,21)

fig, ax = plt.subplots(figsize=(25,7))  # Create a figure and an axes.
plt.grid()
ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

for i in x:
	x_values = [i, i]
	y_values = [measured[i-1], ideal[i-1]]
	ax2.plot(x_values, y_values,color='C1', alpha=0.8)

ax2.plot(x, measured, "C1.", label='Simulated Value')  # Plot some data on the axes.
ax2.plot(x, ideal, "C0.", label='Ideal Value')  # Plot some data on the axes.
ax2.set_xlabel('Neuron #',fontsize=15)  # Add an x-label to the axes.
ax2.set_ylabel('Output Value',fontsize=15)  # Add a y-label to the axes.
ax2.legend(loc=4,fontsize=15)  # Add a legend.
ax2.set_ylim(bottom=0)
ax2.tick_params(axis='both', labelsize=15)
ax.plot(x, percents, "y*", label='Closeness %',alpha=0.6)  # Plot some data on the axes.
ax.plot(x, [avgpercent for i in x], "y--", label='Average Closeness %',alpha=0.6)  # Plot some data on the axes.
ax.fill_between(x, avgpercent-sd, avgpercent+sd, facecolor='yellow', alpha=0.2)
ax.set_ylabel('Closeness (Simulated rel to Ideal) %',fontsize=15)  # Add a y-label to the axes.
ax.set_ylim(bottom=0)
ax.tick_params(axis='both', labelsize=15)
plt.xticks(x)
ax.legend(loc=2,fontsize=15)
plt.show()
