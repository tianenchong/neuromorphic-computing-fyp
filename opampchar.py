import numpy as np
import copy 
from math import exp
from matplotlib import pyplot as plt
import csv


labels = ['2k 2k','50k 50k','500k 500k']
vouts = []
vgnds = []

for a in labels:
	vout = []
	vgnd = []
	with open(a+'.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				print(f'Column names are {", ".join(row)}')
				line_count += 1
			else:
				vout.append([float(row[0]),float(row[1])])
				vgnd.append([float(row[2]),float(row[3])])
				line_count += 1

	vout = np.array(vout)
	vout = np.transpose(vout).tolist()
	vgnd = np.array(vgnd)
	vgnd = np.transpose(vgnd).tolist()
	vouts.append(vout)
	vgnds.append(vgnd)

fig, ax = plt.subplots()  # Create a figure and an axes.

for i in range(len(labels)):
	label_new = labels[i].split(' ')
	ax.plot(vgnds[i][0], vgnds[i][1], label='R1='+label_new[0]+' R0='+label_new[1])  # Plot some data on the axes.
ax.set_xlabel('Input Voltage (V)')  # Add an x-label to the axes.
ax.set_ylabel('Virtual Ground Voltage (V)')  # Add a y-label to the axes.
ax.legend()  # Add a legend.

plt.show()

fig, ax = plt.subplots()  # Create a figure and an axes.

for i in range(len(labels)):
	label_new = labels[i].split(' ')
	ax.plot(vouts[i][0], vouts[i][1], label='R1='+label_new[0]+' R0='+label_new[1])  # Plot some data on the axes.
ax.set_xlabel('Input Voltage (V)')  # Add an x-label to the axes.
ax.set_ylabel('Output Voltage (V)')  # Add a y-label to the axes.
ax.legend()  # Add a legend.

plt.show()
	