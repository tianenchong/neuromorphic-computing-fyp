from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

figure(figsize=(8,8))
x = np.linspace(0,4,100,endpoint=False)
xvertlabel = np.linspace(1,3,9,endpoint=True)
yvertlabel = [1/i for i in xvertlabel]
y = [1/i for i in x]
plt.plot(x,y)
for i in range(len(xvertlabel)):
	plt.plot([xvertlabel[i],xvertlabel[i]],[0,yvertlabel[i]],color='c',linestyle='-',linewidth=0.5)
	plt.plot([xvertlabel[i],0],[yvertlabel[i],yvertlabel[i]],color='c',linestyle='-',linewidth=0.5)
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.xlabel('R') # x label
plt.ylabel('Compressed w') # y label
plt.show() # show the plot