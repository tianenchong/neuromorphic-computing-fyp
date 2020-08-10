import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

totw = 200
totr = 10

def explode(string): 
	li = list(string.split(",")) 
	return li
	
def lim(w,mul):
	if w <= 1/(3*mul):
		return 1/(3*mul)
	elif w >= 1/(1*mul):
		return 1/(1*mul)
	else:
		return w

def	step(w, mul):
	if w <= 1/(3*mul):
		return 1/(3*mul)
	else:
		if w <= 1/(2.75*mul):
			if (w - 1/(3*mul)) < (1/(2.75*mul) - w):
				return 1/(3*mul)
			else:
				return 1/(2.75*mul)
		else:
			if w <= 1/(2.5*mul):
				if (w - 1/(2.75*mul)) < (1/(2.5*mul) - w):
					return 1/(2.75*mul)
				else:
					return 1/(2.5*mul)
			else:
				if w <= 1/(2.25*mul):
					if (w - 1/(2.5*mul)) < (1/(2.25*mul) - w):
						return 1/(2.5*mul)
					else:
						return 1/(2.25*mul)
				else:
					if w <= 1/(2*mul):
						if (w - 1/(2.25*mul)) < (1/(2*mul) - w):
							return 1/(2.25*mul)
						else:
							return 1/(2*mul)
					else:
						if w <= 1/(1.75*mul):
							if (w - 1/(2*mul)) < (1/(1.75*mul) - w):
								return 1/(2*mul)
							else:
								return 1/(1.75*mul)
						else:
							if w <= 1/(1.5*mul):
								if (w - 1/(1.75*mul)) < (1/(1.5*mul) - w):
									return 1/(1.75*mul)
								else:
									return 1/(1.5*mul)
							else:
								if w <= 1/(1.25*mul):
									if (w - 1/(1.5*mul)) < (1/(1.25*mul) - w):
										return 1/(1.5*mul)
									else:
										return 1/(1.25*mul)
								else:
									if w <= 1/(1*mul):
										if (w - 1/(1.25*mul)) < (1/(1*mul) - w):
											return 1/(1.25*mul)
										else:
											return 1/(1*mul)
									else:
										return 1/(1*mul)

		
def comp(w,mul):
	return w*(1/(1*mul)-1/(3*mul))+1/(3*mul)
	
def decomp(round,mul):
	return (round-1/(3*mul))/(1/(1*mul)-1/(3*mul))
	
def compdecomp(w,mul):
	compressed = comp(w,mul)
	round = step(compressed, mul)
	decompressed = decomp(round,mul)
	#if w < 0:
	#	print("before")
	#	print(w)
	#	print("after compressed")
	#	print(compressed)
	#	print("after round")
	#	print(round)
	#	print("after decompressed")
	#	print(decompressed)
	return decompressed

def	stepUniform(w, n):
	#range from 0 - n
	return round(w * n)/n

func = comp

f= open("20denseoriginal.txt","r")
lines = f.readlines()
count = 1
# Strips the newline character 
weights = []
for line in lines: 
	wl = explode(line.strip())
	for w in wl:
		weights.append(float(w))
	if count == totw:
		break;
	count = count + 1

n = 1
bin = 0.001

# fixed bin size
bins = np.arange(-0.1, 1, bin) # fixed bin size

fig, ax = plt.subplots(figsize=(25,7))


plt.xlim([-0.4, 1])


  # Add a legend.


def update(i):
	ax.clear()
	if i == 0:
		ax.set_title('Original weights',fontsize=15)
		ax.hist(weights, bins=bins, label='Weight Frequency')
	else:
		i = i*10
		mweights = [ stepUniform(w,i) for w in weights]
		ax.set_title('Step n = {:d}'.format(i,fontsize=15))
		ax.hist(mweights, bins=bins, label='Weight Frequency')
		
	#print(label)
	# Update the line and the axes (with a new xlabel). Return a tuple of
	# "artists" that have to be redrawn for this frame.
	ax.set_xlabel('Weights (bin size = 0.001)',fontsize=15)
	ax.set_ylabel('Frequency',fontsize=15)
	ax.tick_params(axis='both', labelsize=15)
	return line, ax

save = 1;
	
if __name__ == '__main__':
	# FuncAnimation will call the 'update' function for each frame; here
	# animating over 10 frames, with an interval of 200ms between frames.
	anim = FuncAnimation(fig, update, frames=np.arange(0, totr+1), interval=500)
	if save == 1:
		anim.save('uniform.gif', dpi=80, writer='imagemagick')
	else:
		plt.show()
