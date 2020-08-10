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


def update(a):
	ax.clear()
	i = a%4
	r = int(a/4)+1
	if i == 0:
		ax.set_title('Original weights'.format(i),fontsize=15)
		ax.hist(weights, bins=bins, label='Weight Frequency')
	else:
		mweights = [ comp(w,r) for w in weights]
		if i == 1:
			ax.set_title('Compressing weights ... range [{:.4f}, {:.4f}], Scaled R = {:d}'.format(comp(0,r),comp(1,r),r),fontsize=15)
			ax.hist(mweights, bins=bins, label='Weight Frequency')
		else:
			m2weights = [ step(w,r) for w in mweights]
			if i == 2:
				ax.set_title('Rounding weights ... range [{:.4f}, {:.4f}], Scaled R = {:d}'.format(comp(0,r),comp(1,r),r),fontsize=15)
				ax.hist(m2weights, bins=bins, label='Weight Frequency')
			else:
				m3weights = [ decomp(w,r) for w in m2weights]
				ax.set_title('Decompressing weights ... range [0, 1], Scaled R = {:d}'.format(r),fontsize=15)
				ax.hist(m3weights, bins=bins, label='Weight Frequency')
		
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
	anim = FuncAnimation(fig, update, frames=np.arange(0, totr*4), interval=1000)
	if save == 1:
		anim.save('compdecomp.gif', dpi=80, writer='imagemagick')
	else:
		plt.show()
