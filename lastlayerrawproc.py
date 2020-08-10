import math
a = [0.037776,0.018356,0.031785,0.039524,0.037903,0.032402,0.031561,0.066088,0.029600,0.048405]

sum = 0
for i in a:
	sum = sum + math.exp(i)

b = [math.exp(i)/sum for i in a]
print(b)