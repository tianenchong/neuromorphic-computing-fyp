max_w = 1
max_r = 1/max_w
rrf = [i*250+1000 for i in range(9)]
r = [i*max_r/1000 for i in rrf]
w = [1/i for i in r]
rw = [(i-w[-1])/(w[0]-w[-1]) for i in w]
for i in range(9):
    print(rrf[i],r[i],w[i],rw[i],sep = '\t\t\t')