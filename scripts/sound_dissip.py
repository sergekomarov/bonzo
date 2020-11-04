import h5py
import numpy as np
from scipy.stats import linregress

# calculate damping rate

t = []
Ek = []

with open('hst.txt', 'r') as f:
    print f.readline()
    line = f.readline()
    while line != '':
        vars = np.array(line.split(), dtype='f')
        t.append(vars[1])
        Ek.append(vars[3])
        line = f.readline()

x = np.array(t)
y = np.log(np.array(Ek))

plot(x,y)

d = linregress(x,y)[0]
print d
