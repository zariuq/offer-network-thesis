import numpy as np
#import pylab as pl
import matplotlib.pyplot as plt

def eckp(k, p):
    return k * np.power(p, k)

def best_k(p):
    return -1/np.log(p)

num = 20
ks = range(2, 20, 1)
ksr = np.repeat(np.array([ks]).T, num, axis=1)
ps = np.around(np.linspace(0.7, 1, num, endpoint=False), decimals=3)
gs = eckp(ksr, ps).T

#print("ks: {0}".format(ks))
#print("ps: {0}".format(ps))
#print("gs: {0}".format(gs))

def plotECKP(i):
    plt.subplot(4, 5, i + 1)
    plt.plot(ks, gs[i])
    #plt.xlabel('k-cycle')
    #plt.ylabel('#')
    plt.title('p = {0}'.format(ps[i]))


plt.figure(1)
#plt.suptitle("Expected number of users satisfied given p and k-cycle")
for i in range(0, num):
    plotECKP(i)
plt.show()

'''
bks = best_k(ps)
plt.plot(ps, bks)
plt.show()
'''

'''
X, Y = pl.meshgrid(ks, ps)
Z = eckp(X, Y)
im = pl.imshow(Z, cmap=pl.cm.RdBu)
cset = pl.contour(Z, cmap=pl.cm.Set2)
pl.colorbar(im)
pl.show()
'''
