#!/usr/bin/env python

import numpy as np
#import pylab as pl
import matplotlib.pyplot as plt

def eckp(k, p):
    return k * np.power(p, k)

def best_k(p):
    return -1/np.log(p)



#print("ks: {0}".format(ks))
#print("ps: {0}".format(ps))
#print("gs: {0}".format(gs))

def plotECKP(i):
    plt.subplot(4, 5, i + 1)
    plt.plot(ks, gs[i])
    #plt.xlabel('k-cycle')
    #plt.ylabel('#')
    plt.title('p = {0}'.format(ps[i]))

def plotECKPfun():
    num = 20
    ks = range(2, 20, 1)
    ksr = np.repeat(np.array([ks]).T, num, axis=1)
    ps = np.around(np.linspace(0.7, 1, num, endpoint=False), decimals=3)
    gs = eckp(ksr, ps).T

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

num = 20
ps1 = np.around(np.linspace(0.3, 0.7, num, endpoint=True), decimals=3)
ps2 = np.around(np.linspace(0.7, 1, num, endpoint=False), decimals=3)
ps3 = np.around(np.linspace(0.5, 1, num, endpoint=False), decimals=3)
cs1 = 1 / np.power(ps1, 3)
cs2 = 1 / np.power(ps2, 3)
cs3 = 1 / np.power(ps1, 2)
ppb = 1 - (np.power(ps2, 3) + (np.power(ps2, 3) * (1 - (np.power(ps2, 3)))))
ppb2 = 1 - (np.power(ps3, 2) + (np.power(ps3, 2) * (1 - (np.power(ps3, 2)))))
plt.figure(2)
#plt.suptitle("Number of Rounds to Acceptance")
plt.subplot(2,3,1)
#plt.title("1/p^2")
plt.xlabel("(a)")
plt.plot(ps1,cs3)
plt.subplot(2,3,2)
#plt.title("1/p^3")
plt.xlabel("(b)")
plt.plot(ps1,cs1)
plt.subplot(2,3,3)
#plt.title("1/p^3")
plt.xlabel("(c)")
plt.plot(ps2,cs2)
plt.subplot(2,3,4)
#plt.title("1 - (p^3 + p^3 * (1 - p^3))")
plt.xlabel("(d)")
plt.plot(ps2,ppb)
plt.subplot(2,3,6)
#plt.title("1 - (p^2 + p^2 * (1 - p^2))")
plt.xlabel("(e)")
plt.plot(ps3,ppb2)
plt.show()
