from neo4j.v1 import GraphDatabase
import sys
import numpy as np
import networkx as nx
from graph_gen import *
from graph_control import *
from collections import Counter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

'''
`wish*.csv` files represent the wish lists of the users
format: userID,itemID

`have*.csv` files represent the give-away lists of the users
format: userID,itemID

`transac*.csv` files represent an item sent from a user to anoter
format: GiverUserID,ReceiverUserID,itemID,timestamp

`pair*.csv` files represent a bidirectional transaction between users
where user1 owns item1 and give it to user2
format: format: user1ID,item1ID,user2ID,item2ID,timestamp

`*_dense.csv` files have compressed ID's in order to be loaded in non-sparse data formats.
'''

vecMax = np.vectorize(max)

def pl(x, m, c, c0):
    if c0 < 0:
        return -10**5
    return c0 + c * x**(-m)

def plp(x, m, c, c0):
    if c0 < 0:
        return -10**5
    return c0 + c * (x+0.1)**(-m)

def plm(x, m, c, c0):
    if c0 < 0:
        return -10**5
    return vecMax(c0 + c * x**(-m),1)

def strr(s):
    return str(np.around(s, decimals=2))

def fit_data(title, file_path, loglog):
    data = np.loadtxt(file_path, dtype=int, delimiter=',')
    counts = Counter(data.T[1])

    # [(#), (# of items appearing this many times in a list)]
    item_counts = list(zip(*(list(Counter(list(zip(*list(counts.items())))[1]).items()))))

    popt, pcov = curve_fit(pl, item_counts[0], item_counts[1], maxfev=2000, p0 = np.asarray([3.0,item_counts[1][0],5.0]))
    exponent = strr(popt[2]) + " + " + strr(popt[1]) + " * k^(-" + strr(popt[0]) + ")"

    Xs = np.linspace(item_counts[0][0],item_counts[0][-1],5000,endpoint=True)
    if loglog:
        plt.loglog(Xs, pl(Xs, *popt), '--', label='fit')
        plt.loglog(item_counts[0], item_counts[1], 'ro', label='data')
    else:
        plt.plot(Xs, pl(Xs, *popt), '--', label='fit')
        plt.plot(item_counts[0], item_counts[1], 'ro', label='data')
    plt.ylabel('# items')
    plt.xlabel('# occurences in list')
    plt.title(title + ": " + exponent)
    plt.legend()
    if loglog:
        plt.savefig("plot_output/log" + title + ".png")
    else:
        plt.savefig("plot_output/" + title + ".png")
    #plt.show()

loglog = True

datasets = False

if datasets:
    data = [("Bookmooch6HaveList", "swapit-data/bookmooch/bm6m/have_dense.csv")
            ,("Bookmooch6WishList", "swapit-data/bookmooch/bm6m/wish_dense.csv")
            ,("Bookmooch1HaveList", "swapit-data/bookmooch/bm1y/have_dense.csv")
            ,("Bookmooch1WishList", "swapit-data/bookmooch/bm1y/wish_dense.csv")
            ,("RatebeerHaveList", "swapit-data/ratebeer/have_dense.csv")
            ,("RatebeerWishList", "swapit-data/ratebeer/wish_dense.csv")
            ,("RedditWishList", "swapit-data/reddit/wish.csv")]

    for title, file_path in data:
        fit_data(title, file_path, loglog)
        plt.clf()
        print("Completed %s" % title)


if not datasets:

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "iwin"))

    def degree_analyser(degree_counts, title, co, fun):
        popt, pcov = curve_fit(fun, degree_counts[0], degree_counts[1], maxfev=2000, p0 = np.asarray([3.0,degree_counts[1][0],5.0]))
        exponent = strr(popt[2]) + " + " + strr(popt[1]) + " * k^(-" + strr(popt[0]) + ")"

        xlower = max(1, min(degree_counts[0]))
        xupper = max(degree_counts[0])
        Xs = np.linspace(xlower, xupper,5000,endpoint=True)
        if loglog:
            plt.loglog(Xs, fun(Xs, *popt), '--', label='fit')
            plt.loglog(degree_counts[0], degree_counts[1], co, label='data')
        else:
            plt.plot(Xs, fun(Xs, *popt), '--', label='fit')
            plt.plot(degree_counts[0], dgree_counts[1], co, label='data')
        plt.ylabel('# tasks')
        plt.xlabel('# degree')
        plt.title(title + ": " + exponent)
        plt.legend()
        if loglog:
            plt.savefig("plot_output/a_log_" + title + ".png")
        else:
            plt.savefig("plot_output/a_" + title + ".png")

    #
    #
    #

    # out = offer, in = request

    # 1.88 2.68
    #(G, updates, Nusers, Ntasks) = generate_sfgL(driver, 10000, 20, alpha=0.7,gamma=0.25,beta=0.05,directed_p=0.05,delta_in=0.1,delta_out=1.25)

    # 1.57 3.04
    #(G, updates, Nusers, Ntasks) = generate_sfgL(driver, 10000, 20, alpha=0.85,gamma=0.05,beta=0.1,directed_p=0.05,delta_in=0,delta_out=1)

    # 1.86 1.87
    #(G, updates, Nusers, Ntasks) = generate_sfgL(driver, 10000, 20, alpha=0.3,gamma=0.3,beta=0.4,directed_p=0.1,delta_in=0.5,delta_out=0.5)

    # 1.26 1.34
    #(G, updates, Nusers, Ntasks) = generate_sfgL(driver, 10000, 20, alpha=0.05,gamma=0.05,beta=0.9,directed_p=0.05,delta_in=1.5,delta_out=1.25)

    # 1.53 2.79
    #(G, updates, Nusers, Ntasks) = generate_sfgL(driver, 10000, 20, alpha=0.7,gamma=0.05,beta=0.25,directed_p=0.05,delta_in=0,delta_out=0.25)

    # 1.66 1.88
    #(G, updates, Nusers, Ntasks) = generate_sfgL(driver, 10000, 20, alpha=0.35,gamma=0.15,beta=0.5,directed_p=0.05,delta_in=1.0,delta_out=0.5)

    # 1.43 1.96
    #(G, updates, Nusers, Ntasks) = generate_sfgL(driver, 10000, 20, alpha=0.2,gamma=0.01,beta=0.79,directed_p=0.1,delta_in=0.1,delta_out=0.05)

    # 1.74 1.96
    #(G, updates, Nusers, Ntasks) = generate_sfgL(driver, 10000, 20, alpha=0.25,gamma=0.1,beta=0.65,directed_p=0.05,delta_in=0.01,delta_out=0.01)

    # 1.57 2.05
    #(G, updates, Nusers, Ntasks) = generate_sfgL(driver, 10000, 20, alpha=0.3,gamma=0.05,beta=0.65,directed_p=0.05,delta_in=0.25,delta_out=0.1)

    # 1.66 2.77
    #(G, updates, Nusers, Ntasks) = generate_sfgL(driver, 10000, 20, alpha=0.53,gamma=0.12,beta=0.35,directed_p=0.05,delta_in=0.25,delta_out=0.05)


    degree_counts = list(zip(*list(Counter(G.degree().values()).items())))
    in_degree_counts = list(zip(*list(Counter(G.in_degree().values()).items())[1:] ))
    out_degree_counts = list(zip(*list(Counter(G.out_degree().values()).items())[1:] ))

    fun = pl

    degree_analyser(degree_counts, "deg_tot", 'ro', pl)
    plt.clf()
    degree_analyser(in_degree_counts, "Requests", 'go', pl)
    plt.clf()
    degree_analyser(out_degree_counts, "Offers", 'yo', pl)

    sum = 0
    for i,o in zip(list(G.in_degree().values()),list(G.out_degree().values())):
        sum += abs(i - o)
    print("Average difference in in/out degree: %f" % (sum / len(G.in_degree())))
