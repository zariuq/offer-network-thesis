#!/usr/bin/env python

import numpy as np
import glob
#import pylab as pl
import matplotlib.pyplot as plt
import pickle

## To plot rejection probability related graphs

def eckp(k, p):
    return k * np.power(p, k)

def best_k(p):
    return -1/np.log(p)

#print("ks: {0}".format(ks))
#print("ps: {0}".format(ps))
#print("gs: {0}".format(gs))

def plotECKP(i, ks, gs, ps):
    plt.subplot(4, 5, i + 1)
    plt.plot(ks, gs[i])
    plt.xlabel('k')
    #plt.ylabel('#')
    plt.title('p = {0}'.format(ps[i]))

def plotECKPfun(psl, klower, kupper):
    num = len(psl)
    ks = np.linspace(klower,kupper,5000, endpoint=True)
    ksr = np.repeat(np.array([ks]).T, num, axis=1)
    ps = np.array(psl)#np.around(np.linspace(0.85, 1, num, endpoint=False), decimals=3)
    gs = eckp(ksr, ps).T

    plt.figure(1)
    #plt.suptitle("Expected number of users satisfied given p and k-cycle")
    for i in range(0, num):
        plotECKP(i, ks, gs, ps)
    plt.show()

    # So I want ot see 0.85, 0.95 (k=2 to 50
    # 0.3, 0.6, 0.7, 0.85 (k=2 to 10)

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

def plotNRounds():
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
    plt.ylabel("Expected rounds to acceptance")
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
    plt.xlabel("Acceptance probability p \n (d)")
    plt.ylabel("Probability not accepted in 2 rounds.")
    plt.plot(ps2,ppb)
    plt.subplot(2,3,6)
    #plt.title("1 - (p^2 + p^2 * (1 - p^2))")
    plt.xlabel("Acceptance probability p \n (e)")
    plt.plot(ps3,ppb2)
    plt.show()


## To plot results

def makeTotalMatchDict():
    statListDict = dict()
    ps = []
    for hold in [0, 1]:
        bestTMs = []
        if hold == 1:
            ps = [0.9, 0.7, 0.8, 0.6, 0.5, 0.3]
        else:
            ps = [0.9, 0.7, 0.5, 0.3]
        for p in ps:
            for matchingAlgorithm in ["GSC", "DYN", "GSCPoD"]:
                statListDict['RB1'+str(hold + p) + matchingAlgorithm] = pickle.load(open("results/grid/Ptest2-RB1-"+matchingAlgorithm+"-"+str(p)+"-"+str(hold)+".p", "rb"))
    for matchingAlgorithm in ["GSC", "DYN", "GSCPoD"]:
        statListDict['EN'+str(0 + 0.9) + matchingAlgorithm] = pickle.load(open("results/grid/Ptest2-EN-"+matchingAlgorithm+"-0.9-0.p", "rb"))
    for matchingAlgorithm in ["GSC", "GSCPoD"]:
        statListDict['EN'+str(1 + 0.3) + matchingAlgorithm] = pickle.load(open("results/grid/Ptest3-EN-"+matchingAlgorithm+"-0.3-1.p", "rb"))
        statListDict['BM1'+str(0 + 0.3) + matchingAlgorithm] = pickle.load(open("results/grid/Ptest3-BM1-"+matchingAlgorithm+"-0.3-0.p", "rb"))
        statListDict['BM1'+str(1 + 0.7) + matchingAlgorithm] = pickle.load(open("results/grid/Ptest3-BM1-"+matchingAlgorithm+"-0.7-1.p", "rb"))

    pickle.dump(statListDict, open("results/statListDict.p", "wb"))

    def countMatched(i):
        return i[0][0] * i[1]

    totalMatchDict = dict()
    for fname, runs in statListDict.items():
        tml = []
        for statList in runs:
            pureMatched = sum(statList[0].values())
            statList[1][(0,0)] = 0
            holdMatched = sum(map(countMatched, statList[1].items()))
            tml.append(pureMatched + holdMatched)
        totalMatchDict[fname] = tml

    pickle.dump(totalMatchDict, open("results/totalMatchDict.p", "wb"))

# For Ptest
def makeTotalMatchDictP():
    statListDict = dict()
    ps = []
    for hold in [0, 1]:
        bestTMs = []
        if hold == 1:
            ps = [0.9, 0.8, 0.7]
        else:
            ps = [0.9]
        for p in ps:
            for matchingAlgorithm in ["GSC", "DYN", "GSCPoD", "MAX", "TWO"]:
                statListDict['RB1'+str(hold + p) + matchingAlgorithm] = pickle.load(open("results/grid/Ptest-RB1-"+matchingAlgorithm+"-"+str(p)+"-"+str(hold)+".p", "rb"))

    for matchingAlgorithm in ["GSC", "MAX"]:
        statListDict['EN'+str(1 + 0.3) + matchingAlgorithm] = pickle.load(open("results/grid/Btest-EN-"+matchingAlgorithm+"-0.3-1.p", "rb"))
        statListDict['BM1'+str(1 + 0.5) + matchingAlgorithm] = pickle.load(open("results/grid/Btest-BM1-"+matchingAlgorithm+"-0.5-1.p", "rb"))

    pickle.dump(statListDict, open("results/statListDictP.p", "wb"))

    def countMatched(i):
        return i[0][0] * i[1]

    totalMatchDict = dict()
    for fname, runs in statListDict.items():
        tml = []
        for statList in runs:
            pureMatched = sum(statList[0].values())
            statList[1][(0,0)] = 0
            holdMatched = sum(map(countMatched, statList[1].items()))
            tml.append(pureMatched + holdMatched)
        totalMatchDict[fname] = tml

    pickle.dump(totalMatchDict, open("results/totalMatchDictP.p", "wb"))

def getTotalMatchDict():
    totalMatchDict = pickle.load(open("results/totalMatchDict.p", "rb"))
    return totalMatchDict

def getStatListDict():
    statListDict = pickle.load(open("results/statListDict.p", "rb"))
    return statListDict

def getTotalMatchDictP():
    totalMatchDict = pickle.load(open("results/totalMatchDictP.p", "rb"))
    return totalMatchDict

def getStatListDictP():
    statListDict = pickle.load(open("results/statListDictP.p", "rb"))
    return statListDict

def countMatched(i):
    return i[0][0] * i[1]

def upDateTotalMatched(filename):
    runs = pickle.load(open(filename[:-3] + 'p', "rb"))
    tml = []
    for statList in runs:
        pureMatched = sum(statList[0].values())
        statList[1][(0,0)] = 0
        holdMatched = sum(map(countMatched, statList[1].items()))
        tml.append(pureMatched + holdMatched)
    return tml

# nUtest

colors = {'b': 'b', 'gsc':'y', 't':'g', 'd':'r', 'gscpod':'m'}
mName = {'b': 'MAX', 'gsc':'GSC', 'gscpod': 'GSCPoD','t':'TWO', 'd':'DYN'}

def getDatasets(filenames):
    datasets = dict()
    for filename in filenames:
        data = np.genfromtxt(filename, dtype=str, delimiter=',')
        Gname = data[1][0]
        matchingAlgorithm = data[1][1]
        datasets.setdefault(Gname, dict())
        Gdata = dict()
        for column in data.T[2:]:
            if column[0] == 'totalMatched':
                Gdata[column[0]] = upDateTotalMatched(filename)
            else:
                Gdata[column[0]] = column[1:].astype(float)
        datasets[Gname][matchingAlgorithm] = Gdata
    return datasets

def getDatasetsN(filenames):
    datasets = dict()
    for filename in filenames:
        data = np.genfromtxt(filename, dtype=str, delimiter=',')
        Gname = data[1][0]
        matchingAlgorithm = data[1][1]
        datasets.setdefault(Gname, dict())
        Gdata = dict()
        for column in data.T[2:]:
            Gdata[column[0]] = column[1:].astype(float)
        datasets[Gname][matchingAlgorithm] = Gdata
    return datasets

def getDatasetsP(filenames):
    datasets = dict()
    for filename in filenames:
        data = np.genfromtxt(filename, dtype=str, delimiter=',')
        Gname = data[1][0]
        p = data[1][3]
        Gdata = dict()
        for column in data.T[2:]:
            if column[0] == 'totalMatched':
                Gdata[column[0]] = upDateTotalMatched(filename)
            else:
                Gdata[column[0]] = column[1:].astype(float)
        datasets[p] = Gdata
    return datasets

# name dict
    # alg dict
        # col dict

# Merges two data dictionaries, averaging data when it coincides.
# Can be written more concisely with recursion
def mergeDatasets(ds1, ds2):
    for Gname, Algdict2 in ds2.items():
        if not Gname in ds1:
            ds1[Gname] = Algdict2
        else:
            Algdict1 = ds1[Gname]
            for matchingAlgorithm, Gdata2 in Algdict2.items():
                if not matchingAlgorithm in Algdict1:
                    Algdict1[matchingAlgorithm] = Gdata2
                else:
                    Gdata1 = ds1[matchingAlgorithm]
                    for col, row in Gdata2.items():
                        if not col in Gdata1:
                            Gdata1[col] = row
                        else:
                            Gdata1[col] = (Gdata1[col] + row) / 2

#for Gname, Gdata in datasets.items():
#    for matchingAlgorithm, data in Gdata.items():
#        print("%s - %s: %s" % (Gname, matchingAlgorithm, data))
#        print("")
#        print("")

def plotMetrics(datasets, xLabel, xData, title=""):
    for Gname, Gdata in datasets.items():
        plt.figure(Gname + " " + title, figsize=(11.69, 8.27))
        ax = plt.subplot(1,3,1)
        xTestPoints = Gdata['gsc'][xData]
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            #if not matchingAlgorithm == 't':
            plt.plot(GAdata[xData], GAdata['totalMatched'], colors[matchingAlgorithm]+'--', label=matchingAlgorithm)
        plt.title('Total Matched')
        plt.xlabel(xLabel)
        plt.ylabel('# ORpairs')
        plt.legend()


        ax = plt.subplot(1,3,2)
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            #if not matchingAlgorithm == 't':
            if xData == 'nMatch':
                plt.plot(GAdata[xData], GAdata['avgWaitTimeMatched'] - GAdata[xData] / 2, colors[matchingAlgorithm], label=matchingAlgorithm)
            else:
                plt.plot(GAdata[xData], GAdata['avgWaitTimeMatched'] - 100 / 2, colors[matchingAlgorithm], label=matchingAlgorithm)
            plt.plot(GAdata[xData], GAdata['avgWaitTimeUnmatched'], colors[matchingAlgorithm]+'--')
        plt.title('Wait Time')
        plt.xlabel(xLabel)
        plt.ylabel('# ORpairs')
        plt.legend()

        ax = plt.subplot(1,3,3)
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            #if not matchingAlgorithm == 't':
            plt.plot(GAdata[xData], GAdata['avgMatchSize'], colors[matchingAlgorithm]+'--', label=matchingAlgorithm)
        plt.title('Match Size')
        plt.xlabel(xLabel)
        plt.ylabel('# ORpairs')
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/plot/" + title + "-" + Gname + " " + ".png")
        plt.close()
    #plt.show()

def plotMetricsB(datasets, xLabel, xData, title=""):
    #totalMatchDict = getTotalMatchDictP()
    for Gname, Gdata in datasets.items():
        plt.figure(Gname + " " + title, figsize=(11.69, 8.27))
        ax = plt.subplot(2,3,1)
        xTestPoints = Gdata['gsc'][xData]
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            #if not matchingAlgorithm == 't':
            plt.plot(GAdata[xData], GAdata['totalMatched'], colors[matchingAlgorithm], label=matchingAlgorithm)
            plt.plot(GAdata[xData], GAdata['totalHeld'], colors[matchingAlgorithm]+'--')
        plt.title('Total Matched/Held')
        plt.xlabel(xLabel)
        plt.ylabel('# ORpairs')
        plt.legend()


        ax = plt.subplot(2,3,2)
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            #if not matchingAlgorithm == 't':
            plt.plot(GAdata[xData], GAdata['avgWaitTimeMatched'] - GAdata[xData] / 2, colors[matchingAlgorithm], label=matchingAlgorithm)
            plt.plot(GAdata[xData], GAdata['avgWaitTimeUnmatched'], colors[matchingAlgorithm]+'--')
        plt.title('Wait Time')
        plt.xlabel(xLabel)
        plt.ylabel('# ORpairs')
        plt.legend()

        ax = plt.subplot(2,3,3)
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            #if not matchingAlgorithm == 't':
            if 'avgCycleSize' in GAdata:
                plt.plot(GAdata[xData], GAdata['avgCycleSize'], colors[matchingAlgorithm])
            plt.plot(GAdata[xData], GAdata['avgMatchSize'], colors[matchingAlgorithm]+'--', label=matchingAlgorithm)
        plt.title('Match Size')
        plt.xlabel(xLabel)
        plt.ylabel('# ORpairs')
        plt.legend()

        ax = plt.subplot(2,3,4)
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            if not matchingAlgorithm == 't':
                plt.plot(GAdata[xData], GAdata['avgPoDMatched'], colors[matchingAlgorithm], label=matchingAlgorithm)
                plt.plot(GAdata[xData], GAdata['avgPodUnmatched'], colors[matchingAlgorithm]+'--')
        plt.title('Matched Task Popularity')
        plt.xlabel(xLabel)
        plt.ylabel('Product of Degrees')
        plt.legend()

        ax = plt.subplot(2,3,5)
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            #if not matchingAlgorithm == 't':
            plt.plot(GAdata[xData], GAdata['avgWaitTimeHeldMatched'], colors[matchingAlgorithm], label=matchingAlgorithm)
            plt.plot(GAdata[xData], GAdata['avgWaitTimeHeldUnmatched'], colors[matchingAlgorithm]+'--')
        plt.title('Hold Wait Times')
        plt.xlabel(xLabel)
        plt.ylabel('# Rounds')
        plt.legend()

        ax = plt.subplot(2,3,6)
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            #if not matchingAlgorithm == 't':
            plt.plot(GAdata[xData], GAdata['avgHoldTimeMatched'], colors[matchingAlgorithm], label=matchingAlgorithm)
            plt.plot(GAdata[xData], GAdata['avgHoldTimeUnmatched'], colors[matchingAlgorithm]+'--')
        plt.title('Hold Times')
        plt.xlabel(xLabel)
        plt.ylabel('# Rounds')
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/plot/Ptest-" + Gname + " " + title + ".png")
        plt.close()

def plotMetricsB2(datasets, xLabel, xData, title=""):
    #totalMatchDict = getTotalMatchDict()
    for Gname, Gdata in datasets.items():
        plt.figure(Gname + " " + title, figsize=(11.69, 8.27))
        ax = plt.subplot(2,3,1)
        xTestPoints = Gdata['gsc'][xData]
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            plt.plot(GAdata[xData], GAdata['totalMatched'], colors[matchingAlgorithm], label=matchingAlgorithm)
            plt.plot(GAdata[xData], GAdata['totalHeld'], colors[matchingAlgorithm]+'--')
        plt.title('Total Matched/Held')
        plt.xlabel(xLabel)
        plt.ylabel('# ORpairs')
        plt.legend()


        ax = plt.subplot(2,3,2)
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            #if not matchingAlgorithm == 't':
            plt.plot(GAdata[xData], GAdata['avgWaitTimeMatched'] - GAdata[xData] / 2, colors[matchingAlgorithm], label=matchingAlgorithm)
            plt.plot(GAdata[xData], GAdata['avgWaitTimeUnmatched'], colors[matchingAlgorithm]+'--')
        plt.title('Wait Time')
        plt.xlabel(xLabel)
        plt.ylabel('# ORpairs')
        plt.legend()

        ax = plt.subplot(2,3,3)
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            #if not matchingAlgorithm == 't':
            plt.plot(GAdata[xData], GAdata['avgCycleSize'], colors[matchingAlgorithm])
            plt.plot(GAdata[xData], GAdata['avgMatchSize'], colors[matchingAlgorithm]+'--', label=matchingAlgorithm)
        plt.title('Match Size')
        plt.xlabel(xLabel)
        plt.ylabel('# ORpairs')
        plt.legend()

        ax = plt.subplot(2,3,4)
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            if not matchingAlgorithm == 't':
                plt.plot(GAdata[xData], GAdata['avgPoDMatched'], colors[matchingAlgorithm], label=matchingAlgorithm)
                plt.plot(GAdata[xData], GAdata['avgPodUnmatched'], colors[matchingAlgorithm]+'--')
        plt.title('Matched Task Popularity')
        plt.xlabel(xLabel)
        plt.ylabel('Product of Degrees')
        plt.legend()

        ax = plt.subplot(2,3,5)
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            #if not matchingAlgorithm == 't':
            plt.plot(GAdata[xData], GAdata['avgWaitTimeHeldMatched'], colors[matchingAlgorithm], label=matchingAlgorithm)
            plt.plot(GAdata[xData], GAdata['avgWaitTimeHeldUnmatched'], colors[matchingAlgorithm]+'--')
        plt.title('Hold Wait Times')
        plt.xlabel(xLabel)
        plt.ylabel('# Rounds')
        plt.legend()

        ax = plt.subplot(2,3,6)
        ax.set_xticks(xTestPoints, minor=True)
        for matchingAlgorithm, GAdata in Gdata.items():
            #if not matchingAlgorithm == 't':
            plt.plot(GAdata[xData], GAdata['avgHoldTimeMatched'], colors[matchingAlgorithm], label=matchingAlgorithm)
            plt.plot(GAdata[xData], GAdata['avgHoldTimeUnmatched'], colors[matchingAlgorithm]+'--')
        plt.title('Hold Times')
        plt.xlabel(xLabel)
        plt.ylabel('# Rounds')
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/plot/Ptest2-" + Gname + " " + title + ".png")
        plt.close()

# To make manual toggling easy.
tests = {
         "nMatchtest":False
        ,"NinitialTest":False
        ,"timeTest":False
        ,"series":False
        ,"PtestB":False
        ,"PtestB2-0":False
        ,"PtestB2-1":False
        ,"PvTM":False
        ,"sPotTest":False
        ,"PtestGSC":False
        }

#tests["nMatchtest"] = True
#tests["NinitialTest"] = True
#tests["timeTest"] = True
#tests["series"] = True
#tests["PtestB"] = True
#tests["PtestB2-0"] = True
#tests["PtestB2-1"] = True
#tests["PvTM"] = True
#tests["sPotTest"] = True
tests["PtestGSC"] = True

#makeTotalMatchDict()
#makeTotalMatchDictP()


if tests["nMatchtest"]:
    filenames = glob.glob("results/nUtest*.csv")
    filenames1 = glob.glob("results/nMatchTest-1*.csv")
    #print(filenames)
    datasets = getDatasets(filenames)
    datasets1 = getDatasets(filenames1)
    plotMetrics(datasets, 'Match Frequency', 'nMatch', "nMatch")
    plotMetrics(datasets1, 'Match Frequency', 'nMatch', "nMatch 2")
    #plt.show()

if tests["NinitialTest"]:
    #filenames = glob.glob("results/Ninitialtest*.csv")
    filenames = glob.glob("results/NinitialTest*.csv")
    #print(filenames)
    datasets = getDatasetsN(filenames)
    plotMetrics(datasets, 'Ninitial', 'Ninitial', "Ninitial")
    #plt.show()

    #for Gname, Gdata in datasets.items():
    #    for matchingAlgorithm, data in Gdata.items():
    #        print("%s - %s: %s" % (Gname, matchingAlgorithm, data))

if tests["PtestB"]:
    filenames91 = glob.glob("results/grid/Ptest-RB1-*9-1.csv")
    filenames90 = glob.glob("results/grid/Ptest-RB1-*9-0.csv")
    filenames71 = glob.glob("results/grid/Ptest-RB1-*7-1.csv")
    filenames81 = glob.glob("results/grid/Ptest-RB1-*8-1.csv")
    datasets91 = getDatasets(filenames91)
    datasets90 = getDatasets(filenames90)
    datasets71 = getDatasets(filenames71)
    datasets81 = getDatasets(filenames81)
    plotMetricsB(datasets90, 'Match Frequency', 'nMatch', "p=0.9, hold=0")
    plotMetricsB(datasets71, 'Match Frequency', 'nMatch', "p=0.7, hold=1")
    plotMetricsB(datasets81, 'Match Frequency', 'nMatch', "p=0.8, hold=1")
    plotMetricsB(datasets91, 'Match Frequency', 'nMatch', "p=0.9, hold=1")
    plotMetricsB(getDatasets(glob.glob("results/grid/Btest*0.5*.csv")), 'Match Frequency', 'nMatch', "p=0.5, hold=1")
    plotMetricsB(getDatasets(glob.glob("results/grid/Btest*0.3*.csv")), 'Match Frequency', 'nMatch', "p=0.3, hold=1")
    #plt.show()

if tests["timeTest"]:
    filenames20 = glob.glob("results/timeTest*20.csv")
    filenames100 = glob.glob("results/timeTest*100.csv")
    filenamesGSCPoD = glob.glob("results/timeTest100*.csv")
    filenameMAX = glob.glob("results/timeTestMAX*csv") + glob.glob("results/timeTest-RB1*100.csv")
    #print(filenames20); print(filenames100); print(filenamesGSCPoD); print(filenameMAX)
    datasetList = [("20", getDatasets(filenames20)), ("100", getDatasets(filenames100)), ("100-2", getDatasets(filenamesGSCPoD)), ("100-MAX", getDatasets(filenameMAX))]
    #plotMetrics(datasets, 'Ninitial', 'Ninitial')

    xData = "NORpairs"
    yData = "matchTime"
    xLabel = "# ORpairs"
    yLabel = "Match Time (seconds)"

    def maxDimensions(ax1, ax2):
        (a,b,c,d) = ax1
        (w,x,y,z) = ax2
        return (min(a,w), max(b,x), min(c,y), max(d,z))

    for rnd, datasets in datasetList:
        axes = []
        for Gname, Gdata in datasets.items():
            plt.figure(Gname + " " + rnd, figsize=(11.69, 8.27))
            axes.append(plt.subplot(1,1,1))
            for matchingAlgorithm, GAdata in Gdata.items():
                #if not matchingAlgorithm == 't':
                plt.plot(GAdata[xData], GAdata[yData], colors[matchingAlgorithm]+'--', label=matchingAlgorithm)
            plt.title('Run Time with step size ' + rnd)
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            plt.legend()

        dimensions = (0,0,0,0)
        for ax in axes:
            dimensions = maxDimensions(dimensions, ax.axis())
        for ax in axes:
            ax.axis(dimensions)

    for rnd, datasets in datasetList:
        for Gname, Gdata in datasets.items():
            plt.figure(Gname + " " + rnd, figsize=(11.69, 8.27))
            plt.tight_layout()
            plt.savefig("results/plot/timeTest-" + Gname + " " + rnd + ".png")
            plt.close()
    #plt.show()

if tests["series"]:
    series = getDatasets(glob.glob("results/timeTestLong-*.csv"))

    plt.figure("Graph Size Over Time", figsize=(11.69, 8.27))
    i = 1
    axes = []
    for Gname, Gdata in series.items():
        axes.append(plt.subplot(1,3,i))
        NORseries = Gdata['d']['NORpairs']
        stepSeries = Gdata['d']['Step']
        plt.plot(stepSeries, NORseries, colors['d'], label='d')
        if Gname == "EN":
            NORseries = Gdata['gsc']['NORpairs']
            stepSeries = Gdata['gsc']['Step']
            plt.plot(stepSeries, NORseries, colors['gsc'], label='gsc')
        plt.xlabel('Step')
        plt.ylabel('NORpairs')
        plt.title(Gname + " -- DYN")
        plt.legend()
        plt.gca().set_aspect('equal')
        i += 1

    def maxDimensions(ax1, ax2):
        (a,b,c,d) = ax1
        (w,x,y,z) = ax2
        mi = min(min(a,w), min(c,y))
        ma = max(max(b,x), max(d,z))
        return (mi, ma, mi, ma)

    dimensions = (0,0,0,0)
    for ax in axes:
        dimensions = maxDimensions(dimensions, ax.axis())
    for ax in axes:
        ax.axis(dimensions)

    plt.tight_layout
    plt.savefig("results/plot/timeSeries.png")
    plt.close()

    series = getDatasetsP(glob.glob("results/timeTestLong2-*.csv"))
    plt.figure("Graph Size Over Time - p", figsize=(11.69, 8.27))
    for p, Gdata in series.items():
        NORseries = Gdata['NORpairs']
        stepSeries = Gdata['Step']
        plt.plot(stepSeries, NORseries, colors['gsc'], label='p = ' + str(p))
    plt.xlabel('Step')
    plt.ylabel('NORpairs')
    plt.title("RB1 Graph Size Over Time - p")
    plt.legend()
    plt.gca().set_aspect('equal')

    plt.tight_layout
    plt.savefig("results/plot/timeSeries2.png")
    plt.close()

if tests["PtestB2-0"]:
    datasets90 = getDatasets(glob.glob("results/grid/Ptest2-RB1-*9-0.csv"))
    plotMetricsB2(datasets90, 'Match Frequency', 'nMatch', "p=0.9, hold=0")
    datasets70 = getDatasets(glob.glob("results/grid/Ptest2-RB1-*7-0.csv"))
    plotMetricsB2(datasets70, 'Match Frequency', 'nMatch', "p=0.7, hold=0")
    datasets50 = getDatasets(glob.glob("results/grid/Ptest2-RB1-*5-0.csv"))
    plotMetricsB2(datasets50, 'Match Frequency', 'nMatch', "p=0.5, hold=0")
    datasets30 = getDatasets(glob.glob("results/grid/Ptest2-RB1-*3-0.csv"))
    plotMetricsB2(datasets30, 'Match Frequency', 'nMatch', "p=0.3, hold=0")

if tests["PtestB2-1"]:
    for p in [0.9, 0.7, 0.8, 0.6, 0.5, 0.3]:
        plotMetricsB2(getDatasets(glob.glob("results/grid/Ptest2-RB1-*"+str(p)+"-1.csv"))
                     ,'Match Frequency', 'nMatch', 'p='+str(p)+', hold=1')

        # Initiall matched
        # 0.3 - 90
        # 0.5 - 400
        # 0.7 - 1100
        # 0.9 - 2400
        # 1.0 - 3300


if tests["PvTM"]:
    totalMatchDict = getTotalMatchDict()
    ps = []
    plt.figure("p vs Total Matched", figsize=(16, 4))
    for hold in [0, 1]:
        bestTMs = []
        if hold == 1:
            ps = [0.9, 0.7, 0.8, 0.6, 0.5, 0.3]
        else:
            ps = [0.9, 0.7, 0.5, 0.3]
        for p in ps:
            #ds = getDatasets(glob.glob("results/grid/Ptest2-RB1-*"+str(p)+"-"+str(hold)+".csv"))['RB1']
            best = 0
            #for matchingAlgorithm, GAdata in ds.items():
            #    best = max(best, max(GAdata['totalMatched']))
            for matchingAlgorithm in ['GSC', 'GSCPoD', 'DYN']:
                best = max(best, max(totalMatchDict['RB1'+str(hold + p) + matchingAlgorithm]))
            bestTMs.append(best)
        bestTMsAccounted = []
        if hold == 1:
            bestTMsAccounted.append(bestTMs[0] - (2400 - 2400))
            bestTMsAccounted.append(bestTMs[1] - (2400 - 1100))
            bestTMsAccounted.append(bestTMs[4] - (2400 - 400))
            bestTMsAccounted.append(bestTMs[5] - (2400 - 90))
            plt.subplot(1,4,4)
            plt.plot([0.9, 0.7, 0.5, 0.3], bestTMsAccounted, 'o')
            plt.xlabel('p')
            plt.ylabel('Total Matched')
            plt.title("RB1 - " + str(hold) + " - Accounted")
        if hold == 0:
            bestTMsAccounted.append(bestTMs[0] - (2400 - 2400))
            bestTMsAccounted.append(bestTMs[1] - (2400 - 1100))
            bestTMsAccounted.append(bestTMs[2] - (2400 - 400))
            bestTMsAccounted.append(bestTMs[3] - (2400 - 90))
            plt.subplot(1,4,2)
            plt.plot([0.9, 0.7, 0.5, 0.3], bestTMsAccounted, 'o')
            plt.xlabel('p')
            plt.ylabel('Total Matched')
            plt.title("RB1 - " + str(hold) + " - Accounted")


        plt.subplot(1,4,hold*2+1)
        plt.plot(ps, bestTMs, 'o')
        plt.xlabel('p')
        plt.ylabel('Total Matched')
        plt.title("RB1 - " + str(hold) + " - p vs Total Matched")
    plt.tight_layout
    plt.savefig("results/plot/PvTM.png")
    plt.close()


if tests["sPotTest"]:
    p = 0.9
    plotMetricsB2(getDatasets(glob.glob("results/grid/Ptest2-EN-*"+str(p)+"-0.csv"))
                  ,'Match Frequency', 'nMatch', 'p='+str(p)+', hold=0')
    plotMetricsB2(getDatasets(glob.glob("results/grid/Ptest3-EN-*0.3-1.csv"))
                  ,'Match Frequency', 'nMatch', 'p=0.3, hold=1')
    plotMetricsB2(getDatasets(glob.glob("results/grid/Ptest3-BM1-*0.3-0.csv"))
                  ,'Match Frequency', 'nMatch', 'p=0.3, hold=0')
    plotMetricsB2(getDatasets(glob.glob("results/grid/Ptest3-BM1-*0.7-1.csv"))
                  ,'Match Frequency', 'nMatch', 'p=0.7, hold=1')
#plt.show()


if tests["PtestGSC"]:
    gscd = dict()
    filenames = ''
    for i in [1,2,3]:
        gscd[i] = []
        if i > 1:
            filenames = glob.glob("results/grid/PtestGSC" + str(i) + "*.csv")
        else:
            filenames = glob.glob("results/grid/PtestGSC-*.csv")
        for filen in filenames:
            gscd[i].append(getDatasets([filen]))
    plt.figure("p vs Total Matched", figsize=(11.69, 8.27))
    for num, gscL in gscd.items():
        ps = []
        tm = []
        ma = ''
        ni = 0
        ne = 0
        for gscR in gscL:
            for Gname, Gdata in gscR.items():
                for matchingAlgorithm, GAdata in Gdata.items():
                    ps.append(GAdata['p'])
                    tm.append(GAdata['totalMatched'])
                    ma = matchingAlgorithm
                    ni = GAdata['Ninitial']
                    ne = GAdata['Nend']
        plt.subplot(1,3,num)
        plt.plot(ps, tm, 'o')
        plt.xlabel('p')
        plt.ylabel('Total Matched')
        plt.title("RB1:gsc" + " " + str(ni) + " - " + str(ne))
    plt.tight_layout
    plt.savefig("results/plot/PvTM-GSC.png")
    plt.close()
