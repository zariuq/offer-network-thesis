#!/usr/bin/env python

from neo4j.v1 import GraphDatabase
import sys
import time
import numpy as np
import networkx as nx
import pickle
from graph_gen import *
from graph_control import *


def get_matches(driver, matchingAlgorithm, hold, p, justAddedIDs, stats, G):
    time_pre_match = time.time()
    if matchingAlgorithm == 'b':
        bmatch1(driver)
    elif matchingAlgorithm == 'gsc':
        gscmatch(driver)
    elif matchingAlgorithm == 'gscpod':
        gscPoDmatch(driver)
    elif matchingAlgorithm == 't':
        twomatch(driver)
    elif matchingAlgorithm == 'd':
        dynamicMatchL(driver, justAddedIDs)
    elif matchingAlgorithm == 'n':
        # do nothing
        return (0, 0, 0, [], stats)
    else:
        raise ValueError('Incorrect matching algorithm key.')
    (total, totalPairs, cycles, stats, G) = removecycles(driver, hold, p, stats, G)
    delete_bmatch(driver)
    time_post_match = time.time()
    matchTime = np.around(time_post_match - time_pre_match, decimals = 3)
    return (total, totalPairs, matchTime, cycles, stats)

# Process updates in a batch between matches
def addUpdates(driver, updates):
    if len(updates) > 1000:
        updatesBatchOne = updates[:1000]
        updates = updates[1000:]
        return addUpdates(driver, updatesBatchOne) + addUpdates(driver, updates)
    else:
        userCommands, task_commands, nodeIDsAndOrPairCommands = zip(*updates)
        nodeIDs, orPairCommands = zip(*nodeIDsAndOrPairCommands)
        with driver.session() as session:
            if any(userCommands + task_commands):
                createCommand = " \n".join(userCommands + task_commands)
                with session.begin_transaction() as tx:
                    tx.run(createCommand)
            with session.begin_transaction() as tx:
                for command in orPairCommands:
                    tx.run(command)
        return nodeIDs

# Recursively calculates how long a node was held for
def calcHeldRounds(h):
    if len(h) == 0:
        return 0
    return 1 + max(calcHeldRounds(h[0]), calcHeldRounds(h[1]))

# Recursively calculates node held most times
def calcHeldRoundsWaitTime(h, w):
    if len(h) == 0:
        return 0
    return w + h[2] + h[3] + max(calcHeldRoundsWaitTime(h[0], 0), calcHeldRoundsWaitTime(h[1], 0))

def run_test(driver, matchingAlgorithm, hold, p, Ninitial, Nend, nU, nMatch, G, updates, Gname, results, timeResults, statList, unmatcheable):
    avgCycleLength = []; totals = []; totalPairs = []; times = []
    matchTime = 0
    stats = (dict(), dict(), dict()) # waittimes, holdtimes (history), PoD count

    # Initialize graph
    #print("Initializing Graph")

    # Add the initial 3 ORpairs
    # This has to be done because task/users should be added before ORpairs, and initially more than one have to be added in a step.
    node_commands = []
    for i in range(0,3):
        node_commands.append(updates[i][0])
        node_commands.append(updates[i][1])
    command = "\n".join(node_commands)
    with driver.session().begin_transaction() as tx:
        tx.run(command)
    updates = updates[3:]

    # Add the ORpairs from #0
    init = updates[:(Ninitial-3)]
    updates = updates[(Ninitial-3):]
    nodeIDs = addUpdates(driver, init)
    #print("Creating index on ORpair IDs")
    with driver.session() as session:
        with session.begin_transaction() as tx:
            tx.run("CREATE INDEX ON :ORnode(id) ")

    # Set initial wait to -1 to distinguish initial ORpairs from ones added in each step.
    #print("Setting initial wait time to -1")
    with driver.session() as session:
        session.run("MATCH (n:ORnode) SET n.wait = -1")

    # Do initial matching, but don't collect stats
    #print("Step %s / %s" % (Nend - len(updates), Nend))
    (total, totalPair, matchTime, cycles, stats)  = get_matches(driver, 'gsc', hold, 1, nodeIDs, stats, G)
    stats = (dict(), dict(), dict())

    print("Initially matched %s pairs." % total)

    # Run dynamic test
    while updates:
        #print("Updating next graph to next match round")
        toUpdate = updates[:nMatch]
        updates = updates[nMatch:]
        nodeIDs = addUpdates(driver, toUpdate)
        step = Nend - len(updates)
        print("Step %s / %s. Previous match time %s" % (step, Nend, matchTime))

        (total, totalPair, matchTime, cycles, stats) = get_matches(driver, matchingAlgorithm, hold, p, nodeIDs, stats, G)
        if len(cycles) > 0:
            acceptedCycleLength = np.around(total/len(cycles), decimals=2)
            cycleLength = np.around(sum(map(len,cycles))/ len(cycles), decimals=2)
            avgCycleLength.append((cycleLength, acceptedCycleLength))
        else: cycleLength = 0
        totals.append(total); totalPairs.append(totalPair); times.append(matchTime)
        timeResults.append([Gname, matchingAlgorithm, hold, p, nMatch, step, cycleLength, total, totalPair, matchTime])

    print(list(zip(totals, totalPairs, times)))
    totalMatched = sum(totals)
    if len(avgCycleLength) > 0:
        avgs = list(zip(*avgCycleLength))
        avgMatchSize = np.around((sum(avgs[1]) / len(avgs[1])), decimals=2)
        avgCycleSize = np.around((sum(avgs[0]) / len(avgs[0])), decimals=2)
    else: avgMatchSize = 0; avgCycleSize = 0
    print("Matched a total of %s ORpairs" % totalMatched)
    print("With an average of %f ORpairs per match" %  avgMatchSize)
    waitTimes = stats[0]; holdTimes = stats[1]; podCounts = stats[2] ###

    countUnmatched = 0; countMatched = 0;
    totalWaitTimeMatched = 0; totalWaitTimeUnmatched = 0
    avgWaitTimeUnmatched = 0; avgWaitTimeMatched = 0
    totalHeldMatched = 0; totalHeldUnmatched = 0
    avgHoldTimeMatched = 0; avgHoldTimeUnmatched = 0
    avgWaitTimeHeldMatched = 0; avgWaitTimeHeldUnmatched = 0
    matchedPoD = 0; unmatchedPoD = 0; degrees = G.degree()
    with driver.session() as session:
        result = session.run("MATCH (n:ORnode) RETURN n.wait, count(n.wait), collect(n.offer), collect(n.request) ORDER BY n.wait ")
        #print("Wait rounds : steps : Number of nodes -- unmatched")
        for record in result:
            w = record['n.wait']; c = record['count(n.wait)']
            #print("%s : %s : %s" % (w, w * nMatch, c))
            totalWaitTimeUnmatched += w * nMatch * c
            countUnmatched += c
            for offer, request in zip(record['collect(n.offer)'], record['collect(n.request)']):
                unmatchedPoD += degrees[offer]*degrees[request]
        result = session.run("MATCH (n:ORnode) WHERE n.waitTimes <>  '' RETURN n.wait, n.waitTimes, count(n.waitTimes) ORDER BY count(n.waitTimes) ")
        #print("Number of nodes : Held rounds : Wait Time : History -- unmatched")
        for record in result:
            history = record['n.waitTimes']; lhistory = literal_eval('('+history+')')
            c = record['count(n.waitTimes)']
            totalHeldUnmatched += c
            avgHoldTimeUnmatched += c * calcHeldRounds(lhistory)
            avgWaitTimeHeldUnmatched += c * nMatch * calcHeldRoundsWaitTime(lhistory, record['n.wait'])
            #print("%s : %s : %s : %s" % (record['count(n.waitTimes)'], calcHeldRounds(lhistory), calcHeldRoundsWaitTime(lhistory, record['n.wait']), history))
        #print("Wait rounds : steps : Number of nodes -- matched")
        for wait_time, num_nodes in sorted(waitTimes.items()):
            #print("%s : %s : %s" % (wait_time, wait_time * nMatch, num_nodes))
            totalWaitTimeMatched += wait_time * nMatch * num_nodes
            countMatched += num_nodes
        #print("Hold time : Wait time : Number of nodes -- matched")
        for (hold_time, wait_time), num_nodes in sorted(holdTimes.items()):
            if hold_time > 0:
                totalHeldMatched += num_nodes
                avgHoldTimeMatched += num_nodes * hold_time
                avgWaitTimeHeldMatched += num_nodes * nMatch * wait_time
            #print("%s : %s : %s" % (hold_time, wait_time, num_nodes))
        for pod, count in podCounts.items():
            matchedPoD += pod * count
        if countMatched > 0:
            avgWaitTimeMatched = np.around(totalWaitTimeMatched / countMatched, decimals=2)
            matchedPoD = np.around(matchedPoD / countMatched, decimals=2)
        if totalHeldUnmatched > 0:
            avgHoldTimeUnmatched = np.around(avgHoldTimeUnmatched / totalHeldUnmatched, decimals=2)
            avgWaitTimeHeldUnmatched = np.around(avgWaitTimeHeldUnmatched / totalHeldUnmatched, decimals=2)
        if totalHeldMatched > 0:
            avgHoldTimeMatched = np.around(avgHoldTimeMatched / totalHeldMatched, decimals=2)
            avgWaitTimeHeldMatched = np.around(avgWaitTimeHeldMatched / totalHeldMatched, decimals=2)
        if countUnmatched > 0:
            avgWaitTimeUnmatched = np.around(totalWaitTimeUnmatched / countUnmatched, decimals=2)
            unmatchedPoD = np.around(unmatchedPoD / countUnmatched, decimals=2)
        print("Average wait time -- unmatched: %f" % avgWaitTimeUnmatched)
        print("Average PoD -- unmatched: %f" % unmatchedPoD)
        print("Average wait time -- matched: %f" % avgWaitTimeMatched)
        print("Average PoD -- matched: %f" % matchedPoD)
        results.append([Gname, matchingAlgorithm, hold, p, Ninitial, Nend, nU, nMatch, totalMatched, unmatcheable, avgCycleSize, avgMatchSize, avgWaitTimeMatched, avgWaitTimeUnmatched, (totalHeldMatched + totalHeldUnmatched), avgHoldTimeMatched, avgWaitTimeHeldMatched, avgHoldTimeUnmatched, avgWaitTimeHeldUnmatched, matchedPoD, unmatchedPoD])

        statList.append(stats)

mName = {'b': 'MAX', 'gsc':'GSC', 't':'TWO', 'd':'DYN', 'gscpod':'GSCPoD'}
G = nx.Graph()

# input args: [b gsc t d] [0 1] p Ninitial Nend nU nMatch
if __name__ == "__main__":
    #sys.stdout = open('output', 'w')
    num_args = len(sys.argv)
    if num_args == 3 and sys.argv[1] == "test":
        filename = sys.argv[2]
        matchingAlgorithm = 'b'
        hold = 1
        p = 0.8
        nMatch = 100

        # for Ninitial test, generate the max needed
        #Nend = 7000
        #Nend = 23000

        # For time test
        #Nend = 15000

        Gname = "RB1"
        #Gname = "BM1"
        #Gname = "EN"

        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "iwin"))
        delete_all(driver)

        ## Test 1 with MAX-WEIGHT, so starting low
        ## Moreover, nMatch does not seem to matter that much except for the wait time

        '''
        Ninitial = 400
        Nend = 3400
        nU = 10 # Irrelevant, really
        Graphs = ["RB1"]#, "BM1", "EN"]
        ps = {0: [0.9], 1: [0.7, 0.8, 0.9]}
        matchingAlgorithms = ["gscpod", "gsc", "t", "d", "b"]
        holds = [0,1]
        nMatchRange = {"gsc"   : [1, 5, 10, 25, 50, 100, 250]
                      ,"d"     : [1, 5, 10, 25, 50, 100, 250]
                      ,"gscpod": [5, 10, 25, 50, 100, 250]
                      ,"t"     : [5, 10, 25, 50, 100, 250]
                      ,"b"     : [40, 100, 250, 500]}
        '''

        ## Test 1 without MAX-WEIGHT, so starting high
        ## Moreover, nMatch does not seem to matter that much except for the wait time

        '''
        Ninitial = 10000
        Nend = 15000
        nU = 10 # Irrelevant, really
        Graphs = ["RB1", "EN"]#, "BM1"]
        ps = {0: [0.9, 0.7, 0.5, 0.3], 1: [0.9, 0.7, 0.8, 0.6, 0.5, 0.3]}
        matchingAlgorithms = ["gscpod", "gsc", "d"] # t only for p < 0.7
        holds = [0,1]
        nMatchRange = {"gsc"   : [1, 5, 25, 50, 100]
                      ,"d"     : [1, 5, 25, 50, 100]
                      ,"gscpod": [50, 100]
                      ,"t"     : [5, 25, 100]}
        '''
        params = dict()
        Graphs = []
        matchingAlgorithms = []
        nMatchRange = dict()
        for testNum in [2]:
            if testNum == 0:
                ## Spot test
                filename = "grid/PtestGSC3"
                Ninitial = 10000
                Nend = 15000
                nU = 10 # Irrelevant, really
                Graphs = ["EN"]
                params = {
                          "EN"  : ([1], {1 : [0.3, 0.5, 0.7, 0.9, 1]})
                         }
                matchingAlgorithms = ["gsc"] # t only for p < 0.7
                nMatchRange = {"gsc"   : [20]}

                # Initiall matched
                # 0.3 - 90
                # 0.5 - 400
                # 0.7 - 1100
                # 0.9 - 2400
                # 1.0 - 3300
                '''
                if testNum == 0:
                    ## Spot test
                    filename = "grid/Ptest3"
                    Ninitial = 10000
                    Nend = 15000
                    nU = 10 # Irrelevant, really
                    Graphs = ["EN", "BM1"]
                    params = {
                              "EN"  : ([1], {1 : [0.3]})
                             ,"BM1" : ([0,1], {0 : [0.3], 1 : [0.7]})
                             }
                    matchingAlgorithms = ["gscpod", "gsc"] # t only for p < 0.7
                    nMatchRange = {"gsc"   : [5, 20, 50]
                                  ,"gscpod": [50, 100]}
                '''

            elif testNum == 1:
                # further b test
                filename = "grid/Btest"
                Ninitial = 400
                Nend = 3400
                nU = 10 # Irrelevant, really
                Graphs = ["EN", "BM1"]
                params = {
                           "EN"  : ([1], {1: [0.3]})
                          ,"BM1" : ([1], {1: [0.5]})
                        }
                matchingAlgorithms = ["gsc", "b"]
                nMatchRange = {"gsc"   : [1, 5, 20, 50]
                              ,"b"     : [40, 100]}

            elif testNum == 2:
                filename = "timeTestLong2"
                Ninitial = 100
                Nend = 35000
                nU = 10 # Irrelevant, really
                Graphs = ["RB1"]#"EN", "BM1"]
                params = {
                          "RB1"  : ([1], {1: [0.3, 0.5, 0.7, 0.9, 1]})
                          ,"EN"  : ([0], {0: [1]})
                          ,"BM1" : ([0], {0: [1]})
                        }
                matchingAlgorithms = ["gscpod"]
                nMatchRange = {"gscpod"   : [50]}

            runNum = 1

            #if True:
            for Gname in Graphs:

                # Generate update list for test
                (G, updates, Nusers, Ntasks) = (nx.Graph(), [], 0, 0)

                # 1.54 1.85 1.45
                if Gname == "RB1":
                    (G, updates, Nusers, Ntasks) = generate_sfgL(driver, Nend, nU, alpha=0.35,gamma=0.15,beta=0.5,directed_p=0.05,delta_in=1.0,delta_out=0.5)

                # 1.74 2.35 1.42
                if Gname == "BM1":
                    (G, updates, Nusers, Ntasks) = generate_sfgL(driver, Nend, nU, alpha=0.53,gamma=0.12,beta=0.35,directed_p=0.05,delta_in=0.25,delta_out=0.05)

                # 1.36
                if Gname == "EN":
                    (G, updates, Nusers, Ntasks) = generate_sfgL(driver, Nend, nU, alpha=0.3,gamma=0.3,beta=0.4,directed_p=0.1,delta_in=0.5,delta_out=0.5)

                unmatcheable =  len(list(filter(lambda x: x[0] == '1' or x[1] == '1', G.edges())))
                print("There are %s unmatcheable ORpairs" % unmatcheable)

                holds, ps = params[Gname]
                #if True:
                for hold in holds:
                    for p in ps[hold]:
                        for matchingAlgorithm in matchingAlgorithms:

                            results = [["Gname", "matchingAlgorithm", "hold", "p", "Ninitial", "Nend", "nU", "nMatch", "totalMatched", "unMatcheable", "avgCycleSize", "avgMatchSize", "avgWaitTimeMatched", "avgWaitTimeUnmatched", "totalHeld", "avgHoldTimeMatched", "avgWaitTimeHeldMatched", "avgHoldTimeUnmatched", "avgWaitTimeHeldUnmatched", "avgPoDMatched", "avgPodUnmatched"]]

                            statList = []

                            timeResults = [["Gname", "matchingAlgorithm", "hold", "p", "nMatch", "Step", "avgMatchSize", "matched", "NORpairs", "matchTime"]]

                            runNum += 1
                            for nMatch in nMatchRange[matchingAlgorithm]:

                                print("Starting test #%s-%s: %s %s %s %s  %s:%s" % (testNum, runNum, Gname, matchingAlgorithm, hold, p, nMatch, nMatchRange[matchingAlgorithm]))

                        # For Ninitial test
                        #    for Ninitial in [10, 50, 100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]:
                        #        Nend = Ninitial + 3000

                    #    if True: # Time test

                                current_updates = updates[:Nend]
                                run_test(driver, matchingAlgorithm, hold, p, Ninitial, Nend, nU, nMatch, G, current_updates, Gname, results, timeResults, statList, unmatcheable)
                                delete_all(driver)


                            name = filename + '-' + Gname + '-' + mName[matchingAlgorithm] + '-' + str(p) + '-' + str(hold)
                            np.savetxt("results/"+name+".csv", np.asarray(results), fmt='%s', delimiter=',')
                            pickle.dump(statList, open("results/"+name+".p", "wb"))
                                # inverse is statList = pickle.load(open("results/"+name+".p", "rb"))
                            if testNum == 2:
                                np.savetxt("results/"+name+".csv", np.asarray(timeResults), fmt='%s', delimiter=',')


    elif num_args != 8:
        print("Input args: [b gsc t d n] [0 1] p Ninitial Nend nU nMatch")
    else:
        # Read in data
        matchingAlgorithm = sys.argv[1]
        hold = bool(int(sys.argv[2]))
        p = float(sys.argv[3])
        Ninitial = int(sys.argv[4]);
        Nend = int(sys.argv[5])
        nU = int(sys.argv[6])
        nMatch = int(sys.argv[7])
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "iwin"))
        delete_all(driver)

        results = [["Gname", "matchingAlgorithm", "hold", "p", "Ninitial", "Nend", "nU", "nMatch", "totalMatched", "unMatcheable", "avgCycleSize", "avgMatchSize", "avgWaitTimeMatched", "avgWaitTimeUnmatched", "totalHeld", "avgHoldTimeMatched", "avgWaitTimeHeldMatched", "avgHoldTimeUnmatched", "avgWaitTimeHeldUnmatched", "avgPoDMatched", "avgPodUnmatched"]]

        statList = []

        timeResults = [["Gname", "matchingAlgorithm", "hold", "p", "nMatch", "Step", "avgMatchSize", "matched", "NORpairs", "matchTime"]]

        # Generate update list for test
        Gname = "RB1"
        (G, updates, Nusers, Ntasks) = generate_sfgL(driver, Nend, nU, alpha=0.35,gamma=0.15,beta=0.5,directed_p=0.05,delta_in=1.0,delta_out=0.5)

        unmatcheable =  len(list(filter(lambda x: x[0] == '1' or x[1] == '1', G.edges())))
        print("There are %s unmatcheable ORpairs" % unmatcheable)

        run_test(driver, matchingAlgorithm, hold, p, Ninitial, Nend, nU, nMatch, G, updates, Gname, results, timeResults, statList, unmatcheable)
        #delete_all(driver)

        for result in results:
            print(result)
        for wl, hl, pl in statList:
            print(wl)
            print(hl)
            print(pl)
