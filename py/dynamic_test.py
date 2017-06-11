#!/usr/bin/env python

from neo4j.v1 import GraphDatabase
import sys
import numpy as np
import networkx as nx
from graph_gen import *
from graph_control import *


def get_matches(driver, matchingAlgorithm, hold, p, justAddedIDs, stats, G=None):
    if matchingAlgorithm == 'b':
        bmatch1(driver)
    elif matchingAlgorithm == 'gsc':
        gscmatch(driver)
    elif matchingAlgorithm == 't':
        twomatch(driver)
    elif matchingAlgorithm == 'd':
        dynamicMatch(driver, justAddedIDs)
    else:
        raise ValueError('Incorrect matching algorithm key.')
    (total, totalPairs, cycles, waitTimes, G) = removecycles(driver, hold, p, stats, G)
    return (total, totalPairs, cycles, stats, G)

# Initially, Erdos-Renyi style update
# pt:- % of Ntasks new tasks
# pu:- % of Nusers new users
def next_cycle(driver, taskID, userID, NORnodes, nodeID, NnewTasks, NnewUsers, NnewORnodes, G=None):
    Nusers = userID + NnewUsers; #Ntasks = taskID + NnewTasks;
    generate_users(driver, userID,  Nusers);
    #def update_sfgX(driver, taskID, nodeID, NnewORnodes, Nusers, G):
    (Ntasks, nodeID, G) = update_sfgX(driver, taskID, nodeID, NnewORnodes, Nusers, G)
    NORnodes += NnewORnodes
    #(Ntasks, Nusers, num) = update_er(driver, Ntasks, Nusers, nodeID, NnewORnodes)
    #(Ntasks, Nusers, num) = update_wsg(driver, Ntasks, Nusers, nodeID, NnewORnodes, 2, 0.5)
    #(Ntasks, Nusers, num) = update_sfg(driver, Ntasks, Nusers, nodeID, NnewORnodes)
    #(Ntasks, Nusers, num) = update_plc(driver, Ntasks, Nusers, nodeID, NnewORnodes, 2, 0.3)
    #(Ntasks, Nusers, num) = update_ba(driver, Ntasks, Nusers, nodeID, NnewORnodes, 3)
    # NORnodes = NORnodes + num; nodeID = nodeID + num
    print("NORnodes: %s; nodeid: %s; new: %s" % (NORnodes, nodeID, NnewORnodes))
    return (Ntasks, Nusers, NORnodes, nodeID)

def run_cycle(driver, matchingAlgorithm, hold, Ntasks, Nusers, NORnodes, nodeID, p, stepSize, stats, G=None):
    (total, totalPairs, cycles, stats, G) = get_matches(driver, matchingAlgorithm, hold, p, (nodeID, stepSize), stats, G)
    NORnodes = NORnodes - total
    delete_bipartite(driver); delete_bmatch(driver)
    (pt, pu) = (0.05, 0.05)
    NnewUsers = int(stepSize / 6)
    NnewORnodes = stepSize #int(10 * max(1, total))#int(p * NORnodes)
    (Ntasks, Nusers, NORnodes, nodeID) = next_cycle(driver, Ntasks, Nusers, NORnodes, nodeID
                                           , 1, NnewUsers, NnewORnodes, G)
    return (Ntasks, Nusers, NORnodes, nodeID, total, totalPairs, stats, G)

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

if __name__ == "__main__":
    #sys.stdout = open('output', 'w')
    num_args = len(sys.argv)
    matchingAlgorithm = sys.argv[1]
    hold = bool(int(sys.argv[2]))
    runNumNodes = int(sys.argv[3]);
    stepSize = int(sys.argv[4])
    p = float(sys.argv[5]) #0.6; # probabiltiy of rejecting
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "iwin"))
    delete_all(driver)
    (Ntasks, Nusers, NORnodes, G) = generate_sfgX(driver, stepSize, max(2, int(stepSize/ 3))) # NORnodes #Nusers
    #(Ntasks, Nusers, NORnodes) = generate_er(driver, 20, 50, 150)
    #(Ntasks, Nusers, NORnodes) = generate_plc(driver, 60, 30, 2, 0.3)
    #(Ntasks, Nusers, NORnodes) = generate_ba(driver, 50, 50, 1)
    #(Ntasks, Nusers, NORnodes) = generate_sfg(driver, 120, 30)
    #(Ntasks, Nusers, NORnodes) = generate_wsg(driver, 50, 50, 2, 0.9)
    totals = []; totalPairs = []; waitTimes = dict(); holdTimes = dict()
    stats = (waitTimes, holdTimes)

    nodeID = NORnodes
    i = 0;
    #for i in range(0,int(sys.argv[1])):
    print("nodeID: %s; NORnodes: %s; runNumNodes: %s" % (nodeID, NORnodes, runNumNodes))
    while nodeID < runNumNodes:
        print(">>> Step %s" % i)
        (Ntasks, Nusers, NORnodes, nodeID, total, totalPair, stats, G) = run_cycle(driver, matchingAlgorithm, hold, Ntasks, Nusers, NORnodes, nodeID, p, stepSize, stats, G)
        totals.append(total); totalPairs.append(totalPair)
        i+=1


    print(list(zip(totals, totalPairs)))
    print("Matched a total of %s nodes" % sum(totals))
    waitTimes = stats[0]; holdTimes = stats[1]
    with driver.session() as session:
        result = session.run("MATCH (n:ORnode) RETURN n.wait, count(n.wait) ORDER BY n.wait ")
        print("Wait time : Number of nodes -- unmatched")
        for record in result:
            print("%s : %s" % (record['n.wait'], record['count(n.wait)']))
        result = session.run("MATCH (n:ORnode) WHERE n.waitTimes <>  '' RETURN n.wait, n.waitTimes, count(n.waitTimes) ORDER BY count(n.waitTimes) ")
        print("Number of nodes : Held rounds : Wait Time : History -- unmatched")
        for record in result:
            history = record['n.waitTimes']; lhistory = literal_eval('('+history+')')
            print("%s : %s : %s : %s" % (record['count(n.waitTimes)'], calcHeldRounds(lhistory), calcHeldRoundsWaitTime(lhistory, record['n.wait']), history))
        print("Wait time : Number of nodes -- matched")
        for wait_time, num_nodes in waitTimes.items():
            print("%s : %s" % (wait_time, num_nodes))
        print("Hold time : Wait time : Number of nodes -- matched")
        for (hold_time, wait_time), num_nodes in sorted(holdTimes.items()):
            print("%s : %s : %s" % (hold_time, wait_time, num_nodes))
