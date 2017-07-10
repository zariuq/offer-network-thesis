#!/usr/bin/env python

from neo4j.v1 import GraphDatabase
import sys
import numpy as np
import networkx as nx
from graph_gen import *
from graph_control import *


def get_matches(driver, matchingAlgorithm, hold, p, justAddedIDs, stats):
    if matchingAlgorithm == 'b':
        bmatch1(driver)
        delete_bipartite(driver); delete_bmatch(driver)
    elif matchingAlgorithm == 'gsc':
        gscmatch(driver)
    elif matchingAlgorithm == 't':
        twomatch(driver)
    elif matchingAlgorithm == 'd':
        dynamicMatch(driver, justAddedIDs)
    else:
        raise ValueError('Incorrect matching algorithm key.')
    (total, totalPairs, cycles, stats, G) = removecycles(driver, hold, p, stats)
    return (total, totalPairs, cycles, stats)

# Process updates in a batch between matches
def addUpdates(driver, updates):
    userCommands, task_commands, nodeIDsAndOrPairCommands = zip(*updates)
    nodeIDs, orPairCommands = zip(*nodeIDsAndOrPairCommands)
    createCommand = " \n".join(userCommands + task_commands)
    with driver.session() as session:
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

# input args: [b gsc t d] [0 1] p Ninitial Nend nU nMatch
if __name__ == "__main__":
    #sys.stdout = open('output', 'w')
    num_args = len(sys.argv)
    if num_args != 8:
        print("Input args: [b gsc t d] [0 1] p Ninitial Nend nU nMatch")
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

        # Generate update list for test
        (G, updates, Nusers, Ntasks) = generate_sfgL(driver, Nend, nU, alpha=0.15,gamma=0.15,beta=0.7,directed_p=0.05,delta_in=0.75,delta_out=0.75)

        avgCycleLength = []; totals = []; totalPairs = [];
        waitTimes = dict(); holdTimes = dict()
        stats = (waitTimes, holdTimes)

        # Initialize graph
        init = updates[:Ninitial]
        updates = updates[Ninitial:]
        addUpdates(driver, init)
        with driver.session() as session:
            with session.begin_transaction() as tx:
                tx.run("CREATE INDEX ON :ORnode(id) ")

        # Run dynamic test
        while updates:
            toUpdate = updates[:nMatch]
            updates = updates[nMatch:]
            nodeIDs = addUpdates(driver, toUpdate)
            (total, totalPair, cycles, stats) = get_matches(driver, matchingAlgorithm, hold, p, nodeIDs, stats)
            avgCycleLength.append(total/len(cycles)); totals.append(total); totalPairs.append(totalPair)


        print(list(zip(totals, totalPairs)))
        print("Matched a total of %s ORpairs" % sum(totals))
        print("With an average of %f ORpairs per match" % (sum(avgCycleLength) / len(avgCycleLength)) )
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
