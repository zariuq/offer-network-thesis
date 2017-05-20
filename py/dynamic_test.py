#!/usr/bin/env python

from neo4j.v1 import GraphDatabase
import sys
import numpy as np
from graph_gen import *
from graph_control import *

def get_matches(driver, p, waitTimes):
    bmatch1(driver)
    (total, totalPairs, cycles, waitTimes) = removecycles(driver, p, waitTimes)
    return (total, totalPairs, cycles)

# Initially, Erdos-Renyi style update
# pt:- % of Ntasks new tasks
# pu:- % of Nusers new users
def next_cycle(driver, taskID, userID, NORnodes, nodeID, NnewTasks, NnewUsers, NnewORnodes):
    Ntasks = taskID + NnewTasks; Nusers = userID + NnewUsers
    generate_tasks_users(driver, taskID, userID, Ntasks, Nusers)
    #(Ntasks, Nusers, num) = update_er(driver, Ntasks, Nusers, nodeID, NnewORnodes)
    (Ntasks, Nusers, num) = update_wsg(driver, Ntasks, Nusers, nodeID, NnewORnodes, 2, 0.5)
    #(Ntasks, Nusers, num) = update_sfg(driver, Ntasks, Nusers, nodeID, NnewORnodes)
    #(Ntasks, Nusers, num) = update_plc(driver, Ntasks, Nusers, nodeID, NnewORnodes, 2, 0.3)
    #(Ntasks, Nusers, num) = update_ba(driver, Ntasks, Nusers, nodeID, NnewORnodes, 3)
    print("NORnodes: %s; nodeid: %s; num: %s" % (NORnodes, nodeID, num))
    return (Ntasks, Nusers, NORnodes + num, nodeID + num)

def run_cycle(driver, Ntasks, Nusers, NORnodes, nodeID, waitTimes):
    (total, totalPairs, cycles) = get_matches(driver, 1, waitTimes)
    NORnodes = NORnodes - total
    delete_bipartite(driver); delete_bmatch(driver)
    (pt, pu, p) = (0.05, 0.05, 0.2)
    (Ntasks, Nusers, NORnodes, nodeID) = next_cycle(driver, Ntasks, Nusers, NORnodes, nodeID
                                           , int(pt * Ntasks), int(pu * Nusers), int(p * NORnodes))
    return (Ntasks, Nusers, NORnodes, nodeID, total, totalPairs, waitTimes)


if __name__ == "__main__":
    num_args = len(sys.argv)
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "iwin"))
    delete_all(driver)
    #(Ntasks, Nusers, NORnodes) = generate_er(driver, 20, 50, 150)
    (Ntasks, Nusers, NORnodes) = generate_plc(driver, 60, 30, 6, 0.7)
    #(Ntasks, Nusers, NORnodes) = generate_ba(driver, 50, 50, 1)
    #(Ntasks, Nusers, NORnodes) = generate_sfg(driver, 120, 30)
    #(Ntasks, Nusers, NORnodes) = generate_wsg(driver, 50, 50, 2, 0.9)
    totals = []; pairs = []; waitTimes = dict()

    nodeID = NORnodes
    for i in range(0,int(sys.argv[1])):
        print(">>> Step %s" % i)
        (Ntasks, Nusers, NORnodes, nodeID, total, totalPairs, waitTimes) = run_cycle(driver, Ntasks, Nusers, NORnodes, nodeID, waitTimes)
        totals.append(total); pairs.append(totalPairs)
    print(list(zip(totals, pairs)))
    with driver.session() as session:
        result = session.run("MATCH (n:ORnode) RETURN n.wait, count(n.wait) ORDER BY n.wait ")
        print("Wait time : number of nodes -- unmatched")
        for record in result:
            print("%s : %s" % (record['n.wait'], record['count(n.wait)']))
        print("Wait time : number of nodes -- matched")
        for wait_time, num_nodes in waitTimes.items():
            print("%s : %s" % (wait_time, num_nodes))
