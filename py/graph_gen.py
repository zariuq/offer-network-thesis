#!/usr/bin/env python

from neo4j.v1 import GraphDatabase
import sys
import numpy as np
import numpy.random as npr
import itertools
import networkx as nx
from ast import literal_eval # "[...]" -> [...]
from munkres import munkres

def add_task(task_id):
    return "CREATE (task{0}:Task {{id:\'{0}\'}}) ".format(task_id)

def add_user(user_id):
    return "CREATE (user{0}:User {{id:\'{0}\'}}) ".format(user_id)

def add_ORnode(node_id, node):
    offer   = "task" + str(node[0])
    request = "task" + str(node[1])
    user    = "user" + str(node[2])
    node_name = "OR" + "_".join([offer,request,user,str(node_id)])
    #node_command = "CREATE (ornode:ORnode {{id:\'{1}\', offer:\'{2}\', request:\'{3}\', user:\'{4}\'}}) ".format(node_name, node_id, node[0], node[1], node[2])
    rel_command = " \n".join(
        ["MATCH (offer:Task {{id:\'{0}\'}})".format(node[0])
        ,"MATCH (request:Task {{id:\'{0}\'}})".format(node[1])
        ,"MATCH (user:User {{id:\'{0}\'}})".format(node[2])
        ,"CREATE (ornode:ORnode {{id:\'{1}\', offer:\'{2}\', request:\'{3}\', user:\'{4}\', wait:0}})   ".format(node_name, node_id, node[0], node[1], node[2])
        ,"CREATE (offer)-[:Offer]->(ornode) "
        ,"CREATE (ornode)-[:Request]->(request) "
        ,"CREATE (user)-[:Own]->(ornode) "])
    return rel_command

def generate_er(driver, Ntasks, Nusers, NORnodes):
    print("Generating Random Erdos Renyi Graph")
    p = NORnodes / (Ntasks * (Ntasks - 1)) # So the expected # of nodes is correct
    G = (np.array([*(nx.fast_gnp_random_graph(Ntasks, p, directed=True)).edges()])).transpose()
    num = G.shape[1]
    orUsers = npr.randint(Nusers, size=(num))
    orNodes = list(zip(G[0], G[1], orUsers))
    return generate(driver, Ntasks, Nusers, num, orNodes)

def update_er(driver, Ntasks, Nusers, nodeID, numNewEdges):
    print("Generating Random Erdos Renyi Graph for update")
    p = numNewEdges / (Ntasks * (Ntasks - 1))
    G = (np.array([*(nx.fast_gnp_random_graph(Ntasks, p, directed=True)).edges()])).transpose()
    num = G.shape[1]
    orUsers = npr.randint(Nusers, size=(num))
    orNodes = list(zip(G[0], G[1], orUsers))
    generate_ornodes(driver, nodeID, num, orNodes)
    return (Ntasks, Nusers, num)

# Generate random Watts Strogatz graph
# p :- probaility of rewiring edge
# interesting when Ntasks >> k >> ln(N)
def generate_wsg(driver, Ntasks, Nusers, k, p):
    num = int(k * Ntasks) # NORnodes rounded
    G = (np.array([*(nx.watts_strogatz_graph(Ntasks,k,p)).edges()])).transpose()
    Gd = (np.array([*(nx.watts_strogatz_graph(Ntasks,k,p)).edges()])).transpose()
    orUsers = npr.randint(Nusers,size=(num))
    orNodes = list(zip(G[0], G[1], orUsers[:int(num/2)])) + list(zip(Gd[1], Gd[0], orUsers[int(num/2):]))
    return generate(driver, Ntasks, Nusers, num, orNodes)

def update_wsg(driver, Ntasks, Nusers, nodeID, numNewEdges, k, p):
    print("Generating Barabasi Albert graph for update")
    fakeNtasks = int(numNewEdges / k)
    if not fakeNtasks > k:
        fakeNtasks = k + 1
    print("fakeNtasks: %s" % fakeNtasks)
    G = (np.array([*(nx.watts_strogatz_graph(fakeNtasks,k,p)).edges()])).transpose()
    Gd = (np.array([*(nx.watts_strogatz_graph(fakeNtasks,k,p)).edges()])).transpose()
    num = G.shape[1] + Gd.shape[1]
    orUsers = npr.randint(Nusers,size=(num))
    realTasks = npr.randint(Ntasks,size=(fakeNtasks))
    realTasksd = npr.randint(Ntasks,size=(fakeNtasks))
    orNodes = list(zip(realTasks[G[0]], realTasks[G[1]], orUsers[:G.shape[1]])) + list(zip(realTasksd[Gd[1]], realTasksd[Gd[0]], orUsers[G.shape[1]:]))
    generate_ornodes(driver, nodeID, num, orNodes)
    return (Ntasks, Nusers, num)

def generate_sfg(driver, Ntasks, Nusers):
    G = (np.array([*nx.scale_free_graph(Ntasks).edges()])).transpose()
    num = G.shape[1]
    orUsers = npr.randint(Nusers, size=(num))
    orNodes = list(zip(G[0], G[1], orUsers))
    return generate(driver, Ntasks, Nusers, num, orNodes)

def update_sfg(driver, Ntasks, Nusers, nodeID, numNewEdges):
    print("Generating scale free graph graph for update")
    fakeNtasks = int(numNewEdges / 2) # approximately 2x edges
    print("fakeNtasks: %s" % fakeNtasks)
    G = (np.array([*nx.scale_free_graph(fakeNtasks).edges()])).transpose()
    num = G.shape[1]
    orUsers = npr.randint(Nusers, size=(num))
    realTasks = npr.randint(Ntasks,size=(fakeNtasks))
    orNodes = list(zip(realTasks[G[0]], realTasks[G[1]], orUsers))
    generate_ornodes(driver, nodeID, num, orNodes)
    return (Ntasks, Nusers, num)

# #{edges} = 2*(Ntasks - m)*m
def generate_ba(driver, Ntasks, Nusers, m):
    G = (np.array([*(nx.barabasi_albert_graph(Ntasks,m)).edges()])).transpose()
    Gd = (np.array([*(nx.barabasi_albert_graph(Ntasks,m)).edges()])).transpose()
    num = G.shape[1] + Gd.shape[1]
    orUsers = npr.randint(Nusers,size=(num))
    orNodes = list(zip(G[0], G[1], orUsers[:G.shape[1]])) + list(zip(Gd[1], Gd[0], orUsers[G.shape[1]:]))
    return generate(driver, Ntasks, Nusers, num, orNodes)

def update_ba(driver, Ntasks, Nusers, nodeID, numNewEdges, m):
    print("Generating Barabasi Albert graph for update")
    fakeNtasks = int(numNewEdges / (2 * m)) + m
    print("fakeNtasks: %s" % fakeNtasks)
    G = (np.array([*(nx.barabasi_albert_graph(fakeNtasks,m)).edges()])).transpose()
    Gd = (np.array([*(nx.barabasi_albert_graph(fakeNtasks,m)).edges()])).transpose()
    num = G.shape[1] + Gd.shape[1]
    orUsers = npr.randint(Nusers,size=(num))
    realTasks = npr.randint(Ntasks,size=(fakeNtasks))
    realTasksd = npr.randint(Ntasks,size=(fakeNtasks))
    orNodes = list(zip(realTasks[G[0]], realTasks[G[1]], orUsers[:G.shape[1]])) + list(zip(realTasksd[Gd[1]], realTasksd[Gd[0]], orUsers[G.shape[1]:]))
    generate_ornodes(driver, nodeID, num, orNodes)
    return (Ntasks, Nusers, num)

# #{edges} approx 2*(Ntasks - m)*m*(1+p)
def generate_plc(driver, Ntasks, Nusers, m, p):
    G = (np.array([*(nx.powerlaw_cluster_graph(Ntasks,m,p)).edges()])).transpose()
    Gd = (np.array([*(nx.powerlaw_cluster_graph(Ntasks,m,p)).edges()])).transpose()
    num = G.shape[1] + Gd.shape[1]
    orUsers = npr.randint(Nusers,size=(num))
    orNodes = list(zip(G[0], G[1], orUsers[:G.shape[1]])) + list(zip(Gd[1], Gd[0], orUsers[G.shape[1]:]))
    return generate(driver, Ntasks, Nusers, num, orNodes)

def update_plc(driver, Ntasks, Nusers, nodeID, numNewEdges, m, p):
    print("Generating power law cluster graph for update")
    fakeNtasks = int(numNewEdges / (2 * m * (1 + p))) + m
    if not fakeNtasks > m:
        fakeNtasks = m + 1
    print("fakeNtasks: %s" % fakeNtasks)
    G = (np.array([*(nx.powerlaw_cluster_graph(fakeNtasks,m,p)).edges()])).transpose()
    Gd = (np.array([*(nx.powerlaw_cluster_graph(fakeNtasks,m,p)).edges()])).transpose()
    num = G.shape[1] + Gd.shape[1]
    orUsers = npr.randint(Nusers,size=(num))
    realTasks = npr.randint(Ntasks,size=(fakeNtasks))
    realTasksd = npr.randint(Ntasks,size=(fakeNtasks))
    orNodes = list(zip(realTasks[G[0]], realTasks[G[1]], orUsers[:G.shape[1]])) + list(zip(realTasksd[Gd[1]], realTasksd[Gd[0]], orUsers[G.shape[1]:]))
    generate_ornodes(driver, nodeID, num, orNodes)
    return (Ntasks, Nusers, num)

def generate_user_commands(userID, Nusers):
    node_commands = []
    for i in range(userID, Nusers):
        node_commands.append(add_user(i+1))
    return node_commands

def generate_task_commands(taskID, Ntasks):
    node_commands = []
    for i in range(taskID, Ntasks):
        node_commands.append(add_task(i+1))
    return node_commands

def generate_ornode_commands(nodeID, NnewORnodes, orNodes):
    rel_commands = []
    for i in range(0, NnewORnodes):
        rel_commands.append(add_ORnode(nodeID+i+1, orNodes[i]))
    return rel_commands

def generate_tasks_users(driver, taskID, userID, Ntasks, Nusers):
    with driver.session().begin_transaction() as tx:
        print("Creating task/user cypher commands")
        task_node_commands = generate_task_commands(taskID, Ntasks)
        user_node_commands = generate_user_commands(userID, Nusers)
        print("taskID - Ntasks: %s - %s; userID - Nusers: %s - %s" % (taskID, Ntasks, userID, Nusers))
        node_command = " \n".join(task_node_commands + user_node_commands)
        print("Running transaction:")
        tx.run(node_command)

def generate_ornodes(driver, nodeID, NnewORnodes, orNodes):
    with driver.session().begin_transaction() as tx:
        print("Creating rel commands")
        orNode_commands = generate_ornode_commands(nodeID, NnewORnodes, orNodes)
        print("Running rel transaction of size %s" % len(orNode_commands))
        for command in orNode_commands:
            tx.run(command)

def generate(driver, Ntasks, Nusers, NORnodes, orNodes):
    generate_tasks_users(driver, 0, 0, Ntasks, Nusers)
    print("Creating Task and User indexes.")
    with driver.session().begin_transaction() as tx:
        tx.run("CREATE INDEX ON :Task(id) ")
        tx.run("CREATE INDEX ON :User(id) ")
    generate_ornodes(driver, 0, NORnodes, orNodes)
    print("Creating ORnode index")
    with driver.session().begin_transaction() as tx:
        tx.run("CREATE INDEX ON :ORnode(id)")
    return (Ntasks, Nusers, NORnodes)
