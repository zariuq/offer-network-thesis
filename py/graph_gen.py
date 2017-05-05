from neo4j.v1 import GraphDatabase
import sys
import numpy as np
import numpy.random as npr
import itertools
import networkx as nx
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
        ,"CREATE (ornode:ORnode {{id:\'{1}\', offer:\'{2}\', request:\'{3}\', user:\'{4}\'}})   ".format(node_name, node_id, node[0], node[1], node[2])
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
    generate(driver, Ntasks, Nusers, num, orNodes)

# Generate random Watts Strogatz graph
# p :- probaility of rewiring edge
# interesting when Ntasks >> k >> ln(N)
def generate_wsg(driver, Ntasks, Nusers, k, p):
    num = int(k * Ntasks) # NORnodes rounded
    G = (np.array([*(nx.watts_strogatz_graph(Ntasks,k,p)).edges()])).transpose()
    Gd = (np.array([*(nx.watts_strogatz_graph(Ntasks,k,p)).edges()])).transpose()
    orUsers = npr.randint(Nusers,size=(num))
    orNodes = list(zip(G[0], G[1], orUsers[:int(num/2)])) + list(zip(Gd[1], Gd[0], orUsers[int(num/2):]))
    generate(driver, Ntasks, Nusers, num, orNodes)

def generate_sfg(driver, Ntasks, Nusers):
    G = (np.array([*nx.scale_free_graph(Ntasks).edges()])).transpose()
    num = G.shape[1]
    orUsers = npr.randint(Nusers, size=(num))
    orNodes = list(zip(G[0], G[1], orUsers))
    generate(driver, Ntasks, Nusers, num, orNodes)

# #{edges} = 2*(Ntasks - m)*m
def generate_ba(driver, Ntasks, Nusers, m):
    G = (np.array([*(nx.barabasi_albert_graph(Ntasks,m)).edges()])).transpose()
    Gd = (np.array([*(nx.barabasi_albert_graph(Ntasks,m)).edges()])).transpose()
    num = G.shape[1] + Gd.shape[1]
    orUsers = npr.randint(Nusers,size=(num))
    orNodes = list(zip(G[0], G[1], orUsers[:int(num/2)])) + list(zip(Gd[1], Gd[0], orUsers[int(num/2):]))
    generate(driver, Ntasks, Nusers, num, orNodes)

# #{edges} approx 2*(Ntasks - m)*m
def generate_plc(driver, Ntasks, Nusers, m, p):
    G = (np.array([*(nx.powerlaw_cluster_graph(Ntasks,m,p)).edges()])).transpose()
    Gd = (np.array([*(nx.barabasi_albert_graph(Ntasks,m,p)).edges()])).transpose()
    num = G.shape[1] + Gd.shape[1]
    orUsers = npr.randint(Nusers,size=(num))
    orNodes = list(zip(G[0], G[1], orUsers[:G.shape[1]])) + list(zip(Gd[1], Gd[0], orUsers[G.shape[1]:]))
    generate(driver, Ntasks, Nusers, num, orNodes)

def generate(driver, Ntasks, Nusers, NORnodes, orNodes):
    with driver.session().begin_transaction() as tx:
        print("Creating cypher commands")
        node_commands = []
        for i in range(0, Ntasks):
            node_commands.append(add_task(i+1))
        for i in range(0, Nusers):
            node_commands.append(add_user(i+1))
        node_command = " \n".join(node_commands)
        print("Running transaction.")
        tx.run(node_command)
    print("Creating Task and User indexes.")
    with driver.session().begin_transaction() as tx:
        tx.run("CREATE INDEX ON :Task(id) ")
        tx.run("CREATE INDEX ON :User(id) ")
    with driver.session().begin_transaction() as tx:
        print("Creating rel commands")
        rel_commands = []
        for i in range(0, NORnodes):
            rel_commands.append(add_ORnode(i+1, orNodes[i]))
        print("Running rel transaction")
        for command in rel_commands:
            tx.run(command);
    print("Creating ORnode index")
    with driver.session().begin_transaction() as tx:
        tx.run("CREATE INDEX ON :ORnode(id)")
