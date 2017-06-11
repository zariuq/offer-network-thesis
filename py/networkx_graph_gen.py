"""
Modified by Zar Goertzel

Generators for some directed graphs.

scale_free_graph: scale free directed graph

"""
#    Copyright (C) 2006-2009 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
__author__ ="""Aric Hagberg (hagberg@lanl.gov)\nWillem Ligtenberg (W.P.A.Ligtenberg@tue.nl)"""

__all__ = ['gn_graph', 'gnc_graph', 'gnr_graph','scale_free_graph']

import random

import numpy.random as npr
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import discrete_sequence

def add_task(task_id):
    return "CREATE (task{0}:Task {{id:\'{0}\'}}) ".format(task_id)

def add_user(user_id):
    return "CREATE (:User {{id:\'{0}\'}}) ".format(user_id)

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
        ,"CREATE (ornode:ORnode {{id:\'{1}\', offer:\'{2}\', request:\'{3}\', user:\'{4}\', wait:0, waitTimes:\'\'}})   ".format(node_name, node_id, node[0], node[1], node[2])
        ,"CREATE (offer)-[:Offer]->(ornode) "
        ,"CREATE (ornode)-[:Request]->(request) "
        ,"CREATE (user)-[:Own]->(ornode) "])
    return rel_command

def scale_free_graphX(taskID, nodeID, NORnodes, Nusers,
                     alpha=0.15,
                     beta=0.70,
                     gamma=0.15,
                     directed_p = 0.05,
                     delta_in=0,
                     delta_out=0,
                     create_using=None,
                     seed=None):
    """Return a scale free directed graph.

    Parameters
    ----------
    n : integer
        Number of nodes in graph
    alpha : float
        Probability for adding a new node connected to an existing node
        chosen randomly according to the in-degree distribution.
    beta : float
        Probability for adding an edge between two existing nodes.
        One existing node is chosen randomly according the in-degree
        distribution and the other chosen randomly according to the out-degree
        distribution.
    gamma : float
        Probability for adding a new node conecgted to an existing node
        chosen randomly according to the out-degree distribution.
    directed_p : float in [0, 1]
        Probability of choosing nodes via (in + out)-degree (1) versus in/out-degree (0).
    delta_in : float
        Bias for choosing ndoes from in-degree distribution.
    delta_out : float
        Bias for choosing ndoes from out-degree distribution.
    create_using : graph, optional (default MultiDiGraph)
        Use this graph instance to start the process (default=3-cycle).
    seed : integer, optional
        Seed for random number generator

    Examples
    --------
    >>> G=nx.scale_free_graph(100)

    Notes
    -----
    The sum of alpha, beta, and gamma must be 1.

    References
    ----------
    .. [1] B. Bollob{\'a}s, C. Borgs, J. Chayes, and O. Riordan,
           Directed scale-free graphs,
           Proceedings of the fourteenth annual ACM-SIAM symposium on
           Discrete algorithms, 132--139, 2003.
    """

    def _choose_node(G,distribution,delta):
        cumsum=0.0
        # normalization
        psum=float(sum(distribution.values()))+float(delta)*len(distribution)
        r=random.random()
        for i in range(1,len(distribution)):
            cumsum+=(distribution[str(i)]+delta)/psum
            if r < cumsum:
                break
        return i

    task_commands = []; orPair_commands = []

    if create_using is None:
        # start with 3-cycle
        users = npr.randint(Nusers, size=3)
        G = nx.MultiDiGraph()
        G.add_edge('1','2',key='1',user=str(users[0]))
        G.add_edge('2','3',key='2',user=str(users[1]))
        G.add_edge('3','1',key='3',user=str(users[2]))
        orPair_commands.append(add_ORnode(1, (1, 2, users[0]))) # add to neo4j
        orPair_commands.append(add_ORnode(2, (2, 3, users[1]))) # add to neo4j
        orPair_commands.append(add_ORnode(3, (3, 1, users[2]))) # add to neo4j
        for w in range(0,3):
            task_commands.append(add_task(w+1))
        nodeID = 3; taskID = 3
    else:
        # keep existing graph structure?
        G = create_using
        if not (G.is_directed() and G.is_multigraph()):
            raise nx.NetworkXError(\
                  "MultiDiGraph required in create_using")

    if alpha <= 0:
        raise ValueError('alpha must be >= 0.')
    if beta <= 0:
        raise ValueError('beta must be >= 0.')
    if gamma <= 0:
        raise ValueError('beta must be >= 0.')

    if alpha+beta+gamma !=1.0:
        raise ValueError('alpha+beta+gamma must equal 1.')

    G.name="directed_scale_free_graph(%s,alpha=%s,beta=%s,gamma=%s,delta_in=%s,delta_out=%s)"%(NORnodes,alpha,beta,gamma,delta_in,delta_out)


    users = npr.randint(1, Nusers, size=(NORnodes - nodeID)) # Is it really cheaper to generate at once?
    # seed random number generated (uses None as default)
    random.seed(seed)
    z = {'in':0,'out':0,'mix':0}
    while nodeID < NORnodes:
        r = random.random()
        # random choice in alpha,beta,gamma ranges
        if r<alpha:
            z['in'] += 1
            # alpha
            # add new node v
            taskID += 1; v = taskID; task_commands.append(add_task(taskID));
            # choose w according to in-degree and delta_in
            if r < directed_p:
                w = _choose_node(G, G.in_degree(),delta_in)
            else:
                w = _choose_node(G, G.degree(),delta_in)
        elif r < alpha+beta:
            z['mix'] += 1
            # beta
            # choose v according to out-degree and delta_out
            if r < directed_p:
                while True:
                    v = _choose_node(G, G.out_degree(),delta_out)
                    # choose w according to in-degree and delta_in
                    w = _choose_node(G, G.in_degree(),delta_in)
                    if not v == w:
                        break
            else:
                while True:
                    v = _choose_node(G, G.degree(),delta_out)
                    # choose w according to in-degree and delta_in
                    w = _choose_node(G, G.degree(),delta_in)
                    if not v == w:
                        break
        else:
            z['out'] += 1
            # gamma
            # choose v according to out-degree and delta_out
            if r < directed_p:
                v = _choose_node(G, G.out_degree(),delta_out)
            else:
                v = _choose_node(G, G.degree(),delta_out)
            # add new node w
            taskID += 1; w = taskID; task_commands.append(add_task(taskID));
        userID = users[NORnodes - nodeID - 1]
        nodeID +=1
        G.add_edge(str(v), str(w), key = str(nodeID), user = str(userID)) # add to networkX
        orPair_commands.append(add_ORnode(nodeID, (v, w, userID))) # add to neo4j
        #s = (nodeID, (v, w, userID))
        #print("%s %s" % s)
    print("Chosen by (in, out, mix):: (%s, %s, %s)" % (z['in'], z['out'], z['mix']))

    return G, task_commands, orPair_commands
