#!/usr/bin/env python

from neo4j.v1 import GraphDatabase
import sys
import numpy as np
import itertools
import networkx as nx
from graph_gen import *
from munkres import munkres

# Transform task-centric graph into bipartite graph
# 1) Generate ORnodes for Offer -> () -> Request, and weight-0 link between them
# 2) For each Task, generate weight-1 links for each Offer with each Request
def task_to_bipartite(driver):
    offers = []
    requests = []
    zero_weights = []
    one_weights = []
    with driver.session() as session:
        #1
        print("Getting offers and requests.")
        result = session.run("match (:Task)-[]-(node:ORnode)-[]->(:Task) "
                             "return distinct node ")
        for record in result:
            node = record['node']
            offers.append((node['offer'], node['id'], node['user']))
            requests.append((node['request'], node['id'], node['user']))
            zero_weights.append((node['offer'], node['request'], node['id']))

        #2
        print("Getting tasks.")
        tasks = dict() # {([offers],[requests])}
        result = session.run("match (task:Task)-[]-(node:ORnode)-[]-(:Task) "
                             "return distinct task, node ")
        for record in result:
            #print("\n".join("%s: %s" % (key, record[key]) for key in record.keys()))
            #print("\n")
            node = record['node']
            task = record['task']['id']
            tasks.setdefault(task,([],[]))
            if node['offer'] == task:
                tasks[task][0].append((node['offer'], node['id']))
            elif node['request'] == task:
                tasks[task][1].append((node['request'], node['id']))
            else:
                raise ValueError('Offer or Request should match Task.')

        print("Adding offers and requests.")
        node_commands = []
        for offer in offers:
            node_commands.append("CREATE (offer{0}_{1}:Offer {{task:\'{0}\', node_id:\'{1}\', user:\'{2}\'}}) "
                                .format(offer[0],offer[1],offer[2]))
        for request in requests:
            node_commands.append("CREATE (request{0}_{1}:Request {{task:\'{0}\', node_id:\'{1}\', user:\'{2}\'}}) "
                                .format(request[0],request[1],request[2]))

        node_command = " \n".join(node_commands)
        print("Running transaction.")
        with session.begin_transaction() as tx:
            tx.run(node_command)
        print("Creating Offer and Request indexes")
        with session.begin_transaction() as tx:
            tx.run("CREATE INDEX ON :Offer(node_id) ")
            tx.run("CREATE INDEX ON :Request(node_id) ")

        rel_commands = []
        for edge in zero_weights:
            rel_commands.append("\n".join(
                ["MATCH (offer:Offer {{task:\'{0}\', node_id:\'{1}\'}}) ".format(edge[0],edge[2])
                ,"MATCH (request:Request {{task:\'{0}\', node_id:\'{1}\'}}) ".format(edge[1],edge[2])
                ,"CREATE ((offer)-[:bNullEdge {weight:0}]->(request)) "]))
        print("Adding zero and real edges.")
        for task in tasks.values(): # [([offers],[requests])]
            for offer, request in itertools.product(task[0], task[1]):
                #print("{0} - {1}".format(offer, request))
                rel_commands.append("\n".join(
                    ["MATCH (offer:Offer {{task:\'{0}\', node_id:\'{1}\'}}) "
                        .format(offer[0],offer[1])
                    ,"MATCH (request:Request {{task:\'{0}\', node_id:\'{1}\'}}) "
                        .format(request[0],request[1])
                    ,"CREATE ((offer)<-[:bEdge {weight:1}]-(request)) "]))

        print("Running relation transaction")
        with session.begin_transaction() as tx:
            for command in rel_commands:
                tx.run(command)

# Use cython munrkes algorithm to find maximum weight matching
def bmatch(driver):
    offers = dict()
    iOffers = dict()
    requests = dict()
    iRequests = dict()
    edges = []
    with driver.session() as session:
        print("Getting offers.")
        result = session.run("match (offer:Offer) return offer ")
        num_offers = 0
        for record in result:
            offers[tuple(record['offer'].values())] = num_offers # store the array #
            iOffers[num_offers] = record['offer']
            num_offers += 1
        result = session.run("match (request:Request) return request ")
        num_requests = 0
        print("Getting requests.")
        for record in result:
            requests[tuple(record['request'].values())] = num_requests
            iRequests[num_requests] = record['request']
            num_requests += 1
        print("Getting edges.")
        resultEdge = session.run("match (offer:Offer)-[edge:bEdge]-(request:Request) "
                             "return offer, request, edge ")
        resultNullEdge = session.run("match (offer:Offer)-[edge:bNullEdge]-(request:Request) "
                             "return offer, request, edge ")
        for result in [resultEdge, resultNullEdge]:
            for record in result:
                edges.append((tuple(record['offer'].values()), tuple(record['request'].values())
                            ,-record['edge']['weight']))
    size = max(num_offers, num_requests)
    B = np.full((size, size), np.inf, dtype=np.double)

    for edge in edges:
        i = offers[edge[0]]
        j = requests[edge[1]]
        B[i][j] = edge[2]

    #print("%d %d" % (num_offers, num_requests))
    #print(offers)
    #print(requests)
    #print(edges)
    #for i in range(0,size):
    #    print(B[i])
    print("Matching. Size = {0}".format(size))
    matching = munkres(B)
    print("Generating match graph.")
    with driver.session() as session:
        commands = []
        for match in np.argwhere(matching==True):
            offer = iOffers[match[0]]
            request = iRequests[match[1]]
            if offer['node_id'] != request['node_id']:
                commands.append("MATCH (offer:ORnode)-[]-()-[]-(request:ORnode) "
                               +"WHERE offer.id = \'{0}\' and request.id = \'{1}\' "
                                    .format(offer['node_id'], request['node_id'])
                               +"WITH offer as offer, request as request LIMIT 1 "
                               +"CREATE ((offer)-[:Match]->(request)) ")
        #command = " \n".join(commands)
        print("Running transaction. There are {0} matches".format(len(commands)))
        with session.begin_transaction() as tx:
            for command in commands:
                tx.run(command)

def getcycles(driver):
    handled = dict()
    cycles = []
    with driver.session() as session:
        print("Getting cycles/matches.")
        result = session.run("match p=(o:ORnode)-[:Match*1..]->(o) return tail(nodes(p)) ")
        for record in result:
            cycle = record.values()[0]
            if cycle[0]['id'] in handled:
                continue
            cycles.append(cycle)
            for orNode in cycle:
                handled[orNode['id']] = True
            #print("\n".join("%s" % (node) for node in record.values()[0]))
            #print("\n")
        print("Total %s cycles found." % (len(cycles)))
        print("\n\n".join(" ".join("%s" % (orNode['id']) for orNode in cycle )for cycle in cycles))
    return cycles

def removecycles(driver, p):
    cycles = getcycles(driver)
    commands = []
    for cycle in cycles:
        num = len(cycle)
        acceptance = np.random.binomial(1, p, num)
        match = []
        newmatch = []
        rejector = []
        for i in range(1, num - 1):
            if all(acceptance[i-1:i+2]):
                match.append(cycle[i])
            else:
                newmatch.append((cycle[i-1], cycle[i+1]))
                rejector.append(cycle[i])
        print(acceptance)
        print("[" + ", ".join("\"%s\"" % (orNode['id']) for orNode in match) + "]")
        print("[" + ", ".join("\"(%s, %s)\"" % (orNode1['id'], orNode2['id']) for (orNode1, orNode2) in newmatch) + "]")

        l = "[" + ", ".join("\"%s\"" % (orNode['id']) for orNode in match) + "]"
        commands.append("MATCH (n:ORnode)-[:Match]-() WHERE n.id in {0} set n.test = 1".format(l))
        # above for testing purposes; below really deletes
        #commands.append("MATCH (n:ORnode)-[:Match]-() WHERE n.id in {0} DETACH DELETE n".format(l))
    #with driver.session() as session:
    #    with session.begin_transaction() as tx:
    #        for command in commands:
    #            tx.run(command)

def delete_all(driver):
    with driver.session() as session:
        print("Deleting everything.")
        session.run("MATCH (n)"
                    "DETACH DELETE n")
        session.sync()

def delete_graph(driver):
    with driver.session() as session:
        print("Deleting task-graph.")
        session.run("match (task:Task) "
                    "match (user:User) "
                    "match (ornode:ORnode) "
                    "detach delete task, user, ornode ")
        session.sync()

def delete_bipartite(driver):
    with driver.session() as session:
        print("Deleting bipartite offer-request graph.")
        session.run("match (offer:Offer) "
                    "match (request:Request) "
                    "detach delete offer, request ")
        session.sync()

def delete_bmatch(driver):
    with driver.session() as session:
        print("Deleting matches.")
        session.run("match ()-[match:Match]-()"
                    "detach delete match ")
        session.sync()

def print_instructions():
    print("Error. Use code.")
    print("-d to delete everything. g for graph, tb for bipartite, mb for bipartite match")
    print("-ger [Ntasks Nusers NORnodes] to generate an Erdos-Renyi graph, [] denotes optionality")
    print("-gwsg [Ntasks Nusers [k p]] to generate a Watts Strogatz graph")
    print("-gsfg [Ntasks Nusers] to generate a scale free graph")
    print("-gba [Ntasks Nusers m] to generate a Barabasi Albert graph")
    print("-gplc [Ntasks Nusers m p] to generate a power law cluster graph")
    print("-tb to transform task-centric graph into a bipartite graph.")
    print("-mb to get maximum weight matching via bipartite graph.")
    print("-c to get the cycles/matches")
    print("-r [p] to remove cycles/matches")

if __name__ == "__main__":
    num_args = len(sys.argv)
    if num_args > 1:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "iwin"))
        if sys.argv[1] == "-d":
            if num_args == 3:
                if sys.argv[2] == "g":
                    delete_graph(driver)
                elif sys.argv[2] == "tb":
                    delete_bipartite(driver)
                elif sys.argv[2] == "mb":
                    delete_bmatch(driver)
            else:
                delete_all(driver)
        elif sys.argv[1] == "-ger":
            if num_args == 5:
                generate_er(driver, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
            else:
                generate_er(driver, 50, 50, 50) #default
        elif sys.argv[1] == "-gwsg":
            if num_args == 4:
                Ntasks = int(sys.argv[2])
                k = int(4 * np.log(Ntasks)) # so N >> k >> ln(N)
                generate_wsg(driver, Ntasks, int(sys.argv[3]), k, 0.5)
            elif num_args == 6:
                generate_wsg(driver, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
            else:
                generate_wsg(driver, 50, 50, 2, 0.5)
        elif sys.argv[1] == "-gsfg":
            if num_args == 4:
                generate_sfg(driver, int(sys.argv[2]), int(sys.argv[3]))
            else:
                generate_sfg(driver, 50, 50)
        elif sys.argv[1] == "-gba":
            if num_args == 5:
                generate_ba(driver, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
            else:
                generate_ba(driver, 50, 50, 1) #default
        elif sys.argv[1] == "-gplc":
            if num_args == 6:
                generate_plc(driver, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
            else:
                generate_plc(driver, 50, 50, 1, 0.3) #default
        elif sys.argv[1] == "-tb":
            task_to_bipartite(driver)
        elif sys.argv[1] == "-mb":
            bmatch(driver)
        elif sys.argv[1] == "-c":
            getcycles(driver)
        elif sys.argv[1] == "-r":
            if num_args == 3:
                removecycles(driver, float(sys.argv[2]))
            else:
                removecycles(driver, 0.7)
        else:
            print_instructions()
    else:
        print_instructions()
