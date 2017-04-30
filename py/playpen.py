#!/usr/bin/env python

from neo4j.v1 import GraphDatabase
import sys
import numpy as np
import numpy.random as npr
import itertools
import networkx as nx
from munkres import munkres

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "iwin"))

#session = driver.session()

def add_friends(tx, name, friend_name):
    tx.run("MERGE (a:Person {name: $name}) "
           "MERGE (a)-[:KNOWS]->(friend:Person {name: $friend_name})",
           name=name, friend_name=friend_name)

def print_friends(tx, name):
    for record in tx.run("MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
                         "RETURN friend.name ORDER BY friend.name ", name=name):
        print(record["friend.name"])

def add_task(task_id):
    return "CREATE (task{0}:Task {{id:\'{0}\'}}) ".format(task_id)

def add_user(user_id):
    return "CREATE (user{0}:User {{id:\'{0}\'}}) ".format(user_id)

def add_ORnode(node_id, node):
    offer   = "task" + str(node[0])
    request = "task" + str(node[1])
    user    = "user" + str(node[2])
    node_name = "OR" + "_".join([offer,request,user,str(node_id)])
    node_command = "CREATE ({0}:ORnode {{id:\'{1}\', offer:\'{2}\', request:\'{3}\', user:\'{4}\'}}) ".format(node_name, node_id, node[0], node[1], node[2])
    rel_command = " \n".join(
        ["CREATE ({0})-[:Offer]->({1}) ".format(offer, node_name)
        ,"CREATE ({0})-[:Request]->({1}) ".format(node_name, request)
        ,"CREATE ({0})-[:Own]->({1}) ".format(user, node_name)])
    return (node_command, rel_command)

def genRandomNodes(num, num_options, num_users):
    candidates = npr.randint(num_options,size=(2,num))
    orUsers = npr.randint(num_users,size=(num))
    candidates = list(zip(candidates[0], candidates[1], orUsers))
    #for i in range(0, num):
    #    while candidates[i][0] == candidates[i][1]: #ensure no same-task offers
    #        candidates[i] = tuple(np.append(npr.randint(num_options, size=2), candidates[i][2]))
    diff = 1
    while diff > 0:
        candidates = list(set(candidates))
        diff = num - len(candidates)
        if diff > 0:
            candidates + genRandomNodes(diff, num_options, num_users)
    return candidates

def generate_random(Ntasks, Nusers, NORnodes):
    print("Generating Random Graph")
    ornodes = genRandomNodes(NORnodes, Ntasks, Nusers)
    generate(Ntasks, Nusers, NORnodes, ornodes)

# Generate random Watts Strogatz graph
# p :- probaility of rewiring edge
# interesting when Ntasks >> k >> ln(N)
def generate_wsg(Ntasks, Nusers, NORnodes, p):
    # n = Ntasks
    k = 2 * int(NORnodes / Ntasks)
    num = int((k / 2) * Ntasks) # NORnodes rounded
    G = (np.array([*(nx.watts_strogatz_graph(Ntasks,k,p)).edges()])).transpose()
    orUsers = npr.randint(Nusers,size=(num))
    ornodes = list(zip(G[0], G[1], orUsers))
    generate(Ntasks, Nusers, num, ornodes)

def generate(Ntasks, Nusers, NORnodes, ornodes):
    with driver.session().begin_transaction() as tx:
        print("Creating cypher commands")
        node_commands = []
        rel_commands = []
        for i in range(0, Ntasks):
            node_commands.append(add_task(i+1))
        for i in range(0, Nusers):
            node_commands.append(add_user(i+1))
        for i in range(0, NORnodes):
            (nc, rc) = add_ORnode(i+1, ornodes[i])
            node_commands.append(nc)
            rel_commands.append(rc)
        node_command = " \n".join(node_commands)
        rel_command = " \n".join(rel_commands)
        print("Running transaction.")
        tx.run(node_command + rel_command)

# Transform task-centric graph into bipartite graph
# 1) Generate ORnodes for Offer -> () -> Request, and weight-0 link between them
# 2) For each Task, generate weight-1 links for each Offer with each Request
def task_to_bipartite():
    offers = []
    requests = []
    zero_weights = []
    one_weights = []
    with driver.session() as session:
        #1
        print("Getting offers and requests.")
        result = session.run("match (:Task)-[]-(node:ORnode)-[]->(:Task) "
                             "return node ")
        for record in result:
            node = record['node']
            offers.append((node['offer'], node['id'], node['user']))
            requests.append((node['request'], node['id'], node['user']))
            zero_weights.append((node['offer'], node['request'], node['id']))

        #2
        print("Getting tasks.")
        tasks = dict() # {([offers],[requests])}
        result = session.run("match (task:Task)-[]-(node:ORnode)-[]-(:Task) "
                             "return task, node ")
        for record in result:
            node = record['node']
            task = record['task']['id']
            tasks.setdefault(task,([],[]))
            if node['offer'] == task:
                tasks[task][0].append(node['offer'] + "_" + node['id'])
            else:
                tasks[task][1].append(node['request'] + "_" + node['id'])
            #print("\n".join("%s: %s" % (key, record[key]) for key in record.keys()))
            #print("\n")

        print("Adding offers, requests, and zero-edges.")
        node_commands = []
        rel_commands = []
        for offer in offers:
            node_commands.append("CREATE (offer{0}_{1}:Offer {{task:\'{0}\', node_id:\'{1}\', user:\'{2}\'}}) "
                                .format(offer[0],offer[1],offer[2]))
        for request in requests:
            node_commands.append("CREATE (request{0}_{1}:Request {{task:\'{0}\', node_id:\'{1}\', user:\'{2}\'}}) "
                                .format(request[0],request[1],request[2]))
        for edge in zero_weights:
            rel_commands.append("CREATE ((offer{0}_{2})-[:bNullEdge {{weight:0}}]->(request{1}_{2})) "
                                .format(edge[0],edge[1],edge[2]))
                                #,"CREATE ({0})-[:Own]->({1}) ".format(user, node_name)])
        print("Adding real edges.")
        for task in tasks.values():
            for offer, request in itertools.product(task[0], task[1]):
                rel_commands.append("CREATE ((offer{0})<-[:bEdge {{weight:1}}]-(request{1})) "
                                    .format(offer, request))

        node_command = " \n".join(node_commands)
        rel_command = " \n".join(rel_commands)
        print("Running transaction.")
        with session.begin_transaction() as tx:
            tx.run(node_command + rel_command)

# Use cython munrkes algorithm to find maximum weight matching
def bmatch():
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
    print("Matching.")
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

def delete_all():
    with driver.session() as session:
        session.run("MATCH (n)"
                    "DETACH DELETE n")
        session.sync()

def delete_graph():
    with driver.session() as session:
        session.run("match (task:Task) "
                    "match (user:User) "
                    "match (ornode:ORnode) "
                    "detach delete task, user, ornode ")
        session.sync()

def delete_bipartite():
    with driver.session() as session:
        session.run("match (offer:Offer) "
                    "match (request:Request) "
                    "detach delete offer, request ")
        session.sync()

def delete_bmatch():
    with driver.session() as session:
        session.run("match ()-[match:Match]-()"
                    "detach delete match ")
        session.sync()

def print_instructions():
    print("Error. Use code.")
    print("-d to delete the graph. -d a for all, g for graph, b for bipartite, mb for bipartite match")
    print("-gr [Ntasks Nusers NORnodes] to generate a random graph, [] denotes optionality")
    print("-gwsg [Ntasks Nusers [NORnodes p]] to generate a random Watts Strogatz graph")
    print("-tb to transform task-centric graph into a bipartite graph.")
    print("-mb to get maximum weight matching via bipartite graph.")

if __name__ == "__main__":
    num_args = len(sys.argv)
    if num_args > 1:
        if sys.argv[1] == "-d":
            if num_args == 3:
                if sys.argv[2] == "a":
                    delete_all()
                elif sys.argv[2] == "g":
                    delete_graph()
                elif sys.argv[2] == "b":
                    delete_bipartite()
                elif sys.argv[2] == "mb":
                    delete_bmatch()
            else:
                delete_all()
        elif sys.argv[1] == "-gr":
            if num_args == 5:
                generate_random(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
            else:
                generate_random(50, 100, 50) #default
        elif sys.argv[1] == "-gwsg":
            if num_args == 4:
                Ntasks = int(sys.argv[2])
                NORnodes = 3*int(Ntasks * np.log(Ntasks)) # so N >> k >> ln(N)
                generate_wsg(Ntasks, int(sys.argv[3]), NORnodes, 0.5)
            elif num_args == 6:
                generate_wsg(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
            else:
                generate_wsg(50, 100, 50, 0.5)
        elif sys.argv[1] == "-tb":
            task_to_bipartite()
        elif sys.argv[1] == "-mb":
            bmatch()
        else:
            print_instructions()
    else:
        print_instructions()


'''

result = session.run("PROFILE MATCH (p:Person {name: {name}}) "
                     "RETURN id(p)", {"name": "Arthur"})
summary = result.consume()
print(summary.statement_type)
print(summary.profile)

result = session.run("EXPLAIN MATCH (king), (queen) RETURN king, queen")
summary = result.consume()
for notification in summary.notifications:
    print(notification)

try:
    session.run("This will cause a syntax error").consume()
except CypherError:
    raise RuntimeError("Something really bad has happened!")
finally:
    session.close()

result = session.run("MATCH (knight:Person:Knight) WHERE knight.castle = {castle} "
                     "RETURN id(knight) AS knight_id", {"castle": "Camelot"})
for record in result:
    session.run("MATCH (knight) WHERE id(knight) = {id} "
                "MATCH (king:Person) WHERE king.name = {king} "
                "CREATE (knight)-[:DEFENDS]->(king)", {"id": record["knight_id"], "king": "Arthur"})

search_term = "Sword"
result = session.run("MATCH (weapon:Weapon) WHERE weapon.name CONTAINS {term} "
                     "RETURN weapon.name", {"term": search_term})
print("List of weapons called %r:" % search_term)
for record in result:
    print(record["weapon.name"])

search_term = "Arthur"
result = session.run("MATCH (weapon:Weapon) WHERE weapon.owner CONTAINS {term} "
                     "RETURN weapon.name, weapon.material, weapon.size", {"term": search_term})
print("List of weapons owned by %r:" % search_term)
for record in result:
    print(", ".join("%s: %s" % (key, record[key]) for key in record.keys()))

session = driver.session()
result = session.run("MATCH (knight:Person:Knight) WHERE knight.castle = {castle} "
                     "RETURN knight.name AS name", {"castle": "Camelot"})
retained_result = list(result)
session.close()
for record in retained_result:
    print("%s is a knight of Camelot" % record["name"])


with session.begin_transaction() as tx:
    tx.run("MERGE (:Person {name: 'Guinevere'})")
    tx.success = True

with session.begin_transaction() as tx:
    tx.run("CREATE (:Person {name: 'Merlin'})")
    tx.success = False

session.run("MERGE (a:Person {name:'Arthur', title:'King'})")

result = session.run("MATCH (a:Person) WHERE a.name = 'Arthur' RETURN a.name AS name, a.title AS title")
for record in result:
    print("%s %s" % (record["title"], record["name"]))

session.close()

with session.begin_transaction() as tx:
    tx.run("CREATE (:Person {name: 'Merlin'})")
    tx.success = False


with driver.session() as session:
    session.write_transaction(add_friends, "Arthur", "Guinevere")
    session.write_transaction(add_friends, "Arthur", "Lancelot")
    session.write_transaction(add_friends, "Arthur", "Merlin")
    session.read_transaction(print_friends, "Arthur")
'''
