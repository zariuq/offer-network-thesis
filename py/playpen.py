#!/usr/bin/env python

from neo4j.v1 import GraphDatabase
import sys
import numpy as np
import numpy.random as npr

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "iwin"))

#session = driver.session()

def add_friends(tx, name, friend_name):
    tx.run("MERGE (a:Person {name: $name}) "
           "MERGE (a)-[:KNOWS]->(friend:Person {name: $friend_name})",
           name=name, friend_name=friend_name)

def print_friends(tx, name):
    for record in tx.run("MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
                         "RETURN friend.name ORDER BY friend.name", name=name):
        print(record["friend.name"])

def delete_all(session):
    session.run("MATCH (n)"
                "DETACH DELETE n")
    session.sync()

def add_task(task_id, task_name):
    return "CREATE ({1}:Task {{id:\'{0}\', name:\'{1}\'}}) ".format(task_id, task_name)

def add_user(user_id, user_name):
    return "CREATE ({1}:User {{id:\'{0}\', name:\'{1}\'}}) ".format(user_id, user_name)

def add_ORnode(node_id, offer, request, user):
    node_name = "{0}_{1}".format(offer, request)
    command = " \n".join(
        ["CREATE ({1}:ORnode {{id:\'{0}\', offer:\'{2}\', request:\'{3}\'}}) ".format(node_id, node_name, offer, request)
        ,"CREATE ({0})-[:Offer]->({1}) ".format(offer, node_name)
        ,"CREATE ({0})-[:Request]->({1}) ".format(node_name, request)
        ,"CREATE ({0})-[:Own]->({1})".format(user, node_name)])
    return command

def genRandomNodes(num, num_options, num_users):
    candidates = npr.randint(num_options,size=(2,num))
    orUsers = npr.randint(num_users,size=(num))
    candidates = list(zip(candidates[0], candidates[1], orUsers))
    for i in range(0, num):
        while candidates[i][0] == candidates[i][1]: #ensure no same-task offers
            candidates[i] = tuple(np.append(npr.randint(num_options, size=2), candidates[i][2]))
    diff = 1
    while diff > 0:
        candidates = list(set(candidates))
        diff = num - len(candidates)
        if diff > 0:
            candidates + genRandomNodes(diff, num_options, num_users)
    return candidates

Ntasks = 50
Nusers = 100
NORnodes = 50

def main():
    tasks = ["Task" + str(i) for i in range(1,Ntasks + 1)]
    users = ["User" + str(i) for i in range(1,Nusers + 1)]
    ornodes = genRandomNodes(NORnodes, Ntasks, Nusers)
    with driver.session().begin_transaction() as tx:
        commands = []
        for i in range(0, Ntasks):
            commands.append(add_task(i+1, tasks[i]))
        for i in range(0, Nusers):
            commands.append(add_user(i+1, users[i]))
        for i in range(0, NORnodes):
            commands.append(add_ORnode(i+1, tasks[ornodes[i][0]], tasks[ornodes[i][1]]
                           ,users[ornodes[i][2]]))
        command = " \n".join(commands)
        tx.run(command)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "d":
            with driver.session() as session:
                delete_all(session)
    else:
        main()

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
