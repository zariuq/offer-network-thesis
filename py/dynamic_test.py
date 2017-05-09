#!/usr/bin/env python

from neo4j.v1 import GraphDatabase
import sys
import numpy as np
from graph_gen import *
from graph_control import *

def get_matches(driver):
    task_to_bipartite(driver)
    bmatch(driver)
    getcycles(driver)

if __name__ == "__main__":
    num_args = len(sys.argv)
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "iwin"))
    delete_all(driver)
    generate_plc(driver, 30, 50, 7, 0.6)
    get_matches(driver)
