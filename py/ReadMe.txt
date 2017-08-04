This is a summary of what each Python file does.

Note the MAX-WEIGHT code needs Cython Munkres algorithms to actually run.

distribution.py: Plots and fits the distributions of data files and generated graphs

dynamic_test.py: Manages the running of one dynamic offer network test, generating the graph on an as-needed basis. Various NetworkX graph generators are supported (in comments).

graph_control.py: This file contains the core functionality for interfacing with Neo4j: the matching algorithms, cycle retrieval, cycle removal, HOR processing, and deleting matches or the whole graph. (Note: there is some deprecated code here, such as an initial, less efficient method of implementing MAX-WEIGHT in two steps.)

graphs_gen.py: Code for generating graphs of different types.

graphs.py: Code for generating all the plots in the thesis.

networkx_graph_gen.py: Two modifications of the scale free graph generator from NetworkX. One returns a list of updates to generate the given graph used in the final experiments.

runs.py: Code similar to dynamic_test except graphs can be generated once and used for multiple experiment runs, each collecting stats saved to .csv and .p (pickled) files.
