"""methods for Knowledge discovery;
Analysis of the relation instances
"""
import json

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, write_dot
from networkx.algorithms import approximation

def build_graph():
    """generate graph representation of the training facts
    """
    # G = nx.DiGraph()
    G = nx.Graph()
    with open('data/initial_label_facts.json') as f_in:
        label_triples = json.load(f_in)
        rel_colors = {
            "use":"red",
            "different from":'green',
            "subclass of":'blue', 
            "has quality":'purple',
            "instance of":'yellow', 
            "facet of":'brown'}
    for rel, vals in label_triples.items():
        for pair in vals:
            G.add_edge(pair[0], pair[1], color=rel_colors.get(rel, 'black'))
    # print(len(list(nx.connected_components(G))))
    # print(sorted(d for n, d in G.degree()))
    # print(nx.clustering(G))
    print(approximation.average_clustering(G))
    # write_dot(G, 'test.dot')
    # $ dot -Tpng test.dot >test.png

    return label_triples

build_graph()
