"""
Centrality measures for a directed GraphML network.
"""

import networkx as nx
import csv

PATH = "/Users/minghanfan/Desktop/math479/Capstone/celegansgraph/elegans_scc.graphml"
OUT = "/Users/minghanfan/Desktop/math479/Capstone/classicstat/classicstat_celegans_scc.csv"

G = nx.read_graphml(PATH)

if not G.is_directed():
    G = G.to_directed()

weighted = any('weight' in d for _, _, d in G.edges(data=True))

# invert weights: strength -> distance
H = G.copy()
if weighted:
    for u, v, d in H.edges(data=True):
        if 'weight' in d:
            d['weight'] = 1.0 / d['weight']

dist = {'distance': 'weight'} if weighted else {}

out_close = nx.closeness_centrality(H, **dist)
in_close = nx.closeness_centrality(H.reverse(), **dist)

U = H.to_undirected()
between = nx.betweenness_centrality(U, weight='weight' if weighted else None)

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['node', 'name', 'in_closeness', 'out_closeness', 'betweenness'])
    for n in sorted(G.nodes()):
        w.writerow([n,
                    G.nodes[n].get('name', str(n)),
                    round(in_close[n], 6),
                    round(out_close[n], 6),
                    round(between[n], 6)])

print(f"weighted: {weighted}")
print(f"nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
print(f"saved to {OUT}")