"""
Basic network stats for a region graph.
"""

import os, sys
import networkx as nx
import numpy as np
from collections import Counter

ROOT = "/Users/minghanfan/Desktop/math479/Capstone"
GRAPH_DIR = os.path.join(ROOT, "fruitflygraph")

TARGET = "FB"
path = os.path.join(GRAPH_DIR, f"region_{TARGET}_graph.graphml")
G = nx.read_graphml(path)

N, E = G.number_of_nodes(), G.number_of_edges()

# components
n_weak = nx.number_weakly_connected_components(G)
n_strong = nx.number_strongly_connected_components(G)
largest_wcc = max(nx.weakly_connected_components(G), key=len)
largest_scc = max(nx.strongly_connected_components(G), key=len)

# degree 
in_deg = dict(G.in_degree(weight="weight"))
out_deg = dict(G.out_degree(weight="weight"))
total  = {n: in_deg[n] + out_deg[n] for n in G.nodes()}

# centrality (unweighted)
betw  = nx.betweenness_centrality(G)
close = nx.closeness_centrality(G)
pr = nx.pagerank(G)

# centrality (weighted)
betw_w = nx.betweenness_centrality(G, weight="weight") 
close_w = nx.closeness_centrality(G, distance="weight")
pr_w = nx.pagerank(G, weight="weight")

# reciprocity & clustering
recip = nx.reciprocity(G)
# clustering on undirected projection
G_undir = G.to_undirected()
avg_clust = nx.average_clustering(G_undir)

# top-k helper
def top(d, k=10):
    ranked = sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
    names = nx.get_node_attributes(G, "name")
    lines = []
    for node, val in ranked:
        label = names.get(node, node)
        lines.append(f"  {label:>30s}  {val:.6f}" if isinstance(val, float)
                     else f"  {label:>30s}  {val}")
    return "\n".join(lines)


# print
print(f"=== {TARGET} region graph ===")
print(f"Nodes: {N:,}  |  Edges: {E:,}  |  Density: {nx.density(G):.6f}")
print(f"Weakly connected components:  {n_weak:,}  (largest: {len(largest_wcc):,})")
print(f"Strongly connected components: {n_strong:,}  (largest: {len(largest_scc):,})")
print(f"Reciprocity: {recip:.4f}")
print(f"Avg clustering (undirected): {avg_clust:.4f}")
print()

print(f"Weighted in-degree  — mean {np.mean(list(in_deg.values())):.1f}, "
      f"max {max(in_deg.values())}")
print(f"Weighted out-degree — mean {np.mean(list(out_deg.values())):.1f}, "
      f"max {max(out_deg.values())}")
print()

print("Top 10 by betweenness centrality (unweighted):")
print(top(betw))
print()
print("Top 10 by betweenness centrality (weighted):")
print(top(betw_w))
print()
print("Top 10 by PageRank (unweighted):")
print(top(pr))
print()
print("Top 10 by PageRank (weighted):")
print(top(pr_w))
print()
print("Top 10 by closeness centrality (unweighted):")
print(top(close))
print()
print("Top 10 by closeness centrality (weighted):")
print(top(close_w))
print()
print("Top 10 by weighted in-degree:")
print(top(in_deg))
print()
print("Top 10 by weighted out-degree:")
print(top(out_deg))

# SCC size distribution
scc_sizes = sorted([len(c) for c in nx.strongly_connected_components(G)], reverse=True)
size_counts = Counter(scc_sizes)

print()
print("=== SCC size distribution ===")
print(f"Total SCCs: {len(scc_sizes):,}")
print(f"Giant SCC:  {scc_sizes[0]:,} nodes ({100*scc_sizes[0]/N:.1f}% of graph)")
print(f"Singletons: {size_counts.get(1, 0):,}")
print()
print("Size Count")
for size in sorted(size_counts.keys(), reverse=True)[:20]:
    print(f"  {size:>6,}  {size_counts[size]:>6,}")

# extract giant SCC subgraph
G_scc = G.subgraph(largest_scc).copy()
print()
print(f"=== Giant SCC stats ({TARGET}) ===")
print(f"Nodes: {G_scc.number_of_nodes():,}  |  Edges: {G_scc.number_of_edges():,}  "
      f"|  Density: {nx.density(G_scc):.6f}")
print(f"Reciprocity: {nx.reciprocity(G_scc):.4f}")
print(f"Avg clustering (undirected): {nx.average_clustering(G_scc.to_undirected()):.4f}")

# diameter on giant SCC
if G_scc.number_of_nodes() <= 10000:
    print(f"Diameter: {nx.diameter(G_scc)}")
    print(f"Avg shortest path: {nx.average_shortest_path_length(G_scc):.2f}")
else:
    print("(skipping diameter, giant SCC too large)")

# centrality on giant SCC
print()
scc_betw = nx.betweenness_centrality(G_scc, weight="weight")
scc_pr = nx.pagerank(G_scc, weight="weight")
scc_close = nx.closeness_centrality(G_scc, distance="weight")
scc_in = dict(G_scc.in_degree(weight="weight"))
scc_out = dict(G_scc.out_degree(weight="weight"))

# reuse top() but point it at G_scc
def top_scc(d, k=10):
    ranked = sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
    names = nx.get_node_attributes(G_scc, "name")
    lines = []
    for node, val in ranked:
        label = names.get(node, node)
        lines.append(f"  {label:>30s}  {val:.6f}" if isinstance(val, float)
                     else f"  {label:>30s}  {val}")
    return "\n".join(lines)

print("Giant SCC — Top 10 betweenness:")
print(top_scc(scc_betw))
print()
print("Giant SCC — Top 10 PageRank:")
print(top_scc(scc_pr))
print()
print("Giant SCC — Top 10 closeness:")
print(top_scc(scc_close))
print()
print("Giant SCC — Top 10 weighted in-degree:")
print(top_scc(scc_in))
print()
print("Giant SCC — Top 10 weighted out-degree:")
print(top_scc(scc_out))

# save giant SCC
scc_path = os.path.join(GRAPH_DIR, f"region_{TARGET}_giant_scc.graphml")
nx.write_graphml(G_scc, scc_path)
print(f"\nGiant SCC saved to: {scc_path}")

# turn grapglml into Gephi CSVs for giant SCC
from fruitfly import GEPHI_DIR, export_gephi
export_gephi(G_scc, f"{TARGET}_giant_scc")
print(f"Giant SCC Gephi CSVs saved to: {os.path.join(GEPHI_DIR, f'{TARGET}_giant_scc')}")