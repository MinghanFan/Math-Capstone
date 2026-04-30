"""
Random walk (current-flow) betweenness centrality for directed networks.
Converts to undirected before computing.
"""

import networkx as nx
import csv

ROOT = "/Users/minghanfan/Desktop/math479/Capstone/"
SEED = 13

networks = {
    "celegans": {
        "path": ROOT + "celegansgraph/elegans_scc.graphml",
        "out": ROOT + "celegansrw/rw_betweenness_celegans_scc.csv",
        "weighted": False,
    },
    "fruitfly": {
        "path": ROOT + "fruitflygraph/region_FB_giant_scc.graphml",
        "out": ROOT + "fruitflyrw/rw_betweenness_fruitfly_scc.csv",
        "weighted": True,
    },
}

for name, cfg in networks.items():
    print(f"\n=== {name} ===")
    G = nx.read_graphml(cfg["path"])

    # convert directed to undirected
    U = G.to_undirected()

    # ensure single connected component
    largest_cc = max(nx.connected_components(U), key=len)
    U = U.subgraph(largest_cc).copy()

    print(f"Nodes: {U.number_of_nodes()}, Edges: {U.number_of_edges()}")

    rw_betw = nx.approximate_current_flow_betweenness_centrality(
        U,
        normalized=True,
        weight="weight" if cfg["weighted"] else None,
        solver="lu",
        epsilon=0.1,
        kmax=10000,
        seed=SEED,
    )

    # sort and save
    ranked = sorted(rw_betw.items(), key=lambda x: x[1], reverse=True)

    with open(cfg["out"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "name", "rw_betweenness"])
        for node, val in ranked:
            label = G.nodes[node].get("name", G.nodes[node].get("label", str(node)))
            w.writerow([node, label, round(val, 6)])

    print("Top 10:")
    for node, val in ranked[:10]:
        label = G.nodes[node].get("name", G.nodes[node].get("label", str(node))) 
        print(f"  {node}: {label:>30s}  {val:.6f}")

    print(f"Saved to {cfg['out']}")