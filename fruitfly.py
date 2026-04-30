"""
Build a single-region neuron graph from FlyWire FAFB data and export Gephi CSVs.
"""

import os, csv
import pandas as pd
import networkx as nx

# settings
ROOT = "/Users/minghanfan/Desktop/math479/Capstone"
DATA_DIR = os.path.join(ROOT, "fruitflydata")
GRAPH_DIR = os.path.join(ROOT, "fruitflygraph")
GEPHI_DIR = os.path.join(ROOT, "fruitflygephi")

TARGET  = "FB" #"GNG"
MIN_SYN = 1

for d in [GRAPH_DIR, GEPHI_DIR]:
    os.makedirs(d, exist_ok=True)


# load & merge
def load_data():
    edges = pd.read_csv(os.path.join(DATA_DIR, "connections_princeton.csv.gz"))
    nodes = (
        pd.read_csv(os.path.join(DATA_DIR, "names.csv.gz"))
        .merge(pd.read_csv(os.path.join(DATA_DIR, "classification.csv.gz")),
               on="root_id", how="left")
        .merge(pd.read_csv(os.path.join(DATA_DIR, "consolidated_cell_types.csv.gz")),
               on="root_id", how="left")
        .merge(pd.read_csv(os.path.join(DATA_DIR, "neurons.csv.gz"))[["root_id", "nt_type", "nt_type_score"]],
               on="root_id", how="left")
    )
    for col in ["flow", "super_class", "class", "sub_class",
                 "primary_type", "nt_type", "side", "hemilineage"]:
        if col in nodes.columns:
            nodes[col] = nodes[col].fillna("unknown")
    return edges, nodes


# build graph
def build_region_graph(edges, nodes, target=TARGET, min_syn=MIN_SYN):
    agg = (
        edges[edges["neuropil"] == target]
        .groupby(["pre_root_id", "post_root_id"])["syn_count"]
        .sum().reset_index()
    )
    agg = agg[agg["syn_count"] >= min_syn]

    G = nx.DiGraph()
    node_attrs = nodes.set_index("root_id")

    for nid in set(agg["pre_root_id"]) | set(agg["post_root_id"]):
        attrs = {}
        if nid in node_attrs.index:
            row = node_attrs.loc[nid]
            attrs = {
                "name": str(row.get("name", "")),
                "super_class": str(row.get("super_class", "")),
                "cell_class": str(row.get("class", "")),
                "primary_type": str(row.get("primary_type", "")),
                "nt_type": str(row.get("nt_type", "")),
                "side": str(row.get("side", "")),
            }
        G.add_node(nid, **attrs)

    for _, r in agg.iterrows():
        G.add_edge(r["pre_root_id"], r["post_root_id"], weight=int(r["syn_count"]))

    return G


# export Gephi CSVs
def export_gephi(G, label):
    out = os.path.join(GEPHI_DIR, label)
    os.makedirs(out, exist_ok=True)

    # nodes
    attrs = sorted({k for _, d in G.nodes(data=True) for k in d})
    with open(os.path.join(out, "nodes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Label"] + attrs)
        for node, data in G.nodes(data=True):
            w.writerow([node, data.get("name", str(node))] + [data.get(a, "") for a in attrs])

    # edges
    eattrs = sorted({k for _, _, d in G.edges(data=True) for k in d} - {"weight"})
    with open(os.path.join(out, "edges.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source", "Target", "Type", "Weight"] + eattrs)
        for src, tgt, data in G.edges(data=True):
            w.writerow([src, tgt, "Directed", data.get("weight", 1)]
                       + [data.get(a, "") for a in eattrs])


# main
if __name__ == "__main__":
    edges, nodes = load_data()
    G = build_region_graph(edges, nodes)

    path = os.path.join(GRAPH_DIR, f"region_{TARGET}_graph.graphml")
    nx.write_graphml(G, path)

    print(f"{TARGET} region: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, "
          f"density {nx.density(G):.6f}")

    export_gephi(G, f"region_{TARGET}")