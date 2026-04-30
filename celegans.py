import pandas as pd
import networkx as nx

ROOT = "/Users/minghanfan/Desktop/math479/Capstone"

# read edges
edges = pd.read_csv(
    f"{ROOT}/celegansdata/C-elegans-frontal.txt",
    sep=" ",
    header=None,
    names=["Source", "Target"]
)
edges["Type"] = "Directed"
edges.to_csv(f"{ROOT}/celegansgephi/elegans_edges.csv", index=False)

# read nodes
nodes = pd.read_csv(f"{ROOT}/celegansdata/C-elegans-frontal-meta.csv")
nodes_gephi = nodes.rename(columns={
    "node_id": "Id",
    "name": "Label",
    "posx": "x",
    "posy": "y"
})
nodes_gephi = nodes_gephi[["Id", "Label", "x", "y"]]
nodes_gephi.to_csv(f"{ROOT}/celegansgephi/elegans_nodes.csv", index=False)

# build directed graph
G = nx.from_pandas_edgelist(edges, source="Source", target="Target",
                            create_using=nx.DiGraph())

# extract largest strongly connected component
largest_scc = max(nx.strongly_connected_components(G), key=len)
G_scc = G.subgraph(largest_scc).copy()

# add node attributes
for _, row in nodes_gephi.iterrows():
    n = row["Id"]
    if n in G_scc:
        G_scc.nodes[n]["label"] = row["Label"]
        G_scc.nodes[n]["x"] = row["x"]
        G_scc.nodes[n]["y"] = row["y"]

# filter edges and nodes to SCC
scc_nodes = set(largest_scc)
edges_scc = edges[edges["Source"].isin(scc_nodes) & edges["Target"].isin(scc_nodes)]
nodes_scc = nodes_gephi[nodes_gephi["Id"].isin(scc_nodes)]

edges_scc.to_csv(f"{ROOT}/celegansgephi/elegans_scc_edges.csv", index=False)
nodes_scc.to_csv(f"{ROOT}/celegansgephi/elegans_scc_nodes.csv", index=False)
nx.write_graphml(G_scc, f"{ROOT}/celegansgraph/elegans_scc.graphml")

print(edges.head())
print(nodes_gephi.head())
print(f"original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"largest SCC: {G_scc.number_of_nodes()} nodes, {G_scc.number_of_edges()} edges")