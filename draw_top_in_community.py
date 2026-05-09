import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================
# File paths
# =========================

EDGE_FILE = "/Users/floyd/PycharmProjects/479capstone/data/fly_edges.csv"
NODE_FILE = "/Users/floyd/PycharmProjects/479capstone/fly_result/fly_ALL.csv"

OUTPUT_DIR = "/Users/floyd/PycharmProjects/479capstone/community_figures_2/"


# =========================
# Load graph
# =========================

def load_graph(edge_file):
    edges = pd.read_csv(edge_file)
    edges.columns = edges.columns.str.strip()

    if "Weight" not in edges.columns:
        edges["Weight"] = 1

    edges = edges[["Source", "Target", "Weight"]].dropna()
    edges = edges[edges["Source"] != edges["Target"]]
    edges["Weight"] = pd.to_numeric(edges["Weight"], errors="coerce").fillna(1)
    edges = edges[edges["Weight"] > 0]

    G = nx.from_pandas_edgelist(
        edges,
        source="Source",
        target="Target",
        edge_attr="Weight",
        create_using=nx.DiGraph()
    )

    return G


# =========================
# Helper: rescale node sizes
# =========================

def rescale_sizes(values, min_size=80, max_size=900):
    values = pd.Series(values).fillna(0)

    if values.max() == values.min():
        return [min_size for _ in values]

    scaled = (values - values.min()) / (values.max() - values.min())
    return min_size + scaled * (max_size - min_size)


# =========================
# Draw one centrality figure
# =========================

def draw_centrality_figure(
    G,
    node_df,
    score_col,
    title,
    output_path,
    pos,
    top_k=10,
    color_by_community=True
):
    df = node_df.copy()
    df["Id"] = df["Id"].astype(str)

    # Make sure graph node IDs are strings
    G = nx.relabel_nodes(G, lambda x: str(x))

    # Keep nodes that exist in both graph and table
    df = df[df["Id"].isin(G.nodes())].copy()

    # Top 10 nodes by this centrality
    top_nodes = (
        df.dropna(subset=[score_col])
        .sort_values(score_col, ascending=False)
        .head(top_k)["Id"]
        .tolist()
    )

    top_set = set(top_nodes)

    # Node sizes by centrality score
    score_map = dict(zip(df["Id"], df[score_col]))
    node_sizes = [
        rescale_sizes([score_map.get(n, 0)])[0]
        for n in G.nodes()
    ]

    # The above rescales one at a time, not ideal.
    # Better:
    scores_in_order = [score_map.get(n, 0) for n in G.nodes()]
    node_sizes = rescale_sizes(scores_in_order, min_size=60, max_size=800)

    # Colors
    if color_by_community:
        communities = sorted(df["modularity_class"].dropna().unique())
        cmap = plt.cm.get_cmap("tab10", len(communities))
        comm_to_color = {
            comm: cmap(i)
            for i, comm in enumerate(communities)
        }

        comm_map = dict(zip(df["Id"], df["modularity_class"]))
        node_colors = [
            comm_to_color.get(comm_map.get(n), "lightgray")
            for n in G.nodes()
        ]
    else:
        node_colors = ["white" for _ in G.nodes()]

    # Top-10 nodes get red border
    edge_colors = [
        "red" if n in top_set else "gray"
        for n in G.nodes()
    ]

    linewidths = [
        2.5 if n in top_set else 0.6
        for n in G.nodes()
    ]

    plt.figure(figsize=(12, 8))

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=7,
        width=0.6,
        alpha=0.35,
        edge_color="gray"
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=edge_colors,
        linewidths=linewidths,
        alpha=0.95
    )

    # Add labels only for top 10 nodes
    label_map = dict(zip(df["Id"], df["Label"]))
    top_labels = {
        n: label_map.get(n, n)
        for n in top_set
        if n in G.nodes()
    }

    nx.draw_networkx_labels(
        G,
        pos,
        labels=top_labels,
        font_size=8,
        font_color="black"
    )

    plt.title(title, fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# =========================
# Main
# =========================

def main():
    G = load_graph(EDGE_FILE)
    G = nx.relabel_nodes(G, lambda x: str(x))

    node_df = pd.read_csv(NODE_FILE)
    node_df.columns = node_df.columns.str.strip()
    node_df["Id"] = node_df["Id"].astype(str)

    # Use one fixed layout for all six plots
    # This is important for comparison.
    pos = nx.spring_layout(
        G,
        seed=13,
        k=0.55,
        iterations=200
    )

    metrics = {
        "rw_out_closeness": "Random-walk out-closeness",
        "rw_in_closeness": "Random-walk in-closeness",
        "rw_betweenness": "Random-walk betweenness",
        "normal_out_closeness": "Classical out-closeness",
        "normal_in_closeness": "Classical in-closeness",
        "normal_betweenness": "Classical betweenness"
    }

    for score_col, title in metrics.items():
        output_path = OUTPUT_DIR + f"elegan_{score_col}_top10_community.png"

        draw_centrality_figure(
            G=G,
            node_df=node_df,
            score_col=score_col,
            title=title,
            output_path=output_path,
            pos=pos,
            top_k=10,
            color_by_community=True
        )


if __name__ == "__main__":
    main()

G_nodes = set(str(n) for n in G.nodes())
df_nodes = set(node_df["Id"].astype(str))

missing_nodes = G_nodes - df_nodes

print("Number of graph nodes:", len(G_nodes))
print("Number of nodes in table:", len(df_nodes))
print("Number of graph nodes missing from table:", len(missing_nodes))
print("Missing nodes:", sorted(list(missing_nodes))[:20])