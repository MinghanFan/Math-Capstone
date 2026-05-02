import pandas as pd
import networkx as nx

# =========================
# File paths
# =========================

root = "/Users/floyd/PycharmProjects/479capstone/"

#Input files
#original edge file
EDGE_FILE = root + "data/fly_edges.csv"
#gephi file
GEPHI_NODE_FILE = root + "data/fly_gephi.csv"
#closeness list
OUTPUT_ALL = root + "fly_result/fly_RWC_all.csv"
# Classic centrality stats
CLASSIC_STATS_FILE = root + "fly_result/fly_classic.csv"
# Random-walk betweenness
RW_BETWEENNESS_FILE = root + "fly_result/fly_RWB.csv"


# Outputs files
MERGED_OUTPUT = root + "fly_result/fly_ALL.csv"
COMMUNITY_SUMMARY_OUTPUT = root + "community/fly_community.csv"


# =========================
# Load graph
# =========================

def load_weighted_directed_graph(path):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # If the edge file has no Weight column, automatically set weight = 1
    if "Weight" not in df.columns:
        print("No Weight column found. Automatically setting all edge weights to 1.")
        df["Weight"] = 1

    # Keep required columns
    df = df[["Source", "Target", "Weight"]].dropna(subset=["Source", "Target"])

    # Remove self-loops
    df = df[df["Source"] != df["Target"]]

    # Convert Weight to numeric
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    # If any Weight values are missing or invalid, set them to 1
    df["Weight"] = df["Weight"].fillna(1)

    # Keep only positive weights
    df = df[df["Weight"] > 0]

    G = nx.from_pandas_edgelist(
        df,
        source="Source",
        target="Target",
        edge_attr="Weight",
        create_using=nx.DiGraph()
    )

    return G


# =========================
# Merge all data
# =========================

def merge_all_data():
    rwc_all = pd.read_csv(OUTPUT_ALL)
    gephi_nodes = pd.read_csv(GEPHI_NODE_FILE)
    classic = pd.read_csv(CLASSIC_STATS_FILE)
    rw_btw = pd.read_csv(RW_BETWEENNESS_FILE)

    # Clean column names
    rwc_all.columns = rwc_all.columns.str.strip()
    gephi_nodes.columns = gephi_nodes.columns.str.strip()
    classic.columns = classic.columns.str.strip()
    rw_btw.columns = rw_btw.columns.str.strip()

    print("RWC columns:")
    print(rwc_all.columns.tolist())

    print("\nGephi columns:")
    print(gephi_nodes.columns.tolist())

    print("\nClassic stats columns:")
    print(classic.columns.tolist())

    print("\nRandom-walk betweenness columns:")
    print(rw_btw.columns.tolist())

    # Rename classic stats
    classic = classic.rename(columns={
        "node": "Id",
        "name": "classic_name",
        "in_closeness": "normal_in_closeness",
        "out_closeness": "normal_out_closeness",
        "betweenness": "normal_betweenness"
    })

    # Rename random-walk betweenness
    rw_btw = rw_btw.rename(columns={
        "node": "Id",
        "name": "rw_betweenness_name"
    })

    # Keep useful columns
    gephi_nodes = gephi_nodes[["Id", "Label", "modularity_class"]]

    classic = classic[
        [
            "Id",
            "classic_name",
            "normal_in_closeness",
            "normal_out_closeness",
            "normal_betweenness"
        ]
    ]

    rw_btw = rw_btw[
        [
            "Id",
            "rw_betweenness_name",
            "rw_betweenness"
        ]
    ]

    # Make sure Id types match
    rwc_all["Id"] = rwc_all["Id"].astype(str)
    gephi_nodes["Id"] = gephi_nodes["Id"].astype(str)
    classic["Id"] = classic["Id"].astype(str)
    rw_btw["Id"] = rw_btw["Id"].astype(str)

    # Merge
    merged = pd.merge(
        rwc_all,
        gephi_nodes,
        on="Id",
        how="left"
    )

    merged = pd.merge(
        merged,
        classic,
        on="Id",
        how="left"
    )

    merged = pd.merge(
        merged,
        rw_btw,
        on="Id",
        how="left"
    )

    merged.to_csv(MERGED_OUTPUT, index=False)

    print("\nMerged file saved to:")
    print(MERGED_OUTPUT)

    return merged


# =========================
# Top node community analysis
# =========================

def analyze_top_nodes(df, score_col, top_k=10):
    valid_df = df.dropna(subset=[score_col]).copy()
    top_nodes = valid_df.sort_values(score_col, ascending=False).head(top_k)

    print("\n====================================")
    print(f"Top {top_k} nodes by {score_col}")
    print("====================================")

    display_cols = ["Id", score_col, "modularity_class"]

    if "Label" in top_nodes.columns:
        display_cols.insert(1, "Label")

    if "classic_name" in top_nodes.columns:
        display_cols.insert(2, "classic_name")

    print(top_nodes[display_cols])

    print("\nCommunity distribution:")
    comm_counts = top_nodes["modularity_class"].value_counts()
    print(comm_counts)

    dominant_comm = comm_counts.idxmax()
    concentration_ratio = comm_counts.iloc[0] / top_k

    print("\nDominant community:", dominant_comm)
    print("Concentration ratio:", concentration_ratio)

    return top_nodes, dominant_comm, concentration_ratio


# =========================
# Community structure analysis
# =========================

def analyze_community_structure(G, df, comm_id):
    node_to_comm = dict(zip(df["Id"].astype(str), df["modularity_class"]))

    S = set(
        node for node in G.nodes()
        if str(node) in node_to_comm and node_to_comm[str(node)] == comm_id
    )

    subG = G.subgraph(S).copy()
    subG_undirected = subG.to_undirected()

    internal_edges = 0
    outgoing_cut_edges = 0
    incoming_cut_edges = 0

    for u, v in G.edges():
        u_in = u in S
        v_in = v in S

        if u_in and v_in:
            internal_edges += 1
        elif u_in and not v_in:
            outgoing_cut_edges += 1
        elif not u_in and v_in:
            incoming_cut_edges += 1

    total_out_volume = sum(G.out_degree(n) for n in S)
    total_in_volume = sum(G.in_degree(n) for n in S)

    out_escape_ratio = (
        outgoing_cut_edges / total_out_volume
        if total_out_volume > 0
        else float("nan")
    )

    in_escape_ratio = (
        incoming_cut_edges / total_in_volume
        if total_in_volume > 0
        else float("nan")
    )

    directed_density = nx.density(subG)
    undirected_density = nx.density(subG_undirected)

    summary = {
        "community": comm_id,
        "community_size": len(S),
        "internal_directed_edges": internal_edges,
        "outgoing_cut_edges": outgoing_cut_edges,
        "incoming_cut_edges": incoming_cut_edges,
        "total_out_volume": total_out_volume,
        "total_in_volume": total_in_volume,
        "out_escape_ratio": out_escape_ratio,
        "in_escape_ratio": in_escape_ratio,
        "directed_density": directed_density,
        "undirected_density": undirected_density
    }

    print("\n====================================")
    print(f"Community {comm_id} structure")
    print("====================================")

    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary


# =========================
# Analyze one metric
# =========================

def analyze_one_metric(G, df, score_col, top_k=10):
    top_nodes, dominant_comm, concentration = analyze_top_nodes(
        df,
        score_col=score_col,
        top_k=top_k
    )

    comm_summary = analyze_community_structure(
        G,
        df,
        dominant_comm
    )

    return {
        "type": score_col,
        "dominant_community": dominant_comm,
        "top10_concentration_ratio": concentration,
        **comm_summary
    }


# =========================
# Main
# =========================

def main():
    G = load_weighted_directed_graph(EDGE_FILE)

    print("Graph loaded.")
    print("Nodes:", G.number_of_nodes())
    print("Directed weighted edges:", G.number_of_edges())

    df = merge_all_data()

    metrics = [
        "rw_out_closeness",
        "rw_in_closeness",
        "normal_out_closeness",
        "normal_in_closeness",
        "normal_betweenness",
        "rw_betweenness"
    ]

    summaries = []

    for metric in metrics:
        if metric not in df.columns:
            print(f"\nWarning: {metric} not found. Skipping.")
            continue

        summary = analyze_one_metric(
            G,
            df,
            score_col=metric,
            top_k=10
        )

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(COMMUNITY_SUMMARY_OUTPUT, index=False)

    print("\n====================================")
    print("Final comparison")
    print("====================================")

    final_cols = [
        "type",
        "dominant_community",
        "top10_concentration_ratio",
        "community_size",
        "directed_density",
        "undirected_density",
        "out_escape_ratio",
        "in_escape_ratio"
    ]

    print(summary_df[final_cols])

    print("\nCommunity summary saved to:")
    print(COMMUNITY_SUMMARY_OUTPUT)

    print("\nAnalysis finished.")


if __name__ == "__main__":
    main()