import random
import networkx as nx
import pandas as pd

# files
root = "/Users/minghanfan/Desktop/math479/Capstone/"
EDGE_FILE = root + "fruitflygephi/FB_giant_scc/edges.csv"

OUTPUT_OUT_TOP10 = root + "fruitflyrw/weighted_directed_rw_out_closeness_top10_FB_scc.csv"
OUTPUT_IN_TOP10 = root + "fruitflyrw/weighted_directed_rw_in_closeness_top10_FB_scc.csv"
OUTPUT_ALL = root + "fruitflyrw/weighted_directed_rw_closeness_all_FB_scc.csv"
# settings
SEED = 13
NUM_TARGETS = 10
NUM_SOURCES = 10
NUM_TRIALS = 5
MAX_STEPS = 50000

rng = random.Random(SEED)


def load_weighted_directed_graph(path, component_type="strong"):
    df = pd.read_csv(path)

    # if not Weight column, add default weight of 1
    if "Weight" not in df.columns:
        df["Weight"] = 1

    df = df[["Source", "Target", "Weight"]].dropna()
    df = df[df["Source"] != df["Target"]]
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Weight"])
    df = df[df["Weight"] > 0]

    G = nx.from_pandas_edgelist(
        df,
        source="Source",
        target="Target",
        edge_attr="Weight",
        create_using=nx.DiGraph()
    )

    if component_type == "strong":
        largest_comp = max(nx.strongly_connected_components(G), key=len)
        print("Using largest strongly connected component.")
    elif component_type == "weak":
        largest_comp = max(nx.weakly_connected_components(G), key=len)
        print("Using largest weakly connected component.")
    else:
        raise ValueError("component_type must be either 'strong' or 'weak'.")

    return G.subgraph(largest_comp).copy()


def build_weighted_successors(G):
    """
    For each node u, store outgoing neighbors and their transition weights.

    Example:
        if u -> a has weight 3
        if u -> b has weight 7

        then walker chooses:
            a with probability 3 / 10
            b with probability 7 / 10
    """
    successors = {}

    for node in G.nodes():
        nbrs = []
        weights = []

        for nbr in G.successors(node):
            w = G[node][nbr].get("Weight", 1)
            nbrs.append(nbr)
            weights.append(w)

        successors[node] = (nbrs, weights)

    return successors


def hitting_time_weighted_directed(successors, source, target, max_steps, rng):
    if source == target:
        return 0, False, False

    current = source

    for step in range(1, max_steps + 1):
        nbrs, weights = successors[current]

        if len(nbrs) == 0:
            return max_steps, True, True

        current = rng.choices(nbrs, weights=weights, k=1)[0]

        if current == target:
            return step, False, False

    return max_steps, True, False


def compute_rw_out_closeness(G, num_targets, num_trials, max_steps, rng):
    """
    Random-walk out-closeness:

    For node r:
        start from r,
        randomly sample target nodes,
        measure how quickly walks from r hit those targets.
    """
    nodes = list(G.nodes())
    successors = build_weighted_successors(G)
    results = []

    total_runs = 0
    truncated_count = 0
    stuck_count = 0

    for i, source in enumerate(nodes, start=1):
        possible_targets = [node for node in nodes if node != source]
        targets = rng.sample(possible_targets, min(num_targets, len(possible_targets)))

        total_steps = 0
        source_runs = 0
        source_truncated = 0
        source_stuck = 0

        for target in targets:
            for _ in range(num_trials):
                steps, was_truncated, was_stuck = hitting_time_weighted_directed(
                    successors, source, target, max_steps, rng
                )

                total_steps += steps
                source_runs += 1
                total_runs += 1

                if was_truncated:
                    source_truncated += 1
                    truncated_count += 1

                if was_stuck:
                    source_stuck += 1
                    stuck_count += 1

        avg_hitting_time = total_steps / source_runs
        rw_out_closeness = 1 / avg_hitting_time if avg_hitting_time > 0 else float("inf")

        results.append({
            "Id": source,
            "avg_out_hitting_time": avg_hitting_time,
            "rw_out_closeness": rw_out_closeness,
            "out_truncated_runs": source_truncated,
            "out_stuck_runs": source_stuck,
            "out_total_runs": source_runs,
            "out_truncation_rate": source_truncated / source_runs
        })

        if i % 50 == 0 or i == len(nodes):
            print(f"Out-closeness: {i}/{len(nodes)} done")

    result_df = (
        pd.DataFrame(results)
        .sort_values("rw_out_closeness", ascending=False)
        .reset_index(drop=True)
    )

    return result_df, truncated_count, stuck_count, total_runs


def compute_rw_in_closeness(G, num_sources, num_trials, max_steps, rng):
    """
    Random-walk in-closeness:

    For node r:
        randomly sample source nodes,
        start from those source nodes,
        measure how quickly walks hit r.
    """
    nodes = list(G.nodes())
    successors = build_weighted_successors(G)
    results = []

    total_runs = 0
    truncated_count = 0
    stuck_count = 0

    for i, target in enumerate(nodes, start=1):
        possible_sources = [node for node in nodes if node != target]
        sources = rng.sample(possible_sources, min(num_sources, len(possible_sources)))

        total_steps = 0
        target_runs = 0
        target_truncated = 0
        target_stuck = 0

        for source in sources:
            for _ in range(num_trials):
                steps, was_truncated, was_stuck = hitting_time_weighted_directed(
                    successors, source, target, max_steps, rng
                )

                total_steps += steps
                target_runs += 1
                total_runs += 1

                if was_truncated:
                    target_truncated += 1
                    truncated_count += 1

                if was_stuck:
                    target_stuck += 1
                    stuck_count += 1

        avg_hitting_time = total_steps / target_runs
        rw_in_closeness = 1 / avg_hitting_time if avg_hitting_time > 0 else float("inf")

        results.append({
            "Id": target,
            "avg_in_hitting_time": avg_hitting_time,
            "rw_in_closeness": rw_in_closeness,
            "in_truncated_runs": target_truncated,
            "in_stuck_runs": target_stuck,
            "in_total_runs": target_runs,
            "in_truncation_rate": target_truncated / target_runs
        })

        if i % 50 == 0 or i == len(nodes):
            print(f"In-closeness: {i}/{len(nodes)} done")

    result_df = (
        pd.DataFrame(results)
        .sort_values("rw_in_closeness", ascending=False)
        .reset_index(drop=True)
    )

    return result_df, truncated_count, stuck_count, total_runs


def main():
    G = load_weighted_directed_graph(EDGE_FILE, component_type="strong")

    print("Nodes:", G.number_of_nodes())
    print("Directed weighted edges:", G.number_of_edges())

    out_result, out_truncated, out_stuck, out_total = compute_rw_out_closeness(
        G,
        NUM_TARGETS,
        NUM_TRIALS,
        MAX_STEPS,
        rng
    )

    in_result, in_truncated, in_stuck, in_total = compute_rw_in_closeness(
        G,
        NUM_SOURCES,
        NUM_TRIALS,
        MAX_STEPS,
        rng
    )

    out_top10 = out_result.head(10)
    in_top10 = in_result.head(10)

    out_top10.to_csv(OUTPUT_OUT_TOP10, index=False)
    in_top10.to_csv(OUTPUT_IN_TOP10, index=False)

    merged = pd.merge(
        out_result,
        in_result,
        on="Id",
        how="outer"
    )

    merged.to_csv(OUTPUT_ALL, index=False)

    print("\nTop 10 weighted directed random-walk OUT-closeness:")
    print(out_top10)

    print("\nTop 10 weighted directed random-walk IN-closeness:")
    print(in_top10)

    print(f"\nSaved out-closeness top 10 to: {OUTPUT_OUT_TOP10}")
    print(f"Saved in-closeness top 10 to: {OUTPUT_IN_TOP10}")
    print(f"Saved all results to: {OUTPUT_ALL}")

    print("\nOUT-closeness simulation quality:")
    print(f"Truncated walks: {out_truncated}/{out_total}")
    print(f"Truncation rate: {out_truncated / out_total:.4%}")
    print(f"Stuck walks: {out_stuck}/{out_total}")
    print(f"Stuck rate: {out_stuck / out_total:.4%}")

    print("\nIN-closeness simulation quality:")
    print(f"Truncated walks: {in_truncated}/{in_total}")
    print(f"Truncation rate: {in_truncated / in_total:.4%}")
    print(f"Stuck walks: {in_stuck}/{in_total}")
    print(f"Stuck rate: {in_stuck / in_total:.4%}")


if __name__ == "__main__":
    main()
