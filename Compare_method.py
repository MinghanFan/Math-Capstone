import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


# =========================
# 1. Load data
# =========================

INPUT_CSV = "/Users/floyd/PycharmProjects/479capstone/fly_result/fly_ALL.csv"   # 改成你的 merged csv 文件名
df = pd.read_csv(INPUT_CSV)


# =========================
# 2. Centrality columns
# =========================

centrality_cols = [
    "rw_in_closeness",
    "rw_out_closeness",
    "rw_betweenness",
    "normal_in_closeness",
    "normal_out_closeness",
    "normal_betweenness"
]


# =========================
# 3. Compute all pairwise correlations
# =========================

results = []

for col1, col2 in combinations(centrality_cols, 2):
    data = df[[col1, col2]].dropna().copy()

    pearson_r = data[col1].corr(data[col2], method="pearson")
    spearman_rho = data[col1].corr(data[col2], method="spearman")

    results.append({
        "centrality_1": col1,
        "centrality_2": col2,
        "pearson_r": pearson_r,
        "abs_pearson_r": abs(pearson_r),
        "spearman_rho": spearman_rho,
        "abs_spearman_rho": abs(spearman_rho),
        "n_nodes": len(data)
    })


corr_df = pd.DataFrame(results)


# =========================
# 4. Sort by Pearson and Spearman
# =========================

pearson_ranked = corr_df.sort_values("abs_pearson_r", ascending=False)
spearman_ranked = corr_df.sort_values("abs_spearman_rho", ascending=False)


print("\n==============================")
print("Top pairs by absolute Pearson r")
print("==============================")
print(pearson_ranked[[
    "centrality_1",
    "centrality_2",
    "pearson_r",
    "spearman_rho",
    "n_nodes"
]].to_string(index=False))


print("\n==============================")
print("Top pairs by absolute Spearman rho")
print("==============================")
print(spearman_ranked[[
    "centrality_1",
    "centrality_2",
    "pearson_r",
    "spearman_rho",
    "n_nodes"
]].to_string(index=False))


# =========================
# 5. Save correlation ranking tables
# =========================

corr_df.to_csv("all_15_centrality_pair_correlations.csv", index=False)
pearson_ranked.to_csv("centrality_pairs_ranked_by_pearson.csv", index=False)
spearman_ranked.to_csv("centrality_pairs_ranked_by_spearman.csv", index=False)

print("\nSaved:")
print("all_15_centrality_pair_correlations.csv")
print("centrality_pairs_ranked_by_pearson.csv")
print("centrality_pairs_ranked_by_spearman.csv")