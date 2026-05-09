import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression


# =========================
# 1. Load data
# =========================

INPUT_CSV = "/Users/floyd/PycharmProjects/479capstone/fly_result/fly_ALL.csv"   # 改成你的文件名
df = pd.read_csv(INPUT_CSV)


# =========================
# 2. Columns to compare
# =========================

pairs = [
    ("normal_in_closeness", "rw_in_closeness"),
    ("normal_out_closeness", "rw_out_closeness"),
    ("normal_betweenness", "rw_betweenness"),

    # Optional cross-comparisons
    ("normal_out_closeness", "rw_in_closeness"),
    ("normal_in_closeness", "rw_out_closeness"),
]


# =========================
# 3. Function for scatter plot
# =========================

def plot_correlation(df, x_col, y_col, label_col="Label", output_prefix="correlation"):
    # Keep only valid numeric rows
    data = df[[x_col, y_col, label_col]].dropna().copy()

    x = data[x_col].values.reshape(-1, 1)
    y = data[y_col].values

    # Pearson and Spearman correlation
    pearson_r, pearson_p = pearsonr(data[x_col], data[y_col])
    spearman_rho, spearman_p = spearmanr(data[x_col], data[y_col])

    # Linear regression line
    model = LinearRegression()
    model.fit(x, y)

    x_line = np.linspace(data[x_col].min(), data[x_col].max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(data[x_col], data[y_col], alpha=0.75)
    plt.plot(x_line, y_line, linewidth=2)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(
        f"{y_col} vs {x_col}\n"
        f"Pearson r = {pearson_r:.3f}, Spearman rho = {spearman_rho:.3f}"
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = f"{output_prefix}_{x_col}_vs_{y_col}.png"
    plt.savefig(output_file, dpi=300)
    plt.show()

    print("=" * 70)
    print(f"Comparison: {y_col} vs {x_col}")
    print(f"Pearson r      = {pearson_r:.4f}, p = {pearson_p:.4g}")
    print(f"Spearman rho   = {spearman_rho:.4f}, p = {spearman_p:.4g}")
    print(f"Linear model   = {y_col} = {model.coef_[0]:.6g} * {x_col} + {model.intercept_:.6g}")
    print(f"Saved plot to  = {output_file}")


# =========================
# 4. Run all comparisons
# =========================

for x_col, y_col in pairs:
    plot_correlation(df, x_col, y_col)