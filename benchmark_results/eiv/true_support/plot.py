"""Plot the paired support diagnostic after run_study.py completes."""
import json
import os
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(DIRECTORY/".mplcache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    summaries = json.loads((DIRECTORY/"summary.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    methods = [("full", "Full dictionary EIV", "#606060"),
               ("oracle", "True-support EIV", "#d76328"),
               ("ols", "True-support OLS", "#2878b5")]
    colors = ["#2878b5", "#d76328", "#438747"]
    for column, name in enumerate(("IB", "KdV", "KS")):
        rows = sorted((r for r in summaries if r["name"] == name), key=lambda r: r["noise_ratio"])
        noise = [r["noise_ratio"] for r in rows]
        top, bottom = axes[:, column]
        for method, label, color in methods:
            top.semilogy(noise, [max(r[method]["median_e2"], 1e-12) for r in rows],
                         color=color, marker="o", markersize=4, label=label)
        top.set(title=name, xlabel="Noise ratio", ylabel="Median relative coefficient error E2")
        top.grid(alpha=.2)
        top.legend(fontsize=8)
        for term, (label, color) in enumerate(zip(rows[0]["terms"], colors)):
            for method, style in (("oracle", "-"), ("ols", "--")):
                bottom.plot(noise, [r[method]["mean_coefficient_ratios"][term] for r in rows],
                            color=color, linestyle=style, marker="o", markersize=3,
                            label=f"{label} ({'EIV' if method == 'oracle' else 'OLS'})")
        bottom.axhline(1, color="#777777", linewidth=.8)
        bottom.axhline(0, color="#777777", linewidth=.5)
        bottom.set(xlabel="Noise ratio", ylabel="Mean estimated / true coefficient")
        bottom.grid(alpha=.2)
        bottom.legend(fontsize=7)
    fig.suptitle("EIV with the true support supplied: paired observations, 100 trials per level\n"
                 "EIV selects lambda on the original grid; OLS uses the same corrected design", fontsize=12)
    fig.savefig(DIRECTORY/"diagnostics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
