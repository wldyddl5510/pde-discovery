"""Rebuild the compact paper comparison from saved trials: python report.py."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment import comparison_tables, load_csv, write_tables


def main():
    root = Path(__file__).resolve().parent / "results"
    reference = load_csv(root / "wsindy_paper_reference/trials.csv")
    comparison = load_csv(root / "paper_method_comparison/trials.csv")
    config = json.loads((root / "paper_method_comparison/config.json").read_text())
    expected = sum(1 if m == "WSINDy" else 1+len(config["l1"]) for m in config["methods"]) * len(config["problems"]) * sum(
        config["seeds"] if noise else 1 for noise in config["noise"])
    problems = {"burgers": "Burgers", "kdv": "KdV"}
    lines = ["# PDE recovery", "",
             "Burgers 256×256; KdV 400×601; 43 candidate terms. Planned repeats: 1 at 0% noise, 3 at 20% noise.",
             "W = WENDy; MLE = WENDy-MLE. W:α / MLE:α use λ=αλ_ref; HT keeps the known number of terms.",
             "Errors are mean relative errors, not %. Support / optimizer are counts; time is median seconds.", ""]
    lines += comparison_tables(comparison, paper=True)
    if len(comparison) < expected:
        lines.insert(2, f"**Saved fits: {len(comparison)}/{expected}. The smaller-λ sweep is incomplete; cells use saved fits only. — means unavailable.**\n")
    lines += ["E₂ = ‖ŵ−w★‖₂/‖w★‖₂; E∞ = max relative error on true nonzero terms.",
              "Failed fits remain included; inf = nonfinite error. Optimizer completion does not imply recovery.",
              "[WSINDy Table 5](https://arxiv.org/html/2007.02848v3#S5.T5): noiseless E∞ = 4.3e-5 (Burgers), 3.1e-7 (KdV).", "",
              "![WSINDy coefficient errors versus noise](coefficient_error.png)", "",
              "WSINDy noise sweep: mean errors over 200 seeds per positive noise level, following Figure 6.",
              "WENDy variants were tested only at 0% and 20%; the figure shows our WSINDy measurements.", "",
              "Same PDE extensions and settings: known σ, maxiter=300, tol=1e-8; full covariance.",
              "Runtime: Apple M4 / Python, FFT/covariance up to 4 threads, BLAS 1; includes failed fits, excludes data generation and file writing.",
              "Rescaling follows author code (−1/5), differing from printed Eq. 4.8 (−1/6).", "",
              "Data: [method comparison](paper_method_comparison/summary.md) ·",
              "[WSINDy sweep](wsindy_paper_reference/summary.md) · [legacy baseline](selection_comparison/summary.md).", ""]
    (root / "paper_comparison.md").write_text("\n".join(lines))

    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
        for ax, field, label in zip(axes, ["coefficient_linf", "coefficient_error"], [r"$E_\infty$", r"$E_2$"]):
            for (problem, title), color, marker in zip(problems.items(), ["#0072B2", "#D55E00"], ["o", "s"]):
                rows = [r for r in reference if r["problem"] == problem and r["noise"] > 0]
                noise = sorted({r["noise"] for r in rows})
                errors = [np.mean([r[field] for r in rows if r["noise"] == n]) for n in noise]
                ax.loglog(100*np.array(noise), errors, color=color, marker=marker, linewidth=1.5, label=title)
            ax.set(xlabel=r"Noise $\sigma/\mathrm{RMS}(u)$ (%)", ylabel=f"Mean relative error {label}")
            ax.set_xticks([10, 20, 50, 100], ["10", "20", "50", "100"])
            ax.minorticks_off()
            ax.grid(axis="y", alpha=.2)
        axes[0].legend(frameon=False)
        fig.savefig(root / "coefficient_error.png", dpi=200)
        fig.savefig(root / "coefficient_error.pdf")
        plt.close(fig)
    for folder in ("paper_method_comparison", "wsindy_paper_reference", "selection_comparison"):
        out = root / folder
        write_tables(out, load_csv(out / "trials.csv"), paper=folder != "selection_comparison")


if __name__ == "__main__":
    main()
