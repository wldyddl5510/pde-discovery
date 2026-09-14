"""Rebuild the compact paper comparison from saved trials: python report.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment import load_csv, write_tables


def main():
    root = Path(__file__).resolve().parent / "results"
    reference = load_csv(root / "wsindy_paper_reference/trials.csv")
    comparison = load_csv(root / "paper_method_comparison/trials.csv")
    methods = sorted({r["method"] for r in comparison},
                     key=lambda m: ("MLE" in m, m != "WSINDy", "L1" in m, float(m.split("=")[-1]) if "=" in m else 0))
    problems = {"burgers": "Burgers", "kdv": "KdV"}
    lines = ["# PDE recovery", "",
             "Author datasets; Burgers 256×256, KdV 400×601; 43 candidate terms.",
             "Layout follows [WSINDy Table 5 and Figure 6](https://arxiv.org/html/2007.02848v3#S5.T5).", "",
             "**Noiseless coefficient error E∞** (relative error, not %).", "",
             "| Method | Burgers | KdV |", "|---|---:|---:|",
             "| WSINDy paper, Table 5 | 4.3e-5 | 3.1e-7 |"]
    for method in methods:
        values = [np.mean([r["coefficient_linf"] for r in comparison
                           if r["method"] == method and r["problem"] == problem and r["noise"] == 0])
                  for problem in problems]
        lines.append(f"| {method} | {values[0]:.3g} | {values[1]:.3g} |")
    lines += ["", "**Coefficient error versus noise — WSINDy**", "",
              "![WSINDy coefficient errors versus noise](coefficient_error.png)", "",
              "Means over 200 seeds per noise level. Only WSINDy has a full noise sweep here;",
              "these are our measurements, not digitized Figure 6 values.", "",
              "**Method comparison at 20% noise** · 3 paired seeds per PDE.", "",
              "Support = exact recoveries; E₂ = mean relative error; time = median seconds;",
              "optimizer = successful terminations. L1 labels give α in λ=αλ_ref.", ""]
    for problem, title in problems.items():
        lines += [f"**{title}**", "", "| Method | Support | E₂ | Time (s) | Optimizer |",
                  "|---|---:|---:|---:|---:|"]
        for method in methods:
            rows = [r for r in comparison if r["problem"] == problem and r["method"] == method and r["noise"] == .2]
            support = sum(r["support_exact"] for r in rows)
            error = np.mean([r["coefficient_error"] for r in rows])
            seconds = np.median([r["total_seconds"] for r in rows])
            success = sum(r["optimizer_success"] for r in rows)
            lines.append(f"| {method} | {support}/{len(rows)} | {error:.3g} | {seconds:.4g} | {success}/{len(rows)} |")
        lines.append("")
    lines += ["E₂ = ‖ŵ−w★‖₂/‖w★‖₂; E∞ = maximum relative error on true nonzero terms.",
              "Failed fits remain included (`inf` = nonfinite error); runtime measures termination, not correct recovery.",
              "Three seeds give a small sample: KdV WSINDy support is 2/3 here versus 198/200 in the noise sweep.", "",
              "WENDy HT/L1 are PDE extensions using WSINDy test functions; HT knows the true term count.",
              "Fits use at most 300 iterations, tol=1e-8, and known σ for WENDy variants.",
              "Rescaling follows author code (exponent −1/5), which differs from printed Eq. 4.8 (−1/6).", "",
              "Runtime: Apple M4, Python; FFT/covariance up to 4 threads, BLAS 1. Data generation and file writing excluded.",
              "Paper Table 4 reports 0.12 s / 0.39 s for Burgers / KdV; our single-thread WSINDy medians",
              "at 20% noise are 0.0396 s / 0.133 s (200 seeds). Hardware/language differ.", "",
              "Full tables: [method comparison](paper_method_comparison/summary.md) ·",
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
