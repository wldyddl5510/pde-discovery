# WSINDy benchmark results

The comparison baseline uses physical-unit MSTLS, the author state scale, and the 181-term archived RD library. Primary benchmarks: KS, NLS, RD. Supplementary: IB, KdV, NS, SG.

- [Comparison schedule and current progress](benchmark_results/authors/results.md): 10 noise levels, 50 trials, 7 PDEs, 3,500 requested trials. Completion status and actual sample counts are in the report.
- [Machine-readable summary](benchmark_results/authors/results.summary.csv)
- [Fixed protocol and library metadata](benchmark_results/authors/results.manifest.json)
- [Execution status](benchmark_results/authors/results.status.json)
- [Provenance of reused trials](benchmark_results/authors/results.reuse.json): deterministic first 50 trials at the selected levels, with the interrupted original run preserved separately.
- [Preflight verification](benchmark_results/preflight_final/results.md): all seven original datasets, four noise levels, one draw per level. This is a validation subset.
- [Previous printed-profile smoke results](benchmark_results/printed_previous_smoke/results.md): archived earlier configuration; not the comparison baseline.

Noise ratios are fixed at `0, 0.05, 0.1, 0.2, 0.225, 0.3, 0.4, 0.5, 0.75, 1.0`. The run updates reports automatically. Raw paired instances are specified by each JSONL record's dataset, noise ratio, seed, and the saved NumPy RNG version. See [README](README.md) for reproduction, resumption, and comparison instructions.
