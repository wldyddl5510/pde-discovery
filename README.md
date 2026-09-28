# pde-discovery

Synthetic-data experiments for PDE coefficient estimation.

- [simulation_generation.py](simulation_generation.py): synthetic PDE data.
- [methods.py](methods.py): SINDy, WSINDy, debiased WSINDy, WENDy, and WENDy-MLE estimators.
- [experiments.py](experiments.py): command-line experiment runner.
- [results.md](results.md): Monte Carlo benchmark settings and error/runtime tables.
- [METHODS.md](METHODS.md): method definitions, parameters, and API examples.

Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

Use `python experiments.py --help` for method and experiment options.
Benchmark setups and reproduction commands are in [results.md](results.md).

Run the adapted Burgers Monte Carlo comparison with `OPENBLAS_NUM_THREADS=1 python experiments.py --instance nonlinear_viscous_burgers --append`.

Run the 2D linear `J=1` Monte Carlo control with `OPENBLAS_NUM_THREADS=1 python experiments.py --instance linear_advection_diffusion --noise-ratios 0 0.1 1 --methods sampled-wsindy-ols sampled-wsindy-mstls sampled-debiased-wsindy-ols sampled-debiased-wsindy-mstls --append`.

Run tests:

```sh
python -m unittest discover -s tests -v
```
