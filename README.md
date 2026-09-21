# pde-discovery

Synthetic-data experiments for PDE coefficient estimation.

- [simulation_generation.py](simulation_generation.py): synthetic PDE data.
- [methods.py](methods.py): SINDy, WSINDy, WENDy, and WENDy-MLE estimators.
- [experiments.py](experiments.py): command-line experiment runner.
- [results.md](results.md): experiment settings, error/runtime tables, and validation results.
- [METHODS.md](METHODS.md): method definitions, parameters, and API examples.

Install and run the default five-method comparison:

```sh
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python experiments.py --max-iter 200000 --output results.md
```

For 3D porous medium, add `--instance anisotropic_porous_medium_3d --strides 4 4 4 1`.
Add `--append` to preserve earlier results. The default five methods exclude WENDy and WENDy-MLE.

Use `python experiments.py --help` for method and experiment options.
WENDy-MLE requires a positive `noise_std`; see [method details](METHODS.md#wendy-mle).
The recorded comparisons and reproduction commands are in [results.md](results.md).

Run tests:

```sh
python -m unittest discover -s tests -v
```
