# PHLASH log-Ne MVN bundle

The six population artifacts (`AFR.npz`, `EUR.npz`, `AMR.npz`, `SAS.npz`,
`MID.npz`, and `EAS.npz`) use schema `phlash.aou.log-ne-mvn/v1`. Each artifact
contains the following arrays on the same 10,000-point geometric grid from 100
through 40,000 generations ago:

| array | shape | dtype | meaning |
|---|---:|---|---|
| `time` | `(10000,)` | `float32` | generations ago |
| `mean_log_ne` | `(10000,)` | `float32` | empirical mean of log diploid Ne |
| `covariance_factor` | `(100, 10000)` | `float32` | factor whose transpose-product is the unbiased sample covariance |
| `bootstrap_ne` | `(100, 10000)` | `float32` | retained empirical posterior-median bootstrap curves in diploid Ne units |
| `jitter` | scalar | `float32` | independent log-Ne draw noise; currently exactly zero |

There are also scalar `schema` and `population` identity arrays. The adjacent
JSON file records the artifact hash and size and the path, size, and modification
time for every source fit PKL.

## Exact fit

For one population, let `X` be the `100 x 10000` matrix obtained by evaluating
all posterior models in each PHLASH bootstrap fit on the common time grid,
taking the pointwise posterior median within each fit, and then taking natural
logs. The checked-in model is

```text
mu = mean(X, axis=0)
B  = (X - mu) / sqrt(99)
Sigma = B.T @ B
log Ne draw = mu + z @ B,  z ~ Normal(0, I_100)
```

Thus the ambient MVN has 10,000 dimensions. The mean has 10,000 stored values;
the factor has 1,000,000 stored values, but its rows sum to zero and its rank is
at most 99. A generic rank-99 covariance in 10,000 dimensions has 985,149
degrees of freedom after accounting for factor rotations; adding the mean gives
995,149. These are not 995,149 independently well-estimated scalar effects:
the entire covariance estimate comes from only 100 bootstrap curves and is
singular in the other 9,901 directions.

There is no shrinkage, ridge, taper, PCA truncation, or added diagonal variance
in this MVN fit. `Sigma` is exactly the ordinary unbiased empirical covariance
of log Ne, and `jitter=0`. The upstream PHLASH posterior has its own model and
prior regularization; that is distinct from MVN covariance shrinkage.

“Low-rank Gaussian summary” is more precise than “low-order summary.” The MVN
preserves the empirical first and second moments of log Ne but does not preserve
higher-order shape such as skew, heavy tails, or multimodality. It also reduces
each source fit's full posterior to one pointwise-median curve. Importantly, this
bundle retains those 100 median curves in `bootstrap_ne`, so the empirical
reference used to assess the Gaussian approximation has not been discarded.

## Approximation error

`evaluate_mvn_error.py` uses the same seeded `run_sim.load_mvn_draws` path as
simulation preparation and compares 1,000 drawn curves per population with the
100 stored empirical curves. The plot reports signed relative error for the
pointwise 2.5%, 50%, and 97.5% quantiles. The JSON report also separates exact
analytic MVN marginal error from finite-draw Monte Carlo error and records a
central-distribution log-quantile error, empirical coverage, input hashes, seed,
and source-PKL sizes.

The stored bootstrap curves are an empirical reference, not known biological
truth. Regenerate the committed PNG, PDF, and report with:

```bash
uv run --frozen python -u evaluate_mvn_error.py \
  --mvn-dir mvn --n-draws 1000 --base-seed 42
```

## Original PKLs

The PKLs are not checked in. Their manifests describe 600 files (100 per
population) totaling 125,416,106 bytes (119.61 MiB), so downloading them is not
too large. They preserve the full list of posterior models plus population,
sample IDs, seed, replicate, and fit settings, and enable a stronger comparison
against all posterior curves rather than only the six sets of 100 median curves.
Only unpickle trusted artifacts, and use the matching PHLASH environment because
Python pickle can execute code and model classes must be importable.
