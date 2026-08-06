# SimulatePhase2

This repository has five separate, restartable stages:

1. `generate_map.py` measures the empirical number of segregating biallelic
   SNV positions (`S`) in each 10 kb window and population.
2. `prepare_demographies.py` makes 1,000 deterministic population-specific
   demographic draws from the checked-in PHLASH MVNs.
3. `run_sim.py` simulates ancestry, generates an excess of candidate
   mutations, and retains exactly the empirical number of unmasked,
   biallelic, segregating sites in every window.
4. `generate_cutoffs.py` reduces the completed tree-sequence nulls to compact
   10 kb recent-coalescence cutoff arrays.
5. `generate_cutoff_gamma_smc.py` optionally runs the empirical Gamma-SMC
   within-individual decoder on every completed null and reduces its posterior
   statistic to matching 10 kb cutoff arrays.

The stored value is therefore the **raw, sample-count- and callability-specific
segregating-site count `S`**, not conventional population-genetic theta and not
an inferred per-base mutation probability. No normalization is applied. The
HDF5 records each population's `a_n = sum(1/i, i=1..2N-1)`, allowing a later
Watterson-style estimate `theta_W = S/a_n` or approximate sample-size rescaling
`S_target = S_full * a_n,target/a_n,full`. The latter is approximate when sites
have missing genotypes. Mutation probabilities passed to msprime are proposal
rates used only to create enough candidate sites.

## Installation

Linux, Python 3.10+, `bcftools` with the `+fill-tags` plugin, `awk`, and
`gcloud` (for the default controlled `gs://` inputs) are required.

```bash
git clone git@github.com:soisa001/SimulatePhase2.git
cd SimulatePhase2
uv sync --frozen
```

## Generate the empirical 10 kb map

The production defaults are:

- BCFs:
  `gs://rw-long-reads-transfer-2026-06-17/v9/lrWGS/panel/panel/panel_bubble_split_vcf/aou_lr_phase2_v1.{chrom}.bubble.split.bcf`
- chromosomes: `chr1` through `chr22`
- populations: `AFR, EUR, AMR, SAS, MID, EAS`
- ancestry label: `ancestry_pred_other`
- ancestry table:
  `gs://vwb-aou-datasets-controlled/v9/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv`
- joint-callset exclusions:
  `gs://vwb-aou-datasets-controlled/v9/wgs/short_read/snpindel/aux/qc/flagged_samples.tsv`
- relatedness exclusions:
  `gs://vwb-aou-datasets-controlled/v9/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv`
- hard mask:
  `gs://rw-migration-aou-rw-fa99430f/hardmask.hg38.v4.over99.bed`
- BCF filters retained: `PASS,.`
- minimum per-population genotype call rate: `0.0` (literal any-called
  segregating-site count)
- window size: 10,000 bp
- retained samples: every QC-gated diploid sample available in each population
- optional capped-sample seed: 42, using stable SHA256 ranking
- four chromosome jobs, with two bcftools threads per job

The first BCF header supplies the ordered starting panel. Samples missing from
the ancestry table, labelled OTH, jointly flagged, or related are excluded.
Every remaining sample is retained by default, in BCF panel order, and must
occur in every requested BCF. The exact IDs are persisted in
`mutation_map_work/sample_manifest.tsv` and embedded in the HDF5. To build an
explicit sensitivity map with a fixed cap, use for example
`--samples-per-population 224`; capped selection uses deterministic SHA256 rank
with seed 42 and then restores BCF panel order. An already gated
`sample_id<TAB>population` file supplied with `--sample-manifest` is used exactly
and is never capped.

By default, a population contributes a site whenever `0 < AC < AN`, even if
some selected genotypes are missing. To require a called-allele fraction, add
for example `--min-call-rate 0.99`; the site must then also have
`AN >= ceil(2 * retained diploid samples * 0.99)`. The HDF5, JSON sidecar, and
per-chromosome QC record the expected AN, the applied minimum AN, and how many
otherwise-segregating positions were excluded for low AN in each population.

```bash
uv run python -u generate_map.py \
  --output "$HOME/snv_theta_map.10kb.h5" \
  --work-dir "$HOME/snv_theta_map_work" \
  --cache-dir "$HOME/snv_theta_bcf_cache"
```

Add `--delete-localized` to bound disk usage to the BCFs currently being
processed, at the cost of redownloading them if a completed chromosome must be
rebuilt. Per-chromosome `.npz` checkpoints make ordinary reruns resumable.
The work directory and final artifact are advisory-locked, so accidentally
starting the same launcher twice cannot mix manifests or publish a partial map.

To upload the single compact artifact and its JSON/SHA256 sidecars:

```bash
uv run python -u generate_map.py \
  --output "$HOME/snv_theta_map.10kb.h5" \
  --work-dir "$HOME/snv_theta_map_work" \
  --cache-dir "$HOME/snv_theta_bcf_cache" \
  --upload gs://YOUR_BUCKET/simulate_phase2_maps
```

The final HDF5 is deliberately small. Each chromosome contains one matrix
`S[population, window]`, normally `uint16` because a 10 kb window cannot contain
more than 10,000 distinct SNV positions. A custom window large enough to exceed
the type is safely promoted to `uint32`. Every matrix uses HDF5 gzip level 6,
shuffle, Fletcher32 checksums, and population-major chunks, so one population
row can be read without inflating the other rows. Root metadata records the
window size, total number of windows, population order, sample counts and
Watterson `a_n`, plus the hard-mask source and SHA256. Each chromosome records
its length and window count; starts and ends are implicit from those values.

Window coordinates, callable-base vectors, and mask intervals are intentionally
not duplicated in the final HDF5. `run_sim.py` verifies the recorded mask by
SHA256 and regenerates them at runtime. Resumable generation checkpoints remain
compressed `.npz` files and contain the temporary geometry needed to validate
interrupted work before final assembly. Source BCF fingerprints and QC summaries
remain in the HDF5/JSON provenance.

The checked-in `mvn/mutation_rate_map_perpop_all.h5` is a legacy 20 kb file
without the v2 provenance or compact matrix contract. `run_sim.py`
intentionally rejects it; regenerate the map with `generate_map.py`.

## Run calibrated simulations

The MVNs generated by `soisa001/phlash_ld` use schema
`phlash.aou.log-ne-mvn/v1`: one 10,000-point geometric time grid from 100 to
40,000 generations, one mean log-Ne vector, and a low-rank covariance factor
estimated from the 100 posterior-median bootstrap curves. Materialize the
seeded cache independently if desired. The checked-in `mvn/AFR.npz` through
`mvn/SAS.npz` are accompanied by their
original PHLASH JSON provenance sidecars and a consolidated
`mvn/validation_report.json`. Reproduce the committed validation before drawing
demographies with:

```bash
uv run python -u validate_mvn_artifacts.py --mvn-dir mvn
```

The committed `mvn/plots` figures summarize the same 1,000 deterministic MVN
draws used by simulation preparation. They show the pointwise sample median
and central 95% interval, use the exact 100--40,000-generation support with no
x-axis padding, and fix the diploid-Ne axis at 1,000--80,000. Regenerate all
per-population and combined PNG/PDF figures plus their hash manifest with:

```bash
uv run python -u plot_mvn_summary.py \
  --mvn-dir mvn --n-draws 1000 --base-seed 42
```

The bundle layout, exact low-rank fit, lack of MVN shrinkage, and source-PKL
sizes are documented in [`mvn/README.md`](mvn/README.md). The 100 source
posterior-median bootstrap curves per population are retained inside each NPZ,
which permits a direct empirical-reference check. Reproduce the committed
signed quantile-error plot (PNG and PDF) and its JSON report with the same
seeded draw path used by `run_sim.py`:

```bash
uv run --frozen python -u evaluate_mvn_error.py \
  --mvn-dir mvn --n-draws 1000 --base-seed 42
```

This comparison calls the stored bootstrap curves an empirical reference, not
known biological truth. The report separately records the exact analytic MVN
marginal error and the finite 1,000-draw simulation error.

Materialize the seeded simulation cache with:

```bash
uv run python -u prepare_demographies.py \
  --mvn-dir mvn \
  --demography-cache /scratch.global/soisa001/sims/demographies \
  --n-sims 1000 --demography-epochs 10000 --base-seed 42
```

The simulation runner calls the same cache function, so this phase is optional;
compatible cached files are reused by content key. The key includes the MVN
SHA256, population, NumPy version, number of draws, number of epochs, and seed.
The cache is advisory-locked and atomically published.

```bash
uv run python -u run_sim.py \
  --map "$HOME/snv_theta_map.10kb.h5" \
  --mask "$HOME/hardmask.hg38.v4.over99.bed" \
  --map-snapshot-dir "$HOME/simulate_phase2_map_snapshots" \
  --mvn-dir mvn \
  --demography-cache /scratch.global/soisa001/sims/demographies \
  --sim-dir /scratch.global/soisa001/sims
```

Before inspecting or hashing the HDF5, the runner copies it once into a local
content-addressed snapshot while computing SHA256 in the same read pass. The
snapshot is named `snv_theta_map.<sha256>.h5`; validation, preload, signatures,
and every worker use only that immutable path. Atomic create-once publication
is safe when two launchers start together, and a source file that changes
during copying is rejected. By default snapshots live under
`<demography-cache>/map_snapshots`; use `--map-snapshot-dir` to place this small
local cache explicitly.

`--mask` accepts a local BED/BED.gz file or a `gs://` URI. If omitted, the
runner uses the source URI/path recorded in the map. In either case, its SHA256
must exactly match the map contract, preventing a different mask from silently
changing callable bases. Cloud masks are downloaded once into the
content-addressed `<demography-cache>/mask_snapshots` cache (or
`--mask-cache-dir`).

Before starting the worker pool, the launcher also creates
`simulation_contract.json` at the simulation root under a cross-process lock.
It fixes the algorithm and software versions, map SHA256, rates, retry limit,
and base seed, plus each population's demography key and diploid sample count.
Compatible launchers may run any existing population subset or atomically add
new populations; a global mismatch or conflicting definition of an existing
population is rejected before simulation. A directory containing `.tsz`
outputs but no contract manifest is never adopted silently—use a new
`--sim-dir` (or explicitly reconcile the old outputs) instead.

Simulation defaults are:

- 1,000 simulations per population
- four process workers, with at most eight submitted tasks in memory
- population sample counts read from the map (the full, population-specific
  QC-gated counts for a map generated with defaults); use
  `--samples-per-population 224` to request a smaller common simulation count
- recombination rate `1e-8` per bp per generation
- initial candidate mutation rate `5e-8`
- retry candidate mutation rate `1e-7`, restricted to deficient windows
- at most eight retry draws, always at `1e-7`
- all 10,000 demographic epochs per current PHLASH MVN draw (or the complete
  available grid for shorter legacy artifacts)
- deterministic base seed 42
- output root `/scratch.global/soisa001/sims`, with units under each
  `/scratch.global/soisa001/sims/<pop>/` directory

`--demography-epochs` can explicitly coarsen the grid. A representative local
ancestry benchmark with 224 diploids and a 10 Mb region took 0.90 seconds at
10,000 epochs versus 0.75 seconds at 1,000 epochs; at 1 Mb the corresponding
times were 0.128 and 0.032 seconds. Because chromosome-scale ancestry and exact
mutation calibration dominate production runtime, the default keeps all 10,000
points. Treat these measurements as local development evidence, not an HPC
runtime estimate.

The runner retains useful candidates from the first draw, excludes every
masked or recurrent site, prevents retry collisions, and writes `.tsz` files
atomically. Before publication, tskit independently verifies that every site
is biallelic and segregating and that the complete per-window vector equals
the requested `S` vector. Exhausted retries are errors and never produce a
completed artifact.

At simulation time the runner computes the effective Watterson density in each
window as `theta_W / callable_bp = S / (a_n,map * callable_bp)`.

With the default sample count, the target vector is exactly the stored raw `S`.
If `--samples-per-population N` is nonzero, it deterministically rescales each
window to `round_half_up(S * a_n,N / a_n,map)`. The map sample count, simulation
sample count, both `a_n` values, scale, callable-base total, raw-S total, target
total, and effective-density summary are recorded in the simulation contract
and completion sidecar. This is a Watterson approximation when source sites
have missing genotypes; the original counts remain unchanged in the map.

Exact thinning intentionally conditions each simulated window on the observed
segregating-site count. This preserves the candidate mutations' conditional
frequency distribution but removes ordinary between-replicate Poisson variance
in the count itself. Use an unconditioned mutation-rate model instead if that
count variance is part of the scientific target.

Each `.tsz` contains one compact custom provenance record in production, even
when every target window has zero sites. It stores the complete signed unit
contract, the ancestry, initial-mutation, thinning, and per-retry mutation
seeds, and the realized thinning/retry outcome. Native msprime ancestry and
mutation provenance is disabled so a full demographic model is not repeated
inside every chromosome. A fixed provenance timestamp makes fresh reruns
exactly equal after decompression, including provenance; `.tsz` container bytes
can still differ because the ZIP/Zarr container records its own metadata.

The small `.tsz.json` completion manifest contains the same map, demography,
seed, rate, and sample-count signature. Normal resume checks these signatures
without decompressing every output. It also opens each TSZip ZIP central
directory, so an interrupted/truncated archive is regenerated even if a stale
sidecar happens to exist. Add `--verify-existing` for a full exact-count
revalidation of existing files.

To audit all expected units explicitly before a reduction phase:

```bash
uv run python -u check_sim_completeness.py \
  --sim-dir /scratch.global/soisa001/sims \
  --pops AFR,EUR,AMR,SAS,MID,EAS --chroms 1-22 --n-sims 1000
```

## Generate compact 10 kb TMRCA cutoffs

After all 1,000 tree sequences are complete, reduce them without persisting a
simulation-by-window matrix:

```bash
uv run python -u generate_cutoffs.py \
  --sim-dir /scratch.global/soisa001/sims \
  --pops AFR,EUR,AMR,SAS,MID,EAS --chroms 1-22 \
  --n-sims 1000 --window-size 10000 \
  --threshold-years 4500 --generation-time 25 \
  --p-values 0.01,0.05 --workers 4
```

Each population gets
`/scratch.global/soisa001/sims/<pop>/tmrca_cutoffs.10kb.h5`. For each chromosome
the file stores starts, ends, the 0.01 and 0.05 upper-tail cutoffs, null mean,
SD, minimum, and maximum. Compression, Fletcher32 checksums, the population
simulation contract, and a digest of the 1,000 completion manifests are stored
with the result. Completed chromosome groups are reused on restart.

The p-value contract is the conservative Monte Carlo rule
`(1 + count(null >= observed)) / (R + 1)`. The stored cutoff is tie-safe and an
observation is significant only when `observed > cutoff`. With 1,000 nulls, the
0.01 and 0.05 cutoffs are respectively the 10th and 50th largest null values.

The current compact statistic is explicitly **tree truth**: within each 10 kb
window, tskit computes the span-averaged fraction of all unordered haploid
sample pairs whose local TMRCA is below 180 generations. An empirical
Gamma-SMC value is a mean posterior probability and includes inference error;
do not call the truth cutoff a calibrated empirical Gamma-SMC cutoff without a
matched decode of the simulated data or a separate validation showing that the
two statistics are interchangeable. The HDF5 records this distinction in
`source_kind` and `statistic` metadata.

## Generate empirical-method Gamma-SMC cutoffs

Gamma-SMC consumes the simulation `.tsz` files directly with
`--input-format tsz`; its native entry point loads them with `tszip.load` and
uses the diploid individuals already stored by msprime. No persistent VCF copy
is written by this workflow.

```bash
uv run python -u generate_cutoff_gamma_smc.py \
  --sim-dir /scratch.global/soisa001/sims \
  --pops AFR,EUR,AMR,SAS,MID,EAS --chroms 1-22 --n-sims 1000 \
  --hardmask "$HOME/hardmask.hg38.v4.over99.bed" \
  --gamma-smc-repo "$HOME/gamma_smc_ts" \
  --p-values 0.01,0.05 --decode-workers 4 --decode-threads 1
```

The default decoder contract is the empirical scan contract: fixed
`theta=0.00075`, `rho/theta=0.8`, `mu=1.29e-8`, 4,500 years at 25 years per
generation, `--only_within`, `--recent-call mean`, no heterozygous-site output,
and one output position every 10 kb. The statistic is
`mean_p_tmrca_lt_threshold`, the mean posterior probability across one homolog
pair per diploid. The simulation hardmask contains excluded intervals, whereas
Gamma-SMC expects callable intervals; the script verifies the hardmask SHA256
against `simulation_contract.json` and writes its per-chromosome complement.

Each successful decode is immediately reduced from TSV to a compressed
restart profile. Once a chromosome's HDF5 group is atomically complete, those
profiles are deleted by default; use `--keep-profiles` for diagnostics. A
restart skips compatible chromosome groups or reuses profiles left by an
interrupted chromosome. The final files are
`<sim-dir>/<pop>/gamma_smc_cutoffs.10kb.h5` and record the binary/interface
hashes, decoder parameters, mask provenance, null summaries, and conservative
plus-one Monte Carlo cutoffs.

This is substantially more expensive than the tree-truth compact reducer:
1,000 whole-autosome decodes for each of six populations means 132,000 decoder
invocations. The overall runner therefore defaults to `compact`; use
`--cutoff-mode gamma-smc` or `both` intentionally.

## Run all phases

`run_full_simulation.py` writes a separate timestamped provenance log per phase
under `<sim-dir>/logs`. Every phase is independently callable and idempotent;
simulation is followed by the quick completeness audit, and a cutoff-only run
performs the audit before reading any nulls.

```bash
uv run python -u run_full_simulation.py --phase demography \
  --mvn-dir mvn --sim-dir /scratch.global/soisa001/sims
uv run python -u run_full_simulation.py --phase simulate \
  --map "$HOME/snv_theta_map.10kb.h5" \
  --mask "$HOME/hardmask.hg38.v4.over99.bed" \
  --mvn-dir mvn --sim-dir /scratch.global/soisa001/sims
uv run python -u run_full_simulation.py --phase cutoffs \
  --cutoff-mode compact --sim-dir /scratch.global/soisa001/sims
```

Use `--phase all` for the same sequence in one invocation. For empirical-method
cutoffs, set `--cutoff-mode gamma-smc --gamma-smc-repo "$HOME/gamma_smc_ts"`
and pass the same localized `--mask` used for simulation.

## Local and Slurm launcher

`launch_simulation.py` resolves an explicit test or full profile and then runs
the same phased workflow either directly or as one single-node Slurm job. Its
defaults are deliberately safe:

- `--profile test --mode local`: EUR only, 100 simulations, chromosomes 1--22,
  compact tree-truth plus matched Gamma-SMC cutoffs, and a deep TSZip check;
- `--profile full --mode local`: all six populations and 1,000 simulations,
  with quick sidecar/ZIP checks before both cutoff reducers;
- `--mode slurm`: submit rather than execute, using partition `sioux`, 50
  allocated CPUs, 384 GB RAM, 300 GB local temporary storage, 32 simulation
  workers, and 32 one-thread Gamma-SMC decoders by default.

Local remains the default for both profiles. The test profile itself is also
the CLI default, so the smallest complete end-to-end command is:

```bash
uv run --frozen python -u launch_simulation.py \
  --map "$HOME/snv_theta_map.10kb.h5" \
  --mask "$HOME/hardmask.hg38.v4.over99.bed" \
  --gamma-smc-repo "$HOME/gamma_smc_ts"
```

Run the full workflow locally by adding `--profile full`. Submit that full
profile to Slurm with:

```bash
uv run --frozen python -u launch_simulation.py \
  --profile full --mode slurm \
  --map "$HOME/snv_theta_map.10kb.h5" \
  --mask "$HOME/hardmask.hg38.v4.over99.bed" \
  --gamma-smc-repo "$HOME/gamma_smc_ts"
```

The Slurm launcher writes a content-keyed command contract and job script under
`<sim-dir>/launches`, then records the returned job ID. Repeating an identical
submission while that job remains queued or running does not submit a duplicate.
Use `--dry-run` to print the fully resolved command without requiring inputs or
creating output, and override resources with `--cpus`, `--mem`, `--tmp`,
`--time`, `--workers`, or `--decode-workers`. Additional `sbatch` arguments can
be repeated as `--slurm-extra=...`.

Both cutoff HDF5 files store the requested p-value levels at the root and a
`len(p_value) x n_windows` cutoff matrix for each chromosome. The full
simulation-by-window null matrix is transient by design. For 100 nulls, the
minimum plus-one Monte Carlo p-value is `1/101 = 0.00990099`; consequently the
0.01 cutoff is the largest null value and the 0.05 cutoff is the fifth largest.

## Plot completed simulations

The sanity plotter reads the same map schema and any completed simulation
files; it does not assume 20 kb windows or old v8 population sizes.

```bash
uv run python -u plot_sim_sanity.py \
  --h5 "$HOME/snv_theta_map.10kb.h5" \
  --sim-dir /path/to/persistent/sims \
  --n-sims 10 --workers 4
```

It writes per-population target-versus-realized, diversity, Tajima's D, and
folded-SFS plots plus a cross-population summary under `sim_sanity_plots/`.
