# SimulatePhase2

This repository has two separate, explicit stages:

1. `generate_map.py` measures the empirical number of segregating biallelic
   SNV positions (`theta`) in each 10 kb window and population.
2. `run_sim.py` simulates ancestry, generates an excess of candidate
   mutations, and retains exactly the empirical number of unmasked,
   biallelic, segregating sites in every window.

`theta` is therefore a **sample-count- and callability-specific target count**,
not conventional population-genetic theta and not an inferred per-base mutation
probability. The mutation probabilities passed to msprime are proposal rates
used only to create enough candidate sites.

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
- retained samples: 224 QC-gated diploid samples per population
- sample-selection seed: 42, using stable SHA256 ranking
- four chromosome jobs, with two bcftools threads per job

The first BCF header supplies the ordered starting panel. Samples missing from
the ancestry table, labelled OTH, jointly flagged, or related are excluded.
From the remaining samples, the default build selects exactly 224 diploids per
population using a deterministic SHA256 rank and then restores their BCF panel
order. Every selected ID must occur in every requested BCF. The selected IDs
are persisted in `mutation_map_work/sample_manifest.tsv` and embedded in the
HDF5. Use `--samples-per-population 0` for the complete QC-gated panel, or
supply an already gated `sample_id<TAB>population` manifest with
`--sample-manifest`; an explicit manifest is used exactly and is not capped.

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

The HDF5 embeds the exact ordered population samples, sample hashes/counts,
window coordinates, callable-base counts, normalized hard-mask intervals,
source/header fingerprints, and QC totals. Counts and callable bases are
compressed `uint16` at the default 10 kb resolution (and safely promote to
`uint32` for unusually wide custom windows), so the whole-autosome map is only
one compressed, MB-scale artifact rather than a many-file Zarr directory. GCS
generation, size, and available CRC/MD5 metadata are part of cache and
checkpoint identities.

The checked-in `mvn/mutation_rate_map_perpop_all.h5` is a legacy 20 kb file
without provenance or an embedded mask. `run_sim.py` intentionally rejects it;
regenerate the map with `generate_map.py`.

## Run calibrated simulations

```bash
uv run python -u run_sim.py \
  --map "$HOME/snv_theta_map.10kb.h5" \
  --map-snapshot-dir "$HOME/simulate_phase2_map_snapshots" \
  --mvn-dir mvn \
  --demography-cache "$HOME/simulate_phase2_demographies" \
  --sim-dir /path/to/persistent/sims
```

Before inspecting or hashing the HDF5, the runner copies it once into a local
content-addressed snapshot while computing SHA256 in the same read pass. The
snapshot is named `snv_theta_map.<sha256>.h5`; validation, preload, signatures,
and every worker use only that immutable path. Atomic create-once publication
is safe when two launchers start together, and a source file that changes
during copying is rejected. By default snapshots live under
`<demography-cache>/map_snapshots`; use `--map-snapshot-dir` to place this small
local cache explicitly.

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
- population sample counts read from the map (224 diploids per population for
  a map generated with defaults)
- recombination rate `1e-8` per bp per generation
- initial candidate mutation rate `5e-8`
- retry candidate mutation rate `1e-7`, restricted to deficient windows
- at most eight retry draws, always at `1e-7`
- all 1,000 demographic epochs per legacy MVN draw (or the complete available
  grid for shorter artifacts)
- deterministic base seed 42

`--demography-epochs` can explicitly coarsen very long grids; 1,000 is the
correctness-first default because the checked-in legacy MVNs contain sharp
changes that are visibly distorted at 64 epochs.

The runner retains useful candidates from the first draw, excludes every
masked or recurrent site, prevents retry collisions, and writes `.tsz` files
atomically. Before publication, tskit independently verifies that every site
is biallelic and segregating and that the complete per-window vector equals
the empirical `theta` vector. Exhausted retries are errors and never produce a
completed artifact.

With the default 224 diploids, each simulated population has 448 haplotypes and
`C(448, 2) = 100,128` unordered haplotype pairs. This makes exhaustive
Gamma-SMC pairing practical. If within-individual haplotype pairs are removed,
99,904 pairs remain. For a matched simulation p-value, run the observed scan on
the same persisted 224-person manifest and use the same exhaustive pairing rule
in observed and simulated data; 100,128 pair values are correlated comparisons,
not independent replicates.

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
without decompressing every output; add `--verify-existing` for a full
exact-count revalidation of existing files.

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
